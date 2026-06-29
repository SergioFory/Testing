"""
Gestión de riesgo y tamaño de posición.

Regla central:
  position_size (USD) = (capital × risk_per_trade) / stop_distance_pct

Donde stop_distance = atr_stop_mult × ATR

Esto asegura que en cada operación SIEMPRE arriesgas exactamente
risk_per_trade % del capital, independientemente del activo o la volatilidad.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from config.settings import RISK


@dataclass
class TradeSetup:
    """Representa una operación lista para ejecutar."""
    symbol:        str
    direction:     str          # "long" | "short"
    setup_type:    str
    entry_price:   float
    stop_loss:     float
    take_profit:   float
    position_size: float        # En USD
    position_qty:  float        # En unidades del activo
    atr:           float
    atr_pct:       float
    risk_usd:      float        # USD en riesgo
    reward_usd:    float        # USD potencial de ganancia
    rr_ratio:      float        # reward / risk
    ml_score:      float        # Confianza del modelo (0-1)
    sentiment_score: Optional[float] = None
    notes:         str = ""

    @property
    def stop_distance_pct(self) -> float:
        return abs(self.entry_price - self.stop_loss) / self.entry_price

    @property
    def target_distance_pct(self) -> float:
        return abs(self.take_profit - self.entry_price) / self.entry_price


def calculate_position_size(
    capital:      float,
    entry_price:  float,
    stop_loss:    float,
    risk_per_trade: float = None,
) -> tuple:
    """
    Calcula el tamaño de posición basado en riesgo fijo.

    Args:
        capital:       Capital total disponible en USD
        entry_price:   Precio de entrada
        stop_loss:     Precio del stop loss
        risk_per_trade: Fracción del capital a arriesgar (default: RISK setting)

    Returns:
        (position_size_usd, position_qty, risk_usd)
    """
    if risk_per_trade is None:
        risk_per_trade = RISK["risk_per_trade"]

    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        logger.error("Stop loss igual al precio de entrada. Posición cancelada.")
        return 0, 0, 0

    risk_usd       = capital * risk_per_trade
    position_size  = risk_usd / (stop_distance / entry_price)
    position_qty   = position_size / entry_price

    # Nunca arriesgar más del 10% del capital en una sola operación (safety net)
    max_position = capital * 0.10
    if position_size > max_position:
        position_size = max_position
        position_qty  = position_size / entry_price
        risk_usd      = position_size * (stop_distance / entry_price)
        logger.warning(
            f"Posición reducida al 10% del capital por safety net: "
            f"${position_size:,.0f}"
        )

    return round(position_size, 2), round(position_qty, 6), round(risk_usd, 2)


def build_trade_setup(
    setup_row:   pd.Series,
    capital:     float,
    ml_score:    float,
    sentiment_score: float = None,
) -> Optional[TradeSetup]:
    """
    Construye un TradeSetup completo desde una fila de setup detectado.

    Args:
        setup_row:   Fila del DataFrame de setups (salida de detect_all_setups)
        capital:     Capital actual en USD
        ml_score:    Score del modelo ML (0-1)
        sentiment_score: Score de sentimiento (-1 a +1)

    Returns:
        TradeSetup o None si el setup no es válido.
    """
    entry      = float(setup_row["close"])
    stop_loss  = float(setup_row["stop_loss"])
    take_profit= float(setup_row["take_profit"])
    atr        = float(setup_row.get("atr", entry * 0.02))
    direction  = setup_row["direction"]

    # Validaciones básicas
    if direction == "long":
        if stop_loss >= entry:
            logger.warning("Stop loss >= entry en LONG. Setup inválido.")
            return None
        if take_profit <= entry:
            logger.warning("Take profit <= entry en LONG. Setup inválido.")
            return None
    else:
        if stop_loss <= entry:
            logger.warning("Stop loss <= entry en SHORT. Setup inválido.")
            return None
        if take_profit >= entry:
            logger.warning("Take profit >= entry en SHORT. Setup inválido.")
            return None

    # Calcular posición
    pos_size, pos_qty, risk_usd = calculate_position_size(
        capital, entry, stop_loss
    )
    if pos_size <= 0:
        return None

    reward_usd = pos_size * abs(take_profit - entry) / entry

    # R:R desde la GEOMETRÍA de precios (target_dist / stop_dist). Es exacto y
    # no depende del redondeo del tamaño de posición. Antes se calculaba como
    # reward_usd / risk_usd, donde el redondeo a centavos de risk_usd (tras el
    # safety net) empujaba el ratio por debajo de 1.5 y:
    #   - rechazaba SIEMPRE a Gold (R:R 1.33) y Silver (R:R 1.0) → nunca emitían
    #   - rechazaba a BTC (R:R 1.5) ~6 de cada 10 ciclos por ruido de redondeo
    # El piso baja a 1.0 (reward ≥ risk): solo descarta geometría degenerada.
    # Cada activo define su R:R real en settings (atr_target_mult / atr_stop_mult).
    stop_dist   = abs(entry - stop_loss)
    target_dist = abs(take_profit - entry)
    rr_ratio    = target_dist / (stop_dist + 1e-9)
    if rr_ratio < 1.0 - 1e-6:
        logger.debug(
            f"R:R {rr_ratio:.2f} < 1.0. Setup descartado por geometría degenerada."
        )
        return None

    return TradeSetup(
        symbol        = setup_row.get("symbol", "UNKNOWN"),
        direction     = direction,
        setup_type    = setup_row.get("setup_type", "unknown"),
        entry_price   = entry,
        stop_loss     = stop_loss,
        take_profit   = take_profit,
        position_size = pos_size,
        position_qty  = pos_qty,
        atr           = atr,
        atr_pct       = atr / entry,
        risk_usd      = risk_usd,
        reward_usd    = reward_usd,
        rr_ratio      = round(rr_ratio, 2),
        ml_score      = ml_score,
        sentiment_score = sentiment_score,
    )


class PortfolioRiskManager:
    """
    Controla el riesgo a nivel de portafolio:
    - Máximo N posiciones simultáneas
    - Máxima pérdida diaria
    - Máximo drawdown
    """

    def __init__(self, capital: float = None):
        self.initial_capital  = capital or RISK["capital_inicial"]
        self.current_capital  = self.initial_capital
        self.peak_capital     = self.initial_capital
        self.daily_pnl        = 0.0
        self.open_positions   = []
        self.closed_trades    = []

    @property
    def drawdown(self) -> float:
        """Drawdown actual desde el pico."""
        return (self.peak_capital - self.current_capital) / self.peak_capital

    @property
    def daily_loss_pct(self) -> float:
        return -self.daily_pnl / (self.current_capital + 1e-9)

    def can_open_trade(self, new_setup: TradeSetup) -> tuple:
        """
        Verifica si es seguro abrir la operación.
        Returns: (bool, reason_string)
        """
        # Max posiciones abiertas
        if len(self.open_positions) >= RISK["max_positions"]:
            return False, f"Máximo de {RISK['max_positions']} posiciones alcanzado"

        # Daily drawdown limit
        if self.daily_loss_pct >= RISK["max_daily_loss"]:
            return False, (
                f"Pérdida diaria {self.daily_loss_pct*100:.1f}% "
                f"supera límite {RISK['max_daily_loss']*100:.0f}%"
            )

        # Max drawdown
        if self.drawdown >= RISK["max_drawdown"]:
            return False, (
                f"Drawdown {self.drawdown*100:.1f}% "
                f"supera límite {RISK['max_drawdown']*100:.0f}%"
            )

        # No operar mismo símbolo en ambas direcciones
        for pos in self.open_positions:
            if pos.symbol == new_setup.symbol and pos.direction != new_setup.direction:
                return False, f"Ya hay posición {pos.direction} abierta en {new_setup.symbol}"

        return True, "OK"

    def open_position(self, setup: TradeSetup) -> None:
        self.open_positions.append(setup)
        logger.info(
            f"NUEVA POSICIÓN | {setup.symbol} {setup.direction.upper()} | "
            f"Entry: {setup.entry_price:,.2f} | "
            f"SL: {setup.stop_loss:,.2f} | TP: {setup.take_profit:,.2f} | "
            f"Size: ${setup.position_size:,.0f} | R:R {setup.rr_ratio:.1f}"
        )

    def close_position(self, symbol: str, exit_price: float) -> Optional[dict]:
        """Cierra una posición y actualiza el capital."""
        for i, pos in enumerate(self.open_positions):
            if pos.symbol == symbol:
                if pos.direction == "long":
                    pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

                pnl_usd = pos.position_size * pnl_pct
                self.current_capital += pnl_usd
                self.daily_pnl       += pnl_usd
                self.peak_capital     = max(self.peak_capital, self.current_capital)

                trade = {
                    "symbol":     symbol,
                    "direction":  pos.direction,
                    "entry":      pos.entry_price,
                    "exit":       exit_price,
                    "pnl_usd":    round(pnl_usd, 2),
                    "pnl_pct":    round(pnl_pct * 100, 2),
                    "outcome":    "win" if pnl_usd > 0 else "loss",
                    "ml_score":   pos.ml_score,
                    "rr_ratio":   pos.rr_ratio,
                }
                self.closed_trades.append(trade)
                self.open_positions.pop(i)
                logger.info(
                    f"POSICIÓN CERRADA | {symbol} | "
                    f"PnL: {'+'if pnl_usd>0 else ''}{pnl_usd:,.2f} USD ({pnl_pct*100:+.2f}%)"
                )
                return trade
        logger.warning(f"No se encontró posición abierta para {symbol}")
        return None

    def reset_daily(self) -> None:
        """Llamar al inicio de cada día."""
        self.daily_pnl = 0.0

    def summary(self) -> dict:
        trades = self.closed_trades
        if not trades:
            return {"n_trades": 0}
        wins  = [t for t in trades if t["outcome"] == "win"]
        total_pnl = sum(t["pnl_usd"] for t in trades)
        return {
            "n_trades":        len(trades),
            "win_rate":        len(wins) / len(trades),
            "total_pnl_usd":   round(total_pnl, 2),
            "total_return_pct":round(total_pnl / self.initial_capital * 100, 2),
            "current_capital": round(self.current_capital, 2),
            "drawdown":        round(self.drawdown * 100, 2),
            "avg_rr":          round(np.mean([t["rr_ratio"] for t in trades]), 2),
        }
