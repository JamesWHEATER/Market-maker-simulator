from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from chart_renderer import build_candles, write_interactive_candlestick_html


@dataclass(frozen=True)
class RandomMarketConfig:
    steps: int = 10_000
    seed: int = 42
    initial_fair_value: float = 100.0
    fair_value_step_vol: float = 0.03
    base_spread: float = 0.04
    inventory_skew: float = 0.002
    min_order_size: int = 1
    max_order_size: int = 10


@dataclass
class MarketMaker:
    base_spread: float
    inventory_skew: float
    cash: float = 0.0
    inventory: int = 0

    def quote(self, fair_value: float) -> tuple[float, float]:
        reservation_price = fair_value - self.inventory_skew * self.inventory
        half_spread = self.base_spread / 2
        bid = max(0.01, reservation_price - half_spread)
        ask = max(bid + 0.01, reservation_price + half_spread)
        return bid, ask

    def sell_to_taker(self, price: float, size: int) -> None:
        self.cash += price * size
        self.inventory -= size

    def buy_from_taker(self, price: float, size: int) -> None:
        self.cash -= price * size
        self.inventory += size

    def mark_to_market_pnl(self, fair_value: float) -> float:
        return self.cash + self.inventory * fair_value


@dataclass(frozen=True)
class MarketSnapshot:
    step: int
    fair_value: float
    bid: float
    ask: float
    taker_side: str
    order_size: int
    trade_price: float
    maker_inventory: int
    maker_cash: float
    maker_pnl: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid


def simulate_random_market(config: RandomMarketConfig) -> list[MarketSnapshot]:
    rng = random.Random(config.seed)
    market_maker = MarketMaker(
        base_spread=config.base_spread,
        inventory_skew=config.inventory_skew,
    )
    fair_value = config.initial_fair_value
    history: list[MarketSnapshot] = []

    for step in range(1, config.steps + 1):
        fair_value = max(
            0.01,
            fair_value + rng.gauss(0.0, config.fair_value_step_vol),
        )
        bid, ask = market_maker.quote(fair_value)
        taker_side = rng.choice(("buy", "sell"))
        order_size = rng.randint(config.min_order_size, config.max_order_size)

        if taker_side == "buy":
            trade_price = ask
            market_maker.sell_to_taker(trade_price, order_size)
        else:
            trade_price = bid
            market_maker.buy_from_taker(trade_price, order_size)

        history.append(
            MarketSnapshot(
                step=step,
                fair_value=fair_value,
                bid=bid,
                ask=ask,
                taker_side=taker_side,
                order_size=order_size,
                trade_price=trade_price,
                maker_inventory=market_maker.inventory,
                maker_cash=market_maker.cash,
                maker_pnl=market_maker.mark_to_market_pnl(fair_value),
            )
        )

    return history


def summarize(history: list[MarketSnapshot]) -> dict[str, float]:
    if not history:
        return {}

    trade_prices = [snapshot.trade_price for snapshot in history]
    returns = [
        math.log(trade_prices[index] / trade_prices[index - 1])
        for index in range(1, len(trade_prices))
        if trade_prices[index - 1] > 0
    ]
    pnl_path = [snapshot.maker_pnl for snapshot in history]
    peak_pnl = pnl_path[0]
    max_drawdown = 0.0

    for pnl in pnl_path:
        peak_pnl = max(peak_pnl, pnl)
        max_drawdown = max(max_drawdown, peak_pnl - pnl)

    buy_volume = sum(
        snapshot.order_size for snapshot in history if snapshot.taker_side == "buy"
    )
    sell_volume = sum(
        snapshot.order_size for snapshot in history if snapshot.taker_side == "sell"
    )

    return {
        "steps": float(len(history)),
        "final_fair_value": history[-1].fair_value,
        "final_trade_price": history[-1].trade_price,
        "final_maker_inventory": float(history[-1].maker_inventory),
        "final_maker_pnl": history[-1].maker_pnl,
        "average_spread": mean(snapshot.spread for snapshot in history),
        "average_abs_inventory": mean(
            abs(snapshot.maker_inventory) for snapshot in history
        ),
        "realized_volatility_per_step": pstdev(returns) if len(returns) > 1 else 0.0,
        "max_pnl_drawdown": max_drawdown,
        "buy_volume": float(buy_volume),
        "sell_volume": float(sell_volume),
        "total_volume": float(buy_volume + sell_volume),
    }


def write_history_csv(history: list[MarketSnapshot], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "fair_value",
        "bid",
        "ask",
        "spread",
        "taker_side",
        "order_size",
        "trade_price",
        "maker_inventory",
        "maker_cash",
        "maker_pnl",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot in history:
            writer.writerow(
                {
                    "step": snapshot.step,
                    "fair_value": f"{snapshot.fair_value:.6f}",
                    "bid": f"{snapshot.bid:.6f}",
                    "ask": f"{snapshot.ask:.6f}",
                    "spread": f"{snapshot.spread:.6f}",
                    "taker_side": snapshot.taker_side,
                    "order_size": snapshot.order_size,
                    "trade_price": f"{snapshot.trade_price:.6f}",
                    "maker_inventory": snapshot.maker_inventory,
                    "maker_cash": f"{snapshot.maker_cash:.6f}",
                    "maker_pnl": f"{snapshot.maker_pnl:.6f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a baseline Random Market market-making simulation."
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/random_market.csv"))
    parser.add_argument(
        "--chart-output",
        type=Path,
        default=Path("results/random_market_candles.html"),
    )
    parser.add_argument("--visible-candles", type=int, default=160)
    parser.add_argument("--candle-steps", type=int, default=5)
    parser.add_argument("--initial-fair-value", type=float, default=100.0)
    parser.add_argument("--fair-value-step-vol", type=float, default=0.03)
    parser.add_argument("--base-spread", type=float, default=0.04)
    parser.add_argument("--inventory-skew", type=float, default=0.002)
    parser.add_argument("--min-order-size", type=int, default=1)
    parser.add_argument("--max-order-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RandomMarketConfig(
        steps=args.steps,
        seed=args.seed,
        initial_fair_value=args.initial_fair_value,
        fair_value_step_vol=args.fair_value_step_vol,
        base_spread=args.base_spread,
        inventory_skew=args.inventory_skew,
        min_order_size=args.min_order_size,
        max_order_size=args.max_order_size,
    )
    history = simulate_random_market(config)
    write_history_csv(history, args.output)
    candles = build_candles(history, args.candle_steps)
    write_interactive_candlestick_html(
        candles,
        args.chart_output,
        visible_candles=args.visible_candles,
    )
    
    summary = summarize(history)

    print("Random Market simulation complete")
    print(f"CSV written to: {args.output}")
    print(f"Interactive candlestick chart written to: {args.chart_output}")
    print(f"Candles generated: {len(candles)}")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
