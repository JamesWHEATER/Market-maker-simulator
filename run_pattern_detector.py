import argparse
import csv

from chart_renderer import PriceCandle
from pattern_detector import detect_pattern_universe


parser = argparse.ArgumentParser()
parser.add_argument("csv_file")
args = parser.parse_args()


candles = []

with open(args.csv_file, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        candles.append(
            PriceCandle(
                candle_number=int(row["candle_number"]),
                start_step=int(row["start_step"]),
                end_step=int(row["end_step"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                is_complete=row["is_complete"].lower() == "true",
            )
        )


result = detect_pattern_universe(candles)

print()
print("PATTERN DETECTOR RESULTS")
print("=" * 80)
print(f"Raw candidates:       {len(result.candidates)}")
print(f"Unique causal events: {len(result.events)}")
print()


for i, event in enumerate(result.events, 1):
    print(
        f"{i:>3}. "
        f"{event.pattern_name:<28} "
        f"bars={event.start_index:>4}-{event.end_index:<4} "
        f"available={event.available_at_index:<4} "
        f"execute={event.earliest_execution_index:<4} "
        f"direction={event.expected_direction:<8} "
        f"fit={event.geometry_fit_score:.3f}"
    )