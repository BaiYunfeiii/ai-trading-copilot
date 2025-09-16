from __future__ import annotations

from pathlib import Path

from src.MT5DataProvider import get_rates


def main() -> None:
    symbols = ["XAUUSDm", "BTCUSDm", "ETHUSDm"]
    count = 100
    timeframes = ["D1", "H1", "M5"]

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)


    for symbol in symbols:
        for tf in timeframes:
            df = get_rates(symbol=symbol, timeframe=tf, count=count)
            out_path = output_dir / f"{symbol}_{tf}_{count}.csv"
            df.to_csv(out_path, index=True)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


