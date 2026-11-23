import argparse
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
FEATURES = ["mnth", "hr", "weekday", "weathersit", "temp", "atemp", "hum", "windspeed"]
VARIABLES = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
             "wind_speed_10m", "weather_code"]
WEATHER_NAMES = {1: "Clear / partly cloudy", 2: "Cloudy / fog",
                 3: "Light/moderate precipitation", 4: "Severe precipitation / storm"}


def weather_category(code):
    """Approximate WMO -> UCI mapping, not an exact equivalence."""
    if code in (0, 1, 2):
        return 1
    if code in (3, 45, 48):
        return 2
    if code in (51, 53, 55, 56, 61, 63, 66, 71, 73, 77, 80, 81, 85):
        return 3
    if code in (57, 65, 67, 75, 82, 86, 95, 96, 99):
        return 4
    raise ValueError(f"Unsupported WMO weather code: {code}")


def fetch_weather(latitude, longitude, days):
    params = dict(latitude=latitude, longitude=longitude, hourly=",".join(VARIABLES),
                  current="temperature_2m", timezone="auto", forecast_days=days,
                  temperature_unit="celsius", wind_speed_unit="kmh")
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    with urlopen(url, timeout=30) as response:
        payload = json.load(response)
    if payload.get("error"):
        raise ValueError(payload.get("reason", "Weather API error"))
    return payload, url


def forecast_rows(payload, extra_holidays=()):
    hourly = payload["hourly"]
    units = payload["hourly_units"]
    for key, expected in [("temperature_2m", "°C"), ("apparent_temperature", "°C"),
                          ("relative_humidity_2m", "%"), ("wind_speed_10m", "km/h")]:
        if units.get(key) != expected:
            raise ValueError(f"Unexpected API unit for {key}: {units.get(key)}")
    frame = pd.DataFrame({key: hourly[key] for key in ["time"] + VARIABLES})
    timestamps = pd.to_datetime(frame["time"])
    current = pd.Timestamp(payload["current"]["time"])
    # Dates and hours are in the API's resolved local timezone, not the PC timezone.
    holidays = set(USFederalHolidayCalendar().holidays(
        start=timestamps.min().normalize(), end=timestamps.max().normalize()).date)
    holidays.update(extra_holidays)
    rows = []
    for i, stamp in enumerate(timestamps):
        if stamp <= current or stamp.hour not in (7, 8) or stamp.weekday() >= 5:
            continue
        if stamp.date() in holidays:
            continue
        source = frame.iloc[i]
        values = [source[key] for key in VARIABLES]
        row = {"local_time": stamp.isoformat(), "temperature_c": source["temperature_2m"],
               "apparent_temperature_c": source["apparent_temperature"],
               "humidity_pct": source["relative_humidity_2m"],
               "wind_kmh": source["wind_speed_10m"], "wmo_code": source["weather_code"]}
        try:
            if any(value is None or not math.isfinite(float(value)) for value in values):
                raise ValueError("Missing or non-finite forecast values")
            if not 0 <= row["humidity_pct"] <= 100 or row["wind_kmh"] < 0:
                raise ValueError("Invalid humidity or wind speed")
            category = weather_category(source["weather_code"])
            row.update(mnth=stamp.month, hr=stamp.hour, weekday=(stamp.weekday() + 1) % 7,
                       weathersit=category, weather=WEATHER_NAMES[category],
                       temp=(row["temperature_c"] + 8) / 47,
                       atemp=(row["apparent_temperature_c"] + 16) / 66,
                       hum=row["humidity_pct"] / 100, windspeed=row["wind_kmh"] / 67,
                       input_issue="")
        except (ValueError, TypeError) as exc:
            row["input_issue"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def train_model(csv_path):
    data = pd.read_csv(csv_path)
    data = data[(data["workingday"] == 1) & data["hr"].isin([7, 8])].copy()
    data = data.sort_values(["dteday", "hr"])
    # Split by whole dates: no test-day rows leak into training.
    dates = data["dteday"].drop_duplicates().to_numpy()
    cutoff = dates[int(len(dates) * 0.8)]
    train, test = data[data["dteday"] < cutoff], data[data["dteday"] >= cutoff]
    surge_threshold = float(train["cnt"].quantile(0.75))
    y_train = (train["cnt"] > surge_threshold).astype(int)
    y_test = (test["cnt"] > surge_threshold).astype(int)
    if y_train.nunique() != 2:
        raise ValueError("Training data must contain both surge and non-surge examples")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    model.fit(train[FEATURES], y_train)
    prediction = model.predict(test[FEATURES])
    probability = model.predict_proba(test[FEATURES])[:, 1]
    metrics = dict(test_start=cutoff, test_rows=len(test),
                   accuracy=accuracy_score(y_test, prediction),
                   f1=f1_score(y_test, prediction, zero_division=0),
                   recall=recall_score(y_test, prediction, zero_division=0),
                   auc=roc_auc_score(y_test, probability) if y_test.nunique() == 2 else None)
    # Keep the training-derived target definition fixed when refitting for inference.
    model.fit(data[FEATURES], (data["cnt"] > surge_threshold).astype(int))
    return model, data, surge_threshold, metrics


def predict_plan(rows, model, history, threshold):
    result = rows.copy()
    result["decision_scope"] = "DEMO_ONLY_NOT_OPERATIONALLY_VALIDATED"
    result["surge_probability"] = np.nan
    result["prepare_more_bikes"] = "REVIEW"
    result["recommendation"] = "Manual review required"
    for index, row in result.iterrows():
        issue = row["input_issue"]
        if not issue:
            if row["weathersit"] == 4:
                issue = "Severe weather: manual safety review; no demand recommendation"
            elif row["weathersit"] not in history["weathersit"].unique():
                issue = "Weather category absent from training data"
            else:
                for feature in ["temp", "atemp", "hum", "windspeed"]:
                    if not history[feature].min() <= row[feature] <= history[feature].max():
                        issue = f"{feature} outside historical range"
                        break
        result.loc[index, "input_issue"] = issue
        if issue:
            continue
        features = pd.DataFrame([{key: row[key] for key in FEATURES}], columns=FEATURES)
        probability = float(model.predict_proba(features)[0, 1])
        prepare = probability >= threshold
        result.loc[index, "surge_probability"] = probability
        result.loc[index, "prepare_more_bikes"] = "YES" if prepare else "NO"
        result.loc[index, "recommendation"] = (
            "Prepare extra availability and redistribution staff" if prepare
            else "Routine checks; no extra preparation signal")
    return result


def plot_plan(plan, threshold, location, local_timezone, output_path, metrics, show=False):
    fig, (ax, table_ax) = plt.subplots(
        2, 1, figsize=(13, max(9, 6 + len(plan) * 0.35)),
        gridspec_kw={"height_ratios": [1.3, 1]},
    )
    positions = np.arange(len(plan))
    colors = ["#dc2626" if action == "YES" else "#0f766e" for action in plan["prepare_more_bikes"]]
    bars = ax.bar(positions, plan["surge_probability"] * 100, color=colors)
    for position, bar, (_, row) in zip(positions, bars, plan.iterrows()):
        if pd.isna(row["surge_probability"]):
            ax.text(position, 4, "REVIEW", rotation=90, ha="center", color="#92400e")
        else:
            ax.text(position, bar.get_height() + 2, f"{bar.get_height():.0f}%", ha="center")
    ax.axhline(threshold * 100, color="#475569", linestyle="--", label="Extra preparation threshold")
    ax.set_xticks(positions, [stamp[5:16].replace("T", "\n") for stamp in plan["local_time"]])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Estimated surge probability (%)")
    ax.set_title(f"Weather-based bike preparation demo: {location}\nLocal time: {local_timezone}")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    table_ax.axis("off")
    table_ax.set_title("Forecast weather and preparation signals", fontsize=12, pad=10)
    table_rows = []
    for _, row in plan.iterrows():
        probability = row["surge_probability"]
        table_rows.append([
            row["local_time"][5:16].replace("T", " "),
            row.get("weather", "Unavailable"),
            f"{row['temperature_c']:.1f}", f"{row['humidity_pct']:.0f}",
            f"{row['wind_kmh']:.1f}",
            "--" if pd.isna(probability) else f"{probability:.0%}",
            {"YES": "Prepare extra", "NO": "Routine checks", "REVIEW": "Manual review"}[
                row["prepare_more_bikes"]],
        ])
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["Local time", "Weather", "Temp (C)", "RH (%)", "Wind (km/h)", "Risk", "Demo action"],
        colWidths=[0.13, 0.28, 0.1, 0.09, 0.12, 0.08, 0.2],
        cellLoc="center", bbox=[0, 0, 1, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor("#164e63")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f1f5f9" if r % 2 else "#e2e8f0")
            if c == 6:
                action = plan.iloc[r - 1]["prepare_more_bikes"]
                cell.set_facecolor({"YES": "#fee2e2", "NO": "#ccfbf1", "REVIEW": "#fef3c7"}[action])
    fig.text(0.02, 0.02, "DEMO ONLY | Red = extra signal; green = no extra signal; REVIEW = no prediction.\n"
             f"Historical chronological test: F1={metrics['f1']:.3f}, recall={metrics['recall']:.1%}. "
             "Not validated for current operations; not bike quantities.", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latitude", type=float, default=38.9072)
    parser.add_argument("--longitude", type=float, default=-77.0369)
    parser.add_argument("--location", default="Washington DC")
    parser.add_argument("--days", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-show", action="store_true", help="Save chart without opening a window")
    parser.add_argument("--holiday", type=date.fromisoformat, action="append", default=[],
                        help="Additional non-working date YYYY-MM-DD; repeat as needed")
    args = parser.parse_args()
    if not (-90 <= args.latitude <= 90 and -180 <= args.longitude <= 180 and 0 < args.threshold < 1):
        parser.error("Invalid coordinates or threshold (must be between 0 and 1)")
    print(f"Location: {args.location} ({args.latitude}, {args.longitude})")
    print("Weekends and US federal holidays excluded; add local closures using --holiday YYYY-MM-DD.")
    try:
        payload, url = fetch_weather(args.latitude, args.longitude, args.days)
        rows = forecast_rows(payload, args.holiday)
        if rows.empty:
            print("No future working-day 07:00/08:00 slots in this forecast. No new files written.")
            return 0
        model, history, surge_threshold, metrics = train_model(BASE_DIR / "hour.csv")
        plan = predict_plan(rows, model, history, args.threshold)
        output_dir = BASE_DIR / "allocation_outputs"
        output_dir.mkdir(exist_ok=True)
        # Distinct run folders prevent accidentally using stale forecasts after an API failure.
        run_dir = output_dir / ("weather_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        run_dir.mkdir()
        plan.to_csv(run_dir / "weather_allocation.csv", index=False, encoding="utf-8-sig")
        metadata = dict(fetched_at_utc=datetime.now(timezone.utc).isoformat(), api_url=url,
                        location=args.location, timezone=payload["timezone"], latitude=args.latitude,
                        longitude=args.longitude, decision_threshold=args.threshold,
                        surge_rental_threshold=surge_threshold, validation=metrics,
                        training_features=FEATURES, additional_holidays=[str(day) for day in args.holiday],
                        normalization="temp=(C+8)/47; atemp=(C+16)/66; hum=%/100; wind=kmh/67",
                        limitations="Historical DC demo; WMO mapping approximate; no fleet inventory calculation.")
        (run_dir / "forecast_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (run_dir / "weather_api_response.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Chart saved to: {run_dir / 'weather_allocation.png'}")
        plot_plan(plan, args.threshold, args.location, payload["timezone"], run_dir / "weather_allocation.png",
                  metrics, show=not args.no_show)
        print(f"\nChronological holdout: {metrics}")
        print(f"Surge means rentals > {surge_threshold:.2f}; extra preparation threshold = {args.threshold:.0%}")
        print(plan[["local_time", "temperature_c", "surge_probability", "prepare_more_bikes", "input_issue"]].to_string(index=False))
        print(f"\nSaved forecast, chart and metadata to: {run_dir}")
        print("YES = prepare extra availability; NO = no extra signal, not guaranteed sufficient stock.")
        return 0
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Forecast failed: {exc}. No fallback weather or invented recommendation was used.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
