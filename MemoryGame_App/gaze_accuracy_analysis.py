"""
Gaze Prediction Accuracy Analysis Script

This script analyzes the accuracy of gaze prediction by comparing:
1. Click-time gaze error: Distance between gaze position and click position at click moments
2. Element matching accuracy: Whether gaze was on the same card that was clicked
3. Temporal gaze stability: How stable gaze is before clicks
4. Spatial distribution analysis: Heatmap-like analysis of gaze vs clicks

Usage:
    python gaze_accuracy_analysis.py <session_id>
    python gaze_accuracy_analysis.py 20251213_151001

The script will look for gaze_data_<session_id>.csv in the current directory.
"""

import sys
import os
import csv
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from app_data_paths import get_gaze_data_dir, get_app_data_dir



@dataclass
class GazeSample:
    timestamp_ms: int
    phase: str
    gaze_x: Optional[float]
    gaze_y: Optional[float]
    element_type: str
    card_row: Optional[int]
    card_col: Optional[int]
    card_id: Optional[int]
    card_image_name: str
    game_time_ms: int


@dataclass
class ClickEvent:
    timestamp_ms: int
    phase: str
    gaze_x: Optional[float]
    gaze_y: Optional[float]
    click_x: int
    click_y: int
    element_type: str
    card_row: Optional[int]
    card_col: Optional[int]
    card_id: Optional[int]
    card_image_name: str
    matched: Optional[int]
    game_time_ms: int


def parse_int(value: str) -> Optional[int]:
    """Parse integer from string, return None if empty or invalid."""
    if not value or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_float(value: str) -> Optional[float]:
    """Parse float from string, return None if empty or invalid."""
    if not value or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_gaze_data(filepath: str) -> Tuple[List[GazeSample], List[ClickEvent]]:
    """Load and parse gaze data CSV file."""
    gaze_samples = []
    click_events = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_type = row.get("event_type", "")

            if event_type == "gaze_sample":
                sample = GazeSample(
                    timestamp_ms=int(row["timestamp_ms"]),
                    phase=row["phase"],
                    gaze_x=parse_float(row["gaze_x"]),
                    gaze_y=parse_float(row["gaze_y"]),
                    element_type=row.get("element_type", ""),
                    card_row=parse_int(row.get("card_row", "")),
                    card_col=parse_int(row.get("card_col", "")),
                    card_id=parse_int(row.get("card_id", "")),
                    card_image_name=row.get("card_image_name", ""),
                    game_time_ms=parse_int(row.get("game_time_ms", "0")) or 0,
                )
                gaze_samples.append(sample)

            elif event_type == "click":
                click = ClickEvent(
                    timestamp_ms=int(row["timestamp_ms"]),
                    phase=row["phase"],
                    gaze_x=parse_float(row["gaze_x"]),
                    gaze_y=parse_float(row["gaze_y"]),
                    click_x=int(row["click_x"]),
                    click_y=int(row["click_y"]),
                    element_type=row.get("element_type", ""),
                    card_row=parse_int(row.get("card_row", "")),
                    card_col=parse_int(row.get("card_col", "")),
                    card_id=parse_int(row.get("card_id", "")),
                    card_image_name=row.get("card_image_name", ""),
                    matched=parse_int(row.get("matched", "")),
                    game_time_ms=parse_int(row.get("game_time_ms", "0")) or 0,
                )
                click_events.append(click)

    return gaze_samples, click_events


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def analyze_click_gaze_error(click_events: List[ClickEvent]) -> Dict:
    """
    Analyze the error between gaze position and click position at click moments.
    """
    errors = []
    clicks_with_gaze = 0
    clicks_without_gaze = 0

    for click in click_events:
        if click.gaze_x is not None and click.gaze_y is not None:
            error = euclidean_distance(
                click.gaze_x, click.gaze_y, click.click_x, click.click_y
            )
            errors.append({
                "timestamp_ms": click.timestamp_ms,
                "game_time_ms": click.game_time_ms,
                "gaze_x": click.gaze_x,
                "gaze_y": click.gaze_y,
                "click_x": click.click_x,
                "click_y": click.click_y,
                "error_px": error,
                "card_row": click.card_row,
                "card_col": click.card_col,
                "matched": click.matched,
            })
            clicks_with_gaze += 1
        else:
            clicks_without_gaze += 1

    if not errors:
        return {
            "total_clicks": len(click_events),
            "clicks_with_gaze": 0,
            "clicks_without_gaze": clicks_without_gaze,
            "mean_error_px": None,
            "median_error_px": None,
            "min_error_px": None,
            "max_error_px": None,
            "std_error_px": None,
            "errors": [],
        }

    error_values = [e["error_px"] for e in errors]
    mean_error = sum(error_values) / len(error_values)
    sorted_errors = sorted(error_values)
    median_error = sorted_errors[len(sorted_errors) // 2]
    variance = sum((e - mean_error) ** 2 for e in error_values) / len(error_values)
    std_error = math.sqrt(variance)

    return {
        "total_clicks": len(click_events),
        "clicks_with_gaze": clicks_with_gaze,
        "clicks_without_gaze": clicks_without_gaze,
        "mean_error_px": mean_error,
        "median_error_px": median_error,
        "min_error_px": min(error_values),
        "max_error_px": max(error_values),
        "std_error_px": std_error,
        "errors": errors,
    }


def analyze_element_matching(
    gaze_samples: List[GazeSample], click_events: List[ClickEvent], time_window_ms: int = 500
) -> Dict:
    """
    Analyze whether gaze was on the same card that was clicked.
    Looks at gaze samples within a time window before each click.
    """
    results = []

    for click in click_events:
        if click.card_row is None or click.card_col is None:
            continue

        # Find gaze samples within time window before click
        window_start = click.timestamp_ms - time_window_ms
        relevant_samples = [
            s for s in gaze_samples
            if window_start <= s.timestamp_ms <= click.timestamp_ms
            and s.gaze_x is not None and s.gaze_y is not None
        ]

        if not relevant_samples:
            continue

        # Count how many samples were on the same card
        same_card_count = sum(
            1 for s in relevant_samples
            if s.card_row == click.card_row and s.card_col == click.card_col
        )

        # Count how many samples were on any card
        on_card_count = sum(
            1 for s in relevant_samples
            if s.element_type == "card"
        )

        total_samples = len(relevant_samples)
        same_card_ratio = same_card_count / total_samples if total_samples > 0 else 0
        on_card_ratio = on_card_count / total_samples if total_samples > 0 else 0

        results.append({
            "click_timestamp_ms": click.timestamp_ms,
            "game_time_ms": click.game_time_ms,
            "clicked_card": f"({click.card_row}, {click.card_col})",
            "total_samples": total_samples,
            "same_card_count": same_card_count,
            "same_card_ratio": same_card_ratio,
            "on_card_count": on_card_count,
            "on_card_ratio": on_card_ratio,
            "matched": click.matched,
        })

    if not results:
        return {
            "total_analyzed_clicks": 0,
            "mean_same_card_ratio": None,
            "mean_on_card_ratio": None,
            "perfect_matches": 0,
            "results": [],
        }

    mean_same_card_ratio = sum(r["same_card_ratio"] for r in results) / len(results)
    mean_on_card_ratio = sum(r["on_card_ratio"] for r in results) / len(results)
    perfect_matches = sum(1 for r in results if r["same_card_ratio"] > 0.5)

    return {
        "total_analyzed_clicks": len(results),
        "mean_same_card_ratio": mean_same_card_ratio,
        "mean_on_card_ratio": mean_on_card_ratio,
        "perfect_matches": perfect_matches,
        "perfect_match_ratio": perfect_matches / len(results) if results else 0,
        "time_window_ms": time_window_ms,
        "results": results,
    }


def analyze_gaze_stability(
    gaze_samples: List[GazeSample], click_events: List[ClickEvent], time_window_ms: int = 300
) -> Dict:
    """
    Analyze gaze stability before clicks.
    Lower variance indicates more stable gaze (user was focused on target).
    """
    results = []

    for click in click_events:
        # Find gaze samples within time window before click
        window_start = click.timestamp_ms - time_window_ms
        relevant_samples = [
            s for s in gaze_samples
            if window_start <= s.timestamp_ms <= click.timestamp_ms
            and s.gaze_x is not None and s.gaze_y is not None
        ]

        if len(relevant_samples) < 2:
            continue

        x_values = [s.gaze_x for s in relevant_samples]
        y_values = [s.gaze_y for s in relevant_samples]

        mean_x = sum(x_values) / len(x_values)
        mean_y = sum(y_values) / len(y_values)

        var_x = sum((x - mean_x) ** 2 for x in x_values) / len(x_values)
        var_y = sum((y - mean_y) ** 2 for y in y_values) / len(y_values)

        # Combined spatial variance
        spatial_variance = var_x + var_y
        spatial_std = math.sqrt(spatial_variance)

        # Distance from mean gaze to click
        mean_to_click_dist = euclidean_distance(mean_x, mean_y, click.click_x, click.click_y)

        results.append({
            "click_timestamp_ms": click.timestamp_ms,
            "game_time_ms": click.game_time_ms,
            "num_samples": len(relevant_samples),
            "mean_gaze_x": mean_x,
            "mean_gaze_y": mean_y,
            "click_x": click.click_x,
            "click_y": click.click_y,
            "spatial_std_px": spatial_std,
            "mean_to_click_distance_px": mean_to_click_dist,
            "matched": click.matched,
        })

    if not results:
        return {
            "total_analyzed_clicks": 0,
            "mean_spatial_std_px": None,
            "mean_distance_to_click_px": None,
            "time_window_ms": time_window_ms,
            "results": [],
        }

    mean_spatial_std = sum(r["spatial_std_px"] for r in results) / len(results)
    mean_distance = sum(r["mean_to_click_distance_px"] for r in results) / len(results)

    return {
        "total_analyzed_clicks": len(results),
        "mean_spatial_std_px": mean_spatial_std,
        "mean_distance_to_click_px": mean_distance,
        "time_window_ms": time_window_ms,
        "results": results,
    }


def analyze_element_distribution(gaze_samples: List[GazeSample]) -> Dict:
    """
    Analyze distribution of gaze across different UI elements.
    """
    element_counts = defaultdict(int)
    card_counts = defaultdict(int)

    for sample in gaze_samples:
        if sample.gaze_x is None or sample.gaze_y is None:
            continue

        element_counts[sample.element_type] += 1

        if sample.element_type == "card" and sample.card_row and sample.card_col:
            card_key = f"({sample.card_row}, {sample.card_col})"
            card_counts[card_key] += 1

    total = sum(element_counts.values())

    element_percentages = {
        elem: (count / total * 100) if total > 0 else 0
        for elem, count in element_counts.items()
    }

    return {
        "total_samples": total,
        "element_counts": dict(element_counts),
        "element_percentages": element_percentages,
        "card_distribution": dict(card_counts),
    }


def analyze_phase_accuracy(
    gaze_samples: List[GazeSample], click_events: List[ClickEvent]
) -> Dict:
    """
    Compare accuracy metrics between memorization and play phases.
    """
    phases = ["memorization", "play"]
    results = {}

    for phase in phases:
        phase_samples = [s for s in gaze_samples if s.phase == phase]
        phase_clicks = [c for c in click_events if c.phase == phase]

        # Element distribution for this phase
        element_counts = defaultdict(int)
        for sample in phase_samples:
            if sample.gaze_x is not None:
                element_counts[sample.element_type] += 1

        total = sum(element_counts.values())
        card_percentage = (element_counts.get("card", 0) / total * 100) if total > 0 else 0

        results[phase] = {
            "total_samples": len(phase_samples),
            "total_clicks": len(phase_clicks),
            "samples_with_gaze": sum(1 for s in phase_samples if s.gaze_x is not None),
            "card_gaze_percentage": card_percentage,
            "element_counts": dict(element_counts),
        }

    return results


def generate_report(
    session_id: str,
    gaze_samples: List[GazeSample],
    click_events: List[ClickEvent],
) -> str:
    """Generate a comprehensive accuracy report."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"GAZE PREDICTION ACCURACY REPORT")
    lines.append(f"Session ID: {session_id}")
    lines.append("=" * 70)
    lines.append("")

    # Basic stats
    lines.append("BASIC STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total gaze samples: {len(gaze_samples)}")
    lines.append(f"Total click events: {len(click_events)}")
    samples_with_gaze = sum(1 for s in gaze_samples if s.gaze_x is not None)
    lines.append(f"Samples with valid gaze: {samples_with_gaze} ({samples_with_gaze/len(gaze_samples)*100:.1f}%)" if gaze_samples else "N/A")
    lines.append("")

    # Click-time gaze error
    lines.append("1. CLICK-TIME GAZE ERROR")
    lines.append("-" * 40)
    click_error = analyze_click_gaze_error(click_events)
    lines.append(f"Clicks with gaze data: {click_error['clicks_with_gaze']}/{click_error['total_clicks']}")
    if click_error["mean_error_px"] is not None:
        lines.append(f"Mean error: {click_error['mean_error_px']:.1f} px")
        lines.append(f"Median error: {click_error['median_error_px']:.1f} px")
        lines.append(f"Min error: {click_error['min_error_px']:.1f} px")
        lines.append(f"Max error: {click_error['max_error_px']:.1f} px")
        lines.append(f"Std deviation: {click_error['std_error_px']:.1f} px")
        lines.append("")
        lines.append("Per-click errors:")
        for e in click_error["errors"]:
            lines.append(
                f"  t={e['game_time_ms']:>6}ms | "
                f"gaze=({e['gaze_x']:>6.0f}, {e['gaze_y']:>6.0f}) | "
                f"click=({e['click_x']:>6}, {e['click_y']:>6}) | "
                f"error={e['error_px']:>6.1f}px | "
                f"matched={e['matched']}"
            )
    else:
        lines.append("No gaze data available at click moments")
    lines.append("")

    # Element matching
    lines.append("2. ELEMENT MATCHING ACCURACY (500ms window before click)")
    lines.append("-" * 40)
    element_match = analyze_element_matching(gaze_samples, click_events, time_window_ms=500)
    if element_match["mean_same_card_ratio"] is not None:
        lines.append(f"Analyzed clicks: {element_match['total_analyzed_clicks']}")
        lines.append(f"Mean 'same card' ratio: {element_match['mean_same_card_ratio']*100:.1f}%")
        lines.append(f"Mean 'on any card' ratio: {element_match['mean_on_card_ratio']*100:.1f}%")
        lines.append(f"Perfect matches (>50% on target): {element_match['perfect_matches']}/{element_match['total_analyzed_clicks']} ({element_match['perfect_match_ratio']*100:.1f}%)")
        lines.append("")
        lines.append("Per-click breakdown:")
        for r in element_match["results"]:
            lines.append(
                f"  t={r['game_time_ms']:>6}ms | "
                f"card={r['clicked_card']} | "
                f"samples={r['total_samples']:>2} | "
                f"same_card={r['same_card_ratio']*100:>5.1f}% | "
                f"on_card={r['on_card_ratio']*100:>5.1f}% | "
                f"matched={r['matched']}"
            )
    else:
        lines.append("No data available for element matching analysis")
    lines.append("")

    # Gaze stability
    lines.append("3. GAZE STABILITY BEFORE CLICKS (300ms window)")
    lines.append("-" * 40)
    stability = analyze_gaze_stability(gaze_samples, click_events, time_window_ms=300)
    if stability["mean_spatial_std_px"] is not None:
        lines.append(f"Analyzed clicks: {stability['total_analyzed_clicks']}")
        lines.append(f"Mean spatial std (lower = more stable): {stability['mean_spatial_std_px']:.1f} px")
        lines.append(f"Mean distance from avg gaze to click: {stability['mean_distance_to_click_px']:.1f} px")
        lines.append("")
        lines.append("Per-click breakdown:")
        for r in stability["results"]:
            lines.append(
                f"  t={r['game_time_ms']:>6}ms | "
                f"samples={r['num_samples']:>2} | "
                f"stability={r['spatial_std_px']:>6.1f}px | "
                f"dist_to_click={r['mean_to_click_distance_px']:>6.1f}px | "
                f"matched={r['matched']}"
            )
    else:
        lines.append("No data available for stability analysis")
    lines.append("")

    # Element distribution
    lines.append("4. GAZE ELEMENT DISTRIBUTION")
    lines.append("-" * 40)
    distribution = analyze_element_distribution(gaze_samples)
    lines.append(f"Total valid samples: {distribution['total_samples']}")
    lines.append("Element types:")
    for elem, pct in sorted(distribution["element_percentages"].items(), key=lambda x: -x[1]):
        count = distribution["element_counts"][elem]
        lines.append(f"  {elem:15}: {count:>5} ({pct:>5.1f}%)")
    lines.append("")
    if distribution["card_distribution"]:
        lines.append("Card distribution:")
        for card, count in sorted(distribution["card_distribution"].items()):
            lines.append(f"  Card {card}: {count} samples")
    lines.append("")

    # Phase comparison
    lines.append("5. PHASE COMPARISON")
    lines.append("-" * 40)
    phase_stats = analyze_phase_accuracy(gaze_samples, click_events)
    for phase, stats in phase_stats.items():
        lines.append(f"{phase.upper()} phase:")
        lines.append(f"  Samples: {stats['total_samples']} (with gaze: {stats['samples_with_gaze']})")
        lines.append(f"  Clicks: {stats['total_clicks']}")
        lines.append(f"  Gaze on cards: {stats['card_gaze_percentage']:.1f}%")
    lines.append("")

    # Summary
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    if click_error["mean_error_px"] is not None:
        # Accuracy assessment
        mean_err = click_error["mean_error_px"]
        if mean_err < 100:
            accuracy_grade = "EXCELLENT"
        elif mean_err < 200:
            accuracy_grade = "GOOD"
        elif mean_err < 400:
            accuracy_grade = "MODERATE"
        else:
            accuracy_grade = "POOR"

        lines.append(f"Overall Accuracy Grade: {accuracy_grade}")
        lines.append(f"  - Mean click-time error: {mean_err:.1f} px")
        if element_match["mean_same_card_ratio"] is not None:
            lines.append(f"  - Gaze on target card: {element_match['mean_same_card_ratio']*100:.1f}%")
        if stability["mean_spatial_std_px"] is not None:
            lines.append(f"  - Pre-click gaze stability: {stability['mean_spatial_std_px']:.1f} px std")
    else:
        lines.append("Insufficient data for accuracy assessment")
    lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python gaze_accuracy_analysis.py <session_id>")
        print("Example: python gaze_accuracy_analysis.py 20251213_151001")
        sys.exit(1)

    session_id = sys.argv[1]
    filepath = str(Path(get_gaze_data_dir()) / f"gaze_data_{session_id}.csv")

    if not os.path.exists(filepath):
        # Try looking in current directory
        alt_paths = [
            filepath,
            os.path.join("MemoryGame_App", filepath),
            os.path.join("..", filepath),
            #os.path.join("gaze_data", filepath),
        ]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                filepath = p
                found = True
                break

        if not found:
            print(f"Error: Could not find {filepath}")
            print("Make sure you're running from the correct directory.")
            sys.exit(1)

    print(f"Loading data from: {filepath}")
    gaze_samples, click_events = load_gaze_data(filepath)
    print(f"Loaded {len(gaze_samples)} gaze samples and {len(click_events)} click events")
    print("")

    report = generate_report(session_id, gaze_samples, click_events)
    print(report)

    # Save report to file
    report_path = str(Path(get_app_data_dir()) / f"gaze accuracy reports/gaze_accuracy_report_{session_id}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()


