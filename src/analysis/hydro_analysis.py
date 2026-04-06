#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hydro_analysis.py
=================
Analyse extracted river ice events for hydrological patterns.

Expected input: CSV produced by the extraction pipeline, with columns:
  binding_id, date, river_hint, event_type, snippet

Planned analyses:
  - Freeze / break-up date time series per river
  - Long-term trend detection (is freeze date shifting over decades?)
  - Comparison with instrumental records where available
"""

# TODO: implement after extraction pipeline is validated
# This module is the hydrologist student's primary working area.


def load_events(csv_path: str):
    """Load extracted events from CSV into a pandas DataFrame."""
    import pandas as pd
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def freeze_dates_per_river(df):
    """Return DataFrame of freeze events grouped by river and year."""
    freeze = df[df["event_type"] == "freeze"].copy()
    freeze["year"] = freeze["date"].dt.year
    return freeze.groupby(["river_hint", "year"]).size().reset_index(name="count")


def breakup_dates_per_river(df):
    """Return DataFrame of break-up events grouped by river and year."""
    breakup = df[df["event_type"] == "breakup"].copy()
    breakup["year"] = breakup["date"].dt.year
    return breakup.groupby(["river_hint", "year"]).size().reset_index(name="count")
