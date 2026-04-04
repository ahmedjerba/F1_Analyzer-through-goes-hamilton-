import fastf1 as f1
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import datetime as dt
import seaborn as sns
from fastf1 import plotting
from fastf1 import utils
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter
f1.set_log_level("WARNING")
class F1Analyzer:
    ERAS_BASE_PENALTY = {
        "V10_LIGHT": 0.020,
        "V8_HYBRID": 0.025,
        "V6_TURBO": 0.030,
        "GROUND_EFFECT": 0.035,
    }

    def _calculate_auto_penalty(self, year, grand_prix):
        if year <= 2005:
            era = "V10_LIGHT"
        elif year <= 2013:
            era = "V8_HYBRID"
        elif year <= 2021:
            era = "V6_TURBO"
        else:
            era = "GROUND_EFFECT"
        
        base = self.ERAS_BASE_PENALTY[era]

        POWER_TRACKS = {'Monza', 'Spa-Francorchamps', 'Silverstone', 'Baku City Circuit', 'Las Vegas', 'Jeddah Corniche Circuit'}
        STREET_TRACKS = {'Monaco', 'Singapore Street Circuit', 'Hungaroring', 'Marina Bay Street Circuit'}

        if grand_prix in POWER_TRACKS:
            modifier, cat = 1.2, "POWER"
        elif grand_prix in STREET_TRACKS:
            modifier, cat = 0.7, "STREET"
        else:
            modifier, cat = 1.0, "BALANCED"
            
        return base * modifier, cat
    
    def __init__(self, year, grand_prix, session_type, use_fuel_logic=True):
        self.year = year
        self.grand_prix = grand_prix
        self.session_type = session_type
        self.session = f1.get_session(year, grand_prix, session_type)
        self.session.load()
        if use_fuel_logic:
            self.fuel_penalty, self.circuit_category = self._calculate_auto_penalty(year, grand_prix)
        else:
            self.fuel_penalty, self.circuit_category = 0, "NONE"

        self.session.laps['CircuitCategory'] = self.circuit_category
        self.session.laps['FuelPenaltyFactor'] = self.fuel_penalty
        self.session.laps['CorrectedLapTime'] = (
            self.session.laps['LapTime'].dt.total_seconds() + 
            (self.session.laps['LapNumber'] * self.fuel_penalty)
        )

    def get_clean_laps(self, driver):
        laps = self.session.laps.pick_driver(driver)
        clean_1_laps = laps.loc[
            (laps["PitInTime"].isna()) &
            (laps["PitOutTime"].isna()) &
            (laps["TrackStatus"] == "1")
        ].copy()
        return clean_1_laps

    def audit_data_cleaning(self, driver):
        raw_laps = self.session.laps.pick_driver(driver)
        total_raw = len(raw_laps)

        phys_clean = self.get_clean_laps(driver)
        total_phys = len(phys_clean)

        very_clean = self.get_clean_race_pace_laps(driver)
        total_very = len(very_clean)

        raw_lap_numbers = set(raw_laps["LapNumber"].dropna().astype(int).tolist())
        phys_lap_numbers = set(phys_clean["LapNumber"].dropna().astype(int).tolist())
        very_lap_numbers = set(very_clean["LapNumber"].dropna().astype(int).tolist())

        lost_phys_laps = sorted(raw_lap_numbers - phys_lap_numbers)
        lost_iqr_laps = sorted(phys_lap_numbers - very_lap_numbers)

        lost_phys = len(lost_phys_laps)
        lost_iqr = len(lost_iqr_laps)

        pct_phys = (lost_phys / total_raw) * 100 if total_raw else 0
        pct_iqr = (lost_iqr / total_phys) * 100 if total_phys else 0

        print(f"--- Audit de Nettoyage pour {driver} ---")
        print(f"Tours totaux en session : {total_raw}")
        print(f"Tours perdus (Stands/Drapeaux) : {lost_phys} ({pct_phys:.1f}%)")
        print(f"Numéros des tours perdus (Stands/Drapeaux) : {lost_phys_laps if lost_phys_laps else 'Aucun'}")
        print(f"Tours perdus (IQR/Anomalies) : {lost_iqr} ({pct_iqr:.1f}%)")
        print(f"Numéros des tours perdus (IQR/Anomalies) : {lost_iqr_laps if lost_iqr_laps else 'Aucun'}")
        print(f"Tours restants pour l'analyse : {total_very}")
        print("-" * 40)

        return total_raw, total_phys, total_very, lost_iqr, lost_phys

    def get_clean_race_pace_laps(self, driver):
        laps = self.get_clean_laps(driver)
        series = laps["LapTime"].dt.total_seconds()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        clean_2_laps = laps[(series >= lower_bound) & (series <= upper_bound)]
        return clean_2_laps

    def format_laptime(self, seconds, pos):
        minutes = int(seconds // 60)
        remainder = seconds % 60
        return f"{minutes}:{remainder:06.3f}"

    def plot_drivers_pace(self, session, driver_list, very_clean=False, include_nan_laps=False, use_fuel_logic=False):
        fig, ax = plt.subplots(figsize=(12, 7))

        used_colors = {}

        for driver in driver_list:
            if very_clean:
                laps = self.get_clean_race_pace_laps(driver)
            else:
                laps = self.get_clean_laps(driver)

            if include_nan_laps:
                full_range = range(1, self.session.total_laps + 1)
                laps = laps.set_index("LapNumber").reindex(full_range).reset_index()

            if use_fuel_logic:
                if "CorrectedLapTime" not in laps.columns or laps["CorrectedLapTime"].notna().sum() == 0:
                    continue
                y = laps["CorrectedLapTime"]
            else:
                if laps.empty or laps["LapTime"].notna().sum() == 0:
                    continue
                y = laps["LapTime"].dt.total_seconds()

            x = laps["LapNumber"]

            try:
                base_color = plotting.get_driver_color(driver, session=session)
            except Exception:
                base_color = "#FFFFFF"

            if base_color in used_colors.values():
                rgb = mcolors.to_rgb(base_color)
                color = mcolors.to_hex([min(1, c + 0.4) for c in rgb])
                linestyle = "--"
            else:
                color = base_color
                linestyle = "-"
                used_colors[driver] = base_color

            ax.plot(
                x,
                y,
                color=color,
                label=driver,
                linestyle=linestyle,
                linewidth=2,
                marker="o",
                markersize=3,
                alpha=0.8,
            )

        ax.yaxis.set_major_formatter(FuncFormatter(self.format_laptime))

        title = f"Race Pace Analysis: {session.event['EventName']} {session.event.year}"
        ax.set_title(title, fontsize=15, pad=20)
        ax.set_xlabel("Tour n°", fontsize=12)
        ax.set_ylabel("Temps au tour corrigé" if use_fuel_logic else "Temps au tour", fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        ax.grid(visible=True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        return fig, ax
