import os
import pandas as pd
import numpy as np
import fastf1 as f1

f1.set_log_level("WARNING")
f1.Cache.enable_cache('../cache_strategy')

CIRCUIT_STATS = {
    'Bahrain Grand Prix': {'abrasivity': 5, 'lateral_energy': 3, 'type': 'Traction/Braking'},
    'Saudi Arabian Grand Prix': {'abrasivity': 2, 'lateral_energy': 4, 'type': 'High-Speed Street'},
    'Australian Grand Prix': {'abrasivity': 3, 'lateral_energy': 3, 'type': 'Semi-Permanent'},
    'Japanese Grand Prix': {'abrasivity': 5, 'lateral_energy': 5, 'type': 'High-Energy Technical'},
    'Chinese Grand Prix': {'abrasivity': 3, 'lateral_energy': 4, 'type': 'Technical/Front-Limited'},
    'Miami Grand Prix': {'abrasivity': 3, 'lateral_energy': 3, 'type': 'Street'},
    'Emilia Romagna Grand Prix': {'abrasivity': 3, 'lateral_energy': 3, 'type': 'Old-School Technical'},
    'Monaco Grand Prix': {'abrasivity': 1, 'lateral_energy': 1, 'type': 'Low-Speed Street'},
    'Canadian Grand Prix': {'abrasivity': 2, 'lateral_energy': 2, 'type': 'Stop-and-Go'},
    'Spanish Grand Prix': {'abrasivity': 4, 'lateral_energy': 5, 'type': 'Aero-Reference'},
    'Austrian Grand Prix': {'abrasivity': 2, 'lateral_energy': 3, 'type': 'High-Speed/Short'},
    'British Grand Prix': {'abrasivity': 3, 'lateral_energy': 5, 'type': 'High-Energy Aero'},
    'Hungarian Grand Prix': {'abrasivity': 2, 'lateral_energy': 3, 'type': 'Twisty/Low-Speed'},
    'Belgian Grand Prix': {'abrasivity': 4, 'lateral_energy': 5, 'type': 'Power/High-Energy'},
    'Dutch Grand Prix': {'abrasivity': 3, 'lateral_energy': 5, 'type': 'Banking/High-Energy'},
    'Italian Grand Prix': {'abrasivity': 2, 'lateral_energy': 3, 'type': 'Ultra-High Speed'},
    'Azerbaijan Grand Prix': {'abrasivity': 2, 'lateral_energy': 2, 'type': 'Street/Long-Straight'},
    'Singapore Grand Prix': {'abrasivity': 4, 'lateral_energy': 2, 'type': 'Street/Bumpy'},
    'United States Grand Prix': {'abrasivity': 4, 'lateral_energy': 4, 'type': 'Mix/Bumpy'},
    'Mexico City Grand Prix': {'abrasivity': 2, 'lateral_energy': 2, 'type': 'Altitude/Low-Downforce'},
    'Sao Paulo Grand Prix': {'abrasivity': 3, 'lateral_energy': 3, 'type': 'Technical/Short'},
    'Las Vegas Grand Prix': {'abrasivity': 2, 'lateral_energy': 2, 'type': 'Street/Cold'},
    'Qatar Grand Prix': {'abrasivity': 4, 'lateral_energy': 5, 'type': 'High-Speed/Flat'},
    'Abu Dhabi Grand Prix': {'abrasivity': 3, 'lateral_energy': 3, 'type': 'Technical/Standard'},
}


class F1Analyzer:
    def __init__(self, year, grand_prix, session_type):
        self.year = year
        self.grand_prix = grand_prix
        self.session_type = session_type
        self.session = f1.get_session(year, grand_prix, session_type)
        self.session.load(laps=True, telemetry=True, weather=True, messages=True)

    def get_clean_laps(self, driver=None):
        laps = self.session.laps.copy()
        if driver is not None:
            laps = laps.pick_driver(driver)
        clean_laps = laps.loc[
            (laps['PitInTime'].isna())
            & (laps['PitOutTime'].isna())
            & (laps['TrackStatus'] == '1')
        ].copy()
        return clean_laps

    def add_corrected_lap_time(self, laps):
        laps = laps.copy()
        if 'LapTime' in laps.columns:
            laps['CorrectedLapTime'] = laps['LapTime'].dt.total_seconds()
        else:
            laps['CorrectedLapTime'] = np.nan
        return laps

    def add_delta_to_corrected_lap_time(self, laps, source_column='CorrectedLapTime', output_column='DeltaToBestLapForStint'):
        laps = laps.copy()
        if source_column not in laps.columns:
            laps[output_column] = np.nan
            return laps

        if 'Stint' not in laps.columns:
            laps[output_column] = np.nan
            return laps

        metric = pd.to_numeric(laps[source_column], errors='coerce')
        best_by_stint = metric.groupby(laps['Stint']).transform('min')
        laps[output_column] = metric - best_by_stint
        laps['BestLapForStint'] = best_by_stint
        return laps


def build_monza_extraction_frame(year=2024, grand_prix='Monza', session_type='R'):
    analyzer = F1Analyzer(year, grand_prix, session_type)
    laps = analyzer.get_clean_laps()

    laps = analyzer.add_corrected_lap_time(laps)
    laps = analyzer.add_delta_to_corrected_lap_time(laps, source_column='CorrectedLapTime', output_column='DeltaToBestLapForStint')

    if 'TyreLife' in laps.columns:
        laps['TyreLife'] = pd.to_numeric(laps['TyreLife'], errors='coerce')
    else:
        laps['TyreLife'] = np.nan

    missing_tyre_life = laps['TyreLife'].isna()
    if missing_tyre_life.any():
        laps.loc[missing_tyre_life, 'TyreLife'] = (
            laps.loc[missing_tyre_life]
            .groupby(['Driver', 'Stint'], dropna=False)
            .cumcount() + 1
        )

    track_temp_values = []
    air_temp_values = []
    if hasattr(laps, 'iterlaps'):
        for _, lap in laps.iterlaps():
            try:
                weather = lap.get_weather_data()
                if weather is not None:
                    if 'TrackTemp' in weather.index and pd.notna(weather['TrackTemp']):
                        track_temp_values.append(float(weather['TrackTemp']))
                    if 'AirTemp' in weather.index and pd.notna(weather['AirTemp']):
                        air_temp_values.append(float(weather['AirTemp']))
            except Exception:
                pass

    average_track_temp = float(np.mean(track_temp_values)) if track_temp_values else np.nan
    average_air_temp = float(np.mean(air_temp_values)) if air_temp_values else np.nan

    circuit_key = f'{grand_prix} Grand Prix'
    circuit_info = CIRCUIT_STATS.get(circuit_key, {})

    extracted = pd.DataFrame({
        'Year': year,
        'Event': grand_prix,
        'Driver': laps['Driver'].astype(str),
        'LapNumber': laps['LapNumber'],
        'Stint': laps['Stint'],
        'Compound': laps['Compound'],
        'TyreLife': laps['TyreLife'],
        'CorrectedLapTime': laps['CorrectedLapTime'],
        'DeltaToBestLapForStint': laps['DeltaToBestLapForStint'],
        'BestLapForStint': laps['BestLapForStint'],
        'abrasivity': circuit_info.get('abrasivity', np.nan),
        'lateral_energy': circuit_info.get('lateral_energy', np.nan),
        'circuit_type': circuit_info.get('type', np.nan),
        'AverageTrackTemp_C': average_track_temp,
        'AverageAirTemp_C': average_air_temp,
        'N_Treated_Laps': len(laps),
    }, index=laps.index)

    return analyzer, laps, extracted
