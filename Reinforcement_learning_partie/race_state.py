from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

Pit_Time_MU = 22.5
Pit_Time_SIGMA = 0.8

RACE_COMPOUND_TO_C = {
    'Bahrain': {'Hard': 'C1', 'Medium': 'C2', 'Soft': 'C3'},
    'Saudi Arabia': {'Hard': 'C2', 'Medium': 'C3', 'Soft': 'C4'},
    'Australia': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'Japan': {'Hard': 'C1', 'Medium': 'C2', 'Soft': 'C3'},
    'China': {'Hard': 'C2', 'Medium': 'C3', 'Soft': 'C4'},
    'Miami': {'Hard': 'C2', 'Medium': 'C3', 'Soft': 'C4'},
    'Emilia-Romagna': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'Monaco': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'Canada': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'Spain': {'Hard': 'C1', 'Medium': 'C2', 'Soft': 'C3'},
    'Austria': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'Great Britain': {'Hard': 'C1', 'Medium': 'C2', 'Soft': 'C3'},
    'Hungary': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'Belgium': {'Hard': 'C2', 'Medium': 'C3', 'Soft': 'C4'},
    'Netherlands': {'Hard': 'C1', 'Medium': 'C2', 'Soft': 'C3'},
    'Italy': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'Azerbaijan': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'Singapore': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'USA (Austin)': {'Hard': 'C2', 'Medium': 'C3', 'Soft': 'C4'},
    'Mexico': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'Brazil': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'Las Vegas': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
    'Qatar': {'Hard': 'C1', 'Medium': 'C2', 'Soft': 'C3'},
    'Abu Dhabi': {'Hard': 'C3', 'Medium': 'C4', 'Soft': 'C5'},
}

GP_CODE_TO_RACE = {
    'BAH': 'Bahrain',
    'SAU': 'Saudi Arabia',
    'AUS': 'Australia',
    'JPN': 'Japan',
    'CHN': 'China',
    'MIA': 'Miami',
    'EMI': 'Emilia-Romagna',
    'MON': 'Monaco',
    'CAN': 'Canada',
    'ESP': 'Spain',
    'AUT': 'Austria',
    'GBR': 'Great Britain',
    'HUN': 'Hungary',
    'BEL': 'Belgium',
    'NED': 'Netherlands',
    'ITA': 'Italy',
    'AZE': 'Azerbaijan',
    'SIN': 'Singapore',
    'USA': 'USA (Austin)',
    'MEX': 'Mexico',
    'BRA': 'Brazil',
    'LVS': 'Las Vegas',
    'QAT': 'Qatar',
    'ABU': 'Abu Dhabi',
}

RACE_NAME_TO_COMPOUND_KEY = {
    'Imola': 'Emilia-Romagna',
    'Baku': 'Azerbaijan',
    'Austin': 'USA (Austin)',
    'Melbourne': 'Australia',
}

TYRE_LIMITS_2024 = {
    'Bahrain': {'SOFT': 16, 'MEDIUM': 24, 'HARD': 34},
    'Saudi Arabia': {'SOFT': 22, 'MEDIUM': 34, 'HARD': 48},
    'Australia': {'SOFT': 14, 'MEDIUM': 26, 'HARD': 38},
    'Japan': {'SOFT': 15, 'MEDIUM': 22, 'HARD': 32},
    'China': {'SOFT': 17, 'MEDIUM': 25, 'HARD': 35},
    'Miami': {'SOFT': 18, 'MEDIUM': 28, 'HARD': 42},
    'Emilia-Romagna': {'SOFT': 20, 'MEDIUM': 30, 'HARD': 44},
    'Monaco': {'SOFT': 30, 'MEDIUM': 45, 'HARD': 60},
    'Canada': {'SOFT': 20, 'MEDIUM': 32, 'HARD': 48},
    'Spain': {'SOFT': 16, 'MEDIUM': 24, 'HARD': 34},
    'Austria': {'SOFT': 18, 'MEDIUM': 28, 'HARD': 40},
    'Great Britain': {'SOFT': 17, 'MEDIUM': 25, 'HARD': 35},
    'Hungary': {'SOFT': 18, 'MEDIUM': 28, 'HARD': 40},
    'Belgium': {'SOFT': 14, 'MEDIUM': 22, 'HARD': 30},
    'Netherlands': {'SOFT': 20, 'MEDIUM': 30, 'HARD': 42},
    'Italy': {'SOFT': 18, 'MEDIUM': 28, 'HARD': 40},
    'Azerbaijan': {'SOFT': 18, 'MEDIUM': 28, 'HARD': 42},
    'Singapore': {'SOFT': 18, 'MEDIUM': 28, 'HARD': 42},
    'USA (Austin)': {'SOFT': 16, 'MEDIUM': 24, 'HARD': 36},
    'Mexico': {'SOFT': 22, 'MEDIUM': 35, 'HARD': 50},
    'Brazil': {'SOFT': 18, 'MEDIUM': 28, 'HARD': 40},
    'Las Vegas': {'SOFT': 20, 'MEDIUM': 32, 'HARD': 48},
    'Qatar': {'SOFT': 16, 'MEDIUM': 22, 'HARD': 30},
    'Abu Dhabi': {'SOFT': 18, 'MEDIUM': 28, 'HARD': 40},
}

RACE_NUMBER_TO_GP_ID = {
    23: 'Bahrain',
    24: 'Saudi Arabia',
    25: 'Australia',
    26: 'Suzuka',
    27: 'China',
    28: 'Miami',
    29: 'Imola',
    30: 'Monaco',
    31: 'Canada',
    32: 'Spain',
    33: 'Austria',
    34: 'Silverstone',
    35: 'Hungary',
    36: 'Spa',
    37: 'Zandvoort',
    38: 'Monza',
    39: 'Baku',
    40: 'Singapore',
    41: 'Austin',
    42: 'Mexico',
    43: 'Interlagos',
    44: 'Las Vegas',
    45: 'Qatar',
    46: 'Abu Dhabi',
}

C_HARDNESS = {'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5}


def normalize_race_name_for_compound(race_name: Optional[str]) -> Optional[str]:
    if race_name is None:
        return None
    key = str(race_name).strip()
    return RACE_NAME_TO_COMPOUND_KEY.get(key, key)


def get_compound_hardness(race_name: str, compound: str, default: Optional[int] = None) -> Optional[int]:
    if race_name is None or compound is None:
        return default

    normalized_race = normalize_race_name_for_compound(race_name)
    race_map = RACE_COMPOUND_TO_C.get(normalized_race)
    if race_map is None:
        return default

    c_compound = race_map.get(str(compound).strip().title())
    if c_compound is None:
        return default

    return C_HARDNESS.get(c_compound, default)


def get_compound_hardness_by_gp_code(gp_code: str, compound: str, default: Optional[int] = None) -> Optional[int]:
    if gp_code is None:
        return default

    race_name = GP_CODE_TO_RACE.get(str(gp_code).strip().upper())
    if race_name is None:
        return default

    return get_compound_hardness(race_name, compound, default)


def get_compound_label_from_hardness(
    race_name: str,
    compound_hardness: int,
    default: Optional[str] = None,
) -> Optional[str]:
    """Return Soft/Medium/Hard label from hardness (1..5) for a given GP."""
    if race_name is None or compound_hardness is None:
        return default

    normalized_race = normalize_race_name_for_compound(race_name)
    race_map = RACE_COMPOUND_TO_C.get(normalized_race)
    if race_map is None:
        return default

    target_hardness = int(compound_hardness)
    for label, c_code in race_map.items():
        if C_HARDNESS.get(c_code) == target_hardness:
            return str(label).upper()

    return default


def get_compound_label_from_hardness_by_gp_code(
    gp_code: str,
    compound_hardness: int,
    default: Optional[str] = None,
) -> Optional[str]:
    if gp_code is None:
        return default

    race_name = GP_CODE_TO_RACE.get(str(gp_code).strip().upper())
    if race_name is None:
        return default

    return get_compound_label_from_hardness(race_name, compound_hardness, default)


def get_compound_from_pit_action(action: str, default: Optional[str] = None) -> Optional[str]:
    """Convert PIT_* action string to compound label (SOFT/MEDIUM/HARD)."""
    if action is None or not str(action).startswith('PIT_'):
        return default
    return str(action).split('_', 1)[1].strip().upper()


def get_tyre_life_pct(race_number, gp_id, compound_label, tyre_life, stint_length=None, compound_hardness=None):
    # 1. Sécurité sur TyreLife (ne doit JAMAIS être None)
    safe_tl = float(tyre_life) if tyre_life is not None else 0.0
    
    # 2. Mapping avec "MEDIUM" par défaut si on ne trouve rien
    if compound_label is None and compound_hardness is not None:
        compound_label = get_compound_label_from_hardness_by_gp_code(race_number, compound_hardness)
        if compound_label is None:
            compound_label = "MEDIUM" # <-- Ton choix par défaut

    gp_name = gp_id or RACE_NUMBER_TO_GP_ID.get(race_number)
    gp_name = normalize_race_name_for_compound(gp_name) if gp_name else None

    # 3. Récupération de la limite avec triple sécurité
    limit = None
    if gp_name and compound_label:
        tyre_limits = TYRE_LIMITS_2024.get(gp_name, {})
        limit = tyre_limits.get(compound_label.upper())

    if limit is None:
        # Fallback ultime : stint_length ou 30 tours
        limit = stint_length if (stint_length and stint_length > 0) else 30.0

    return safe_tl / max(float(limit), 1.0)

@dataclass
class RaceState:
    lap: int
    total_laps: int
    tyre_life: float
    compound: str
    compound_hardness: int
    fuel_load: float
    delta_to_best: float
    delta_velocity: float
    stint_number: float
    stint_length: float
    tyre_life_pct: float
    prev_stint_max_delta: float
    track_temp: float
    abrasivity: float
    lateral: float
    race_number: int
    team_encoded: int
    compound_x_abrasivity: float
    compound_x_lateral: float
    lateral_stress_cumul: float
    abrasive_stress_cumul: float
    stress_x_temp: float
    compound_x_tyrelife: float
    rival_tyre_life: float
    rival_compound_hardness: int
    rival_delta_to_best: float
    rival_fuel_load: float
    rival_stint: float
    gap_to_rival: float
    gp_id: str = ''

    GP_STRATEGY_DATA = {
        'Bahrain': {'compounds': [1, 2, 3], 'wear_factor': 0.85, 'pit_loss': 23.0},
        'Suzuka': {'compounds': [1, 2, 3], 'wear_factor': 0.80, 'pit_loss': 22.5},
        'Spain': {'compounds': [1, 2, 3], 'wear_factor': 0.82, 'pit_loss': 23.0},
        'Austria': {'compounds': [3, 4, 5], 'wear_factor': 0.88, 'pit_loss': 20.5},
        'Silverstone': {'compounds': [1, 2, 3], 'wear_factor': 0.75, 'pit_loss': 20.0},
        'Spa': {'compounds': [2, 3, 4], 'wear_factor': 0.90, 'pit_loss': 23.5},
        'Zandvoort': {'compounds': [1, 2, 3], 'wear_factor': 0.82, 'pit_loss': 18.0},
        'Qatar': {'compounds': [1, 2, 3], 'wear_factor': 0.70, 'pit_loss': 25.0},
        'Saudi Arabia': {'compounds': [2, 3, 4], 'wear_factor': 1.10, 'pit_loss': 24.0},
        'Melbourne': {'compounds': [3, 4, 5], 'wear_factor': 1.05, 'pit_loss': 21.0},
        'Miami': {'compounds': [2, 3, 4], 'wear_factor': 1.15, 'pit_loss': 24.5},
        'Monaco': {'compounds': [3, 4, 5], 'wear_factor': 1.50, 'pit_loss': 25.0},
        'Canada': {'compounds': [3, 4, 5], 'wear_factor': 1.25, 'pit_loss': 22.5},
        'Baku': {'compounds': [3, 4, 5], 'wear_factor': 1.15, 'pit_loss': 26.0},
        'Singapore': {'compounds': [3, 4, 5], 'wear_factor': 1.20, 'pit_loss': 28.5},
        'Las Vegas': {'compounds': [3, 4, 5], 'wear_factor': 1.35, 'pit_loss': 27.0},
        'China': {'compounds': [2, 3, 4], 'wear_factor': 0.95, 'pit_loss': 25.5},
        'Imola': {'compounds': [3, 4, 5], 'wear_factor': 1.00, 'pit_loss': 23.0},
        'Hungary': {'compounds': [3, 4, 5], 'wear_factor': 0.92, 'pit_loss': 23.5},
        'Monza': {'compounds': [3, 4, 5], 'wear_factor': 1.05, 'pit_loss': 24.0},
        'Austin': {'compounds': [2, 3, 4], 'wear_factor': 0.88, 'pit_loss': 24.0},
        'Mexico': {'compounds': [3, 4, 5], 'wear_factor': 1.10, 'pit_loss': 24.5},
        'Interlagos': {'compounds': [3, 4, 5], 'wear_factor': 0.95, 'pit_loss': 21.0},
        'Abu Dhabi': {'compounds': [3, 4, 5], 'wear_factor': 1.10, 'pit_loss': 24.0},
    }

    def _get_gp_name_for_limits(self):
        gp_name = self.gp_id or self.RACE_NUMBER_TO_GP_ID.get(self.race_number)
        if gp_name is None:
            return None
        return normalize_race_name_for_compound(gp_name)

    def is_terminal(self) -> bool:
        return self.lap >= self.total_laps

    def to_features(self) -> pd.DataFrame:
        tl = self.tyre_life
        return pd.DataFrame([{
            'CompoundEncoded': self.compound_hardness,
            'TyreLife': tl,
            'TrackTemp': self.track_temp,
            'FuelLoad': self.fuel_load,
            'Abrasivity': self.abrasivity,
            'LateralEnergy': self.lateral,
            'DeltaToBest': self.delta_to_best,
            'LapNumber': self.lap,
            'Stint': self.stint_number,
            'RaceNumber': self.race_number,
            'TeamEncoded': self.team_encoded,
            'delta_velocity': self.delta_velocity,
            'lateral_stress_cumul': self.lateral_stress_cumul,
            'abrasive_stress_cumul': self.abrasive_stress_cumul,
            'stress_x_temp': self.stress_x_temp,
            'compound_x_abrasivity': self.compound_x_abrasivity,
            'compound_x_lateral': self.compound_x_lateral,
            'compound_x_tyrelife': self.compound_x_tyrelife,
            'prev_stint_max_delta': self.prev_stint_max_delta,
            'stint_length': self.stint_length,
            'tyre_life_pct': get_tyre_life_pct(
                self.race_number,
                self.gp_id,
                self.compound,
                tl,
                stint_length=self.stint_length,
                compound_hardness=self.compound_hardness,
            ),
        }])

    def to_rival_features(self) -> pd.DataFrame:
        # Valeurs par defaut robustes pour eviter les None dans les calculs.
        tl = self.rival_tyre_life if self.rival_tyre_life is not None else 0.0
        tt = self.track_temp if self.track_temp is not None else 30.0
        ch = self.rival_compound_hardness if self.rival_compound_hardness is not None else 3.0
        rf = self.rival_fuel_load if self.rival_fuel_load is not None else 50.0
        ab = self.abrasivity if self.abrasivity is not None else 1.0
        lat = self.lateral if self.lateral is not None else 1.0
        rdb = self.rival_delta_to_best if self.rival_delta_to_best is not None else 0.0
        rstint = self.rival_stint if self.rival_stint is not None else 1.0
        return pd.DataFrame([{
            'CompoundEncoded': float(ch),
            'TyreLife': float(tl),
            'TrackTemp': float(tt),
            'FuelLoad': float(rf),
            'Abrasivity': float(ab),
            'LateralEnergy': float(lat),
            'DeltaToBest': float(rdb),
            'LapNumber': self.lap,
            'Stint': float(rstint),
            'RaceNumber': self.race_number,
            'TeamEncoded': self.team_encoded,
            'delta_velocity': 0.0,
            'lateral_stress_cumul': lat * tl,
            'abrasive_stress_cumul': ab * tl,
            'stress_x_temp': lat * tt * tl,
            'compound_x_abrasivity': ch * ab,
            'compound_x_lateral': ch * lat,
            'compound_x_tyrelife': ch * tl,
            'prev_stint_max_delta': 0.0,
            'stint_length': self.stint_length,
            'tyre_life_pct': get_tyre_life_pct(
                self.race_number,
                self.gp_id,
                None,
                tl,
                stint_length=self.stint_length,
                compound_hardness=ch,
            ),
        }])

    @classmethod
    def from_dataset_row(
        cls,
        driver_row: pd.Series,
        rival_row: pd.Series,
        total_laps: int,
        gap: float,
    ) -> 'RaceState':
        return cls(
            lap=int(driver_row['LapNumber']),
            total_laps=total_laps,
            tyre_life=float(driver_row['TyreLife']),
            compound=str(driver_row.get('Compound', 'UNKNOWN')),
            compound_hardness=int(driver_row['CompoundEncoded']),
            fuel_load=float(driver_row['FuelLoad']),
            delta_to_best=float(driver_row['DeltaToBest']),
            delta_velocity=float(driver_row['delta_velocity']),
            stint_number=float(driver_row['Stint']),
            stint_length=float(driver_row['stint_length']),
            tyre_life_pct=float(driver_row['tyre_life_pct']),
            prev_stint_max_delta=float(driver_row['prev_stint_max_delta']),
            track_temp=float(driver_row['TrackTemp']),
            abrasivity=float(driver_row['Abrasivity']),
            lateral=float(driver_row['LateralEnergy']),
            race_number=int(driver_row['RaceNumber']),
            team_encoded=int(driver_row['TeamEncoded']),
            compound_x_abrasivity=float(driver_row['compound_x_abrasivity']),
            compound_x_lateral=float(driver_row['compound_x_lateral']),
            lateral_stress_cumul=float(driver_row['lateral_stress_cumul']),
            abrasive_stress_cumul=float(driver_row['abrasive_stress_cumul']),
            stress_x_temp=float(driver_row['stress_x_temp']),
            compound_x_tyrelife=float(driver_row['compound_x_tyrelife']),
            rival_tyre_life=float(rival_row['TyreLife']),
            rival_compound_hardness=int(rival_row['CompoundEncoded']),
            rival_delta_to_best=float(rival_row['DeltaToBest']),
            rival_fuel_load=float(rival_row['FuelLoad']),
            rival_stint=float(rival_row['Stint']),
            gap_to_rival=gap,
        )

    def _estimate_stint_length(self, compound: int) -> float:
        base_stint_map = {1: 48, 2: 45, 3: 38, 4: 28, 5: 20}
        gp_id = self.gp_id or RACE_NUMBER_TO_GP_ID.get(self.race_number)
        gp_data = self.GP_STRATEGY_DATA.get(gp_id)
        if not gp_data:
            return base_stint_map.get(compound, 30)
        if compound not in gp_data['compounds']:
            print(f"Warning: Compound C{compound} non autorise a {gp_id}")
        base_len = base_stint_map.get(compound, 30)
        return base_len * gp_data['wear_factor']

    def get_pit_loss(self) -> float:
        gp_id = self.gp_id or RACE_NUMBER_TO_GP_ID.get(self.race_number)
        return self.GP_STRATEGY_DATA.get(gp_id, {}).get('pit_loss', 23.0)

    def transition(self, action: str, model, noise: float = 0.0) -> 'RaceState':
        new = deepcopy(self)
        new.lap += 1
        new.fuel_load -= 1
        new.rival_fuel_load -= 1
        rival_delta = model.predict(self.to_rival_features())[0]
        rival_delta = max(0, rival_delta + np.random.normal(0, 0.1))

        new.rival_delta_to_best = rival_delta
        new.rival_tyre_life += 1

        if action == 'STAY_OUT':
            delta_pred = model.predict(self.to_features())[0]
            delta_pred = max(0, delta_pred + noise)
            prev_delta = new.delta_to_best
            new.delta_to_best = delta_pred
            new.delta_velocity = new.delta_to_best - prev_delta
            new.tyre_life += 1
            new.tyre_life_pct = get_tyre_life_pct(
                new.race_number,
                new.gp_id,
                new.compound,
                new.tyre_life,
                stint_length=new.stint_length,
                compound_hardness=new.compound_hardness,
            )

            new.lateral_stress_cumul = new.lateral * new.tyre_life
            new.abrasive_stress_cumul = new.abrasivity * new.tyre_life
            new.stress_x_temp = new.lateral * new.track_temp * new.tyre_life
            new.compound_x_tyrelife = new.compound_hardness * new.tyre_life
            new.gap_to_rival += (rival_delta - delta_pred)
        else:
            compound_str = get_compound_from_pit_action(action, default=new.compound)
            new.compound = compound_str
            gp_name = self.gp_id or RACE_NUMBER_TO_GP_ID.get(self.race_number)
            new.compound_hardness = get_compound_hardness(gp_name, compound_str, self.compound_hardness)
            pit_time = np.random.normal(self.get_pit_loss(), Pit_Time_SIGMA)
            new.gap_to_rival -= pit_time

            new.prev_stint_max_delta = new.delta_to_best
            new.tyre_life = 1
            new.delta_to_best = 0.0
            new.delta_velocity = 0.0
            new.stint_number += 1
            new.stint_length = self._estimate_stint_length(new.compound_hardness)
            new.tyre_life_pct = get_tyre_life_pct(
                new.race_number,
                new.gp_id,
                new.compound,
                new.tyre_life,
                stint_length=new.stint_length,
                compound_hardness=new.compound_hardness,
            )

            new.compound_x_abrasivity = new.compound_hardness * new.abrasivity
            new.compound_x_lateral = new.compound_hardness * new.lateral
            new.lateral_stress_cumul = new.lateral * new.tyre_life
            new.abrasive_stress_cumul = new.abrasivity * new.tyre_life
            new.stress_x_temp = new.lateral * new.track_temp * new.tyre_life
            new.compound_x_tyrelife = new.compound_hardness * new.tyre_life

        return new
