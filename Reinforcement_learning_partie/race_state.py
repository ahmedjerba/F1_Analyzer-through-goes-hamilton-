from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

Pit_Time_MU = 22.5
Pit_Time_SIGMA = 0.8

F1_SEASON_2024_CONFIG = {
    'Bahrain': {
        'code': 'BAH', 'round': 1, 
        'compounds': {'Hard': 1, 'Medium': 2, 'Soft': 3},
        'wear_factor': 0.85, 'pit_loss': 23.0,
        'tyre_limits': {'Hard': 34, 'Medium': 24, 'Soft': 16}
    },
    'Saudi Arabia': {
        'code': 'SAU', 'round': 2,
        'compounds': {'Hard': 2, 'Medium': 3, 'Soft': 4},
        'wear_factor': 1.10, 'pit_loss': 24.0,
        'tyre_limits': {'Hard': 48, 'Medium': 34, 'Soft': 22}
    },
    'Australia': { # Remplacement de 'Melbourne'
        'code': 'AUS', 'round': 3,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 1.05, 'pit_loss': 21.0,
        'tyre_limits': {'Hard': 38, 'Medium': 26, 'Soft': 14}
    },
    'Japan': { # Remplacement de 'Suzuka'
        'code': 'JPN', 'round': 4,
        'compounds': {'Hard': 1, 'Medium': 2, 'Soft': 3},
        'wear_factor': 0.80, 'pit_loss': 22.5,
        'tyre_limits': {'Hard': 32, 'Medium': 22, 'Soft': 15}
    },
    'China': {
        'code': 'CHN', 'round': 5,
        'compounds': {'Hard': 2, 'Medium': 3, 'Soft': 4},
        'wear_factor': 0.95, 'pit_loss': 25.5,
        'tyre_limits': {'Hard': 35, 'Medium': 25, 'Soft': 17}
    },
    'Miami': {
        'code': 'MIA', 'round': 6,
        'compounds': {'Hard': 2, 'Medium': 3, 'Soft': 4},
        'wear_factor': 1.15, 'pit_loss': 24.5,
        'tyre_limits': {'Hard': 42, 'Medium': 28, 'Soft': 18}
    },
    'Emilia-Romagna': { # Remplacement de 'Imola'
        'code': 'EMI', 'round': 7,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 1.00, 'pit_loss': 23.0,
        'tyre_limits': {'Hard': 44, 'Medium': 30, 'Soft': 20}
    },
    'Monaco': {
        'code': 'MON', 'round': 8,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 1.50, 'pit_loss': 25.0,
        'tyre_limits': {'Hard': 60, 'Medium': 45, 'Soft': 30}
    },
    'Canada': {
        'code': 'CAN', 'round': 9,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 1.25, 'pit_loss': 22.5,
        'tyre_limits': {'Hard': 48, 'Medium': 32, 'Soft': 20}
    },
    'Spain': {
        'code': 'ESP', 'round': 10,
        'compounds': {'Hard': 1, 'Medium': 2, 'Soft': 3},
        'wear_factor': 0.82, 'pit_loss': 23.0,
        'tyre_limits': {'Hard': 34, 'Medium': 24, 'Soft': 16}
    },
    'Austria': {
        'code': 'AUT', 'round': 11,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 0.88, 'pit_loss': 20.5,
        'tyre_limits': {'Hard': 40, 'Medium': 28, 'Soft': 18}
    },
    'Great Britain': { # Remplacement de 'Silverstone'
        'code': 'GBR', 'round': 12,
        'compounds': {'Hard': 1, 'Medium': 2, 'Soft': 3},
        'wear_factor': 0.75, 'pit_loss': 20.0,
        'tyre_limits': {'Hard': 35, 'Medium': 25, 'Soft': 17}
    },
    'Hungary': {
        'code': 'HUN', 'round': 13,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 0.92, 'pit_loss': 23.5,
        'tyre_limits': {'Hard': 40, 'Medium': 28, 'Soft': 18}
    },
    'Belgium': { # Remplacement de 'Spa'
        'code': 'BEL', 'round': 14,
        'compounds': {'Hard': 2, 'Medium': 3, 'Soft': 4},
        'wear_factor': 0.90, 'pit_loss': 23.5,
        'tyre_limits': {'Hard': 30, 'Medium': 22, 'Soft': 14}
    },
    'Netherlands': { # Remplacement de 'Zandvoort'
        'code': 'NED', 'round': 15,
        'compounds': {'Hard': 1, 'Medium': 2, 'Soft': 3},
        'wear_factor': 0.82, 'pit_loss': 18.0,
        'tyre_limits': {'Hard': 42, 'Medium': 30, 'Soft': 20}
    },
    'Italy': { # Remplacement de 'Monza'
        'code': 'ITA', 'round': 16,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 1.05, 'pit_loss': 24.0,
        'tyre_limits': {'Hard': 40, 'Medium': 28, 'Soft': 18}
    },
    'Azerbaijan': { # Remplacement de 'Baku'
        'code': 'AZE', 'round': 17,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 1.15, 'pit_loss': 26.0,
        'tyre_limits': {'Hard': 42, 'Medium': 28, 'Soft': 18}
    },
    'Singapore': {
        'code': 'SIN', 'round': 18,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 1.20, 'pit_loss': 28.5,
        'tyre_limits': {'Hard': 42, 'Medium': 28, 'Soft': 18}
    },
    'USA': { # Remplacement de 'Austin'
        'code': 'USA', 'round': 19,
        'compounds': {'Hard': 2, 'Medium': 3, 'Soft': 4},
        'wear_factor': 0.88, 'pit_loss': 24.0,
        'tyre_limits': {'Hard': 36, 'Medium': 24, 'Soft': 16}
    },
    'Mexico': {
        'code': 'MEX', 'round': 20,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 1.10, 'pit_loss': 24.5,
        'tyre_limits': {'Hard': 50, 'Medium': 35, 'Soft': 22}
    },
    'Brazil': { # Remplacement de 'Interlagos'
        'code': 'BRA', 'round': 21,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 0.95, 'pit_loss': 21.0,
        'tyre_limits': {'Hard': 40, 'Medium': 28, 'Soft': 18}
    },
    'Las Vegas': {
        'code': 'LVS', 'round': 22,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 1.35, 'pit_loss': 27.0,
        'tyre_limits': {'Hard': 48, 'Medium': 32, 'Soft': 20}
    },
    'Qatar': {
        'code': 'QAT', 'round': 23,
        'compounds': {'Hard': 1, 'Medium': 2, 'Soft': 3},
        'wear_factor': 0.70, 'pit_loss': 25.0,
        'tyre_limits': {'Hard': 30, 'Medium': 22, 'Soft': 16}
    },
    'Abu Dhabi': {
        'code': 'ABU', 'round': 24,
        'compounds': {'Hard': 3, 'Medium': 4, 'Soft': 5},
        'wear_factor': 1.10, 'pit_loss': 24.0,
        'tyre_limits': {'Hard': 40, 'Medium': 28, 'Soft': 18}
    }
}
def get_gp_param(gp_name, param_name, sub_param=None):
    """
    Récupère un paramètre spécifique pour un Grand Prix donné.
    
    Arguments:
    - gp_name (str): Le nom du Grand Prix (ex: 'Bahrain', 'Japan').
    - param_name (str): Le paramètre souhaité (ex: 'wear_factor', 'tyre_limits').
    - sub_param (str, optionnel): La clé spécifique si le paramètre est un dictionnaire (ex: 'Soft').
    
    Retourne:
    - La valeur du paramètre demandé, ou un message d'erreur si introuvable.
    """
    # 1. Vérification du Grand Prix
    if gp_name not in F1_SEASON_2024_CONFIG:
        raise KeyError(f"Le Grand Prix '{gp_name}' n'existe pas dans le calendrier.")
        
    gp_data = F1_SEASON_2024_CONFIG[gp_name]
    
    # 2. Vérification du paramètre principal
    if param_name not in gp_data:
        raise KeyError(f"Le paramètre '{param_name}' n'existe pas pour {gp_name}.")
        
    # 3. Gestion des paramètres imbriqués (ex: on veut juste la limite des pneus Soft)
    if sub_param is not None:
        if isinstance(gp_data[param_name], dict):
            if sub_param in gp_data[param_name]:
                return gp_data[param_name][sub_param]
            else:
                raise KeyError(f"Le sous-paramètre '{sub_param}' n'existe pas dans '{param_name}'.")
        else:
            raise ValueError(f"Le paramètre '{param_name}' n'est pas un dictionnaire, impossible d'utiliser sub_param.")
            
    # 4. Retour normal
    return gp_data[param_name]

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
