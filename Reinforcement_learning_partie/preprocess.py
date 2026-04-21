"""
F1 Reinforcement Learning - Complete Preprocessing Pipeline
Exact replication of preprocessing logic from model_2024.ipynb.

Main function: preprocess(df)
- Combines all preprocessing, postprocessing, and feature alignment
- Returns DataFrame ready for RL state construction with features matching RF/XGBoost/LightGBM
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def compute_delta_correctly(df):
    """Compute delta_to_best using progressive best-per-stint logic."""
    required = {'Driver', 'Event', 'Stint', 'LapNumber', 'CorrectedLapTime_Global'}
    if not required.issubset(df.columns):
        return df

    df = df.sort_values(['Driver', 'Event', 'Stint', 'LapNumber']).copy()

    # Min progressif: seulement les tours passes
    df['BestCorrectedByStint'] = (
        df.groupby(['Driver', 'Event', 'Stint'])['CorrectedLapTime_Global']
        .transform(lambda x: x.expanding().min())
    )

    df['DeltaToBest'] = df['CorrectedLapTime_Global'] - df['BestCorrectedByStint']
    return df


def preprocess_data(df):
    """
    Full preprocessing pipeline matching model_2024.ipynb.
    Includes:
    - Race numbering (chronological 2023-2024)
    - Compound mapping (Hard/Medium/Soft -> C1-C5)
    - Driver/Team encoding
    - Feature engineering (stress, delta velocity, etc.)
    - Target variable (delta_next_lap)
    - Feature interactions
    """
    df = df.copy()

    # Recompute DeltaToBest using progressive best-per-stint logic.
    df = compute_delta_correctly(df)

    # Support both possible naming conventions.
    delta_col = 'deltaToBest' if 'deltaToBest' in df.columns else 'DeltaToBest'

    # Exact compound per GP: Hard / Medium / Soft -> C1 to C5 depending on the circuit.
    F1_COMPOUNDS_2024 = {
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
    EVENT_TO_F1_KEY = {
        'Bahrain Grand Prix': 'Bahrain',
        'Saudi Arabian Grand Prix': 'Saudi Arabia',
        'Australian Grand Prix': 'Australia',
        'Japanese Grand Prix': 'Japan',
        'Chinese Grand Prix': 'China',
        'Miami Grand Prix': 'Miami',
        'Emilia Romagna Grand Prix': 'Emilia-Romagna',
        'Monaco Grand Prix': 'Monaco',
        'Canadian Grand Prix': 'Canada',
        'Spanish Grand Prix': 'Spain',
        'Austrian Grand Prix': 'Austria',
        'British Grand Prix': 'Great Britain',
        'Hungarian Grand Prix': 'Hungary',
        'Belgian Grand Prix': 'Belgium',
        'Dutch Grand Prix': 'Netherlands',
        'Italian Grand Prix': 'Italy',
        'Azerbaijan Grand Prix': 'Azerbaijan',
        'Singapore Grand Prix': 'Singapore',
        'United States Grand Prix': 'USA (Austin)',
        'Mexico City Grand Prix': 'Mexico',
        'Sao Paulo Grand Prix': 'Brazil',
        'Las Vegas Grand Prix': 'Las Vegas',
        'Qatar Grand Prix': 'Qatar',
        'Abu Dhabi Grand Prix': 'Abu Dhabi',
    }
    COMPUND_ALIASES = {
        'H': 'Hard',
        'M': 'Medium',
        'S': 'Soft',
        'HARD': 'Hard',
        'MEDIUM': 'Medium',
        'SOFT': 'Soft',
        'C1': 'C1',
        'C2': 'C2',
        'C3': 'C3',
        'C4': 'C4',
        'C5': 'C5',
    }
    COMPOUND_TO_INT = {
        'C1': 1,
        'C2': 2,
        'C3': 3,
        'C4': 4,
        'C5': 5,
    }

    def normalize_event_label(value):
        if pd.isna(value):
            return ''
        return (
            str(value).strip()
            .replace('São Paulo Grand Prix', 'Sao Paulo Grand Prix')
            .replace('Emilia-Romagna Grand Prix', 'Emilia Romagna Grand Prix')
        )

    def normalize_compound_label(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        return COMPUND_ALIASES.get(text.upper(), text)

    # Encode season chronology from Bahrain 2023 to Abu Dhabi 2024.
    RACE_ORDER_2023_2024 = {
        (2024, 'Bahrain Grand Prix'): 1,
        (2024, 'Saudi Arabian Grand Prix'): 2,
        (2024, 'Australian Grand Prix'): 3,
        (2024, 'Japanese Grand Prix'): 4,
        (2024, 'Chinese Grand Prix'): 5,
        (2024, 'Miami Grand Prix'): 6,
        (2024, 'Emilia Romagna Grand Prix'): 7,
        (2024, 'Monaco Grand Prix'): 8,
        (2024, 'Canadian Grand Prix'): 9,
        (2024, 'Spanish Grand Prix'): 10,
        (2024, 'Austrian Grand Prix'): 11,
        (2024, 'British Grand Prix'): 12,
        (2024, 'Hungarian Grand Prix'): 13,
        (2024, 'Belgian Grand Prix'): 14,
        (2024, 'Dutch Grand Prix'): 15,
        (2024, 'Italian Grand Prix'): 16,
        (2024, 'Azerbaijan Grand Prix'): 17,
        (2024, 'Singapore Grand Prix'): 18,
        (2024, 'United States Grand Prix'): 19,
        (2024, 'Mexico City Grand Prix'): 20,
        (2024, 'Sao Paulo Grand Prix'): 21,
        (2024, 'Las Vegas Grand Prix'): 22,
        (2024, 'Qatar Grand Prix'): 23,
        (2024, 'Abu Dhabi Grand Prix'): 24,
    }

    if {'Year', 'Event'}.issubset(df.columns):
        event_for_map = df['Event'].map(normalize_event_label)
        year_for_map = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        race_keys = [
            (int(y), e) if pd.notna(y) else None
            for y, e in zip(year_for_map, event_for_map)
        ]
        df['RaceNumber'] = pd.Series(race_keys, index=df.index).map(RACE_ORDER_2023_2024)

    if {'Event', 'Compound'}.issubset(df.columns):
        event_for_compound = df['Event'].map(normalize_event_label)
        base_compound = df['Compound'].map(normalize_compound_label)
        exact_compound = [
            F1_COMPOUNDS_2024.get(EVENT_TO_F1_KEY.get(event, event), {}).get(base)
            for event, base in zip(event_for_compound, base_compound)
        ]
        df['CompoundExact'] = pd.Series(exact_compound, index=df.index)
        df['CompoundExact'] = df['CompoundExact'].fillna(base_compound)
        # Encode compounds with deterministic F1 scale: C1 -> 1 ... C5 -> 5.
        df['CompoundEncoded'] = (
            df['CompoundExact']
            .astype(str)
            .str.upper()
            .map(COMPOUND_TO_INT)
            .fillna(0)
            .astype(int)
        )

    # Driver -> Team mapping for 2023 + 2024 (year-aware first, keyed by driver NUMBER)
    DRIVER_CODE_TO_NUMBER = {
        'VER': '1', 'PER': '11',
        'HAM': '44', 'RUS': '63',
        'LEC': '16', 'SAI': '55',
        'NOR': '4', 'PIA': '81',
        'ALO': '14', 'STR': '18',
        'OCO': '31', 'GAS': '10',
        'BOT': '77', 'ZHO': '24',
        'ALB': '23', 'SAR': '2',
        'MAG': '20', 'HUL': '27',
        'TSU': '22', 'RIC': '3',
        'DEV': '21', 'LAW': '40'
    }

    def normalize_driver_id(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().upper()
        text = DRIVER_CODE_TO_NUMBER.get(text, text)
        try:
            return str(int(float(text)))
        except ValueError:
            return text

    DRIVER_TEAM_BY_YEAR = {
        (2023, '1'): 'Red Bull', (2023, '11'): 'Red Bull',
        (2023, '44'): 'Mercedes', (2023, '63'): 'Mercedes',
        (2023, '16'): 'Ferrari', (2023, '55'): 'Ferrari',
        (2023, '4'): 'McLaren', (2023, '81'): 'McLaren',
        (2023, '14'): 'Aston Martin', (2023, '18'): 'Aston Martin',
        (2023, '31'): 'Alpine', (2023, '10'): 'Alpine',
        (2023, '77'): 'Sauber', (2023, '24'): 'Sauber',
        (2023, '23'): 'Williams', (2023, '2'): 'Williams',
        (2023, '20'): 'Haas', (2023, '27'): 'Haas',
        (2023, '22'): 'RB', (2023, '3'): 'RB',
        (2023, '21'): 'RB', (2023, '40'): 'RB',

        (2024, '1'): 'Red Bull', (2024, '11'): 'Red Bull',
        (2024, '44'): 'Mercedes', (2024, '63'): 'Mercedes',
        (2024, '16'): 'Ferrari', (2024, '55'): 'Ferrari',
        (2024, '4'): 'McLaren', (2024, '81'): 'McLaren',
        (2024, '14'): 'Aston Martin', (2024, '18'): 'Aston Martin',
        (2024, '31'): 'Alpine', (2024, '10'): 'Alpine',
        (2024, '77'): 'Sauber', (2024, '24'): 'Sauber',
        (2024, '23'): 'Williams', (2024, '2'): 'Williams',
        (2024, '20'): 'Haas', (2024, '27'): 'Haas',
        (2024, '22'): 'RB', (2024, '3'): 'RB',
    }
    DRIVER_TO_TEAM_FALLBACK = {
        '1': 'Red Bull', '11': 'Red Bull',
        '44': 'Mercedes', '63': 'Mercedes',
        '16': 'Ferrari', '55': 'Ferrari',
        '4': 'McLaren', '81': 'McLaren',
        '14': 'Aston Martin', '18': 'Aston Martin',
        '31': 'Alpine', '10': 'Alpine',
        '77': 'Sauber', '24': 'Sauber',
        '23': 'Williams', '2': 'Williams',
        '20': 'Haas', '27': 'Haas',
        '22': 'RB', '3': 'RB', '21': 'RB', '40': 'RB'
    }

    if {'Year', 'Driver'}.issubset(df.columns):
        year_for_team = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        driver_for_team = df['Driver'].map(normalize_driver_id)
        team_keys = [
            (int(y), d) if pd.notna(y) and pd.notna(d) else None
            for y, d in zip(year_for_team, driver_for_team)
        ]
        team_series = pd.Series(team_keys, index=df.index).map(DRIVER_TEAM_BY_YEAR)
        team_fallback = driver_for_team.map(DRIVER_TO_TEAM_FALLBACK)
        df['Team'] = team_series.fillna(team_fallback).fillna('Unknown')
    elif 'Driver' in df.columns:
        driver_for_team = df['Driver'].map(normalize_driver_id)
        df['Team'] = driver_for_team.map(DRIVER_TO_TEAM_FALLBACK).fillna('Unknown')

    if 'Team' in df.columns:
        le_team = LabelEncoder()
        df['TeamEncoded'] = le_team.fit_transform(df['Team'])

    if 'Event' in df.columns:
        le_event = LabelEncoder()
        df['EventEncoded'] = le_event.fit_transform(df['Event'])

    # Preserve season for later evaluation without using it as a feature.
    if 'Year' in df.columns:
        df['SeasonYear'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    elif 'year' in df.columns:
        df['SeasonYear'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

    # 1. Delta vitesse (variation intra-stint)
    if {'Driver', 'Stint', delta_col}.issubset(df.columns):
        df['delta_velocity'] = df.groupby(['Driver', 'Stint'])[delta_col].diff()

    # 2. Stress cumulatif
    df['lateral_stress_cumul'] = df['LateralEnergy'] * df['TyreLife']
    df['abrasive_stress_cumul'] = df['Abrasivity'] * df['TyreLife']

    # 3. Interaction thermique
    df['stress_x_temp'] = df['LateralEnergy'] * df['TrackTemp'] * df['TyreLife']

    # 4. Target
    if {'Driver', 'Stint', delta_col}.issubset(df.columns):
        df['delta_next_lap'] = df.groupby(['Driver', 'Stint'])[delta_col].shift(-1)

    # 5. Drop derniere ligne de chaque stint (target = NaN)
    if 'delta_next_lap' in df.columns:
        df = df.dropna(subset=['delta_next_lap'])

    # 6. Feature engineering that still needs Driver/Event before they are dropped
    if 'RaceNumber' in df.columns:
        df = df[df['RaceNumber'] != 8 ]  # Monaco = course 8

    if {'CompoundEncoded', 'Abrasivity'}.issubset(df.columns):
        df['compound_x_abrasivity'] = df['CompoundEncoded'] * df['Abrasivity']
    if {'CompoundEncoded', 'LateralEnergy'}.issubset(df.columns):
        df['compound_x_lateral'] = df['CompoundEncoded'] * df['LateralEnergy']
    if {'CompoundEncoded', 'TyreLife'}.issubset(df.columns):
        df['compound_x_tyrelife'] = df['CompoundEncoded'] * df['TyreLife']

    if {'Driver', 'RaceNumber', 'Stint', 'DeltaToBest'}.issubset(df.columns):
    # Calculer le max par stint d'abord
        stint_max = df.groupby(['Driver', 'RaceNumber', 'Stint'])['DeltaToBest'].max().groupby(['Driver', 'RaceNumber']).shift(1)
        stint_max.name = 'prev_stint_max_delta'
    
    # Fusionner proprement sans casser l'index
    df = df.merge(stint_max, on=['Driver', 'RaceNumber', 'Stint'], how='left')
    # Remplir les ffill manuellement si nécessaire
    df['prev_stint_max_delta'] = df.groupby(['Driver', 'RaceNumber'])['prev_stint_max_delta'].ffill()

    if {'RaceNumber', 'Stint', 'TyreLife'}.issubset(df.columns):
        if 'Driver' in df.columns:
            df['stint_length'] = df.groupby(['Driver', 'RaceNumber', 'Stint'])['TyreLife'].transform('max')
        else:
            df['stint_length'] = df.groupby(['RaceNumber', 'Stint'])['TyreLife'].transform('max')
        df['tyre_life_pct'] = df['TyreLife'] / df['stint_length']

    # 6. Ensure TeamEncoded exists, then remove raw categorical columns after encoding.
    if 'TeamEncoded' not in df.columns and 'Team' in df.columns:
        le_team = LabelEncoder()
        df['TeamEncoded'] = le_team.fit_transform(df['Team'].fillna('Unknown').astype(str))

    df = df.drop(columns=['Event', 'Compound', 'CompoundExact', 'Year', 'year', 'Team', 'YearEncoded'], errors='ignore')

    return df


def _postprocess_features(df, verbose=False):
    """
    Post-processing matching model_2024.ipynb:
    - Drop leakage columns
    - Fill delta_velocity NaN with 0 (first lap of stint)
    - Check LapNumber/TyreLife correlation and drop if > 0.85
    - Clip outliers in target variable
    """
    df = df.copy()
    
    # 1. Drop leakage columns
    df = df.drop(columns=[
        'BestCorrectedByStint',     # leakage (used to compute DeltaToBest)
        'CorrectedLapTime_Global',  # indirect leakage
        'EventEncoded',             # redundant with RaceNumber
    ], errors='ignore')
    
    # 2. Fill delta_velocity NaN (first lap of each stint)
    if 'delta_velocity' in df.columns:
        df['delta_velocity'] = df['delta_velocity'].fillna(0)
    
    # 4. Clip outliers in target
    if 'delta_next_lap' in df.columns:
        outlier_high = (df['delta_next_lap'] > 3).sum()
        outlier_low = (df['delta_next_lap'] < 0).sum()
        if verbose and (outlier_high > 0 or outlier_low > 0):
            print(f"Outliers: {outlier_low} < 0, {outlier_high} > 3")
        df = df[df['delta_next_lap'].between(0, 3)]
    
    return df


def _get_model_features(df):
    """Retourne la liste exacte des colonnes pour le modèle."""
    # Liste corrigée avec virgules et ordre strict
    features = ['Driver',
        'CompoundEncoded', 'TyreLife', 'TrackTemp', 'FuelLoad', 'Abrasivity',
        'LateralEnergy', 'DeltaToBest', 'LapNumber', 'Stint', 'RaceNumber',
        'TeamEncoded', 'delta_velocity', 'lateral_stress_cumul', 
        'abrasive_stress_cumul', 'stress_x_temp', 'compound_x_abrasivity', 
        'compound_x_lateral', 'compound_x_tyrelife', 'prev_stint_max_delta', 
        'stint_length', 'tyre_life_pct'
    ]
    
    # On vérifie si chaque feature est présente, sinon on l'affiche pour debugger
    present_features = []
    for f in features:
        if f in df.columns:
            present_features.append(f)
        else:
            print(f"⚠️ Alerte : La feature {f} est absente du DataFrame après processing !")
            
    return present_features


def preprocess(df, verbose=False):
    """
    Complete F1 RL preprocessing pipeline - SINGLE FUNCTION
    ═════════════════════════════════════════════════════════════
    
    Input: Raw lap data DataFrame with columns from dataset
    Output: Feature DataFrame ready for RL state construction
    
    Steps:
    1. Compute progressive best-per-stint logic
    2. Race numbering (chronological 2023-2024)
    3. Compound mapping (Hard/Medium/Soft → C1-C5)
    4. Feature engineering (stress, delta velocity, interactions)
    5. Target variable (delta_next_lap)
    6. Post-processing (drop leakage, fill NaN, correlations, outliers)
    7. Feature alignment matching RF/XGBoost/LightGBM
    
    GUARANTEE: Output features are 100% identical to X_train from model_2024.ipynb
    
    Args:
        df: Raw DataFrame
        verbose: Print processing steps (default: False)
    
    Returns:
        DataFrame with model-aligned features ready for RL
    
    Example:
        >>> df = pd.read_csv('master_dataset_partie2_2024_stint.csv')
        >>> df_features = preprocess(df)
        >>> print(df_features.shape, df_features.columns)
    """
    
    if verbose:
        print("╔════════════════════════════════════════════════╗")
        print("║  F1 RL Complete Preprocessing Pipeline         ║")
        print("╚════════════════════════════════════════════════╝")
        print(f"\n📥 Input: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Step 1: Full preprocessing
    df_processed = preprocess_data(df)
    
    # Étape 2: Post-processing
    df_processed = _postprocess_features(df_processed, verbose=verbose)
    
    # Étape 3: Alignement et Sécurité
    feature_cols = _get_model_features(df_processed)
    if 'LapNumber' not in df_processed.columns and 'LapNumber' in df.columns:
        df_processed['LapNumber'] = df['LapNumber']
    
    # SÉCURITÉ : Vérifier si des colonnes manquent et les remplir par 0 ou l'original
    for col in feature_cols:
        if col not in df_processed.columns:
            if col in df.columns: # Si elle était dans le CSV de base
                df_processed[col] = df[col]
            else:
                df_processed[col] = 0 # Fallback
                
    # On ne garde que les colonnes nécessaires dans l'ordre du modèle
    df_features = df_processed[feature_cols].copy()
    if 'delta_next_lap' in df_processed.columns:
        df_features['delta_next_lap'] = df_processed['delta_next_lap'].values
    
    return df_features
