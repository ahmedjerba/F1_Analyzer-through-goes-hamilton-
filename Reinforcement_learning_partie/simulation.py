import numpy as np 
from copy import deepcopy
from race_state import RaceState
from race_state import *



def _resolve_gp_code_from_state(state):
    """Best-effort GP code resolution for get_compound_hardness_by_gp_code."""
    gp_code = getattr(state, 'gp_code', None)
    if gp_code:
        return str(gp_code).strip().upper()

    gp_name = getattr(state, 'gp_id', None)
    if not gp_name and hasattr(state, 'race_number'):
        gp_name = RACE_NUMBER_TO_GP_ID.get(state.race_number)

    if gp_name:
        for code, race_name in GP_CODE_TO_RACE.items():
            if race_name == gp_name:
                return code

    return None

def simulate_one_scenario(init_state,model,test_action,test_lap,df_preprocessed,noise_level=0.05):
    """
    Simulate one scenario of the race
    Parameters
    ----------
    init_state : dict
        The initial state           
    model : object
        The model to use for the simulation
    test_action : list
        The action to test
    test_lap : int
        The lap to test
    df_preprocessed : pd.DataFrame
        The preprocessed dataset
    noise_level : float, optional
        The level of noise to add to the simulation, by default 0.05
    Returns
    -------
    dict
        The final state of the simulation
    """
    state=deepcopy(init_state)
    model_features = list(model.feature_names_in_)
    
    
    current_rival_compound_str=get_compound_label_from_hardness_by_gp_code(state.RaceNumber, state.rival_compound_hardness)
    current_rival_compound_int=state.rival_compound_hardness
    rival_limit=state.estimate_length_limit(current_rival_compound_int)
    pit_loss=state.get_pit_loss()
    while state.lap<state.total_laps:
        action='STAY_OUT'
        if state.lap==test_lap:
            action=test_action
        rival_pitted_this_lap = False
        lap_context = df_preprocessed[
            (df_preprocessed['RaceNumber'] == state.RaceNumber) & 
            (df_preprocessed['LapNumber'] == state.lap)
        ]
        if not lap_context.empty:
            row = lap_context.iloc[0]
            # On met à jour l'état avec les données du preprocess
            state.track_temp = row['TrackTemp']
            state.abrasivity = row['Abrasivity']
            state.lateral = row['LateralEnergy']
            state.team_encoded = row['TeamEncoded']
        
        # Le rival regarde s'il a atteint 85% de la limite de son pneu actuel
        if state.rival_tyre_life >= (rival_limit * 0.85):
            rival_pitted_this_lap = True
            
            # --- CHOIX DU PROCHAIN PNEU DU RIVAL ---
            laps_remaining = state.total_laps - state.lap
            
            if laps_remaining < 15:
                current_rival_compound_str = 'SOFT'
                if state.RaceNumber is not None:
                    current_rival_compound_int = get_compound_hardness_by_gp_code(state.RaceNumber, 'SOFT')
                else:
                    gp_name = state.gp_id or RACE_NUMBER_TO_GP_ID.get(state.race_number)
                    current_rival_compound_int = get_compound_hardness(gp_name, 'SOFT', current_rival_compound_int)
                
            elif laps_remaining < 30:
                current_rival_compound_str = 'MEDIUM'
                if state.RaceNumber is not None:
                    current_rival_compound_int = get_compound_hardness_by_gp_code(state.RaceNumber, 'MEDIUM')
                else:
                    gp_name = state.gp_id or RACE_NUMBER_TO_GP_ID.get(state.race_number)
                    current_rival_compound_int = get_compound_hardness(gp_name, 'MEDIUM', current_rival_compound_int)
            else:
                current_rival_compound_str = 'HARD'
                if state.RaceNumber is not None:
                    current_rival_compound_int = get_compound_hardness_by_gp_code(state.RaceNumber, 'HARD')
                else:
                    gp_name = state.gp_id or RACE_NUMBER_TO_GP_ID.get(state.race_number)
                    current_rival_compound_int = get_compound_hardness(gp_name, 'HARD', current_rival_compound_int)
            # --- MISE À JOUR DE LA LIMITE ---
            # On recalcule la limite pour le nouveau pneu choisi
            rival_limit = state.estimate_length_limit(current_rival_compound_int)
            state.rival_tyre_life = 0
        current_noise = np.random.normal(0, noise_level)
        state = state.transition(action, model, noise=current_noise)
        if rival_pitted_this_lap:
            current_rival_compound_str = get_compound_label_from_hardness_by_gp_code(state.RaceNumber, current_rival_compound_int)  
            state.rival_compound_hardness = current_rival_compound_int
            state.rival_tyre_life = 0
            state.time += pit_loss
    return state.gap_to_rival
    
        
    