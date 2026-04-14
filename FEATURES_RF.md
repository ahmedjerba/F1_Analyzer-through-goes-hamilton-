# Features envoyées au modèle RF

Ce fichier est la source de lecture rapide pour les features passées au modèle de régression random forest dans le pipeline F1.

Références principales:
- [Reinforcement_learning_partie/preprocess.py](Reinforcement_learning_partie/preprocess.py)
- [Reinforcement_learning_partie/Reinforcement.ipynb](Reinforcement_learning_partie/Reinforcement.ipynb)
- [Models.ipynb/model_2024.ipynb](Models.ipynb/model_2024.ipynb)

## Règle d'ordre

L'ordre des colonnes doit rester strictement identique à celui du modèle:

1. `CompoundEncoded`
2. `TyreLife`
3. `TrackTemp`
4. `FuelLoad`
5. `Abrasivity`
6. `LateralEnergy`
7. `DeltaToBest`
8. `LapNumber`
9. `Stint`
10. `RaceNumber`
11. `TeamEncoded`
12. `delta_velocity`
13. `lateral_stress_cumul`
14. `abrasive_stress_cumul`
15. `stress_x_temp`
16. `compound_x_abrasivity`
17. `compound_x_lateral`
18. `compound_x_tyrelife`
19. `prev_stint_max_delta`
20. `stint_length`
21. `tyre_life_pct`

## Détails de calcul

| Feature | Calcul / origine | Remarque |
|---|---|---|
| `CompoundEncoded` | Encodage du composé exact (`C1` à `C5`) avec `LabelEncoder` | Calculé pendant le preprocessing à partir de `Event` + `Compound` |
| `TyreLife` | Valeur brute du dataset / état courant | Vie du pneu au tour considéré |
| `TrackTemp` | Valeur brute du dataset / état courant | Température piste |
| `FuelLoad` | Valeur brute du dataset / état courant | Charge carburant |
| `Abrasivity` | Valeur brute du dataset / profil circuit | Abrasivité du circuit |
| `LateralEnergy` | Valeur brute du dataset / profil circuit | Énergie latérale du circuit |
| `DeltaToBest` | `CorrectedLapTime_Global - BestCorrectedByStint` | `BestCorrectedByStint` est le meilleur temps corrigé cumulé jusqu’au tour courant dans le même stint |
| `LapNumber` | Numéro de tour | Repris du dataset ou de la ligne de l’état |
| `Stint` | Numéro de stint | Repris du dataset ou de l’état |
| `RaceNumber` | Ordre chronologique de la course dans la saison | Mappage 2023-2024 |
| `TeamEncoded` | Encodage de l’équipe avec `LabelEncoder` | Déduit depuis `Driver` + `Year` puis fallback par pilote |
| `delta_velocity` | `groupby(['Driver', 'Stint'])[DeltaToBest].diff()` puis `fillna(0)` | Variation intra-stint du delta |
| `lateral_stress_cumul` | `LateralEnergy * TyreLife` | Feature dérivée de stress |
| `abrasive_stress_cumul` | `Abrasivity * TyreLife` | Feature dérivée de stress |
| `stress_x_temp` | `LateralEnergy * TrackTemp * TyreLife` | Interaction thermique |
| `compound_x_abrasivity` | `CompoundEncoded * Abrasivity` | Interaction composé / circuit |
| `compound_x_lateral` | `CompoundEncoded * LateralEnergy` | Interaction composé / circuit |
| `compound_x_tyrelife` | `CompoundEncoded * TyreLife` | Interaction composé / usure |
| `prev_stint_max_delta` | Max de `DeltaToBest` sur le stint précédent, puis `ffill` par pilote et course | Utilise `groupby(['Driver', 'RaceNumber', 'Stint'])['DeltaToBest'].max().groupby(['Driver', 'RaceNumber']).shift(1)` |
| `stint_length` | Max de `TyreLife` dans le stint | `groupby(['Driver', 'RaceNumber', 'Stint'])['TyreLife'].transform('max')` quand `Driver` existe |
| `tyre_life_pct` | `TyreLife / stint_length` | En `RaceState`, la valeur est recalculée via `get_tyre_life_pct(...)` |

## Points importants

- Le pipeline final garde uniquement ces 21 colonnes, dans cet ordre.
- Si une colonne manque au moment du preprocessing, elle est reconstruite si possible, sinon remplie à `0`.
- Les colonnes de fuite ou non utilisées par le modèle sont écartées avant l’entraînement et avant l’inférence.
- La fonction de référence pour l’alignement des features est `_get_model_features(df)` dans [Reinforcement_learning_partie/preprocess.py](Reinforcement_learning_partie/preprocess.py).

## Version courte

Si tu veux vérifier rapidement ce qui entre au RF, prends simplement la liste `features` ci-dessus et garde exactement le même ordre partout où tu appelles `model.predict(...)`.
