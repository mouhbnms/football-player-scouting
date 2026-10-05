import pandas as pd
from sklearn.preprocessing import MinMaxScaler


SCOUTING_ATTRIBUTES = [
    'movement_acceleration',
    'movement_sprint_speed',
    'skill_dribbling',
    'skill_ball_control',
    'movement_agility',
    'movement_balance',
    'attacking_short_passing',
    'skill_long_passing',
    'mentality_vision',
    'skill_curve',
    'skill_fk_accuracy',
    'attacking_crossing',
    'attacking_finishing',
    'attacking_volleys',
    'power_shot_power',
    'power_long_shots',
    'mentality_positioning',
    'mentality_composure',
    'mentality_interceptions',
    'defending_awareness',
    'defending_standing_tackle',
    'defending_sliding_tackle',
    'attacking_heading_accuracy',
    'power_jumping',
    'power_strength',
    'power_stamina',
    'skill_moves',
    'weak_foot',
    'mentality_aggression',
    'movement_reactions',
    'mentality_penalties',
    'goalkeeping_diving',
    'goalkeeping_handling',
    'goalkeeping_kicking',
    'goalkeeping_positioning',
    'goalkeeping_reflexes'
]


ABILITY_GROUPS = {
    'pace': [
        'movement_acceleration',
        'movement_sprint_speed'
    ],

    'dribbling_control': [
        'skill_dribbling',
        'skill_ball_control'
    ],

    'agility': [
        'movement_agility',
        'movement_balance'
    ],

    'passing_technique': [
        'attacking_short_passing',
        'skill_long_passing'
    ],

    'shooting_power': [
        'power_shot_power',
        'power_long_shots'
    ],

    'defending': [
        'mentality_interceptions',
        'defending_awareness'
    ],

    'tackling': [
        'defending_standing_tackle',
        'defending_sliding_tackle'
    ],

    'aerial': [
        'attacking_heading_accuracy',
        'power_jumping'
    ],

    'physical': [
        'power_strength',
        'power_stamina'
    ],
    'goalkeeping': [
        'goalkeeping_diving',
        'goalkeeping_handling',
        'goalkeeping_kicking',
        'goalkeeping_positioning',
        'goalkeeping_reflexes'
    ]
}


INDIVIDUAL_ABILITIES = [
    'attacking_finishing',
    'attacking_volleys',
    'attacking_crossing',
    'mentality_vision',
    'mentality_positioning',
    'mentality_composure',
    'movement_reactions',
    'mentality_aggression',
    'skill_moves',
    'weak_foot',
    'mentality_penalties',
    'skill_curve',
    'skill_fk_accuracy',

]


def normalize_attributes(df):
    """Normalize scouting attributes to [0, 1]."""

    df = df.copy()

    scaler = MinMaxScaler()

    df[SCOUTING_ATTRIBUTES] = scaler.fit_transform(
        df[SCOUTING_ATTRIBUTES]
    )

    return df


def create_ability_features(df):
    df = df.copy()

    for group_name, features in ABILITY_GROUPS.items():
        df[group_name] = df[features].mean(axis=1)

    df["stamina"] = df["power_stamina"]

    return df


def get_base_playstyle(playstyle):
    if playstyle.endswith("+"):
        return playstyle[:-1]
    return playstyle


def prepare_playstyles(df):
    df = df.copy()

    df["playstyle_list"] = (
        df["playstyles"]
        .fillna("")
        .str.split(", ")
    )

    df["playstyle_list"] = (
        df["playstyle_list"]
        .apply(lambda styles: [style for style in styles if style])
    )

    df["base_playstyles"] = (
        df["playstyle_list"]
        .apply(
            lambda styles: list(
                dict.fromkeys(
                    get_base_playstyle(style)
                    for style in styles
                )
            )
        )
    )

    return df

ABILITY_FEATURES = (
    list(ABILITY_GROUPS.keys())
    + INDIVIDUAL_ABILITIES
    + ["stamina"]
)