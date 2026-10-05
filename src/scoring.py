import pandas as pd

POSITION_WEIGHTS = {

    'GK': {
        'goalkeeping': 1.0
    },

    'CB': {
        'defending': 0.30,
            'tackling': 0.30,
            'aerial': 0.15,
            'physical': 0.12,
            'pace': 0.08,
            'passing_technique': 0.05
    },

    'CDM': {
        'defending': 0.25,
            'tackling': 0.20,
            'passing_technique': 0.18,
            'mentality_vision': 0.12,
            'physical': 0.10,
            'mentality_aggression': 0.08,
            'dribbling_control': 0.07
    },

    'CM': {
        'passing_technique': 0.20,
            'mentality_vision': 0.15,
            'dribbling_control': 0.15,
            'defending': 0.12,
            'tackling': 0.10,
            'shooting_power': 0.10,
            'attacking_finishing': 0.08,
            'mentality_positioning': 0.05,
            'agility': 0.05
    },

    'CAM': {
        'dribbling_control': 0.18,
            'mentality_vision': 0.15,
            'skill_moves': 0.12,
            'attacking_finishing': 0.12,
            'passing_technique': 0.10,
            'shooting_power': 0.08,
            'mentality_positioning': 0.08,
            'agility': 0.07,
            'skill_curve': 0.05,
            'mentality_composure': 0.05
    },

    'ST': {
        'attacking_finishing': 0.25,
            'shooting_power': 0.15,
            'mentality_positioning': 0.15,
            'aerial': 0.12,
            'dribbling_control': 0.08,
            'pace': 0.08,
            'physical': 0.07,
            'skill_moves': 0.05,
            'mentality_composure': 0.05
    },

    'LW': {
        'dribbling_control': 0.20,
            'pace': 0.18,
            'skill_moves': 0.15,
            'attacking_finishing': 0.12,
            'agility': 0.10,
            'attacking_crossing': 0.08,
            'mentality_positioning': 0.07,
            'shooting_power': 0.05,
            'mentality_vision': 0.03,
            'mentality_composure': 0.02
    },

    'RW': {
        'dribbling_control': 0.20,
            'pace': 0.18,
            'skill_moves': 0.15,
            'attacking_finishing': 0.12,
            'agility': 0.10,
            'attacking_crossing': 0.08,
            'mentality_positioning': 0.07,
            'shooting_power': 0.05,
            'mentality_vision': 0.03,
            'mentality_composure': 0.02
    },

    'LM': {
        'pace': 0.17,
            'dribbling_control': 0.15,
            'attacking_crossing': 0.14,
            'skill_moves': 0.12,
            'agility': 0.10,
            'attacking_finishing': 0.10,
            'mentality_positioning': 0.07,
            'passing_technique': 0.06,
            'mentality_vision': 0.05,
            'stamina': 0.04
    },

    'RM': {
        'pace': 0.17,
            'dribbling_control': 0.15,
            'attacking_crossing': 0.14,
            'skill_moves': 0.12,
            'agility': 0.10,
            'attacking_finishing': 0.10,
            'mentality_positioning': 0.07,
            'passing_technique': 0.06,
            'mentality_vision': 0.05,
            'stamina': 0.04
    },

    'LB': {
        'defending': 0.20,
            'tackling': 0.18,
            'pace': 0.15,
            'attacking_crossing': 0.15,
            'agility': 0.08,
            'dribbling_control': 0.07,
            'stamina': 0.06,
            'physical': 0.05,
            'passing_technique': 0.04,
            'mentality_positioning': 0.02
    },

    'RB': {
        'defending': 0.20,
            'tackling': 0.18,
            'pace': 0.15,
            'attacking_crossing': 0.15,
            'agility': 0.08,
            'dribbling_control': 0.07,
            'stamina': 0.06,
            'physical': 0.05,
            'passing_technique': 0.04,
            'mentality_positioning': 0.02
    }
}




PRIORITY_MULTIPLIERS = {
    'very_low': 0.5,
    'low': 0.75,
    'medium': 1.0,
    'high': 1.5,
    'very_high': 2.0
}


def score_players(df, weights):
    score = 0

    for ability, weight in weights.items():
        score += df[ability] * weight

    return score


def build_custom_weights(
    base_weights,
    priorities
):
    custom_weights = base_weights.copy()

    for ability, priority in priorities.items():

        if ability in custom_weights:
            custom_weights[ability] *= (
                PRIORITY_MULTIPLIERS[priority]
            )

    total = sum(
        custom_weights.values()
    )

    return {
        ability: weight / total
        for ability, weight
        in custom_weights.items()
    }


def explain_player(player, weights):
    explanation = []

    for ability, weight in weights.items():

        player_value = player[ability]

        contribution = (
            player_value * weight
        )

        explanation.append({
            'ability': ability,
            'player_value': player_value,
            'weight': weight,
            'contribution': contribution
        })

    return (
    pd.DataFrame(explanation)
    .sort_values(
        'contribution',
        ascending=False
    )
    .reset_index(drop=True)
    )
    