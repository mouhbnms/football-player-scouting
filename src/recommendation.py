from .scoring import (
    POSITION_WEIGHTS,
    score_players,
    build_custom_weights,
    explain_player
)


PLAYSTYLE_MATCH_VALUES = {
    'none': 0.0,
    'base': 0.75,
    'plus': 1.0
}

PLAYSTYLE_BONUS_MAX = 0.05


def apply_filter(df, column, value):

    if value is None:
        return df

    if isinstance(value, list):
        return df[
            df[column].isin(value)
        ]

    return df[
        df[column] == value
    ]


def get_position_candidates(
    df,
    position
):
    return df[
        df['eligible_positions'].str.contains(
            rf'\b{position}\b',
            regex=True
        )
    ].copy()


def get_position_fit(
    player,
    target_position
):
    if player['position'] == target_position:
        return 'Primary'

    if (
        target_position
        in player['eligible_positions'].split()
    ):
        return 'Alternate'

    return 'Not eligible'


def get_playstyle_match(player_playstyles, desired_playstyle):
    # Handle list of desired playstyles recursively or via sum
    if isinstance(desired_playstyle, list):
        return sum(
            get_playstyle_match(player_playstyles, style) 
            for style in desired_playstyle
        )
    
    # Single string logic
    plus_version = desired_playstyle + '+'

    if plus_version in player_playstyles:
        return PLAYSTYLE_MATCH_VALUES['plus']

    if desired_playstyle in player_playstyles:
        return PLAYSTYLE_MATCH_VALUES['base']

    return PLAYSTYLE_MATCH_VALUES['none']


def calculate_playstyle_score(
    player_playstyles,
    desired_playstyles
):
    if not desired_playstyles:
        return None

    matches = [
        get_playstyle_match(
            player_playstyles,
            desired
        )
        for desired in desired_playstyles
    ]

    return sum(matches) / len(matches)


def recommend_players(
    df,
    position,
    min_age=None,
    max_age=None,
    gender=None,
    league=None,
    nationality=None,
    continent=None,
    preferred_foot=None,
    priorities=None,
    desired_playstyles=None,
    top_n=20
):

    candidates = get_position_candidates(
        df,
        position
    )

    if min_age is not None:
        candidates = candidates[
            candidates['age'] >= min_age
        ]

    if max_age is not None:
        candidates = candidates[
            candidates['age'] <= max_age
        ]

    candidates = apply_filter(
        candidates,
        'gender',
        gender
    )

    candidates = apply_filter(
        candidates,
        'league',
        league
    )

    candidates = apply_filter(
        candidates,
        'nationality',
        nationality
    )

    candidates = apply_filter(
        candidates,
        'continent',
        continent
    )

    candidates = apply_filter(
        candidates,
        'preferred_foot',
        preferred_foot
    )

    weights = POSITION_WEIGHTS[position]

    if priorities is not None:
        weights = build_custom_weights(
            weights,
            priorities
        )

    candidates['profile_score'] = (
        score_players(
            candidates,
            weights
        )
    )

    if desired_playstyles:

        candidates['playstyle_score'] = (
            candidates['playstyle_list']
            .apply(
                lambda styles:
                calculate_playstyle_score(
                    styles,
                    desired_playstyles
                )
            )
        )

        candidates['playstyle_bonus'] = (
            candidates['playstyle_score']
            * PLAYSTYLE_BONUS_MAX
        )

        candidates['final_score'] = (
            candidates['profile_score']
            + candidates['playstyle_bonus']
        )

    else:

        candidates['playstyle_score'] = None
        candidates['playstyle_bonus'] = 0

        candidates['final_score'] = (
            candidates['profile_score']
        )

    candidates['position_fit'] = candidates.apply(
        lambda player:
        get_position_fit(
            player,
            position
        ),
        axis=1
    )

    return (
        candidates
        .sort_values(
            'final_score',
            ascending=False
        )
        .head(top_n)
    )