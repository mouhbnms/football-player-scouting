import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


from src.scoring import score_players

# ============================================================
# HELPERS
# ============================================================

def _get_player(
    df,
    player_id,
):
    """
    Return one player from the dataframe.
    """

    target = df[
        df["player_id"] == player_id
    ]

    if target.empty:
        raise ValueError(
            f"Player with ID {player_id} was not found."
        )

    return target.iloc[0]


def _calculate_similarity(
    target,
    candidates,
    ability_features,
):
    """
    Calculate cosine similarity between the target player
    and every candidate.

    Returns values between 0 and 1.
    """

    target_vector = (
        target[
            ability_features
        ]
        .to_numpy(dtype=float)
        .reshape(1, -1)
    )

    candidate_matrix = (
        candidates[
            ability_features
        ]
        .to_numpy(dtype=float)
    )

    similarities = cosine_similarity(
        target_vector,
        candidate_matrix,
    )[0]

    return similarities


# ============================================================
# PLAYER SIMILARITY
# ============================================================

def find_similar_players(
    df,
    player_id,
    ability_features,
    n_neighbors=10,
    position=None,
):
    """
    Find players with similar ability profiles.

    Parameters
    ----------
    df : pandas.DataFrame
        Player dataset.

    player_id : int
        ID of the player we want to compare against.

    ability_features : list
        Features used to calculate similarity.

    n_neighbors : int
        Number of players to return.

    position : str or None
        Optional position filter.
    """

    target = _get_player(
        df,
        player_id,
    )

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    if position is not None:

        candidates = df[
            df["eligible_positions"].str.contains(
                rf"\b{position}\b",
                regex=True,
                na=False,
            )
        ].copy()

    else:

        candidates = df.copy()

    # Remove target itself
    candidates = candidates[
        candidates["player_id"] != player_id
    ].copy()

    if candidates.empty:
        raise ValueError(
            "No eligible candidate players were found."
        )

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    similarities = _calculate_similarity(
        target=target,
        candidates=candidates,
        ability_features=ability_features,
    )

    candidates["similarity"] = (
        similarities * 100
    )

    # --------------------------------------------------------
    # Position fit
    # --------------------------------------------------------

    target_position = target[
        "position"
    ]

    candidates["position_fit"] = candidates.apply(
        lambda player: (
            "Primary"
            if player["position"] == target_position
            else "Alternate"
            if target_position
            in player["eligible_positions"].split()
            else "Other"
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results = (
        candidates
        .sort_values(
            "similarity",
            ascending=False,
        )
        .head(n_neighbors)
        .reset_index(drop=True)
    )

    return results


# ============================================================
# PLAYER REPLACEMENTS
# ============================================================

def find_replacements(
    df,
    player_id,
    ability_features,
    custom_weights,
    desired_playstyles=None,
    min_age=16,
    max_age=40,
    gender="All",
    preferred_foot="All",
    nationalities=None,
    leagues=None,
    clubs=None,
    position=None,
    n_neighbors=10,
):
    """
    Find recruitment-oriented replacements.

    The final score combines:

    - position-specific profile score: 75%
    - player similarity: 25%
    - optional playstyle bonus inside the profile score
    """

    target = _get_player(
        df,
        player_id,
    )

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    if position is None:
        position = target["position"]

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidates = df[
        df["eligible_positions"].str.contains(
            rf"\b{position}\b",
            regex=True,
            na=False,
        )
    ].copy()

    # Remove target
    candidates = candidates[
        candidates["player_id"] != player_id
    ].copy()

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    candidates = candidates[
        candidates["age"].between(
            min_age,
            max_age,
        )
    ]

    # --------------------------------------------------------
    # Gender
    # --------------------------------------------------------

    if gender != "All":

        candidates = candidates[
            candidates["gender"] == gender
        ]

    # --------------------------------------------------------
    # Preferred foot
    # --------------------------------------------------------

    if preferred_foot != "All":

        candidates = candidates[
            candidates["preferred_foot"]
            == preferred_foot
        ]

    # --------------------------------------------------------
    # Nationality
    # --------------------------------------------------------

    if nationalities:

        candidates = candidates[
            candidates["nationality"].isin(
                nationalities
            )
        ]

    # --------------------------------------------------------
    # League
    # --------------------------------------------------------

    if leagues:

        candidates = candidates[
            candidates["league"].isin(
                leagues
            )
        ]

    # --------------------------------------------------------
    # Club
    # --------------------------------------------------------

    if clubs:

        candidates = candidates[
            candidates["club"].isin(
                clubs
            )
        ]

    if candidates.empty:

        raise ValueError(
            "No players satisfy the replacement constraints."
        )

    # ========================================================
    # PROFILE SCORE
    # ========================================================

    candidates["profile_score"] = score_players(
        candidates,
        custom_weights,
    )

    # ========================================================
    # PLAYSTYLE SCORE
    # ========================================================

    if desired_playstyles:

        candidates["playstyle_score"] = (
            candidates["base_playstyles"]
            .apply(
                lambda styles: (
                    sum(
                        1.0
                        if desired + "+" in styles
                        else 0.75
                        if desired in styles
                        else 0.0
                        for desired in desired_playstyles
                    )
                    / len(desired_playstyles)
                )
            )
        )

        candidates["playstyle_bonus"] = (
            candidates["playstyle_score"]
            * 0.05
        )

    else:

        candidates["playstyle_score"] = 0.0
        candidates["playstyle_bonus"] = 0.0

    # --------------------------------------------------------
    # Recruitment score
    # --------------------------------------------------------

    candidates["recruitment_score"] = (
        candidates["profile_score"]
        + candidates["playstyle_bonus"]
    )

    # ========================================================
    # SIMILARITY
    # ========================================================

    similarities = _calculate_similarity(
        target=target,
        candidates=candidates,
        ability_features=ability_features,
    )

    candidates["similarity"] = (
        similarities * 100
    )

    # ========================================================
    # FINAL REPLACEMENT SCORE
    # ========================================================

    candidates["replacement_score"] = (
        candidates["recruitment_score"] * 0.75
        + candidates["similarity"].div(100) * 0.25
    )

    # ========================================================
    # POSITION FIT
    # ========================================================

    candidates["position_fit"] = candidates.apply(
        lambda player: (
            "Primary"
            if player["position"] == position
            else "Alternate"
        ),
        axis=1,
    )

    # ========================================================
    # SORT
    # ========================================================

    results = (
        candidates
        .sort_values(
            "replacement_score",
            ascending=False,
        )
        .head(n_neighbors)
        .reset_index(drop=True)
    )

    return results