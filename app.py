import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import textwrap

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.preprocessing import (
    load_data,
    clean_data,
    add_age,
)

from src.features import (
    normalize_attributes,
    create_ability_features,
    prepare_playstyles,
    ABILITY_FEATURES,
)

from src.scoring import (
    score_players,
    build_custom_weights,
    explain_player,
    POSITION_WEIGHTS,
)

from src.recommendation import (
    get_position_candidates,
)

from src.similarity import (
    find_similar_players,
    find_replacements,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = PROJECT_ROOT / "Data" / "players_cleaned.csv"

PLAYSTYLE_BONUS_MAX = 0.05

PRIORITY_MULTIPLIERS = {
    "very_low": 0.5,
    "low": 0.75,
    "medium": 1.0,
    "high": 1.5,
    "very_high": 2.0,
}

POSITIONS = [
    "GK", "CB", "RB", "LB", "CDM", "CM", "CAM", "RM", "LM", "RW", "LW", "ST",
]

PRIORITY_OPTIONS = [
    "very_low", "low", "medium", "high", "very_high",
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Football Player Scouting",
    page_icon="⚽",
    layout="wide",
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_players():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_age(df)
    df = normalize_attributes(df)
    df = create_ability_features(df)
    df = prepare_playstyles(df)
    missing_ability_features = [
        feature
        for feature in ABILITY_FEATURES
        if feature not in df.columns
    ]

    if missing_ability_features:
        raise ValueError(
            f"Missing ability features: {missing_ability_features}"
        )
    
    for col in ["base_playstyles", "playstyle_list", "positions", "eligible_positions"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    # ---------------------------
    return df


@st.cache_data
def get_player_lookup(df):
    lookup = {}
    for row in df[["player_id", "first_name", "last_name", "club", "position"]].itertuples(index=False):
        lookup[row.player_id] = {
            "first_name": row.first_name,
            "last_name": row.last_name,
            "club": row.club,
            "position": row.position,
        }
    return lookup


@st.cache_data
def get_player_ids(df):
    return df["player_id"].tolist()


@st.cache_data
def get_available_playstyles(df):
    return sorted(
        {
            style
            for styles in df["base_playstyles"]
            for style in styles
        }
    )


# ============================================================
# LOAD DATA
# ============================================================

try:
    df = load_players()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

player_lookup = get_player_lookup(df)
player_ids = get_player_ids(df)
available_playstyles = get_available_playstyles(df)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def player_label(player_id):
    player = player_lookup[player_id]
    return f"{player['first_name']} {player['last_name']} — {player['club']}"

def search_players(df, search_text):
    if not search_text:
        return df

    search_text = search_text.lower().strip()

    full_name = (
        df["first_name"].fillna("") + " " +
        df["last_name"].fillna("")
    ).str.lower()

    return df[full_name.str.contains(search_text, na=False)]

def position_fit(player, target_position):
    if player["position"] == target_position:
        return "Primary"
    if target_position in str(player.get("eligible_positions", "")).split():
        return "Alternate"
    return "Other"

def get_player_info(df, player_id):
    return df[df["player_id"] == player_id].iloc[0]


def display_player_card(player, title="Selected player"):
    st.markdown(f"### {title}")

    st.markdown(
        f"""
        <div style="
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            background-color: rgba(128,128,128,0.05);
            margin-bottom: 1rem;
        ">
            <h3 style="margin-bottom: 0.4rem;">
                {player['first_name']} {player['last_name']}
            </h3>
            <p style="margin-bottom: 0.8rem; color: #888;">
                {player['club']} · {player['position']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Age", int(player["age"]))

    with col2:
        st.metric("Overall", int(player["overall_rating"]))

    with col3:
        st.metric("Position", player["position"])

    with col4:
        st.metric("Foot", player["preferred_foot"])


def display_score_bar(label, value):
    value = max(0.0, min(100.0, float(value)))

    st.markdown(
        f"""
<div style="margin-bottom: 0.6rem;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
        <span>{label}</span>
        <strong>{value:.1f}</strong>
    </div>
    <div style="width: 100%; height: 8px; background: rgba(128,128,128,0.2); border-radius: 10px; overflow: hidden;">
        <div style="width: {value:.1f}%; height: 100%; border-radius: 10px; background: #4CAF50;"></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

def calculate_playstyle_score(player_playstyles, desired_playstyles):
    if not desired_playstyles:
        return 0.0

    scores = []
    for desired in desired_playstyles:
        plus_version = desired + "+"
        if plus_version in player_playstyles:
            scores.append(1.0)
        elif desired in player_playstyles:
            scores.append(0.75)
        else:
            scores.append(0.0)

    return sum(scores) / len(scores)


def get_filtered_candidates(
    df,
    target_position,
    min_age,
    max_age,
    selected_gender,
    selected_foot,
    selected_nationalities,
    selected_leagues,
    selected_clubs,
):
    candidates = get_position_candidates(df, target_position)
    candidates = candidates[candidates["age"].between(min_age, max_age)].copy()

    if selected_gender != "All":
        candidates = candidates[candidates["gender"] == selected_gender]

    if selected_foot != "All":
        candidates = candidates[candidates["preferred_foot"] == selected_foot]

    if selected_nationalities:
        candidates = candidates[candidates["nationality"].isin(selected_nationalities)]

    if selected_leagues:
        candidates = candidates[candidates["league"].isin(selected_leagues)]

    if selected_clubs:
        candidates = candidates[candidates["club"].isin(selected_clubs)]

    return candidates.copy()


def display_recommendations(
    recommendations,
    target_position,
    top_n,
    custom_weights,
):
    if recommendations.empty:
        st.warning("No players satisfy the selected criteria.")
        return

    top_players = recommendations.head(top_n).copy()

    st.header(f"🏆 Top {len(top_players)} recommendations")

    st.caption(
        f"Players ranked for the **{target_position}** recruitment profile."
    )

    # ------------------------------------------------------------
    # Ranking table
    # ------------------------------------------------------------

    display_columns = [
        "first_name",
        "last_name",
        "club",
        "position",
        "age",
        "overall_rating",
        "final_score",
    ]

    display_columns = [
        col for col in display_columns
        if col in top_players.columns
    ]

    table = top_players[display_columns].copy()

    table.insert(
        0,
        "Rank",
        range(1, len(table) + 1)
    )

    table["final_score"] = (
        table["final_score"] * 100
    ).round(1)

    table = table.rename(
        columns={
            "first_name": "First name",
            "last_name": "Last name",
            "club": "Club",
            "position": "Position",
            "age": "Age",
            "overall_rating": "Overall",
            "final_score": "Scouting score",
        }
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "Rank",
                width="small",
            ),
            "Scouting score": st.column_config.ProgressColumn(
                "Scouting score",
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
            "Overall": st.column_config.NumberColumn(
                "Overall",
                format="%d",
            ),
        },
    )

    # ------------------------------------------------------------
    # Player analysis
    # ------------------------------------------------------------

    st.divider()

    st.subheader("🔎 Player analysis")

    rec_player_ids = top_players["player_id"].tolist()

    selected_player_id = st.selectbox(
        "Select a recommended player",
        rec_player_ids,
        format_func=player_label,
        key="recommended_player_select",
    )

    selected_player = top_players[
        top_players["player_id"] == selected_player_id
    ].iloc[0]

    display_player_card(
        selected_player,
        title="Selected player",
    )

    fit = position_fit(
        selected_player,
        target_position,
    )

    st.write(f"**Position fit:** {fit}")

    # ------------------------------------------------------------
    # Playstyles
    # ------------------------------------------------------------

    player_playstyles = selected_player["base_playstyles"]

    if player_playstyles:
        st.write(
            "**Playstyles:** "
            + " · ".join(player_playstyles)
        )
    else:
        st.write("**Playstyles:** None listed")

    # ------------------------------------------------------------
    # Why this player?
    # ------------------------------------------------------------

    st.divider()

    st.subheader("💡 Why this player?")

    st.caption(
        "The strongest contributions to the player's scouting score."
    )

    explanation = explain_player(
        selected_player,
        custom_weights,
    )

    explanation["player_value"] = (
        explanation["player_value"] * 100
    )

    explanation["weight"] = (
        explanation["weight"] * 100
    )

    explanation["contribution"] = (
        explanation["contribution"] * 100
    )

    explanation = explanation.sort_values(
        "contribution",
        ascending=False,
    )

    # Show top contributors visually
    top_contributors = explanation.head(8)

    for _, row in top_contributors.iterrows():
        display_score_bar(
            row["ability"].replace("_", " ").title(),
            row["contribution"],
        )

    # Detailed table
    with st.expander("View detailed scoring breakdown"):

        detail = explanation.rename(
            columns={
                "ability": "Ability",
                "player_value": "Player value",
                "weight": "Weight (%)",
                "contribution": "Contribution",
            }
        )

        st.dataframe(
            detail[
                [
                    "Ability",
                    "Player value",
                    "Weight (%)",
                    "Contribution",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")

app_mode = st.sidebar.radio(
    "Choose Mode",
    options=[
        "🎯 Player Recommendations",
        "🔄 Player Similarity",
        "🔁 Find Replacements",
    ],
)


# ============================================================
# TITLE
# ============================================================

st.title("⚽ Football Player Scouting & Recruitment")

st.markdown(
    """
    **Data-driven player discovery, similarity analysis, and replacement scouting.**
    
    Built using EA FC 27 player ratings and a transparent attribute-based
    recommendation system.
    """
)


# ============================================================
# MODE 1 — PLAYER RECOMMENDATIONS
# ============================================================

if app_mode == "🎯 Player Recommendations":

    st.header("🎯 Player recommendations")
    st.markdown("Define the recruitment profile and search for suitable players.")

    with st.sidebar.form("recommendation_form"):
        st.subheader("Recruitment criteria")

        target_position = st.selectbox(
            "Target position", POSITIONS, key="recommendation_position"
        )
        min_age, max_age = st.slider(
            "Age range", min_value=16, max_value=40, value=(16, 30), key="recommendation_age"
        )

        gender_options = ["All"] + sorted(df["gender"].dropna().unique().tolist())
        selected_gender = st.selectbox("Gender", gender_options, key="recommendation_gender")

        foot_options = ["All"] + sorted(df["preferred_foot"].dropna().unique().tolist())
        selected_foot = st.selectbox("Preferred foot", foot_options, key="recommendation_foot")

        selected_nationalities = st.multiselect(
            "Nationality",
            sorted(df["nationality"].dropna().unique().tolist()),
            placeholder="All nationalities",
            key="recommendation_nationalities",
        )

        selected_leagues = st.multiselect(
            "League",
            sorted(df["league"].dropna().unique().tolist()),
            placeholder="All leagues",
            key="recommendation_leagues",
        )

        selected_clubs = st.multiselect(
            "Club",
            sorted(df["club"].dropna().unique().tolist()),
            placeholder="All clubs",
            key="recommendation_clubs",
        )

        top_n = st.slider(
            "Number of players", min_value=5, max_value=25, value=10, key="recommendation_top_n"
        )

        st.subheader("Attribute priorities")
        base_weights = POSITION_WEIGHTS[target_position]
        priorities = {}

        for ability in base_weights:
            priorities[ability] = st.selectbox(
                ability.replace("_", " ").title(),
                PRIORITY_OPTIONS,
                index=2,
                key=f"recommendation_priority_{ability}",
            )

        st.subheader("Playstyle preferences")
        desired_playstyles = st.multiselect(
            "Desired playstyles", available_playstyles, key="recommendation_playstyles"
        )

        search_recommendations = st.form_submit_button(
            "🔎 Find players", width="stretch"
        )

    if search_recommendations:
        custom_weights = build_custom_weights(base_weights, priorities)

        with st.spinner("Searching players..."):
            candidates = get_filtered_candidates(
                df=df,
                target_position=target_position,
                min_age=min_age,
                max_age=max_age,
                selected_gender=selected_gender,
                selected_foot=selected_foot,
                selected_nationalities=selected_nationalities,
                selected_leagues=selected_leagues,
                selected_clubs=selected_clubs,
            )

            if candidates.empty:
                st.session_state["rec_results"] = None
                st.warning("No players satisfy the selected criteria.")
            else:
                candidates["profile_score"] = score_players(candidates, custom_weights)

                if desired_playstyles:
                    candidates["playstyle_score"] = candidates["base_playstyles"].apply(
                        lambda styles: calculate_playstyle_score(styles, desired_playstyles)
                    )
                    candidates["playstyle_bonus"] = candidates["playstyle_score"] * PLAYSTYLE_BONUS_MAX
                else:
                    candidates["playstyle_score"] = 0.0
                    candidates["playstyle_bonus"] = 0.0

                candidates["final_score"] = (
                    candidates["profile_score"] + candidates["playstyle_bonus"]
                )

                candidates = candidates.sort_values("final_score", ascending=False).reset_index(drop=True)

                # Store with custom state keys to prevent key collisions with form inputs
                st.session_state["rec_results"] = candidates
                st.session_state["rec_target_position"] = target_position
                st.session_state["rec_custom_weights"] = custom_weights

    if st.session_state.get("rec_results") is not None:
        display_recommendations(
            recommendations=st.session_state["rec_results"],
            target_position=st.session_state["rec_target_position"],
            top_n=st.session_state["recommendation_top_n"],
            custom_weights=st.session_state["rec_custom_weights"],
        )


# ============================================================
# MODE 2 — PLAYER SIMILARITY
# ============================================================

elif app_mode == "🔄 Player Similarity":

    st.header("🔄 Player similarity")
    st.markdown("Find players with a similar attribute profile using cosine similarity.")

    with st.form("similarity_form"):
        search_text = st.text_input("Search players", placeholder="Enter player name...")

        matching_players = search_players(df, search_text)


        if matching_players.empty:
            st.warning("No players found.")
            similarity_player = None
        else:
            similarity_player = st.selectbox(
            "Select player",
            matching_players["player_id"].tolist(),
            format_func=player_label,
            key="similarity_player"
        )
        similarity_position = st.selectbox(
            "Position filter",
            [None] + POSITIONS,
            format_func=lambda x: "All positions" if x is None else x,
            key="similarity_position",
        )
        similarity_n = st.slider(
            "Number of similar players", min_value=5, max_value=25, value=10, key="similarity_n"
        )
        search_similarity = st.form_submit_button("🔎 Find similar players", width="stretch")

    if search_similarity:
        with st.spinner("Calculating player similarity..."):
            similar_players = find_similar_players(
                df=df,
                player_id=similarity_player,
                ability_features=ABILITY_FEATURES,
                n_neighbors=similarity_n,
                position=similarity_position,
            )
        st.session_state["similar_players"] = similar_players

    if "similar_players" in st.session_state:
        similar_players = st.session_state["similar_players"]

        st.divider()

        st.subheader("Similar players")

        similarity_columns = [
            "first_name", "last_name", "club", "position", "eligible_positions",
            "age", "overall_rating", "similarity", "position_fit",
        ]
        similarity_columns = [col for col in similarity_columns if col in similar_players.columns]

        result = similar_players[similarity_columns].copy()
        result.insert(
        0,
        "Rank",
        range(1, len(result) + 1),
    )
        result["similarity"] = result["similarity"].round(1)
        result = result.rename(
            columns={
                "first_name": "First name",
                "last_name": "Last name",
                "club": "Club",
                "position": "Position",
                "eligible_positions": "Eligible positions",
                "age": "Age",
                "overall_rating": "Overall",
                "similarity": "Similarity (%)",
                "position_fit": "Position fit",
            }
        )

        st.dataframe(
        result,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "Rank",
                width="small",
            ),
            "Similarity (%)": st.column_config.ProgressColumn(
                "Similarity",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )


# ============================================================
# MODE 3 — FIND REPLACEMENTS
# ============================================================

elif app_mode == "🔁 Find Replacements":

    st.header("🔁 Find replacements")
    st.markdown("Find players who can replace an existing player.")

    with st.sidebar.form("replacement_form"):
        st.subheader("Replacement criteria")

        search_text = st.text_input("Search for player to replace", placeholder="Enter player name...", key="replacement_search")
        matching_players = search_players(df, search_text)

        if matching_players.empty:
            st.warning("No players found.")
            replacement_player = None
        else:
            replacement_player = st.selectbox(
            "Select player to replace",
            matching_players["player_id"].tolist(),
            format_func=player_label,
            key="replacement_player"
        )
        replacement_min_age, replacement_max_age = st.slider(
            "Age range", min_value=16, max_value=40, value=(16, 24), key="replacement_age_range"
        )

        replacement_gender_options = ["All"] + sorted(df["gender"].dropna().unique().tolist())
        replacement_gender = st.selectbox(
            "Gender", replacement_gender_options, key="replacement_gender"
        )

        replacement_foot_options = ["All"] + sorted(df["preferred_foot"].dropna().unique().tolist())
        replacement_foot = st.selectbox(
            "Preferred foot", replacement_foot_options, key="replacement_foot"
        )

        replacement_nationalities = st.multiselect(
            "Nationality",
            sorted(df["nationality"].dropna().unique().tolist()),
            placeholder="All nationalities",
            key="replacement_nationalities",
        )

        replacement_leagues = st.multiselect(
            "League",
            sorted(df["league"].dropna().unique().tolist()),
            placeholder="All leagues",
            key="replacement_leagues",
        )

        replacement_clubs = st.multiselect(
            "Club",
            sorted(df["club"].dropna().unique().tolist()),
            placeholder="All clubs",
            key="replacement_clubs",
        )

        replacement_position = st.selectbox(
            "Replacement position", ["Same as player"] + POSITIONS, key="replacement_position"
        )

        replacement_n = st.slider(
            "Number of replacements", min_value=5, max_value=25, value=10, key="replacement_n"
        )

        st.subheader("Attribute priorities")
        priority_position = (
            replacement_position if replacement_position != "Same as player" else "ST"
        )
        base_weights = POSITION_WEIGHTS[priority_position]
        priorities = {}

        for ability in base_weights:
            priorities[ability] = st.selectbox(
                ability.replace("_", " ").title(),
                PRIORITY_OPTIONS,
                index=2,
                key=f"replacement_priority_{ability}",
            )

        st.subheader("Playstyle preferences")
        replacement_playstyles = st.multiselect(
            "Desired playstyles", available_playstyles, key="replacement_playstyles"
        )

        search_replacements = st.form_submit_button(
            "🔎 Find replacements", width="stretch"
        )

    if search_replacements:
        target = df[df["player_id"].eq(replacement_player)].iloc[0]
        target_position = target["position"]

        final_position = (
            target_position if replacement_position == "Same as player" else replacement_position
        )
        custom_weights = build_custom_weights(POSITION_WEIGHTS[final_position], priorities)

        with st.spinner("Finding replacements..."):
            try:
                replacements = find_replacements(
                    df=df,
                    player_id=replacement_player,
                    ability_features=ABILITY_FEATURES,  # Added required feature list
                    custom_weights=custom_weights,      # Fixed keyword argument name
                    desired_playstyles=replacement_playstyles,
                    min_age=replacement_min_age,
                    max_age=replacement_max_age,
                    gender=replacement_gender,
                    preferred_foot=replacement_foot,
                    nationalities=replacement_nationalities,
                    leagues=replacement_leagues,
                    clubs=replacement_clubs,
                    position=final_position,             # Fixed parameter name
                    n_neighbors=replacement_n,           # Fixed parameter name
                )
                st.session_state["replacement_results"] = replacements
            except ValueError as e:
                st.warning(str(e))
                st.session_state["replacement_results"] = None

    if st.session_state.get("replacement_results") is not None:
        replacements = st.session_state["replacement_results"]

        st.subheader("Candidate Replacements")

        display_cols = [
            "first_name",
            "last_name",
            "club",
            "position",
            "eligible_positions",  
            "position_fit",
            "age",
            "overall_rating",
            "similarity",
            "replacement_score",
        ]
        display_cols = [c for c in display_cols if c in replacements.columns]

        formatted = replacements[display_cols].copy()
        formatted["similarity"] = formatted["similarity"].round(1)
        formatted["replacement_score"] = (formatted["replacement_score"] * 100).round(1)

        formatted = formatted.rename(
            columns={
                "first_name": "First Name",
                "last_name": "Last Name",
                "club": "Club",
                "position": "Position",
                "eligible_positions": "Eligible Positions",
                "position_fit": "Fit",
                "age": "Age",
                "overall_rating": "Overall",
                "similarity": "Similarity (%)",
                "replacement_score": "Score (%)",
            }
        )
        formatted.insert(
    0,
    "Rank",
    range(1, len(formatted) + 1),
)
        st.dataframe(
    formatted,
    width="stretch",
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn(
            "Rank",
            width="small",
        ),
        "Score (%)": st.column_config.ProgressColumn(
            "Replacement score",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        ),
        "Similarity (%)": st.column_config.ProgressColumn(
            "Similarity",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        ),
    },
)
# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Football Player Scouting & Recruitment Recommendation "
    "System — based on EA FC 27 player ratings."
)