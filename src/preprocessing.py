import pandas as pd


DROP_COLUMNS = [
    'common_name',
    'height_cm',
    'weight_kg',
    'edition',
    'snapshot_date'
]


def load_data(path):
    """Load the FC27 player dataset."""
    df = pd.read_csv(path)

    df['birthdate'] = pd.to_datetime(
        df['birthdate']
    )

    return df


def clean_data(df):
    """Apply the project's basic cleaning steps."""

    df = df.copy()

    df = df.drop(
        columns=[
            column
            for column in DROP_COLUMNS
            if column in df.columns
        ]
    )

    df['eligible_positions'] = (
        df['position']
        + ' '
        + df['alternate_positions'].fillna('')
    ).str.strip()

    return df


def add_age(
    df,
    reference_date='2026-09-12'
):
    """Calculate player age relative to a reference date."""

    df = df.copy()

    reference_date = pd.Timestamp(
        reference_date
    )
    
    df['age'] = (
        reference_date.year
        - df['birthdate'].dt.year
        - (
            (
                df['birthdate'].dt.month
                > reference_date.month
            )
            |
            (
                (
                    df['birthdate'].dt.month
                    == reference_date.month
                )
                &
                (
                    df['birthdate'].dt.day
                    > reference_date.day
                )
            )
        ).astype(int)
    )

    return df


def prepare_data(
    path,
    reference_date='2026-09-12'
):
    """Complete data-loading and preparation pipeline."""

    df = load_data(path)

    df = clean_data(df)

    df = add_age(
        df,
        reference_date
    )

    return df