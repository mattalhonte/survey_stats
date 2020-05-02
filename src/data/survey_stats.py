import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from IPython.display import display, Markdown
from statsmodels.stats.weightstats import ttest_ind
from typing import List, Tuple
import os


def aov_quick(df: pd.DataFrame, dvar: str, ivar: str) -> pd.DataFrame:
    formula = f"{dvar} ~ {ivar}"
    lm = ols(formula, df).fit()
    aov_table = sm.stats.anova_lm(lm, typ=2)
    return aov_table


def clean_and_add_response_values_from_df(
    df: pd.DataFrame, question_txts: pd.DataFrame, var: str
) -> pd.DataFrame:
    df_to_use = df.copy().dropna(subset=[var])
    resp_list = question_txts.loc[var]["response_vals"]
    df_to_use[var] = df_to_use[var].replace(
        dict(zip(np.sort(df_to_use[var].unique()), resp_list))
    )
    return df_to_use


def get_descs_for_cat(
    df: pd.DataFrame, question_txts: pd.DataFrame, dvar: str, ivar: str
) -> pd.DataFrame:
    df_to_use = clean_and_add_response_values_from_df(df, question_txts, ivar)
    return df_to_use.groupby(ivar)[dvar].describe()


def test_and_descs(
    df: pd.DataFrame, question_txts: pd.DataFrame, dvar: str, ivar: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aov = aov_quick(df, dvar, ivar)
    descs = get_descs_for_cat(df, question_txts, dvar, ivar)
    return aov, descs


def display_question_text(question_txts: pd.DataFrame, var: str) -> None:
    display(Markdown(f"**{var}**"))
    display(Markdown(f"""{question_txts.loc[var]["text"]}"""))


def display_test_and_descs(
    df: pd.DataFrame, question_txts: pd.DataFrame, dvar: str, ivar: str
) -> None:
    display(Markdown(f"**{ivar}**"))
    display(Markdown(f"""{question_txts.loc[ivar]["text"]}"""))
    display(*test_and_descs(df, question_txts, dvar, ivar))


def get_between_thrshs(
    df: pd.DataFrame, dvar: str, ivars: List[str], upper: float, lower: float = 0
) -> List[str]:
    aovs = (aov_quick(df, dvar, x) for x in ivars)
    pred = (lambda x, y: lambda z: x < z < y)(lower, upper)

    return [x.index[0] for x in aovs if pred(x.iloc[0, 3])]


def all_anovas_between_thrshs(
    df: pd.DataFrame,
    question_txts: pd.DataFrame,
    dvars: List[str],
    ivars: List[str],
    upper: float,
    lower: float = 0,
) -> None:
    for dvar in dvars:
        sig_test = get_between_thrshs(df, dvar, ivars, upper, lower)
        if sig_test:
            display(Markdown(f"# {dvar}"))
            for ivar in sig_test:
                display_question_text(question_txts, dvar)
                display_test_and_descs(df, question_txts, dvar, ivar)
                display(Markdown("<br>"))
            display(Markdown("<br>"))


def t_test_quick(df: pd.DataFrame, dvar: str, ivar: str) -> pd.DataFrame:

    results_df = pd.DataFrame(
        pd.Series(
            dict(
                zip(
                    ("tstat", "p-value", "df"),
                    ttest_ind(
                        df[df[ivar] == 0][dvar].dropna(),
                        df[df[ivar] == 1][dvar].dropna(),
                    ),
                )
            )
        )
    ).transpose()
    results_df.index = [ivar]

    return results_df


def t_test_and_descs(
    df: pd.DataFrame, question_txts: pd.DataFrame, dvar: str, ivar: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aov = t_test_quick(df, dvar, ivar)
    descs = get_descs_for_cat(df, question_txts, dvar, ivar)
    return aov, descs


def display_t_test_and_descs(
    df: pd.DataFrame, question_txts: pd.DataFrame, dvar: str, ivar: str
) -> None:
    display(Markdown(f"**{ivar}**"))
    display(Markdown(f"""{question_txts.loc[ivar]["text"]}"""))
    display(*t_test_and_descs(df, question_txts, dvar, ivar))


def get_ts_between_thrshs(
    df: pd.DataFrame, dvar: str, ivars: List[str], upper: float, lower: float = 0
) -> List[str]:
    aovs = (t_test_quick(df, dvar, x) for x in ivars)
    pred = (lambda x, y: lambda z: x < z < y)(lower, upper)

    return [x.index[0] for x in aovs if pred(x.iloc[0, 1])]


def add_new_texts(
    filepath: str, survey_texts: List[Tuple[str, str, List[str]]]
) -> None:
    new_texts = pd.DataFrame(survey_texts, columns=["name", "text", "response_vals"])
    if os.path.isfile(filepath):
        existing_texts = pd.read_json(filepath)
        new_texts = (pd.concat([new_texts, existing_texts, ], ignore_index=True)
                    .drop_duplicates(subset="name"))

    new_texts.to_json(filepath)

def all_ts_between_thrshs(
    df: pd.DataFrame,
    question_txts: pd.DataFrame,
    dvars: List[str],
    ivars: List[str],
    upper: float,
    lower: float = 0,
) -> None:
    for dvar in dvars:
        sig_test = get_ts_between_thrshs(df, dvar, ivars, upper, lower)
        if sig_test:
            display(Markdown(f"# {dvar}"))
            for ivar in sig_test:
                display_question_text(question_txts, dvar)
                display_t_test_and_descs(df, question_txts, dvar, ivar)
                display(Markdown("<br>"))
            display(Markdown("<br>"))


def show_group_bys_with_questions(
    df,
    question_txts: pd.DataFrame,
    categorical1: str,
    categorical2: str,
    continuous: str,
) -> None:
    display(Markdown(f"**{categorical1}**"))
    display(Markdown(f"""{question_txts.loc[categorical1]["text"]}"""))
    display(Markdown(f"**{categorical2}**"))
    display(Markdown(f"""{question_txts.loc[categorical2]["text"]}"""))
    display(
        clean_and_add_response_values_from_df(df, question_txts, categorical1)
        .groupby([categorical1, categorical2])[continuous]
        .mean()
        .unstack(fill_value=0)
    )
