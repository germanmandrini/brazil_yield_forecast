import sys
import pandas as pd
import numpy as np
import pyreadr
from plotnine import *
from sklearn.metrics import r2_score as r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from math import sqrt

pd.set_option("display.max_rows", 100)  # show all columns of tables


def load_sim():
    sim_n = "dul_weather"
    sim_ls = list()
    for sim_n in ["dul_weather", "ll15_weather"]:
        sim_df = pyreadr.read_r(

            f"./data/sim_{sim_n}.rds"
        )
        sim_df = sim_df[None]
        sim_df["sim_type"] = sim_n.replace("_weather", "")
        sim_ls.append(sim_df)
    sim_df = pd.concat(sim_ls, ignore_index=True)
    sim_df.columns = sim_df.columns.str.lower()
    sim_df["region_id"] = sim_df["region_id"].astype(int)

    yield_sim_df = (
        sim_df.groupby(
            [
                "sim_type",
                "cultivar",
                "region_id",
                "year",
            ]
        )
        .agg(yield_tn_ha=("yield", "max"))
        .reset_index()
    )
    yield_sim_df["yield_tn_ha"] = yield_sim_df["yield_tn_ha"] / 1000
    yield_sim_df

    return yield_sim_df


# col_rel, cols_grp = "yield_obs", ["cultivar", "region_id"]
def make_relative(yield_cal_df, col_rel, cols_grp):
    max_s = yield_cal_df.groupby(cols_grp)[col_rel].transform("max")
    min_s = yield_cal_df.groupby(cols_grp)[col_rel].transform("min")
    rel_s = (yield_cal_df[col_rel] - min_s) / (max_s - min_s)
    return rel_s


def plot_obs_vs_pred_vit(d, x_var, y_var):
    d_plot = d[[x_var, y_var]].dropna()
    p1 = max(max(d_plot[x_var]), max(d_plot[y_var]))
    p0 = min(min(d_plot[x_var]), min(d_plot[y_var]))
    corr_val, _ = pearsonr(d_plot[x_var], d_plot[y_var])
    corr_label = "Corr(Pearson)= " + str(round(corr_val, 2))

    reg = smf.ols(f"{x_var} ~ {y_var}", data=d)
    reg_fit = reg.fit()
    r2_val = reg_fit.rsquared
    r2_label = "R2 = " + str(round(r2_val, 2))

    # r2_label = "R2 = " + str(r2_score(d_plot[x_var], d_plot[y_var]).round(2))
    rmse_val = sqrt(mean_squared_error(d_plot[x_var], d_plot[y_var]))
    RMSE_label = "RMSE = " + str(round(rmse_val, 2))

    x_lab = (p1 - p0) * 0.7 + p0
    y_lab = (p1 - p0) * 0.1 + p0

    g = (
        ggplot()
        + geom_point(
            data=d_plot, mapping=aes(x=x_var, y=y_var), size=1, color="#709e33"
        )  #
        + coord_fixed()
        + geom_abline(alpha=0.2, linetype="dashed")
        + ylim(p0, p1)
        + xlim(p0, p1)
        #   xlab('State change (%)')+
        #   ylab('County change (%)')+
        # + annotate("text", x=x_lab, y=y_lab, label=RMSE_label, color="#709e33")
        + theme(
            figure_size=[3.5, 3.5],
            aspect_ratio=1,
        )
        + theme_ff()
    )
    # g.save(os.path.abspath('./figures/validation1.svg'))
    return g


def plot_ll15_vs_dul(yield_sim_df):
    sim_type_comp_df = pd.pivot_table(
        yield_sim_df,
        index=["cultivar", "region_id", "year"],
        values="yield_tn_ha",
        columns="sim_type",
    ).reset_index()
    g = plot_obs_vs_pred_vit(sim_type_comp_df, x_var="dul", y_var="ll15")

    g = g + labs(x="DUL Yield (tn/ha)", y="LL15 Yield (tn/ha)")
    display(g)


def plot_yield_boxplots(yield_sim_df, yield_obs_df):
    yield_sim_df["sim_type"] = pd.Categorical(
        yield_sim_df["sim_type"], categories=["ll15", "dul"], ordered=True
    )
    yield_sim_df = yield_sim_df.sort_values(["cultivar", "sim_type"], ascending=True)

    yield_sim_df["cultivar_sim_type"] = (
        yield_sim_df["cultivar"] + "_" + yield_sim_df["sim_type"].astype(str)
    )

    yield_obs_tmp = yield_obs_df.loc[:, ["region_id", "year", "yield_obs"]].rename(
        {"yield_obs": "yield_tn_ha"}, axis=1
    )
    yield_obs_tmp["cultivar_sim_type"] = "observed"
    yield_obs_tmp["sim_type"] = "observed"

    yield_long_df = pd.concat([yield_sim_df, yield_obs_tmp], ignore_index=True)

    # Get the right order for the new variable
    yield_long_df["cultivar_sim_type"]
    cultivar_sim_type_ord = list(yield_sim_df["cultivar_sim_type"].unique())
    cultivar_sim_type_ord.append("observed")
    yield_long_df["cultivar_sim_type"] = pd.Categorical(
        yield_long_df["cultivar_sim_type"],
        categories=cultivar_sim_type_ord,
        ordered=True,
    )

    p = (
        ggplot(yield_long_df)
        + geom_boxplot(
            aes(
                x="cultivar_sim_type",
                y="yield_tn_ha",
                fill="sim_type",
            ),
            color="black",
        )
        + labs(
            y="Yield (tn/ha)",
            x="",
            fill="",
        )
        + scale_fill_manual(
            values={"dul": FBN_GREEN, "ll15": FBN_RED, "observed": FBN_BLUE},
            labels={
                "dul": "Drained Upper Limit (DUL)",
                "ll15": "15Bar lower limit (LL15)",
                "observed": "Observed",
            },
        )
        # + theme_minimal()
        + theme_ff()
        + theme(
            axis_text_x=element_text(
                angle=45,
                # va="bottom",
                # ha="left",
                # y=0.95,
                # x=50,
                # size=15, color="#8c8c8c"
            ),
            legend_position="bottom",
        )
    )

    display(p)


# df, group_cols, observed_col, predicted_col=yield_cal_df.copy(),["cultivar", "region_id"],"yield_obs","yield_sim",
def calculate_metrics_by_group(df, group_cols, observed_col, predicted_col):
    """
    Calculate RMSE, R2, and correlation between observed and predicted variables by group.

    Parameters:
    - df: DataFrame
    - group_cols: list of str, columns to group by
    - observed_col: str, name of the column containing observed variable
    - predicted_col: str, name of the column containing predicted variable

    Returns:
    - result_df: DataFrame, containing RMSE, R2, and correlation by group
    """
    result_ls = list()

    # Group by specified columns
    grouped_df = df.groupby(group_cols)

    for group_name, group_data in grouped_df:
        observed_values = group_data[observed_col]
        predicted_values = group_data[predicted_col]

        rmse = np.sqrt(mean_squared_error(observed_values, predicted_values))
        r2 = r2_score(observed_values, predicted_values)
        correlation = observed_values.corr(predicted_values)

        grp_info = group_data[group_cols].drop_duplicates().reset_index(drop=True)

        grp_metric = pd.DataFrame(
            {
                # **{f"{col.lower()}": [val] for col, val in zip(group_cols, group_name)},
                "rmse": [rmse],
                "r2": [r2],
                "correlation": [correlation],
            },
            index=[0],
        )
        group_result = pd.concat([grp_info, grp_metric], axis=1)

        result_ls.append(group_result)

    result_df = pd.concat(result_ls, ignore_index=True)
    return result_df


def sim_calibration(yield_sim_df, yield_obs_df):
    yield_cal_df = pd.merge(yield_sim_df, yield_obs_df, how="left")
    yield_cal_df = (
        yield_cal_df.loc[yield_cal_df["yield_obs"] > 0].reset_index(drop=True).copy()
    )
    yield_cal_df.rename({"yield_tn_ha": "yield_sim"}, axis=1, inplace=True)
    cols_grp = ["cultivar", "region_id", "sim_type"]
    yield_cal_df["yield_obs_rel"] = make_relative(
        yield_cal_df, col_rel="yield_obs", cols_grp=cols_grp
    )
    yield_cal_df["yield_sim_rel"] = make_relative(
        yield_cal_df, col_rel="yield_sim", cols_grp=cols_grp
    )

    # Best RMSE ---------------------------------------
    rmse_comp_df = calculate_metrics_by_group(
        df=yield_cal_df,
        group_cols=["cultivar", "sim_type"],
        observed_col="yield_obs",
        predicted_col="yield_sim",
    ).sort_values("rmse", ascending=True)
    display(rmse_comp_df)
    rmse_comp_df.to_csv("./data/figures/rmse_comp.csv", index=False)

    # MG6 dul
    yield_filt_df = yield_cal_df.loc[
        (yield_cal_df["cultivar"] == "MG6") & (yield_cal_df["sim_type"] == "dul")
    ].copy()
    g = plot_obs_vs_pred_vit(yield_filt_df, x_var="yield_obs", y_var="yield_sim")
    g = g + labs(x="Yield Obs. (tn/ha)", y="Yield Sim. (tn/ha)")
    display(g)

    out_path = "./data/figures/simple_obs_vs_pred.png"
    g.save(
        out_path, dpi=300, verbose=False
    )  # ,height=4, width=6 g = plot_obs_vs_pred_vit(yield_filt_df, x_var="yield_obs", y_var="yield_sim")
    if False:
        # Best metric by region Approach ---------------------------------------
        metrics_rel_df = calculate_metrics_by_group(
            df=yield_cal_df,
            group_cols=["cultivar", "region_id", "sim_type"],
            observed_col="yield_obs_rel",
            predicted_col="yield_sim_rel",
        )
        # display(metrics_rel_df)
        metrics_abs_df = calculate_metrics_by_group(
            df=yield_cal_df,
            group_cols=["cultivar", "region_id", "sim_type"],
            observed_col="yield_obs",
            predicted_col="yield_sim",
        )
        # display(metrics_abs_df)

        # Relative ---------------------------------------
        metrics_best_df = (
            metrics_rel_df.sort_values("rmse")
            .groupby("region_id")
            .first()
            .reset_index()
        )
        metrics_best_df = (
            metrics_rel_df.sort_values("correlation")
            .groupby("region_id")
            .last()
            .reset_index()
        )
        yield_filt_df = pd.merge(yield_cal_df, metrics_best_df, how="inner")

        metrics_overall_df = calculate_metrics_by_group(
            df=yield_filt_df,
            group_cols=["commodity"],
            observed_col="yield_obs_rel",
            predicted_col="yield_sim_rel",
        )
        display(metrics_overall_df)

        g = plot_obs_vs_pred_vit(yield_filt_df, x_var="yield_obs", y_var="yield_sim")
        g = g + labs(x="Yield Obs. (tn/ha)", y="Yield Sim. (tn/ha)")
        display(g)
        out_path = f"./data/figures/{sim_n}_method_abs_disp_abs.png"
        g.save(out_path, dpi=300, verbose=False)  # ,height=4, width=6

        g = plot_obs_vs_pred_vit(
            yield_filt_df, x_var="yield_obs_rel", y_var="yield_sim_rel"
        )
        g = g + labs(x="Relative Yield Obs. (tn/ha)", y="Relative Yield Sim. (tn/ha)")
        display(g)
        out_path = f"./data/figures/{sim_n}_method_abs_disp_rel.png"
        g.save(out_path, dpi=300, verbose=False)  # ,height=4, width=6

        # Absolute ---------------------------------------
        metrics_best_df = (
            metrics_abs_df.sort_values("rmse", ascending=True)
            .groupby("region_id")
            .first()
            .reset_index()
        )
        metrics_best_df = (
            metrics_abs_df.sort_values("correlation", ascending=True)
            .groupby("region_id")
            .last()
            .reset_index()
        )

        yield_filt_df = pd.merge(yield_cal_df, metrics_best_df, how="inner")

        metrics_abs_overall_df = calculate_metrics_by_group(
            df=yield_filt_df,
            group_cols=["commodity"],
            observed_col="yield_obs",
            predicted_col="yield_sim",
        )
        display(metrics_abs_overall_df)

        g = plot_obs_vs_pred_vit(yield_filt_df, x_var="yield_obs", y_var="yield_sim")
        g = g + labs(x="Yield Obs. (tn/ha)", y="Yield Sim. (tn/ha)")
        display(g)
        out_path = f"./data/figures/{sim_n}_method_rel_disp_abs.png"
        g.save(out_path, dpi=300, verbose=False)  # ,height=4, width=6

        g = plot_obs_vs_pred_vit(
            yield_filt_df, x_var="yield_obs_rel", y_var="yield_sim_rel"
        )
        g = g + labs(x="Relative Yield Obs. (tn/ha)", y="Relative Yield Sim. (tn/ha)")
        display(g)
        out_path = f"./data/figures/{sim_n}_method_rel_disp_rel.png"
        g.save(out_path, dpi=300, verbose=False)  # ,height=4, width=6


if __name__ == "__main__":
    yield_obs_df = pd.read_excel(
        "./data/MT Soybeans Case Study Dataset.xlsx",
        sheet_name="yield",
    )
    yield_obs_df["year"] = yield_obs_df["crop_year"].str[0:4].astype(int)
    yield_obs_df.rename({"yield": "yield_obs"}, axis=1, inplace=True)
    yield_sim_df = load_sim()
    plot_ll15_vs_dul(yield_sim_df)
    plot_yield_boxplots(yield_sim_df, yield_obs_df)
    sim_calibration(yield_sim_df, yield_obs_df)
