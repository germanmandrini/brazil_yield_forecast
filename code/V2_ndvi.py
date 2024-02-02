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
import catboost as cb

from V1_calibration import plot_obs_vs_pred_vit, calculate_metrics_by_group


pd.set_option("display.max_rows", 100)  # show all columns of tables


def load_sim_daily():
    sim_df = pyreadr.read_r(
        f"./data/sim_dul_weather.rds"
    )
    sim_df = sim_df[None]
    sim_df.columns = sim_df.columns.str.lower()
    sim_df["region_id"] = sim_df["region_id"].astype(int)
    sim_df = sim_df.loc[sim_df["cultivar"] == "MG6"].copy()
    sim_df.columns
    sim_df.rename(
        {
            "year": "sow_year",
            "soybean.phenology.stage": "stage_num",
            "soybean.phenology.currentstagename": "stage_name",
            "weather.rain": "rain",
            "weather.mint": "mint",
            "weather.maxt": "maxt",
            "weather.radn": "radn",
        },
        axis=1,
        inplace=True,
    )
    sim_df["date"] = pd.to_datetime(sim_df["date"])
    sim_df["year"] = sim_df["date"].dt.year
    sim_df["woy"] = sim_df["date"].dt.isocalendar().week
    sim_df["doy"] = sim_df["date"].dt.dayofyear

    # Fill stage_name column
    # sim_df["stage_name"] = sim_df["stage_name"].replace("", np.nan)
    # ref_df = sim_df[["stage_num", "stage_name"]].dropna()
    # ref_df["stage_num"] = ref_df["stage_num"].astype(int)
    # ref_df = ref_df.drop_duplicates()
    if False:
        # Save Stage Reference
        ref_df = sim_df.loc[
            (sim_df["region_id"] == 5101) & (sim_df["sow_year"] == 2020),
            ["date", "stage_num", "stage_name"],
        ].dropna()
        ref_df.to_csv(
            "./data/figures/stage_ref.csv", index=False
        )

    # Fill stage_name column
    cols_sim_id = ["region_id", "cultivar", "sow_year"]
    sim_df["stage_name"] = sim_df["stage_name"].replace("", np.nan)
    sim_df["stage_name_fill"] = sim_df.groupby(cols_sim_id)["stage_name"].ffill()
    # Do not ffill the last one
    sim_df.loc[
        ~(sim_df["stage_name"] == "HarvestRipe")
        & (sim_df["stage_name_fill"] == "HarvestRipe"),
        "stage_name_fill",
    ] = np.nan

    if False:  # check
        sim_df.loc[
            (sim_df["region_id"] == 5101) & (sim_df["sow_year"] == 2020),
            ["date", "stage_num", "stage_name", "stage_name_fill"],
        ].dropna(subset=["stage_name_fill"])

    # check
    sim_df.groupby(cols_sim_id)["date"].count()
    # Dedup dates for each crop year
    sim_df["date_min"] = pd.to_datetime(sim_df["sow_year"].astype(str) + "-05-29")
    sim_df["date_max"] = pd.to_datetime((sim_df["sow_year"] + 1).astype(str) + "-05-28")
    sim_df = sim_df.loc[
        sim_df["date"].between(sim_df["date_min"], sim_df["date_max"])
    ].copy()
    sim_df.drop(["date_min", "date_max"], axis=1, inplace=True)
    sim_df.reset_index(drop=True, inplace=True)

    return sim_df


def load_yield_obs():
    yield_obs_df = pd.read_excel(
        "./data/MT Soybeans Case Study Dataset.xlsx",
        sheet_name="yield",
    )
    yield_obs_df["sow_year"] = yield_obs_df["crop_year"].str[0:4].astype(int)
    yield_obs_df.rename({"yield": "yield_obs"}, axis=1, inplace=True)
    yield_obs_df["region_id"] = yield_obs_df["region_id"].astype(int)
    yield_obs_df["yield_obs_l1"] = yield_obs_df.groupby("region_id")[
        "yield_obs"
    ].shift()
    yield_obs_df["yield_obs_diff"] = (
        yield_obs_df["yield_obs"] - yield_obs_df["yield_obs_l1"]
    )

    yield_obs_df.loc[(yield_obs_df["region_id"] == 5104)]
    return yield_obs_df


def load_ndvi():
    ndvi_df = pd.read_excel(
        "./data/MT Soybeans Case Study Dataset.xlsx",
        sheet_name="ndvi",
    )
    ndvi_df["date"] = pd.to_datetime(ndvi_df["date"])
    ndvi_df["year"] = ndvi_df["date"].dt.year
    ndvi_df["month"] = ndvi_df["date"].dt.month
    ndvi_df["region_id"] = ndvi_df["region_id"].astype(int)

    # Get the sow_year from July to June
    ndvi_df["sow_year"] = ndvi_df["year"]
    ndvi_df.loc[ndvi_df["month"] < 7, "sow_year"] = ndvi_df["sow_year"] - 1

    return ndvi_df


def load_weather():
    weather_df = pd.read_excel(
        "./data/MT Soybeans Case Study Dataset.xlsx",
        sheet_name="weather",
    )
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    weather_df["year"] = weather_df["date"].dt.year
    weather_df["month"] = weather_df["date"].dt.month

    # Get the sow_year from July to June
    weather_df["sow_year"] = weather_df["year"]
    weather_df.loc[weather_df["month"] < 7, "sow_year"] = weather_df["sow_year"] - 1

    weather_df["region_id"] = weather_df["region_id"].astype(int)

    return weather_df


def load_planting():
    planting_df = pd.read_excel(
        "./data/MT Soybeans Case Study Dataset.xlsx",
        sheet_name="planting",
    )
    planting_df["date"] = pd.to_datetime(planting_df["date"])
    planting_df["year"] = planting_df["date"].dt.year
    planting_df["woy"] = planting_df["date"].dt.isocalendar().week
    planting_df["month"] = planting_df["date"].dt.month
    planting_df["region_id"] = planting_df["region_id"].astype(int)
    planting_df["sow_year"] = planting_df["crop_year"].str[0:4].astype(int)
    return planting_df


def load_harvest():
    harvest_df = pd.read_excel(
        "./data/MT Soybeans Case Study Dataset.xlsx",
        sheet_name="harvest",
    )
    harvest_df["date"] = pd.to_datetime(harvest_df["date"])
    harvest_df["year"] = harvest_df["date"].dt.year
    harvest_df["woy"] = harvest_df["date"].dt.isocalendar().week
    harvest_df["region_id"] = harvest_df["region_id"].astype(int)
    harvest_df["sow_year"] = harvest_df["crop_year"].str[0:4].astype(int)
    return harvest_df


def compare_rain(sim_daily_df, weather_df):
    rain_apsim_df = (
        sim_daily_df.drop_duplicates(["region_id", "date"])
        .groupby(["region_id", "year"])
        .agg(rain_apsim=("rain", "sum"), n_apsim=("date", "count"))
        .reset_index()
    )
    rain_apsim_df = rain_apsim_df.loc[rain_apsim_df["n_apsim"] >= 365]
    rain_vit_df = (
        weather_df.groupby(["region_id", "year"])
        .agg(rain_vit=("prcp", "sum"), n_vit=("date", "count"))
        .reset_index()
    )
    rain_vit_df = rain_vit_df.loc[rain_vit_df["n_vit"] >= 365]

    rain_comp_df = pd.merge(rain_vit_df, rain_apsim_df)
    p = plot_obs_vs_pred_vit(d=rain_comp_df, x_var="rain_vit", y_var="rain_apsim")
    display(p)


def plot_yield_trends(yield_obs_df):
    # --------------------------------------------------------------------------------
    # Is there a year trend in Yield?

    plot_df = yield_obs_df.dropna(subset=["yield_obs"]).copy()

    p = (
        ggplot(
            plot_df,
        )
        + geom_line(
            aes(
                x="sow_year",
                y="yield_obs",
                color="factor(region_id)",
                # group="sow_year",
                # linetype="year_group",
            ),
            size=1,
        )
        + scale_colour_manual(values=FBN_palette)
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + labs(x="", y="Obs. Yield (tn/ha)", color="Region")
        + scale_x_continuous(
            labels=plot_df["sow_year"].tolist(),
            breaks=plot_df["sow_year"].tolist(),
        )
        + theme_ff()
        + theme(
            # figure_size=[4, 3],
            legend_position="bottom"
        )
    )
    display(p)

    # plot_df["yield_obs_3yr"] = plot_df.groupby(["region_id"])["yield_obs"].transform(
    #     lambda x: x.rolling(window=3, min_periods=1).mean()
    # )

    p = (
        ggplot(
            plot_df,
        )
        + geom_line(
            aes(
                x="sow_year",
                y="yield_obs_diff",
                color="factor(region_id)",
                # group="sow_year",
                # linetype="year_group",
            ),
            size=1,
        )
        # + geom_line(
        #     aes(
        #         x="sow_year",
        #         y="yield_obs_3yr",
        #         color="factor(region_id)",
        #         # group="sow_year",
        #         # linetype="year_group",
        #     ),
        #     size=1,
        # )
        + scale_colour_manual(values=FBN_palette)
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + labs(x="", y="Yield Change (tn/ha)", color="Region")
        + scale_x_continuous(
            labels=plot_df["sow_year"].tolist(),
            breaks=plot_df["sow_year"].tolist(),
        )
        + theme_ff()
        + theme(
            # figure_size=[4, 3],
            legend_position="bottom"
        )
    )
    display(p)


def plot_apsim_yield_boxplot():
    sim_daily_df.columns
    sim_yearly_df = (
        sim_daily_df.groupby(["region_id", "sow_year"])
        .agg(
            yield_sim=("yield", "max"),
        )
        .reset_index()
    )

    # Yield Boxplot
    p = (
        ggplot()
        + geom_boxplot(
            sim_yearly_df,
            aes(
                x="factor(region_id)",
                y="yield_sim",
                group="region_id",
            ),
            color="#709e33",
            # outlier_size=0,
            # outlier_shape="",
        )
        + geom_point(
            sim_yearly_df.loc[sim_yearly_df["sow_year"] == 2023],
            aes(x="factor(region_id)", y="yield_sim", color="factor(sow_year)"),
            size=2,
        )
        # + ylim(2.9, 3.5)
        + labs(
            y="Sim. Yield (tn/ha)",
            x="Region",
            color="",
        )
        # + scale_colour_manual(values=FBN_palette)
        + scale_color_manual(values=[FBN_YELLOW], labels=["2023/2024"])
        # + scale_x_discrete(limits=x_lims, breaks=breaks)
        + theme_ff()
        + theme(
            # figure_size=[4, 3],
            legend_position="bottom",
            text=element_text(size=12),  # Set font size for all text elements
        )
    )

    display(p)


def prepare_preds_yield(sim_daily_df, ndvi_df, yield_obs_df, planting_df):
    # Check max stage on second week
    stage_woy2 = (
        sim_daily_df.loc[sim_daily_df["woy"] <= 2]
        .groupby(["region_id", "sow_year"])["stage_num"]
        .max()
        .sort_values()
    )

    stage_df = sim_daily_df[
        ["date", "sow_year", "region_id", "stage_num", "stage_name"]
    ].drop_duplicates()
    stage_df["stage_cat"] = np.nan
    stage_df.loc[stage_df["stage_num"].between(1.5, 4), "stage_cat"] = "vg"
    stage_df.loc[stage_df["stage_num"].between(4.0001, 7), "stage_cat"] = "fw"
    if False:
        stage_df.loc[
            (stage_df["region_id"] == 5101) & (stage_df["sow_year"] == 2020),
            ["date", "stage_num", "stage_name", "stage_cat"],
        ].dropna()

        stage_df.loc[
            (stage_df["region_id"] == 5101)
            & (stage_df["sow_year"] == 2020)
            & (stage_df["stage_num"].between(1.01, 2))
        ]

    # NDVI
    ndvi_df2 = pd.merge(ndvi_df, stage_df, how="left")
    ndvi_sum = (
        ndvi_df2.groupby(["region_id", "sow_year", "stage_cat"])
        .agg(ndvi=("ndvi", "mean"))
        .reset_index()
    )

    # Weather
    if False:
        # Add a new stage, that groups the critical period stages
        cp_df = sim_daily_df.loc[
            sim_daily_df["stage_num_str"].isin(["3", "4", "5"])
        ].copy()
        cp_df["stage_num_str"] = "cp"

        # Add a new stage, that groups the grain filling stages
        gf_df = sim_daily_df.loc[
            sim_daily_df["stage_num_str"].isin(["6", "7", "8"])
        ].copy()
        gf_df["stage_num_str"] = "gf"

        sim_daily_df2 = pd.concat([sim_daily_df, cp_df, gf_df], ignore_index=True)
    sim_daily_df2 = pd.merge(
        sim_daily_df,
        stage_df[["date", "region_id", "sow_year", "stage_cat"]],
        how="left",
    )
    weather_sum = (
        sim_daily_df2.groupby(["region_id", "sow_year", "stage_cat"])
        .agg(
            maxt=("maxt", "mean"),
            # mint=("mint", "mean"), # Keep only one temp
            rain=("rain", "sum"),
            radn=("radn", "mean"),
            stage_length=("date", "count"),
        )
        .reset_index()
    )

    preds_yield_df = pd.merge(weather_sum, ndvi_sum, how="left")

    # Limit Predictors to stage 7 (woy 2)
    # preds_yield_df = preds_yield_df.loc[
    #     preds_yield_df["stage_num"].astype(int) <= 7
    # ].copy()

    # Pivot the table
    preds_yield_wide_df = pd.pivot_table(
        preds_yield_df,
        index=["region_id", "sow_year"],
        columns="stage_cat",
        # values=["ndvi", "stage_length"],
        # aggfunc="mean",
    ).reset_index()

    # Flatten the MultiIndex columns
    preds_yield_wide_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
        for col in preds_yield_wide_df.columns
    ]
    # Add sim yield
    sim_yearly_df = (
        sim_daily_df.groupby(["region_id", "sow_year"])
        .agg(
            yield_sim=("yield", "max"),
        )
        .reset_index()
    )

    preds_yield_wide_df = pd.merge(preds_yield_wide_df, sim_yearly_df, how="left")

    # Add APSIM Maturity Indicator
    apsim_stage_df = (
        sim_daily_df[sim_daily_df["woy"] == 2]
        .groupby(["region_id", "sow_year"])
        .agg(apsim_stage_woy2=("stage_num", "max"))
        .reset_index()
    )
    preds_yield_wide_df = pd.merge(preds_yield_wide_df, apsim_stage_df, how="left")
    # Add Planting Date
    # planting_date_df = (
    #     planting_df.loc[planting_df["pct_planting_cumulative"] > 0.5]
    #     .groupby(["region_id", "sow_year"])
    #     .agg(planting50_woy=("woy", "min"))
    #     .reset_index()
    # )

    # preds_yield_wide_df = pd.merge(preds_yield_wide_df, planting_date_df, how="left")

    # Add obs yield
    preds_yield_wide_df = pd.merge(
        preds_yield_wide_df,
        yield_obs_df[
            ["sow_year", "region_id", "yield_obs", "yield_obs_l1", "yield_obs_diff"]
        ],
        how="left",
    )

    return preds_yield_wide_df


def check_variation(df, var):
    if var not in df.columns:
        return False

    v = df[var].copy()
    # Test: less than 50% are nan
    test1 = (v.isna().sum() / len(v)) < 0.5

    # Replace 0 pi by nan (iacorn has many 0 when missing)
    v = v.replace({"0": np.nan, 0: np.nan})
    # Test: some variation in values
    # test2 = v.std() > 0.1

    # Test: a few unique values
    test3 = len(v.unique()) > 10

    return test1 & test3


# df, y_var =testing_df.copy(), "yield_obs_diff"
def cb_pool_data(df, y_var, test=False):
    """
    Returns a pool dataset for catboost

    df (GeoDataFrame): with _x and _y columns for candidates
    test (bool): true if the data is for testing (unlabeled)

    Returns:
        cb.Pool
    """
    cat_features = ["region_id"]
    exclude_var = [
        "sow_year",
        # "ndvi_fw",
        # "ndvi_vg",
        "yield_obs_diff",
        "yield_obs",
        "yield_obs_l1",
    ]

    df = df.dropna(subset=y_var)

    x_vars = df.columns.difference(cat_features + [y_var] + exclude_var)

    if False:
        x_vars2 = list()
        for var in x_vars:
            if check_variation(df, var):
                x_vars2.append(var)
    else:
        x_vars2 = list(x_vars)

    X_df = df[x_vars2 + cat_features].fillna(-999)

    if not test:
        y_train = df[y_var]
        X_train = X_df.copy()
        return X_train, y_train, cat_features


year_n = 2023
show_plot = True


def pred_with_cb(preds_yield_wide_df, year_n, show_plot=True):
    # preds_yield_wide_df["region_id"] = preds_yield_wide_df["region_id"].astype(str)
    model = cb.CatBoostRegressor(
        loss_function="RMSE",
        logging_level="Silent",
        allow_writing_files=False,
    )

    training_df = (
        preds_yield_wide_df.loc[preds_yield_wide_df["sow_year"] != year_n]
        .reset_index(drop=True)
        .copy()
    )
    testing_df = (
        preds_yield_wide_df.loc[preds_yield_wide_df["sow_year"] == year_n]
        .reset_index(drop=True)
        .copy()
    )
    y_var = "yield_obs"
    X_train, y_train, cat_features = cb_pool_data(training_df, y_var, test=False)
    cols_selected = list(X_train.columns)
    X_test = testing_df[cols_selected].fillna(-999)

    model.fit(
        X=X_train,
        y=y_train,
        cat_features=cat_features,
    )
    if show_plot & (year_n == 2023):
        _ = plot_feature_importances(model)

    if False:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # summarize the effects of all the features
        shap.summary_plot(
            shap_values,
            X_train,
            # feature_names=boston.feature_names[sorted_feature_importance],
        )

        # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
        shap.dependence_plot("planting_pct_10", shap_values, X_train, xmin=0)
        shap.dependence_plot("building_cnt_10km_x", shap_values, X_test, xmin=0)

        # Selecting the variable for interaction
        shap.dependence_plot(
            "ndvi_fw",
            shap_values,
            X_train,
            interaction_index="rain_fw",
            xmin=0,
        )

    if "diff" in y_var:
        testing_df["yield_cb_diff"] = model.predict(X_test)
        testing_df["yield_cb"] = (
            testing_df["yield_cb_diff"] + testing_df["yield_obs_l1"]
        )
        if False:
            testing_df.loc[(testing_df["region_id"] == 5104)]
    else:
        testing_df["yield_cb"] = model.predict(X_test)
    return testing_df


def pred_loyo_yield(preds_yield_wide_df, show_plot=True):
    res_ls = list()
    for year_n in preds_yield_wide_df["sow_year"].unique():
        res_tmp = pred_with_cb(preds_yield_wide_df, year_n, show_plot)
        res_ls.append(res_tmp)
    yield_forecast_preds_df = pd.concat(res_ls, ignore_index=True)
    if False:
        yield_forecast_preds_df.loc[(yield_forecast_preds_df["region_id"] == 5104)]

    # Create a new column for grouping sow_year
    yield_forecast_preds_df["year_group"] = yield_forecast_preds_df["sow_year"].apply(
        lambda year: "2023" if year == 2023 else "Other Years"
    )
    return yield_forecast_preds_df


# d, x_var, y_var=yield_forecast_preds_df.copy(), "yield_obs", "yield_cb"
def plot_obs_vs_pred_cb(d, x_var, y_var):
    d_plot = d[[x_var, y_var, "region_id"]].dropna()
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
            data=d_plot,
            mapping=aes(x=x_var, y=y_var, color="factor(region_id)"),
            size=1.5,
            # color="#709e33",
        )
        + coord_fixed()
        + geom_abline(alpha=0.2, linetype="dashed")
        + ylim(p0, p1)
        + xlim(p0, p1)
        #   xlab('State change (%)')+
        #   ylab('County change (%)')+
        + annotate("text", x=x_lab, y=y_lab, label=RMSE_label, color="#709e33")
        + annotate("text", x=x_lab, y=y_lab + 0.2, label=corr_label, color="#709e33")
        + scale_colour_manual(values=FBN_palette)
        + theme(
            figure_size=[3.5, 3.5],
            aspect_ratio=1,
            # text=element_text(size=12),  # Set font size for all text elements
        )
        + theme_ff()
    )

    return g


def explore_preds_yield(yield_forecast_preds_df):
    # --------------------------------
    # Explore one predictor

    # plot_df = yield_forecast_preds_df[
    #     ["region_id", "year_group", "sow_year", "planting50_woy", "yield_cb"]
    # ].dropna()
    # plot_df["planting50_woy"] = plot_df["planting50_woy"].astype(int)
    # p = (
    #     ggplot(
    #         plot_df,
    #         aes(
    #             x="planting50_woy",
    #             y="yield_cb",
    #         ),
    #     )
    #     + geom_point(aes(color="year_group"), size=1)
    #     + geom_smooth(method="lm", color=FBN_GREEN, se=False)
    #     # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
    #     # + geom_smooth(
    #     #     aes(group="factor(region_id)"), method="lm", se=False, color=FBN_GREEN
    #     # )
    #     + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
    #     + labs(y="Pred. Yield (tn/ha)", color="")
    #     + theme_ff()
    # )
    # display(p)
    # --------------------------------
    # Explore one predictor (Time Series)
    # p = (
    #     ggplot(
    #         yield_forecast_preds_df,
    #         aes(
    #             x="yield_obs_l1",
    #             y="yield_cb",
    #         ),
    #     )
    #     + geom_point(aes(color="year_group"), size=1)
    #     + geom_smooth(method="lm", color=FBN_GREEN, se=False)
    #     # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
    #     # + geom_smooth(
    #     #     aes(group="factor(region_id)"), method="lm", se=False, color=FBN_GREEN
    #     # )
    #     + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
    #     + labs(y="Pred. Yield (tn/ha)", color="")
    #     + theme_ff()
    #     + theme(
    #         text=element_text(size=12),  # Set font size for all text elements
    #     )
    # )
    # display(p)

    # --------------------------------
    # Explore one predictor
    p = (
        ggplot(
            yield_forecast_preds_df,
            aes(
                x="ndvi_fw",
                y="yield_cb",
            ),
        )
        + geom_point(aes(color="year_group"), size=1)
        + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(
        #     aes(group="factor(region_id)"), method="lm", se=False, color=FBN_GREEN
        # )
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + labs(y="Pred. Yield (tn/ha)", color="")
        + theme_ff()
        + theme(
            text=element_text(size=12),  # Set font size for all text elements
        )
    )
    display(p)

    # --------------------------------
    # Explore one predictor
    p = (
        ggplot(
            yield_forecast_preds_df,
            aes(
                x="maxt_fw",
                y="yield_cb",
            ),
        )
        + geom_point(aes(color="year_group"), size=1)
        + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(
        #     aes(group="factor(region_id)"), method="lm", se=False, color=FBN_GREEN, linetype="dashed"
        # )
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + labs(y="Pred. Yield (tn/ha)", color="")
        + theme_ff()
        + theme(
            text=element_text(size=12),  # Set font size for all text elements
        )
    )
    display(p)

    # --------------------------------
    # Explore one predictor
    p = (
        ggplot(
            yield_forecast_preds_df,
            aes(
                x="rain_vg",
                y="yield_cb",
            ),
        )
        + geom_point(aes(color="year_group"), size=1)
        + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(
        #     aes(group="factor(region_id)"), method="lm", se=False, color=FBN_GREEN, linetype="dashed"
        # )
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + labs(y="Pred. Yield (tn/ha)", color="")
        + theme_ff()
        + theme(
            text=element_text(size=12),  # Set font size for all text elements
        )
    )
    display(p)

    # --------------------------------
    # Explore one predictor
    p = (
        ggplot(
            yield_forecast_preds_df,
            aes(
                x="rain_fw",
                y="yield_cb",
            ),
        )
        + geom_point(aes(color="year_group"), size=1)
        + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(
        #     aes(group="factor(region_id)"), method="lm", se=False, color=FBN_GREEN
        # )
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + labs(y="Pred. Yield (tn/ha)", color="")
        + theme_ff()
        + theme(
            text=element_text(size=12),  # Set font size for all text elements
        )
    )
    display(p)

    # --------------------------------
    # Yield Boxplot
    p = (
        ggplot()
        + geom_boxplot(
            yield_forecast_preds_df,
            aes(
                x="factor(region_id)",
                y="yield_cb",
                group="region_id",
            ),
            color="#709e33",
            # outlier_size=0,
            # outlier_shape="",
        )
        + geom_point(
            yield_forecast_preds_df.loc[yield_forecast_preds_df["sow_year"] == 2023],
            aes(x="factor(region_id)", y="yield_cb", color="factor(sow_year)"),
            size=2,
        )
        + ylim(2.9, 3.5)
        + labs(
            y="Pred. Yield (tn/ha)",
            x="Region",
            color="",
        )
        # + scale_colour_manual(values=FBN_palette)
        + scale_color_manual(values=[FBN_YELLOW], labels=["2023/2024"])
        # + scale_x_discrete(limits=x_lims, breaks=breaks)
        + theme_ff()
        + theme(
            # figure_size=[4, 3],
            legend_position="bottom",
            text=element_text(size=12),  # Set font size for all text elements
        )
    )

    display(p)
    # --------------------------------
    # Obs vs pred
    g = plot_obs_vs_pred_cb(
        d=yield_forecast_preds_df, x_var="yield_obs", y_var="yield_cb"
    )
    g = g + labs(x="Pred. Yield (tn/ha)", y="Obs. Yield (tn/ha)", color="Region")
    display(g)
    # --------------------------------
    # Performance Metrics

    df_tmp = yield_forecast_preds_df[["yield_obs", "yield_cb"]].dropna().copy()
    df_tmp["cultivar"] = "MG6"

    rmse_comp_df = calculate_metrics_by_group(
        df=df_tmp,
        group_cols=["cultivar"],
        observed_col="yield_obs",
        predicted_col="yield_cb",
    )
    display(rmse_comp_df)


def adjust_yield_forecast(yield_forecast_preds_df, yield_obs_df):
    """
    Adjust the yield cb relative to the mean prediction.
    So if the yield cb is 1.1 from the yield cb mean, we multiply the yield obs x 1.1
    Is this better???
    It also fills 2023 production predictions

    """
    yield_forecast_adj_df = yield_forecast_preds_df.copy()
    # Get mean predictions

    yield_forecast_adj_df["yield_cb_mean"] = yield_forecast_adj_df.groupby("region_id")[
        "yield_cb"
    ].transform("mean")

    # Get ratio
    yield_forecast_adj_df["forecast_ratio"] = (
        yield_forecast_adj_df["yield_cb"] / yield_forecast_adj_df["yield_cb_mean"]
    )

    # Add Mean Obs Yield
    yield_forecast_adj_df["yield_obs_mean"] = yield_forecast_adj_df.groupby(
        "region_id"
    )["yield_obs"].transform("mean")

    yield_forecast_adj_df["yield_cb_adj"] = (
        yield_forecast_adj_df["yield_obs_mean"]
        * yield_forecast_adj_df["forecast_ratio"]
    )

    # Add area
    yield_forecast_adj_df = pd.merge(
        yield_forecast_adj_df,
        yield_obs_df[["region_id", "sow_year", "planted_area", "production"]],
        how="left",
    )

    # Fill yields with forecast
    yield_forecast_adj_df.loc[
        yield_forecast_adj_df["yield_obs"].isna(), "yield_obs"
    ] = yield_forecast_adj_df["yield_cb_adj"]

    production_forecast = (
        yield_forecast_adj_df["planted_area"] * yield_forecast_adj_df["yield_obs"]
    )
    yield_forecast_adj_df.loc[
        yield_forecast_adj_df["production"].isna(), "production"
    ] = production_forecast

    yield_forecast_adj_df.loc[yield_forecast_adj_df["sow_year"] == 2023].to_csv(
        "./data/figures/preds2023.csv", index=False
    )
    return yield_forecast_adj_df


def plot_production_timeline(prod_forecast_year_df):
    prod_forecast_year_df = (
        yield_forecast_adj_df.dropna(subset=["production", "planted_area"])
        .groupby(["sow_year"])
        .agg(
            production=("production", "sum"),
            planted_area=("planted_area", "sum"),
            # n=("production", "count"),
        )
        .reset_index()
    ).copy()

    prod_forecast_year_df["production"] = prod_forecast_year_df["production"] / 1e6
    prod_forecast_year_df["planted_area"] = prod_forecast_year_df["planted_area"] / 1e6

    p = (
        ggplot(
            prod_forecast_year_df,
            aes(x="sow_year", y="planted_area"),
        )
        + geom_line(color=FBN_GREEN, size=2)
        + labs(x="", y="Planted Area (million ha)")
        + scale_x_continuous(
            labels=prod_forecast_year_df["sow_year"].tolist(),
            breaks=prod_forecast_year_df["sow_year"].tolist(),
        )
        + theme_ff()
    )
    display(p)

    p = (
        ggplot(
            prod_forecast_year_df,
            aes(x="sow_year", y="production"),
        )
        + geom_line(color=FBN_GREEN, size=2)
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + labs(x="", y="Production (million tn)")
        + scale_x_continuous(
            labels=prod_forecast_year_df["sow_year"].tolist(),
            breaks=prod_forecast_year_df["sow_year"].tolist(),
        )
        + scale_y_continuous(
            breaks=range(0, int(prod_forecast_year_df["production"].max()) + 1, 5)
        )  # Set breaks for y-axis
        + theme_ff()
    )
    display(p)


def save_season_characterization(yield_forecast_adj_df):
    yield_forecast_adj_df.columns
    cols_sum = [
        "maxt_vg",
        "maxt_fw",
        "ndvi_vg",
        "ndvi_fw",
        "radn_vg",
        "radn_fw",
        "rain_vg",
        "rain_fw",
        "stage_length_vg",
        "stage_length_fw",
        "apsim_stage_woy2",
    ]
    season_sum_df = (
        yield_forecast_adj_df.groupby("sow_year")[cols_sum]
        .mean()
        .round(2)
        .reset_index()
    )
    season_sum_df.to_csv(
        "./data/figures/season_sum.csv", index=False
    )
    # Check soil water variables
    sim_daily_df.columns

    pd.pivot_table(
        sim_daily_df.loc[
            (sim_daily_df["doy"] == 10), ["sow_year", "region_id", "esw_total"]
        ],
        index="sow_year",
        columns="region_id",
    )


if __name__ == "__main__":
    yield_obs_df = load_yield_obs()
    ndvi_df = load_ndvi()
    weather_df = load_weather()
    planting_df = load_planting()
    sim_daily_df = load_sim_daily()
    plot_yield_trends(yield_obs_df)
    plot_apsim_yield_boxplot(sim_daily_df)
    # compare_rain(sim_daily_df, weather_df)
    preds_yield_wide_df = prepare_preds_yield(
        sim_daily_df, ndvi_df, yield_obs_df, planting_df
    )
    yield_forecast_preds_df = pred_loyo_yield(preds_yield_wide_df, show_plot=True)
    explore_preds_yield(yield_forecast_preds_df)
    yield_forecast_adj_df = adjust_yield_forecast(yield_forecast_preds_df, yield_obs_df)
    yield_forecast_adj_df.to_parquet(
        "./data/export/yield_forecast_adj_df.parquet"
    )

    plot_production_timeline(yield_forecast_adj_df)
    save_season_characterization(yield_forecast_adj_df)
