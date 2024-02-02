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

from V2_ndvi import (
    load_sim_daily,
    load_yield_obs,
    load_ndvi,
    load_weather,
    load_harvest,
    load_planting,
    check_variation,
)


pd.set_option("display.max_rows", 100)  # show all columns of tables


def prepare_preds_harvest(
    planting_df, weather_sum_df, ndvi_sum_df, harvest_df, sim_daily_df
):
    planting_sum_df = (
        planting_df.loc[planting_df["month"].isin([9, 10, 11])]
        .groupby(["region_id", "sow_year", "month"])
        .agg(planting_pct=("pct_planting_cumulative", "max"))
    )

    weather_sum_df = (
        weather_df.loc[weather_df["month"].isin([9, 10, 11, 12])]
        .groupby(["region_id", "sow_year", "month"])
        .agg(
            tmax=("tmax", "mean"),
            # tmin=("tmin", "mean"),
            rain=("prcp", "sum"),
        )
    )

    ndvi_sum_df = (
        ndvi_df.loc[ndvi_df["month"].isin([10, 11, 12])]
        .groupby(["region_id", "sow_year", "month"])
        .agg(
            ndvi=("ndvi", "mean"),
        )
    )

    full_df = (
        planting_sum_df.merge(
            weather_sum_df, how="outer", left_index=True, right_index=True
        )
        .merge(ndvi_sum_df, how="outer", left_index=True, right_index=True)
        .reset_index()
    )

    # Pivot the table
    full_wide_df = pd.pivot_table(
        full_df,
        index=["region_id", "sow_year"],
        columns="month",
        # values=["ndvi", "stage_length"],
        # aggfunc="mean",
    ).reset_index()

    # Flatten the MultiIndex columns
    full_wide_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] != "" else col[0] for col in full_wide_df.columns
    ]

    # Add APSIM Maturity Indicator
    apsim_stage_df = (
        sim_daily_df[sim_daily_df["woy"] == 2]
        .groupby(["region_id", "sow_year"])
        .agg(apsim_stage_woy2=("stage_num", "max"))
        .reset_index()
    )
    full_wide_df = pd.merge(full_wide_df, apsim_stage_df, how="left")

    # Add APSIM Soil Water
    apsim_esw_df = (
        sim_daily_df[sim_daily_df["doy"] == 15]
        .groupby(["region_id", "sow_year"])
        .agg(esw_1=("esw_total", "max"))
        .reset_index()
    )
    full_wide_df = pd.merge(full_wide_df, apsim_esw_df, how="left")

    # Add harvest
    if False:
        harvest_df.loc[
            (harvest_df["sow_year"] == 2020) & (harvest_df["region_id"] == 5104)
        ]
    harvest_tmp1 = harvest_df.loc[
        harvest_df["woy"].between(0, 20),
        ["region_id", "sow_year", "woy", "pct_harvested_cumulative"],
    ]

    # Add 2023 to harvest (missing)
    harvest_tmp2 = harvest_tmp1.loc[(harvest_tmp1["sow_year"] == 2021)].copy()
    harvest_tmp2["sow_year"] = 2023
    harvest_tmp2["pct_harvested_cumulative"] = np.nan
    harvest_tmp = pd.concat([harvest_tmp1, harvest_tmp2], ignore_index=True)

    preds_harvest_df = pd.merge(
        harvest_tmp,
        full_wide_df,
        how="left",
    )
    return preds_harvest_df


# df = training_df.copy()
def cb_pool_data(df, test=False):
    """
    Returns a pool dataset for catboost

    df (GeoDataFrame): with _x and _y columns for candidates
    test (bool): true if the data is for testing (unlabeled)

    Returns:
        cb.Pool
    """
    cat_features = ["region_id"]
    y_var = "pct_harvested_cumulative"
    exclude_var = ["sow_year"]

    df = df.dropna(subset=y_var)

    # x_vars = df.columns.difference(cat_features + [y_var] + exclude_var)
    x_vars = [
        "woy",
        # "ndvi_10",
        # "ndvi_11",
        # "ndvi_12",
        "planting_pct_9",
        "planting_pct_10",
        "planting_pct_11",
        # "rain_9",
        "rain_10",
        "rain_11",
        "rain_12",
        # "tmax_9",
        # "tmax_10",
        # "tmax_11",
        # "tmax_12",
        # "apsim_stage_woy2",
        "esw_1",
    ]

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


def pred_with_cb_harvest(preds_harvest_df, year_n, show_plot=True):
    # yield_forecast_adj_df["region_id"] = yield_forecast_adj_df["region_id"].astype(str)
    model = cb.CatBoostRegressor(
        loss_function="RMSE",
        logging_level="Silent",
        allow_writing_files=False,
    )

    training_df = (
        preds_harvest_df.loc[
            (preds_harvest_df["sow_year"] != year_n)  # & (preds_harvest_df["woy"] == 8)
        ]
        .reset_index(drop=True)
        .copy()
    )
    testing_df = (
        preds_harvest_df.loc[
            (preds_harvest_df["sow_year"] == year_n)  # & (preds_harvest_df["woy"] == 8)
        ]
        .reset_index(drop=True)
        .copy()
    )

    X_train, y_train, cat_features = cb_pool_data(training_df, test=False)
    cols_selected = list(X_train.columns)
    X_test = testing_df[cols_selected]

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
            "planting_pct_10",
            shap_values,
            X_train,
            interaction_index="rain_12",
            xmin=0,
        )

    testing_df["harvest_pct_cb"] = model.predict(X_test)
    return testing_df


def pred_loyo_harvest(preds_harvest_df, show_plot=True):
    res_ls = list()
    for year_n in preds_harvest_df["sow_year"].unique():
        res_tmp = pred_with_cb_harvest(preds_harvest_df, year_n, show_plot)
        res_ls.append(res_tmp)
    harvest_forecast_df = pd.concat(res_ls, ignore_index=True)

    return harvest_forecast_df


def fill_some_variables(harvest_forecast_df, yield_forecast_adj_df):
    harvest_forecast_df.loc[
        harvest_forecast_df["pct_harvested_cumulative"].isna(),
        "pct_harvested_cumulative",
    ] = harvest_forecast_df["harvest_pct_cb"]

    # Add Yield
    harvest_forecast_df = pd.merge(
        harvest_forecast_df,
        yield_forecast_adj_df[["region_id", "sow_year", "yield_obs", "planted_area"]],
        how="left",
    )

    harvest_forecast_df["harvested_area_cb"] = (
        harvest_forecast_df["planted_area"] * harvest_forecast_df["harvest_pct_cb"]
    )

    harvest_forecast_df["harvested_production_cb"] = (
        harvest_forecast_df["harvested_area_cb"] * harvest_forecast_df["yield_obs"]
    )

    # Needed for plotting in several parts
    harvest_forecast_df["year_group"] = harvest_forecast_df["sow_year"].apply(
        lambda year: "2023" if year == 2023 else "Other Years"
    )
    return harvest_forecast_df


def explore_obs_harvest(harvest_forecast_df):
    harvest_forecast_df.loc[
        (harvest_forecast_df["region_id"] == 5101)
        & (harvest_forecast_df["sow_year"] == 2020),
        [
            "region_id",
            "sow_year",
            "woy",
            "pct_harvested_cumulative",
            "harvest_pct_cb",
            "harvested_area_cb",
            "harvested_production_cb",
        ],
    ]
    harvest_forecast_df["region_id"].unique()
    plot_df = harvest_forecast_df.loc[
        (harvest_forecast_df["woy"] == 8)
        # & (harvest_forecast_df["region_id"] == 5104)
    ].copy()

    # --------------------------------
    # Explore individual Predictors
    plot_df.columns
    p = (
        ggplot(
            plot_df,
            aes(
                x="planting_pct_11",
                y="pct_harvested_cumulative*100",
            ),
        )
        + geom_point(aes(color="factor(region_id)"), size=1)
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
        + geom_smooth(aes(color="factor(region_id)"), method="lm", se=False)
        + scale_colour_manual(values=FBN_palette)
        + labs(y="Obs. Harvest Progress WOY 8 (%)", color="Region")
        + theme_ff()
        + theme(
            text=element_text(size=12),  # Set font size for all text elements
        )
    )
    display(p)

    p = (
        ggplot(
            plot_df,
            aes(
                x="rain_12",
                y="pct_harvested_cumulative*100",
            ),
        )
        + geom_point(aes(color="factor(region_id)"), size=1)
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
        + geom_smooth(aes(color="factor(region_id)"), method="lm", se=False)
        + scale_colour_manual(values=FBN_palette)
        + labs(y="Obs. Harvest Progress WOY 8 (%)", color="Region")
        + theme_ff()
        + theme(
            text=element_text(size=12),  # Set font size for all text elements
        )
    )
    display(p)

    p = (
        ggplot(
            plot_df,
            aes(
                x="tmax_12",
                y="pct_harvested_cumulative*100",
            ),
        )
        + geom_point(aes(color="factor(region_id)"), size=1)
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
        + geom_smooth(aes(color="factor(region_id)"), method="lm", se=False)
        + scale_colour_manual(values=FBN_palette)
        + labs(y="Obs. Harvest Progress WOY 8 (%)", color="Region")
        + theme_ff()
        + theme(
            text=element_text(size=12),  # Set font size for all text elements
        )
    )
    display(p)

    p = (
        ggplot(
            plot_df,
            aes(
                x="ndvi_12",
                y="pct_harvested_cumulative*100",
            ),
        )
        + geom_point(aes(color="factor(region_id)"), size=1)
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
        + geom_smooth(aes(color="factor(region_id)"), method="lm", se=False)
        + scale_colour_manual(values=FBN_palette)
        + labs(y="Obs. Harvest Progress WOY 8 (%)", color="Region")
        + theme_ff()
        + theme(
            text=element_text(size=12),  # Set font size for all text elements
        )
    )
    display(p)


def explore_preds_harvest(harvest_forecast_df):
    harvest_forecast_df.loc[
        (harvest_forecast_df["region_id"] == 5101)
        & (harvest_forecast_df["sow_year"] == 2020),
        [
            "region_id",
            "sow_year",
            "woy",
            "pct_harvested_cumulative",
            "harvest_pct_cb",
            "harvested_area_cb",
            "harvested_production_cb",
        ],
    ]
    harvest_forecast_df["region_id"].unique()
    plot_df = harvest_forecast_df.loc[
        (harvest_forecast_df["woy"] == 8)
        # & (harvest_forecast_df["region_id"] == 5104)
    ].copy()

    # --------------------------------
    # Explore individual Predictors
    plot_df.columns
    p = (
        ggplot(
            plot_df,
            aes(
                x="planting_pct_10",
                y="pct_harvested_cumulative*100",
            ),
        )
        + geom_point(aes(color="year_group"), size=1)
        + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(
        #     aes(group="factor(region_id)"), method="lm", se=False, color=FBN_GREEN
        # )
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + labs(y="Pred. Harvest Progress WOY 8 (%)", color="")
        + theme_ff()
    )
    display(p)

    p = (
        ggplot(
            plot_df,
            aes(
                x="rain_12",
                y="harvest_pct_cb*100",
            ),
        )
        + geom_point(aes(color="factor(region_id)"), size=1)
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        # + geom_smooth(aes(group="region_id"), method="lm", color=FBN_GREEN, se=False)
        + geom_smooth(aes(color="factor(region_id)"), method="lm", se=False)
        + scale_colour_manual(values=FBN_palette)
        + labs(y="Pred. Harvest Progress WOY 8 (%)", color="Region")
        + theme_ff()
    )
    display(p)

    p = (
        ggplot(
            plot_df,
            aes(
                x="tmax_12",
                y="pct_harvested_cumulative*100",
            ),
        )
        + geom_point(aes(color="year_group"), size=1)
        + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + labs(y="Pred. Harvest Progress WOY 8 (%)", color="")
        + theme_ff()
    )
    display(p)

    p = (
        ggplot(
            plot_df,
            aes(
                x="rain_11",
                y="pct_harvested_cumulative*100",
            ),
        )
        + geom_point(aes(color="year_group"), size=1)
        + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + labs(y="Pred. Harvest Progress WOY 8 (%)", color="")
        + theme_ff()
    )
    display(p)

    p = (
        ggplot(
            plot_df.loc[plot_df["apsim_stage_woy2"] > 6],
            aes(
                x="apsim_stage_woy2",
                y="pct_harvested_cumulative*100",
            ),
        )
        + geom_point(aes(color="year_group"), size=1)
        + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + labs(y="Pred. Harvest Progress WOY 8 (%)", color="")
        + theme_ff()
    )
    display(p)


def regression_model(preds_harvest_df):
    import statsmodels.api as sm

    df = pd.DataFrame(preds_harvest_df.loc[preds_harvest_df["woy"] == 8]).dropna()

    # Convert region_id to a categorical variable
    # df["region_id"] = pd.Categorical(df["region_id"])

    X = pd.get_dummies(
        df[
            [
                # "woy",
                # "planting_pct_9",
                # "planting_pct_10",
                "planting_pct_11",
                # "rain_10",
                # "rain_11",
                "rain_12",
                "tmax_12",
                "region_id",
            ]
        ],
        columns=["region_id"],
        drop_first=True,
        dtype=int,
    )
    # Add a constant term to the independent variables for the intercept
    X = sm.add_constant(X)
    X
    # Fit the linear regression model
    model = sm.OLS(df["pct_harvested_cumulative"], X).fit()

    # Print the regression summary
    print(model.summary())

    res = model.summary()
    res = pd.DataFrame(res.tables[1])
    res.to_csv("./data/figures/regression.csv", index=False)


def plot_planting_progress_curve(planting_df, yield_obs_df):
    # --------------------------------
    # Planting Progress Curve (Total, not by region. For that we need to add up the areas)
    # Add planting area
    planting_plot_df = pd.merge(
        planting_df, yield_obs_df[["region_id", "sow_year", "planted_area"]], how="left"
    )
    planting_plot_df["planted_area"] = (
        planting_plot_df["planted_area"] * planting_plot_df["pct_planting_cumulative"]
    )

    planting_sum_df = (
        planting_plot_df.groupby(["sow_year", "woy"])
        .agg(
            planted_area=("planted_area", "sum"),
            n=("sow_year", "count"),
        )
        .reset_index()
    )

    planting_sum_df["planted_area"] = planting_sum_df["planted_area"] / 1e6

    # Now calculate the percentage, based on the total areas
    total_planted_area = planting_sum_df.groupby(["sow_year"])[
        "planted_area"
    ].transform("max")
    planting_sum_df["pct_planting_cumulative"] = (
        planting_sum_df["planted_area"] / total_planted_area
    )

    planting_sum_df["year_group"] = planting_sum_df["sow_year"].apply(
        lambda year: "2023" if year == 2023 else "Other Years"
    )

    p = (
        ggplot(
            planting_sum_df,
        )
        + geom_line(
            aes(
                x="woy",
                y="pct_planting_cumulative",
                color="year_group",
                group="sow_year",
                linetype="year_group",
            ),
            size=1,
        )
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + labs(x="Week Of Year", y="Planting Progress (%)", color="", linetype="")
        # + scale_x_continuous(
        #     labels=prod_forecast_year_df["sow_year"].tolist(),
        #     breaks=prod_forecast_year_df["sow_year"].tolist(),
        # )
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + scale_linetype_manual(values={"2023": "solid", "Other Years": "dashed"})
        # + facet_wrap("~region_id")
        + theme_ff()
        + theme(
            # figure_size=[4, 3],
            legend_position="bottom"
        )
    )
    display(p)


def plot_year_trends(planting_df, harvest_df, yield_obs_df):
    # --------------------------------------------------------------------------------
    # Is there a year trend in planting dates?

    planting50_df = (
        planting_df.loc[planting_df["pct_planting_cumulative"] >= 0.5]
        .groupby(["region_id", "sow_year"])
        .first()
        .reset_index()
    )

    p = (
        ggplot(
            planting50_df,
        )
        + geom_line(
            aes(
                x="sow_year",
                y="woy",
                color="factor(region_id)",
                # group="sow_year",
                # linetype="year_group",
            ),
            size=1,
        )
        + scale_colour_manual(values=FBN_palette)
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + labs(x="", y="Planting 50% (WOY)", color="Region")
        + scale_x_continuous(
            labels=planting50_df["sow_year"].tolist(),
            breaks=planting50_df["sow_year"].tolist(),
        )
        + theme_ff()
        + theme(
            # figure_size=[4, 3],
            legend_position="bottom"
        )
    )
    display(p)

    # --------------------------------------------------------------------------------
    # Is there a year trend in harvest dates?

    harvest50_df = (
        harvest_df.loc[harvest_df["pct_harvested_cumulative"] >= 0.5]
        .groupby(["region_id", "sow_year"])
        .first()
        .reset_index()
    )

    p = (
        ggplot(
            harvest50_df,
        )
        + geom_line(
            aes(
                x="sow_year",
                y="woy",
                color="factor(region_id)",
                # group="sow_year",
                # linetype="year_group",
            ),
            size=1,
        )
        + scale_colour_manual(values=FBN_palette)
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + labs(x="", y="Harvest 50% (WOY)", color="Region")
        + scale_x_continuous(
            labels=harvest50_df["sow_year"].tolist(),
            breaks=harvest50_df["sow_year"].tolist(),
        )
        + theme_ff()
        + theme(
            # figure_size=[4, 3],
            legend_position="bottom"
        )
    )
    display(p)

    # --------------------------------------------------------------------------------
    # Is there a year trend in Yield?

    plot_df = yield_obs_df.dropna(subset=["yield_obs"])

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


def explore_harvest_curve(harvest_forecast_df):
    # --------------------------------
    # By Region

    p = (
        ggplot(
            harvest_forecast_df,
        )
        + geom_line(
            aes(
                x="woy",
                y="harvest_pct_cb",
                color="year_group",
                group="sow_year",
                linetype="year_group",
            ),
            size=1,
        )
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + labs(x="Week Of Year", y="Harvest Cum. Progress (%)", color="", linetype="")
        # + scale_x_continuous(
        #     labels=prod_forecast_year_df["sow_year"].tolist(),
        #     breaks=prod_forecast_year_df["sow_year"].tolist(),
        # )
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + scale_linetype_manual(values={"2023": "solid", "Other Years": "dashed"})
        + facet_wrap("~region_id")
        + theme_ff()
        + theme(
            # figure_size=[4, 3],
            legend_position="bottom"
        )
    )
    display(p)

    # --------------------------------
    # Total Harvest (not by region)
    # Add planted area

    harvest_progress_curve_df = (
        harvest_forecast_df.groupby(["sow_year", "woy"])
        .agg(
            harvested_area_cb=("harvested_area_cb", "sum"),
            n=("sow_year", "count"),
        )
        .reset_index()
    )
    # Calculate the percentage over the maximum area
    harvest_progress_curve_df["harvest_pct_cb"] = harvest_progress_curve_df[
        "harvested_area_cb"
    ] / harvest_progress_curve_df.groupby("sow_year")["harvested_area_cb"].transform(
        "max"
    )

    harvest_progress_curve_df["year_group"] = harvest_progress_curve_df[
        "sow_year"
    ].apply(lambda year: "2023" if year == 2023 else "Other Years")

    p = (
        ggplot(
            harvest_progress_curve_df,
        )
        + geom_line(
            aes(
                x="woy",
                y="harvest_pct_cb",
                color="year_group",
                group="sow_year",
                linetype="year_group",
            ),
            size=1,
        )
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + labs(x="Week Of Year", y="Harvest Cum. Progress (%)", color="", linetype="")
        # + scale_x_continuous(
        #     labels=prod_forecast_year_df["sow_year"].tolist(),
        #     breaks=prod_forecast_year_df["sow_year"].tolist(),
        # )
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + scale_linetype_manual(values={"2023": "solid", "Other Years": "dashed"})
        # + facet_wrap("~region_id")
        + theme_ff()
        + theme(
            # figure_size=[4, 3],
            legend_position="bottom"
        )
    )
    display(p)

    # --------------------------------
    # Harvested Production Curve

    harvest_production_curve_df = (
        harvest_forecast_df.groupby(["sow_year", "woy"])
        .agg(
            harvested_production_cb=("harvested_production_cb", "sum"),
            n=("sow_year", "count"),
        )
        .reset_index()
    )

    harvest_production_curve_df["harvested_production_cb"] = (
        harvest_production_curve_df["harvested_production_cb"] / 1e6
    )

    harvest_production_curve_df["year_group"] = harvest_production_curve_df[
        "sow_year"
    ].apply(lambda year: "2023" if year == 2023 else "Other Years")

    p = (
        ggplot(
            harvest_production_curve_df,
        )
        + geom_line(
            aes(
                x="woy",
                y="harvested_production_cb",
                color="year_group",
                group="sow_year",
                linetype="year_group",
            ),
            size=1,
        )
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + labs(
            x="Week Of Year", y="Harvested Volume (million tn)", color="", linetype=""
        )
        # + scale_x_continuous(
        #     labels=prod_forecast_year_df["sow_year"].tolist(),
        #     breaks=prod_forecast_year_df["sow_year"].tolist(),
        # )
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + scale_linetype_manual(values={"2023": "solid", "Other Years": "dashed"})
        # + facet_wrap("~region_id")
        + theme_ff()
        + theme(
            # figure_size=[4, 3],
            legend_position="bottom",
            text=element_text(size=12),  # Set font size for all text elements
        )
    )
    display(p)


def plot_aspim_development_curve(sim_daily_df):
    sim_daily_df.columns

    apsim_dev_df = sim_daily_df.loc[
        (sim_daily_df["region_id"] == 5101)
        # & (sim_daily_df["sow_year"] == 2020)
        & (sim_daily_df["stage_num"] > 1),
        [
            "region_id",
            "sow_year",
            "date",
            "woy",
            "stage_num",
        ],
    ]

    # Find the last September 1st before each date
    last_sep_1st = pd.to_datetime(apsim_dev_df["sow_year"].astype(str) + "-09-01")
    apsim_dev_df["days_sep1"] = (apsim_dev_df["date"] - last_sep_1st).dt.days

    apsim_dev_df["year_group"] = apsim_dev_df["sow_year"].apply(
        lambda year: "2023" if year == 2023 else "Other Years"
    )

    p = (
        ggplot(
            apsim_dev_df,
        )
        + geom_line(
            aes(
                x="days_sep1",
                y="stage_num",
                color="year_group",
                group="sow_year",
                linetype="year_group",
            ),
            size=1,
        )
        # + geom_smooth(method="lm", color=FBN_GREEN, se=False)
        + labs(x="Date", y="Planting Progress (%)", color="", linetype="")
        # + scale_x_continuous(
        #     labels=prod_forecast_year_df["sow_year"].tolist(),
        #     breaks=prod_forecast_year_df["sow_year"].tolist(),
        # )
        + scale_color_manual(values={"2023": FBN_RED, "Other Years": FBN_YELLOW})
        + scale_linetype_manual(values={"2023": "solid", "Other Years": "dashed"})
        # + facet_wrap("~region_id")
        + theme_ff()
        + theme(
            # figure_size=[4, 3],
            legend_position="bottom"
        )
    )
    display(p)


if __name__ == "__main__":
    yield_obs_df = load_yield_obs()
    ndvi_df = load_ndvi()
    weather_df = load_weather()
    sim_daily_df = load_sim_daily()
    harvest_df = load_harvest()
    planting_df = load_planting()

    preds_harvest_df = prepare_preds_harvest(
        planting_df, weather_df, ndvi_df, harvest_df, sim_daily_df
    )
    harvest_forecast_df = pred_loyo_harvest(preds_harvest_df, show_plot=True)
    yield_forecast_adj_df = pd.read_parquet(
        "./data/export/yield_forecast_adj_df.parquet"
    )
    harvest_forecast_df = fill_some_variables(
        harvest_forecast_df, yield_forecast_adj_df
    )
    explore_obs_harvest(harvest_forecast_df)
    explore_preds_harvest(harvest_forecast_df)
    plot_planting_progress_curve(planting_df, yield_obs_df)
    explore_harvest_curve(harvest_forecast_df)
    plot_year_trends(planting_df, harvest_df, yield_obs_df)
    plot_aspim_development_curve(sim_daily_df)
    regression_model(preds_harvest_df)
