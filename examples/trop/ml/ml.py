import pandas as pd
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from node import gather, logger
from trop import flow, time_list, time_list_small
from trop.reference.location import ngl_us, ngl_station
from trop.resource.cds import era5
from trop.resource.mars import elda
from trop.resource.ngl import *
from trop.resource.ztd_nwm import ztd_nwm
from trop.resource.preprocess import ztd_qc, ztd_merge
from trop import cfg

from trop import PROJ_ROOT
from trop.reference import times


@flow.task(local=True)
def ml(ztd_merge, ztd_gnss_dataset, time_limit=60, feature_cols=["lon", "lat", ], target="res",test_size=0.0,):
    df = ztd_merge
    _, _, location,_,_,_ = ztd_gnss_dataset

    df['res']=df['ztd_nwm']-df['ztd_gnss']

    from sklearn.model_selection import train_test_split
    if test_size!=0:
        train_site, test_site = train_test_split(
            location,
            test_size=test_size,
            random_state=42,
        )
        test_data = df[df['site'].isin(test_site.site.values)].copy()
        train_data = df[df['site'].isin(train_site.site.values)].copy()
    else:
        train_data=df[df['site'].isin(location.site.values)].copy()
        test_data=None
    from autogluon.tabular import TabularPredictor

    hp = get_hyperparameter_config('zeroshot')
    keep_models = ['CAT', 'XGB', 'RF', 'XT', 'GBM', 'KNN']
    hp = {k: v for k, v in hp.items() if k in keep_models}
    logger.info(f"All Avaliable Models: {hp.keys()}")
    # for k in hp:
    #     if not isinstance(hp[k], dict):
    #         hp[k] = {}
    #     if 'ag_args_fit' not in hp[k]:
    #         hp[k]['ag_args_fit'] = {}
    #     hp[k]['ag_args_fit']['max_time_limit_ratio'] = 0.4  # 单模型最多训练 300 秒

    predictor = TabularPredictor(label=target,path=PROJ_ROOT/"storage"/"ml_temp",
                                 eval_metric="root_mean_squared_error",
                                 verbosity=0).fit(
        train_data=train_data[feature_cols + [target]],
        time_limit=time_limit,
        num_bag_folds=10,
        save_bag_folds=True,  # 保留 OOF 预测数据
        excluded_model_types=['NN_TORCH', 'FASTAI'],
        refit_full=False,
        set_best_to_refit_full=False,
        hyperparameters=hp,
    )

    # print("best")
    # predictor = TabularPredictor(label=target,path=PROJ_ROOT/"storage"/"ml_temp", eval_metric="root_mean_squared_error", verbosity=2).fit(
    #     train_data=train_data[feature_cols + [target]],
    #     time_limit=time_limit,
    #     presets="best_quality",
    # )

    train_data['pred'] = predictor.predict_oof(train_data=train_data)  # (Series of OOF predictions) :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
    if test_size!=0:
        test_data['pred'] = predictor.predict(test_data)

    def report(df):
        origin_std = (df['res']).std()
        ml_std = (df['pred'] - df['res']).std()
        print("ml_std: ", ml_std)
        print("origin_std: ", origin_std)
        if ml_std > origin_std:
            logger.warning("ml_std > origin_std")

    report(train_data)
    if test_data is not None:
        report(test_data)

    predictor.save_space()
    return train_data, test_data, predictor

def mls(region,nwm_file_preset,time_preset):
    cfg.region = region
    cfg.times = cfg.times_presets[time_preset]
    cfg.nwm_file = cfg.nwm_file_presets[nwm_file_preset]
    times_list=times().get()
    ml_list=[]
    for time in times_list:
        cfg.time=str(time)
        ml_list.append(ml())
    return gather(ml_list,cache=True)


if __name__ == "__main__":
    # for region in ['us', 'au', 'jp']:
    #     for time_preset in ['month1','month7']:
    #         mls(region=region,nwm_file_preset='era5',time_preset=time_preset).generate()
    #
    # for region in ['us']:
    #     for time_preset in ['month1','month7']:
    #         mls(region=region,nwm_file_preset='eda',time_preset=time_preset).get()
    # cfg.ml.test_size=0.2
    #
    # for region in ['all']:
    #     for time_preset in ['month7','month1','full']:
    #         mls(region=region,
    #             nwm_file_preset='eda_global',
    #             time_preset=time_preset).get()
    #         mls(region=region,
    #             nwm_file_preset='era5',
    #             time_preset=time_preset).get()l


    # cfg.ztd_ngl_dataset=cfg.ztd_ngl_dataset_presets.normal
    # cfg.ztd_ngl_dataset.site_threshold = 0.95
    cfg.ml.test_size=0.0

    for region in ['all']:
        for time_preset in ['full']:
            for dataset in ['eda_global','lwda']:

                ms = mls(region=region, nwm_file_preset=dataset, time_preset=time_preset).get()

    cfg.ztd_nwm_agg.use_member = 0

    for region in ['all']:
        for time_preset in ['full']:
            for dataset in ['eda_global','lwda']:
                ms = mls(region=region, nwm_file_preset=dataset, time_preset=time_preset).get()

    cfg.ztd_nwm_agg.use_member = [1,10]

    for region in ['all']:
        for time_preset in ['full']:
            for dataset in ['eda_global','lwda']:
                ms = mls(region=region, nwm_file_preset=dataset, time_preset=time_preset).get()
    # cfg.ml.test_size=0.2
    #
    # for region in ['all']:
    #     for time_preset in ['month7','month1','full']:
    #         mls(region=region,
    #             nwm_file_preset='eda_global',
    #             time_preset=time_preset).get()l
    #         mls(region=region,
    #             nwm_file_preset='lwda',
    #             time_preset=time_preset).get()
