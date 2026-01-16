import geopandas as gpd, json, pandas as pd, plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from trop.reference.location import _resolve_regions,LAND_DIR,_load_country_boundaries   # 你的陆地数据目录

# ------------------------------------------------------------
# ① 读一次陆地，多次复用
# ------------------------------------------------------------
_LAND_JSON = None
def _get_land_json():
    global _LAND_JSON
    if _LAND_JSON is None:
        shp = next(Path(LAND_DIR).glob("*.shp"))
        gdf = gpd.read_file(shp).to_crs("EPSG:4326")
        js  = json.loads(gdf.to_json())
        for i, f in enumerate(js["features"]):
            f["id"] = i
        _LAND_JSON = js
    return _LAND_JSON


# ------------------------------------------------------------
# ② 包装函数：接口 ≈ px.scatter_geo
# ------------------------------------------------------------
def scatter_geo_land(
    data_frame,
    lat="lat",
    lon="lon",
    projection="mercator",
    zoom="scatter",                      # 'scatter' 或 'world'
    land_color="#E6E6E6",
    land_border="#000",
    land_border_width=0.0,
    width=None,
    height=None,
    **px_kwargs,                       # color / hover_* / color_continuous_scale 等都丢进去
):
    """
    给 px.scatter_geo 加一层灰色陆地底图。
    其他参数完全继承 px.scatter_geo（lat/lon 默认已改为 'lat'/'lon'）。
    """
    # ---------- 1. 先用 PX 把“散点+色条”完整画完 ----------
    px_fig = px.scatter_geo(
        data_frame,
        lat=lat,
        lon=lon,
        **px_kwargs
    )

    # ---------- 2. 再把 PX 图包装进新的 go.Figure ----------
    fig = go.Figure(px_fig)            # 直接拷贝 data + layout（含 coloraxis！）

    # ---------- 3. 加灰色陆地图层 ----------
    land_json = _get_land_json()
    land_ids  = [f["id"] for f in land_json["features"]]

    land_trace = go.Choropleth(
        geojson           = land_json,
        locations         = land_ids,
        z                 = [0]*len(land_ids),
        colorscale        = [[0, land_color], [1, land_color]],
        showscale         = False,
        hoverinfo         = "skip",
        marker_line_color = land_border,
        marker_line_width = land_border_width,
    )

    # --- A. 先把 PX 图整体复制过来（保留 coloraxis 等） ---
    fig = go.Figure(px_fig)

    # --- B. 把陆地面层追加到最后，再把它挪到最前面 ---
    fig.add_trace(land_trace)            # 现在陆地在最上面
    fig.data = (fig.data[-1],) + fig.data[:-1]   # 把最后一条移到最前 ⇒ 底图在最底



    # ---------- 4. 地理视图参数 ----------
    geo_args = dict(
        projection_type = projection,
        visible         = False,       # 关海岸线 / 国家线
    )

    if zoom == "world":
        geo_args["fitbounds"] = "geojson"   # 全世界
    else:                                   # zoom == 'scatter'
        pad = 5
        lon0, lon1 = data_frame[lon].min()-pad, data_frame[lon].max()+pad
        lat0, lat1 = data_frame[lat].min()-pad, data_frame[lat].max()+pad
        geo_args.update(lonaxis_range=[lon0, lon1], lataxis_range=[lat0, lat1])

    fig.update_geos(**geo_args)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0),
                      width=width, height=height)

    return fig

import geopandas as gpd, json, plotly.express as px, plotly.graph_objects as go

us_region = _resolve_regions(_load_country_boundaries())["us"]  # GeoDataFrame
# ------------------------------------------------------------
# ① 把 `us_region` 缓存成 GeoJSON（只做一次）
# ------------------------------------------------------------
import json

_US_JSON = None


def _get_us_json(us_region_gdf):
    """
    us_region_gdf : GeoDataFrame，可能包含多条记录；需为 EPSG:4326
    """
    global _US_JSON
    if _US_JSON is None:
        # 转成 GeoJSON
        js = json.loads(us_region_gdf.to_crs(4326).to_json())

        # 每个 feature 都要有 id（Plotly 匹配时用）
        for f in js["features"]:
            f["id"] = "us"

        _US_JSON = js
    return _US_JSON


import geopandas as gpd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from trop.reference.location import _resolve_regions, LAND_DIR, _load_country_boundaries

# ------------------------------------------------------------
# 1. 计算并缓存所有区域 GeoDataFrame（做一次）
# ------------------------------------------------------------
_COUNTRY_GDF = None
def _get_country_gdf():
    global _COUNTRY_GDF
    if _COUNTRY_GDF is None:
        # 直接调用已有的函数，加载带 ISO_A2/CONTINENT 的国家边界
        _COUNTRY_GDF = _load_country_boundaries()
    return _COUNTRY_GDF


_REGION_GDFS: dict[str, gpd.GeoDataFrame] = None
def _get_region_gdfs():
    global _REGION_GDFS
    if _REGION_GDFS is None:
        countries = _get_country_gdf()
        _REGION_GDFS = _resolve_regions(countries)
    return _REGION_GDFS

# ------------------------------------------------------------
# 2. 按区域名缓存其 GeoJSON
# ------------------------------------------------------------
_REGION_JSONS: dict[str, dict] = {}
def _get_region_json(region_name: str) -> dict:
    """
    返回 region_name 对应的 geojson dict，用于 Plotly Choropleth。
    """
    if region_name not in _REGION_JSONS:
        regs = _get_region_gdfs()
        if region_name not in regs:
            raise KeyError(f"Unknown region: {region_name}")
        gdf = regs[region_name]
        js = json.loads(gdf.to_json())
        # Plotly needs each feature 一个 id，使用 region_name 保证唯一
        for feat in js["features"]:
            feat["id"] = region_name
        _REGION_JSONS[region_name] = js
    return _REGION_JSONS[region_name]

# ------------------------------------------------------------
# 3. 通用化 scatter_geo_region
# ------------------------------------------------------------
def scatter_geo_region(
    data_frame,
    region_name: str = "us",
    lat: str = "lat",
    lon: str = "lon",
    projection: str = "mercator",
    zoom: str = "scatter",        # 'scatter' 或 'region'
    land_color: str = "rgb(0.9,0.9,0.9)",
    land_border: str = "#000",
    land_border_width: float = 0.0,
    width: int = None,
    height: int = None,
    **px_kwargs,                 # 其余 px.scatter_geo 参数
):
    """
    在指定 region_name 的底图上绘制散点。只要该区域在 _compute_regions 定义里，就能直接用。

    Parameters
    ----------
    data_frame : pd.DataFrame
        包含 lat/lon 列的数据。
    region_name : str
        参考 REGION_DEFS 中定义的键，比如 'us', 'eu', 'au'。
    zoom : {'scatter','region'}
        'scatter' → 缩放到散点区；'region' → 整个区域显示。
    px_kwargs : dict
        其他全部透传给 px.scatter_geo。
    """
    # 1) 散点层
    px_fig = px.scatter_geo(
        data_frame,
        lat=lat,
        lon=lon,
        projection=projection,
        **px_kwargs
    )

    # 2) 区域底图
    region_js = _get_region_json(region_name)
    land_trace = go.Choropleth(
        geojson=region_js,
        locations=[region_name],
        z=[0],
        showscale=False,
        colorscale=[[0, land_color], [1, land_color]],
        hoverinfo="skip",
        marker_line_color=land_border,
        marker_line_width=land_border_width,
    )

    # 3) 合并并调整图层顺序
    fig = go.Figure(px_fig)
    fig.add_trace(land_trace)
    # 把底图移到最前（即最底层）
    fig.data = (fig.data[-1],) + fig.data[:-1]

    # 4) Geo 布局
    geo_cfg = dict(projection_type=projection, visible=False)
    if zoom == "region":
        geo_cfg["fitbounds"] = "geojson"
    else:  # zoom == 'scatter'
        pad = 3
        lon0, lon1 = data_frame[lon].min() - pad, data_frame[lon].max() + pad
        lat0, lat1 = data_frame[lat].min() - pad, data_frame[lat].max() + pad
        geo_cfg.update(lonaxis_range=[lon0, lon1], lataxis_range=[lat0, lat1])

    fig.update_geos(**geo_cfg)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0),
                      width=width, height=height)
    return fig

# ------------------------------------------------------------
# 4. 用法示例
# ------------------------------------------------------------
# fig_us = scatter_geo_region(df, region_name='us', color='rms', size='altitude')
# fig_eu = scatter_geo_region(df, region_name='eu', zoom='region',
#                             color='temperature', hover_name='site')

# ------------------------------------------------------------
# ② 主函数：scatter_geo_us
# ------------------------------------------------------------
def scatter_geo_us(
    data_frame,
    us_region_gdf= us_region,
    lat="lat",
    lon="lon",
    projection="albers usa",
    zoom="scatter",      # 'scatter' 或 'us'
    land_color="#E6E6E6",
    land_border="#000",
    land_border_width=0,
    width=None,
    height=None,
    **px_kwargs,         # color / hover_* / size / log_color … 一律透传
):
    """
    在美国大陆轮廓（灰底 / 可调边框）上绘点。

    Parameters
    ----------
    data_frame : pd.DataFrame
        含 lat / lon 的站点数据。
    us_region_gdf : gpd.GeoDataFrame
        单条多边形（美国大陆）。可复用前文的 regs["us"]。
    zoom : {'scatter','us'}
        'scatter' → 包住散点 ± padding；'us' → 完整放大到整张美国。
    **px_kwargs
        其他所有 px.scatter_geo 的参数都可以直接写进来。
    """
    # ---------- 1. 散点层 ----------
    px_fig = px.scatter_geo(
        data_frame,
        lat=lat,
        lon=lon,
        **px_kwargs
    )

    # ---------- 2. 美国底图 ----------
    us_json = _get_us_json(us_region_gdf)
    land_trace = go.Choropleth(
        geojson           = us_json,
        locations         = ["us"],
        z                 = [0],
        showscale         = False,
        colorscale        = [[0, land_color], [1, land_color]],
        hoverinfo         = "skip",
        marker_line_color = land_border,
        marker_line_width = land_border_width,
    )

    # ---------- 3. 合并并调整图层顺序 ----------
    fig = go.Figure(px_fig)         # 复制 px 图
    fig.add_trace(land_trace)       # 先加到底 → 再挪到最底
    fig.data = (fig.data[-1],) + fig.data[:-1]

    # ---------- 4. Geo 视图参数 ----------
    geo_cfg = dict(
        projection_type = projection,
        visible         = False,    # 关掉默认海岸线 / 国家线
    )
    if zoom == "us":
        geo_cfg["fitbounds"] = "geojson"
    else:         # zoom == 'scatter'
        pad = 3
        lon0, lon1 = data_frame[lon].min() - pad, data_frame[lon].max() + pad
        lat0, lat1 = data_frame[lat].min() - pad, data_frame[lat].max() + pad
        geo_cfg.update(lonaxis_range=[lon0, lon1], lataxis_range=[lat0, lat1])

    fig.update_geos(**geo_cfg)
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        width=width,
        height=height,
    )
    fig.update_traces(
        marker=dict(
            size = 3,
            opacity =1,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=0.2,
                color='black'
            ),
            colorscale = 'Spectral',
            # cmin =  df['rms'].min(),
            # color = df['rms'],
            # cmax = df['rms'].max(),
            # colorbar=dict(
            #     title=dict(
            #         text="RMSE (mm)"
            #     )
            # )
        ),
        selector=dict(type="scattergeo")  # 仅作用于散点地理图 trace:contentReference[oaicite:0]{index=0}
    )
    return fig


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trop.plot.plot import _get_us_json


def plot_violin_color_geo2(
        df,
        value,  # 数值列
        us_region_gdf=us_region,  # 美国大陆 Polygon
        lat="lat",
        lon="lon",
        range_x=(0, 30),
        colorscale="Spectral_r",  # 三图统一色标
        projection="albers usa",
        zoom="scatter",
        land_color="#E6E6E6",
        land_border="#000",
        land_border_width=0,
        width=750,  # 总宽度
        height=500,  # 总高度
        top_xy_margin=0.15,  # violin / heat‑bar 左右留白比例
        xy_position="bottom",  # ★ 新增：'top' | 'bottom'
        title=None,
        color_continuous_midpoint=None,
        **px_kwargs
):
    """
    三联图：
    ─ 当 xy_position='top'  →  violin + heat‑bar 在上，地图在下
    ─ 当 xy_position='bottom'(默认) → 地图在上，violin + heat‑bar 在下
    所有色条隐藏；三幅子图共享同一色标。
    """

    # ---------- 0. 行序 & 行高 ----------
    if xy_position not in {"top", "bottom"}:
        raise ValueError("xy_position must be 'top' or 'bottom'.")

    if xy_position == "bottom":
        specs = [[{"type": "geo"}], [{"type": "xy"}], [{"type": "xy"}]]
        row_heights = [1.00, 0.08, 0.02]  # 地图大，XY 小
        geo_row, vio_row, heat_row = 1, 2, 3
    else:  # 'top'
        specs = [[{"type": "xy"}], [{"type": "xy"}], [{"type": "geo"}]]
        row_heights = [0.08, 0.02, 1.00]  # XY 小，地图大
        vio_row, heat_row, geo_row = 1, 2, 3

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=row_heights,
        specs=specs
    )

    # ---------- 1. 地图 ----------
    px_fig = px.scatter_geo(
        df,
        lat=lat,
        lon=lon,
        color=value,
        range_color=range_x,
        projection=projection,
        template="simple_white",
        **px_kwargs
    )

    us_json = _get_us_json(us_region_gdf)
    land_trace = go.Choropleth(
        geojson=us_json,
        locations=["us"],
        z=[0],
        showscale=False,
        colorscale=[[0, land_color], [1, land_color]],
        hoverinfo="skip",
        marker_line_color=land_border,
        marker_line_width=land_border_width,
    )

    fig.add_trace(land_trace, row=geo_row, col=1)
    for tr in px_fig.data:
        fig.add_trace(tr, row=geo_row, col=1)

    geo_cfg = dict(projection_type=projection, visible=False)
    if zoom == "us":
        geo_cfg["fitbounds"] = "geojson"
    else:
        pad = 3
        lon0, lon1 = df[lon].min() - pad, df[lon].max() + pad
        lat0, lat1 = df[lat].min() - pad, df[lat].max() + pad
        geo_cfg.update(lonaxis_range=[lon0, lon1],
                       lataxis_range=[lat0, lat1])
    fig.update_geos(row=geo_row, col=1, **geo_cfg)

    fig.update_traces(
        selector=dict(type="scattergeo"),
        marker=dict(
            size=4,
            opacity=1,
            symbol="circle",
            line=dict(width=0.2, color="black"),
            colorscale=colorscale,
            cmin=range_x[0],
            cmax=range_x[1],
            showscale=False
        )
    )

    # ---------- 2. Violin ----------
    pxv = px.violin(
        df, x=value,
        points=False,
        orientation="h",
        template="simple_white"
    )
    pxv.update_traces(
        side="positive",
        meanline_visible=True,
        bandwidth=0.5,
        spanmode="hard",
        fillcolor="rgb(0.9,0.9,0.9)",
        line_color="black",
        line_width=1,
        showlegend=False
    )
    fig.add_trace(pxv.data[0], row=vio_row, col=1)
    fig.update_xaxes(row=vio_row, col=1,
                     showline=False, ticks='',
                     showticklabels=False, range=range_x)
    fig.update_yaxes(row=vio_row, col=1,
                     ticks='', showline=False)

    # ---------- 3. Heat‑bar ----------
    x_band = np.linspace(*range_x, 512)
    z_band = np.linspace(*range_x, x_band.size)[None, :]

    heat_trace = go.Heatmap(
        x=x_band,
        y=[0],
        z=z_band,
        showscale=False,
        zmin=range_x[0],
        zmax=range_x[1],
        colorscale=colorscale,
        coloraxis="coloraxis",
        color_continuous_midpoint=color_continuous_midpoint,
        name="colorband"
    )
    fig.add_trace(heat_trace, row=heat_row, col=1)

    fig.update_xaxes(row=heat_row, col=1,
                     showline=False, ticks='inside', title=title,
                     showticklabels=True, range=range_x)
    fig.update_yaxes(row=heat_row, col=1,
                     ticks='', showline=False,
                     showticklabels=False)

    # ---------- 4. 横向留白 ----------
    m = float(top_xy_margin)
    fig.update_xaxes(row=vio_row, col=1, domain=[m, 1 - m])
    fig.update_xaxes(row=heat_row, col=1, domain=[m, 1 - m])

    # ---------- 5. Layout ----------
    fig.update_layout(
        width=width,
        height=height,
        template="simple_white",
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorscale=colorscale,
        coloraxis_showscale=False
    )

    return fig

# 示例调用 1：默认（地图上，XY 下）
# fig = plot_violin_color_geo(df, value='rms', range_x=(0, 30))
# fig.show()

# 示例调用 2：提琴图 + 色带上方显示
# fig = plot_violin_color_geo(df, value='rms', range_x=(0, 30), xy_position='top')
# fig.show()
def plot_violin_color(df, value, range_x=(0, 20)):
    pxv = px.violin(
        df,
        x=value,  # 数值
        points="outliers",  # 显示异常点（也可 'all'）
        # box=True,
        orientation="h",
        template="simple_white"
    )
    pxv.update_traces(
        # pointpos =-2,
        side="positive",  # 半边
        meanline_visible=True,  # 均值线
        bandwidth=0.5,
        spanmode="hard",  # → 强制小提琴不超出实际 min/max
        fillcolor="rgb(0.95,0.95,0.95)",  # 浅灰填充
        line_color="black",  # 灰色轮廓
        line_width=1,
        # showlegend=False
    )
    pxv.update_layout(xaxis_showline=False,
                      yaxis_ticks='',
                      yaxis_showline=False,
                      width=400, height=200)

    # pxv.show()
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.1],
        vertical_spacing=0.1
    )
    fig.add_trace(pxv.data[0], row=1, col=1)
    fig.update_xaxes(row=1, col=1, showline=False, ticks='', showticklabels=False)
    fig.update_yaxes(row=1, col=1, ticks='', showline=False)
    fig.update_layout(width=600, height=250, template='simple_white')

    x_band = np.linspace(*range_x, 256)
    z_band = np.linspace(*range_x, x_band.size)[None, :]
    fig.add_trace(
        go.Heatmap(
            x=x_band,
            y=[0],
            z=z_band,
            showscale=False,
            zmin=range_x[0], zmax=range_x[1],
            colorscale="Spectral_r",
            name="colorband"
        ),
        row=2, col=1
    )
    fig.update_xaxes(row=2, col=1, showline=False, ticks='inside', showticklabels=True)
    fig.update_yaxes(row=2, col=1, ticks='', showline=False, showticklabels=False)
    return fig
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trop.plot.plot import _get_us_json


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trop.plot.plot import _get_us_json


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trop.plot.plot import _get_us_json

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trop.plot.plot import _get_us_json


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trop.plot.plot import _get_us_json


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trop.plot.plot import _get_us_json


def plot_violin_color_geo2(
        df,
        value,                        # 数值列
        us_region_gdf=us_region,      # 美国大陆 Polygon
        lat="lat",
        lon="lon",
        range_x=(0, 30),
        colorscale="Spectral_r",      # 三图统一色标
        projection="albers usa",
        zoom="scatter",
        background_color="rgb(0.92,0.92,0.92)",
        land_border="#000",
        land_border_width=0,
        width=750,                    # 总宽度
        height=500,                   # 总高度
        top_xy_margin=0.15,           # violin / heat‑bar 左右留白比例
        xy_position="bottom",         # 'top' | 'bottom' → violin+heat‑bar 位置
        show_stats=True,             # 是否标注 min / mean / max
        title=None,
        template=None,
        **px_kwargs
    ):
    """
    三联图：
    ─ xy_position='bottom' (默认)：地图在上，violin + heat‑bar 在下
    ─ xy_position='top'           ：violin + heat‑bar 在上，地图在下

    若 show_stats=True，则在 violin 图上方标注 min / mean / max（保留 1 位小数）。
    所有色条隐藏；三幅子图共享同一色标。
    """

    # ---------- 0. 行序 & 行高 ----------
    if xy_position not in {"top", "bottom"}:
        raise ValueError("xy_position must be 'top' or 'bottom'.")

    if xy_position == "bottom":
        specs       = [[{"type": "geo"}], [{"type": "xy"}], [{"type": "xy"}]]
        row_heights = [1.00, 0.08, 0.02]
        geo_row, vio_row, heat_row = 1, 2, 3
    else:  # 'top'
        specs       = [[{"type": "xy"}], [{"type": "xy"}], [{"type": "geo"}]]
        row_heights = [0.08, 0.02, 1.00]
        vio_row, heat_row, geo_row = 1, 2, 3

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,          # ★ 更贴合 violin 与色带
        row_heights=row_heights,
        specs=specs
    )

    # ---------- 1. 散点地理分布 ----------
    px_fig = px.scatter_geo(
        df,
        lat=lat,
        lon=lon,
        color=value,
        range_color=range_x,
        projection=projection,
        template=template,
        **px_kwargs
    )

    us_json = _get_us_json(us_region_gdf)
    land_trace = go.Choropleth(
        geojson=us_json,
        locations=["us"],
        z=[0],
        showscale=False,
        colorscale=[[0, background_color], [1, background_color]],
        hoverinfo="skip",
        marker_line_color=land_border,
        marker_line_width=land_border_width,
    )

    fig.add_trace(land_trace, row=geo_row, col=1)
    for tr in px_fig.data:
        fig.add_trace(tr, row=geo_row, col=1)

    geo_cfg = dict(projection_type=projection, visible=False)
    if zoom == "us":
        geo_cfg["fitbounds"] = "geojson"
    else:
        pad = 3
        lon0, lon1 = df[lon].min() - pad, df[lon].max() + pad
        lat0, lat1 = df[lat].min() - pad, df[lat].max() + pad
        geo_cfg.update(lonaxis_range=[lon0, lon1],
                       lataxis_range=[lat0, lat1])
    fig.update_geos(row=geo_row, col=1, **geo_cfg)

    fig.update_traces(
        selector=dict(type="scattergeo"),
        marker=dict(
            size=6,
            opacity=1,
            symbol="circle",
            line=dict(width=0.5, color="black"),
            colorscale=colorscale,
            cmin=range_x[0],
            cmax=range_x[1],
            showscale=False
        )
    )

    # ---------- 2. Violin ----------
    pxv = px.violin(
        df, x=value,
        points=False,
        orientation="h",
        template=template
    )
    pxv.update_traces(
        side="positive",
        meanline_visible=True,
        bandwidth=0.5,
        spanmode="hard",
        fillcolor=background_color,
        line_color="black",
        line_width=0.8,
        showlegend=False
    )
    fig.add_trace(pxv.data[0], row=vio_row, col=1)
    fig.update_xaxes(row=vio_row, col=1,
                     showline=False, ticks='',
                     showticklabels=False, range=range_x)
    fig.update_yaxes(row=vio_row, col=1,
                     ticks='', showline=False)

    # ---------- 3. Heat‑bar ----------
    x_band = np.linspace(*range_x, 512)
    z_band = np.linspace(*range_x, x_band.size)[None, :]

    heat_trace = go.Heatmap(
        x=x_band,
        y=[0],
        z=z_band,
        showscale=False,
        zmin=range_x[0],
        zmax=range_x[1],
        colorscale=colorscale,
        coloraxis="coloraxis",
        name="colorband"
    )
    fig.add_trace(heat_trace, row=heat_row, col=1)

    fig.update_xaxes(row=heat_row, col=1,
                     showline=False, ticks='inside',
                     showticklabels=True, range=range_x,title=title)
    fig.update_yaxes(row=heat_row, col=1,
                     ticks='', showline=False,
                     showticklabels=False)

    # ---------- 4. 横向留白 ----------
    m = float(top_xy_margin)
    fig.update_xaxes(row=vio_row,  col=1, domain=[m, 1 - m])
    fig.update_xaxes(row=heat_row, col=1, domain=[m, 1 - m])

    # ---------- 5. 统计标注（提琴图上方） ----------
    if show_stats:
        vals = df[value].dropna()
        stats = {
            "min":  (round(vals.min(),  1),4),
            "mean": (round(vals.mean(), 1),28),
            "max":  (round(vals.max(),  1),4)
        }
        for key, (x_val,yshift) in stats.items():
            fig.add_annotation(
                x=x_val,
                y=0,
                text=f"{x_val}",
                showarrow=False,
                # font=dict(size=10, color="black"),
                xanchor="center",
                yanchor="bottom",   # ★ 上方
                yshift=yshift,           #   ↑ 抬离曲线
                row=vio_row, col=1
            )

    # ---------- 6. Layout ----------
    fig.update_layout(
        width=width,
        height=height,
        template=template,
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorscale=colorscale,
        coloraxis_showscale=False
    )

    return fig


# ----------- 使用示例 -------------
# fig = plot_violin_color_geo(df, value='rms', range_x=(0, 30),
#                             xy_position='bottom', show_stats=True)
# fig.show()



# ------------------ 使用示例 ------------------
# fig = plot_violin_color_geo(
#         df, value='rms', range_x=(0, 30),
#         xy_position='bottom', show_stats=True,
#         title="RMSE Spatial Distribution"
# )
# fig.show()



# ------------------ 使用示例 ------------------
# fig = plot_violin_color_geo(
#         df, value='rms', range_x=(0, 30),
#         xy_position='bottom',
#         show_stats=True,
#         title="RMSE (mm)"            # 色带下方标题
# )
# fig.show()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path  # 推荐使用 pathlib处理路径


# ==============================================================================
# 1. 辅助函数 (Helper Functions)
#    - 这些函数功能单一，保持不变，但添加了文档字符串。
# ==============================================================================

def mad_std(x: pd.Series) -> float:
    """
    计算基于中位数绝对偏差（MAD）的稳健标准差。
    MAD is a robust measure of the variability of a univariate sample of quantitative data.

    Args:
        x (pd.Series): 输入数据序列。

    Returns:
        float: 稳健标准差。
    """
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def build_bins(series: pd.Series, step: float) -> tuple[list[float], list[str]]:
    """
    根据给定的步长为Series数据创建分箱边界和标签。

    Args:
        series (pd.Series): 用于分箱的数值型Series。
        step (float): 每个分箱的宽度。

    Returns:
        tuple[list[float], list[str]]: 返回分箱边界(edges)和标签(labels)的元组。
    """
    lo, hi = max(0, series.min()), series.max()
    # 确保边界列表至少有两个值
    edges = [lo] + [v for v in np.arange(step, hi, step) if v > lo] + [hi]
    if len(edges) < 2:
        edges = [lo, hi]

    labels = [f"[{edges[i]:.1f}, {edges[i + 1]:.1f})" for i in range(len(edges) - 1)]
    return edges, labels


def assign_bins(df: pd.DataFrame, col: str, edges: list[float], labels: list[str], min_bin_size: int) -> pd.DataFrame:
    """
    为DataFrame中的列分配分箱，并过滤掉样本量过小的分箱。

    Args:
        df (pd.DataFrame): 输入的DataFrame。
        col (str): 需要分箱的列名。
        edges (list[float]): 分箱边界。
        labels (list[str]): 分箱标签。
        min_bin_size (int): 每个分箱所需的最小样本数。

    Returns:
        pd.DataFrame: 带有新分箱列 ('<col>_bin') 且已过滤的DataFrame。
    """
    bin_col_name = f'{col}_bin'

    # 使用 pd.cut 进行分箱，include_lowest=True确保包含最小值
    binned = pd.cut(df[col], bins=edges, labels=labels, right=False, include_lowest=True)

    df = df.copy()
    df[bin_col_name] = pd.Categorical(binned, categories=labels, ordered=True)

    # 按分箱计算样本数量，并过滤
    counts = df.groupby(bin_col_name, observed=True)[col].transform('count')
    return df[counts >= min_bin_size].dropna(subset=[bin_col_name])


# ==============================================================================
# 2. 核心绘图函数 (Core Plotting Function)
#    - 将所有绘图逻辑封装于此，通过参数控制，不依赖全局变量。
# ==============================================================================

def plot_binned_residuals(
        df: pd.DataFrame,
        bin_col: str,
        res_col: str,
        bin_width: float,
        x_axis_title: str,
        color: str = "rgba(50,136,189,1)",
        square_std: bool = False,
        min_bin_size: int = 250,
) -> go.Figure:
    """
    创建一个三行一列的图，分析残差与某个变量（如sigma）分箱后的关系。

    - 第1行: Bias vs. Bins
    - 第2行: STD (or Variance) vs. Bins
    - 第3行: Violin plot of residuals per bin

    Args:
        df (pd.DataFrame): 包含数据的DataFrame。
        bin_col (str): 用于分箱的列名 (e.g., 'ztd_nwm_sigma')。
        res_col (str): 残差列名 (e.g., 'res_cor')。
        bin_width (float): 分箱宽度。
        x_axis_title (str): X轴标题。
        color (str): 绘图颜色。
        square_std (bool, optional): 如果为True，第二行绘制方差(STD^2)而不是标准差. Defaults to False.
        min_bin_size (int, optional): 每个分箱的最小样本数. Defaults to 250.

    Returns:
        go.Figure: 生成的Plotly图形对象。
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes='columns', shared_yaxes='rows',
        row_heights=[1, 1, 2], vertical_spacing=0.04
    )

    # --- 数据准备 ---
    edges, labels = build_bins(df[bin_col], bin_width)
    subset = assign_bins(df, bin_col, edges, labels, min_bin_size)

    if subset.empty:
        print(f"Warning: No data left for column '{bin_col}' after filtering bins with size < {min_bin_size}.")
        # 返回一个空图或带有提示的图
        fig.update_layout(title_text=f"No data to plot for {bin_col}", showlegend=False)
        return fig

    bin_col_name = f'{bin_col}_bin'

    # --- 计算统计量 ---
    stats = subset.groupby(bin_col_name, observed=True)[res_col].agg(['mean', 'std']).reset_index()
    bias = stats.set_index(bin_col_name)['mean']
    std = stats.set_index(bin_col_name)['std']

    y_axis_std_title = "STD (mm)"
    if square_std:
        std = np.power(std, 2)
        y_axis_std_title = "Variance (mm²)"

    # --- 绘图 ---
    # Row 1: Bias
    fig.add_trace(go.Scatter(
        x=bias.index, y=bias.values,
        text=[f"{v:.1f}" for v in bias.values],
        textposition="top center", cliponaxis=False,
        mode='lines+markers+text', marker_color=color, line_color=color,
        name='Bias'
    ), row=1, col=1)

    # Row 2: STD / Variance
    fig.add_trace(go.Scatter(
        x=std.index, y=std.values,
        text=[f"{v:.1f}" for v in std.values],
        textposition="top center", cliponaxis=False,
        mode='lines+markers+text', marker_color=color, line_color=color,
        name='STD'
    ), row=2, col=1)

    # Row 3: Violin Plots
    for label in labels:
        y_data = subset.loc[subset[bin_col_name] == label, res_col]
        if y_data.empty:
            continue

        fig.add_trace(go.Violin(
            x=y_data.map(lambda x: label),  # 更稳健的X轴赋值方式
            y=y_data,
            scalemode='width', scalegroup='all_violins', width=1.2,
            fillcolor=color.replace("1)", "0.5)"),
            line_color=color,
            box_visible=False, points="all", meanline_visible=True,
            side='positive', spanmode='hard',
            jitter=0.05, pointpos=-0.2,
            marker={'size': 1.5, 'opacity': 0.8},
            line={'width': 1.5},
            showlegend=False
        ), row=3, col=1)

    # --- 布局与样式 ---
    fig.update_layout(
        width=750, height=1000, template='simple_white',
        margin=dict(l=60, r=50, b=100, t=50),
        showlegend=False,
        font=dict(family='Arial, sans-serif', color="black", size=14)
    )
    fig.update_xaxes(showline=False, ticks="", showgrid=True)  # , gridwidth=1, gridcolor='LightGray')
    fig.update_xaxes(title_text=x_axis_title, row=3, col=1, showline=True, linecolor='black')

    fig.update_yaxes(title_text="Bias (mm)", row=1, col=1, range=[-2.5, 2.5], showline=True, linecolor='black')
    fig.update_yaxes(title_text=y_axis_std_title, row=2, col=1, showline=True, linecolor='black')

    range_res = np.quantile(np.abs(df[res_col]), 0.999)
    fig.update_yaxes(title_text="Residual (mm)", row=3, col=1, range=[-range_res, range_res], showline=True,
                     linecolor='black')
    fig.update_xaxes(showline=False)
    fig.update_xaxes(showline=True,row=3,ticks='outside')
    return fig

def plot_binned_residuals2(
        df: pd.DataFrame,
        bin_col: str,
        res_col: str,
        bin_width: float,
        x_axis_title: str,
        color: str = "rgba(50,136,189,1)",
        square_std: bool = False,
        min_bin_size: int = 250,
        fit_std_line: bool = True,
) -> go.Figure:
    """
    创建一个三行一列的图，分析残差与某个变量（如sigma）分箱后的关系。

    - 第1行: Bias vs. Bins
    - 第2行: STD (or Variance) vs. Bins
    - 第3行: Violin plot of residuals per bin

    Args:
        df (pd.DataFrame): 包含数据的DataFrame。
        bin_col (str): 用于分箱的列名 (e.g., 'ztd_nwm_sigma')。
        res_col (str): 残差列名 (e.g., 'res_cor')。
        bin_width (float): 分箱宽度。
        x_axis_title (str): X轴标题。
        color (str): 绘图颜色。
        square_std (bool, optional): 如果为True，第二行绘制方差(STD^2)而不是标准差. Defaults to False.
        min_bin_size (int, optional): 每个分箱的最小样本数. Defaults to 250.
        fit_std_line (bool, optional): 如果为True，对STD图拟合线性模型并绘制拟合直线和方程. Defaults to False.

    Returns:
        go.Figure: 生成的Plotly图形对象。
    """
    import re

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes='columns', shared_yaxes='rows',
        row_heights=[1, 1, 2], vertical_spacing=0.04
    )

    # --- 数据准备 ---
    edges, labels = build_bins(df[bin_col], bin_width)
    subset = assign_bins(df, bin_col, edges, labels, min_bin_size)

    if subset.empty:
        print(f"Warning: No data left for column '{bin_col}' after filtering bins with size < {min_bin_size}.")
        fig.update_layout(title_text=f"No data to plot for {bin_col}", showlegend=False)
        return fig

    bin_col_name = f'{bin_col}_bin'

    # --- 计算统计量 ---
    stats = subset.groupby(bin_col_name, observed=True)[res_col].agg(['mean', 'std']).reset_index()
    bias = stats.set_index(bin_col_name)['mean']
    std = stats.set_index(bin_col_name)['std']

    y_axis_std_title = "STD (mm)"
    if square_std:
        std = np.power(std, 2)
        y_axis_std_title = "Variance (mm²)"

    # --- 绘图 ---
    # Row 1: Bias
    fig.add_trace(go.Scatter(
        x=bias.index.astype(str), y=bias.values,
        text=[f"{v:.1f}" for v in bias.values],
        textposition="top center", cliponaxis=False,
        mode='lines+markers+text', marker_color=color, line_color=color,
        name='Bias'
    ), row=1, col=1)

    # Row 2: STD / Variance
    fig.add_trace(go.Scatter(
        x=std.index.astype(str), y=std.values,
        text=[f"{v:.1f}" for v in std.values],
        textposition="top center", cliponaxis=False,
        mode='lines+markers+text', marker_color=color, line_color=color,
        name='STD'
    ), row=2, col=1)

    # 如果请求拟合直线，则对STD进行线性拟合并绘制
    if fit_std_line:
        def interval_mid(obj):
            """返回区间或区间字符串的中点"""
            if isinstance(obj, pd.Interval):
                return (obj.left + obj.right) / 2
            s = str(obj)
            m = re.match(r"\s*[\[\(]?\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*[)\]]?", s)
            if m:
                return (float(m.group(1)) + float(m.group(2))) / 2
            try:
                return float(s)
            except ValueError:
                return np.nan

        x_centers = stats[bin_col_name].map(interval_mid).values.astype(float)
        y_vals = std.values.astype(float)

        # 过滤掉无法解析为数值的label
        mask = np.isfinite(x_centers) & np.isfinite(y_vals)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(x_centers[mask], y_vals[mask], 1)
            y_line = slope * x_centers + intercept
            fig.add_trace(go.Scatter(
                x=std.index.astype(str), y=y_line,
                mode='lines', line=dict(color='black', dash='dash'),
                name='Fit'
            ), row=2, col=1)
            # 在图中添加方程文本
            fig.add_annotation(
                x=std.index.astype(str)[-1], y=y_line[-1],
                text=f"y = {slope:.1f}x + {intercept:.1f}",
                xanchor="left", yanchor="top",  # ★ 上方
                showarrow=False, row=2, col=1,
                # font=dict(size=12, family='Arial, sans-serif')
            )
        else:
            print("Warning: Not enough valid points to fit STD line.")

    # Row 3: Violin Plots
    for label in labels:
        y_data = subset.loc[subset[bin_col_name] == label, res_col]
        if y_data.empty:
            continue

        fig.add_trace(go.Violin(
            x=y_data.map(lambda _: str(label)),
            y=y_data,
            scalemode='width', scalegroup='all_violins', width=1.2,
            fillcolor=color.replace("1)", "0.5)"),
            line_color=color,
            box_visible=False, points="all", meanline_visible=True,
            side='positive', spanmode='hard',
            jitter=0.05, pointpos=-0.2,
            marker={'size': 1.5, 'opacity': 0.8},
            line={'width': 1},
            showlegend=False
        ), row=3, col=1)

    # --- 布局与样式 ---
    fig.update_layout(
        width=750, height=1000, template='simple_white',
        margin=dict(l=60, r=50, b=100, t=50),
        showlegend=False,
        font=dict(family='Arial, sans-serif', color='black', size=14)
    )
    fig.update_xaxes(showline=False, ticks='', showgrid=True)
    fig.update_xaxes(title_text=x_axis_title, row=3, col=1, showline=True, linecolor='black')

    fig.update_yaxes(title_text='Bias (mm)', row=1, col=1, range=[-2.5, 2.5], showline=True, linecolor='black')
    fig.update_yaxes(title_text=y_axis_std_title, row=2, col=1, showline=True, linecolor='black')

    range_res = np.quantile(np.abs(df[res_col]), 0.999)
    fig.update_yaxes(title_text='Residual (mm)', row=3, col=1, range=[-range_res, range_res], showline=True,
                     linecolor='black')
    fig.update_xaxes(showline=False)

    return fig
def plot_binned_residuals6(
        df: pd.DataFrame,
        bin_col: str,
        res_col: str,
        bin_width: float,
        x_axis_title: str,
        color: str = "rgba(50,136,189,1)",
        square_std: bool = False,
        min_bin_size: int = 250,
        fit_sqrt_sigma2_line: bool = False,
) -> go.Figure:
    """Plot residual statistics versus a binned predictor.

    Rows
    ----
    1. **Bias**  – mean residual per bin
    2. **STD** (or **Variance** if ``square_std=True``)
       Optional overlay of the non‑linear model
       :math:`\mathrm{STD} = \sqrt{k\,\sigma^2 + B}` when
       ``fit_sqrt_sigma2_line=True``.
    3. Distribution of residuals per bin (violins)

    Parameters
    ----------
    df, bin_col, res_col, bin_width
        See original signature.
    square_std : bool, default *False*
        Show residual variance instead of STD in row‑2 **unless** the
        non‑linear fit is requested.  When ``fit_sqrt_sigma2_line=True`` the
        Y‑axis is forced to STD to match the fitted curve.
    min_bin_size : int, default 250
        Minimum observations per bin; smaller bins are discarded.
    fit_sqrt_sigma2_line : bool, default *False*
        If *True* perform OLS on
        :math:`y^2 = k\,\sigma^2 + B` (where :math:`y=`STD) and overlay the
        resulting curve :math:`STD=\sqrt{k\,\sigma^2+B}` with its equation.
    """
    import re

    # ---- create sub‑plots ----
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes="columns",
        shared_yaxes="rows",
        row_heights=[1, 1, 2],
        vertical_spacing=0.04,
    )

    # ---- binning ----
    edges, labels = build_bins(df[bin_col], bin_width)
    subset = assign_bins(df, bin_col, edges, labels, min_bin_size)
    if subset.empty:
        fig.update_layout(title_text=f"No data after filtering (<{min_bin_size} per bin)")
        return fig

    bin_col_name = f"{bin_col}_bin"

    # ---- per‑bin stats ----
    stats = (
        subset.groupby(bin_col_name, observed=True)[res_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    bias = stats.set_index(bin_col_name)["mean"]
    std_raw = stats.set_index(bin_col_name)["std"]  # always STD here

    # ---- choose row‑2 series ----
    if fit_sqrt_sigma2_line:
        y_series = std_raw              # keep STD for model match
        y_axis_title = "STD (mm)"
    else:
        if square_std:
            y_series = std_raw.pow(2)
            y_axis_title = "Variance (mm²)"
        else:
            y_series = std_raw
            y_axis_title = "STD (mm)"

    # ---- Row‑1 Bias ----
    fig.add_trace(
        go.Scatter(
            x=bias.index.astype(str),
            y=bias.values,
            mode="lines+markers+text",
            marker_color=color,
            line_color=color,
            text=[f"{v:.1f}" for v in bias.values],
            textposition="top center",
            name="Bias",
        ),
        row=1,
        col=1,
    )

    # ---- Row‑2 STD / Variance ----
    fig.add_trace(
        go.Scatter(
            x=y_series.index.astype(str),
            y=y_series.values,
            mode="lines+markers+text",
            marker_color=color,
            line_color=color,
            text=[f"{v:.1f}" for v in y_series.values],
            textposition="top center",
            name="STD" if y_axis_title.startswith("STD") else "Variance",
        ),
        row=2,
        col=1,
    )

    # ---- helper: label → bin centre σ ----
    def _mid(lbl):
        if isinstance(lbl, pd.Interval):
            return 0.5 * (lbl.left + lbl.right)
        s = str(lbl)
        m = re.match(r"[\[\(]?\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*[)\]]?", s)
        if m:
            return 0.5 * (float(m.group(1)) + float(m.group(2)))
        try:
            return float(s)
        except ValueError:
            return np.nan

    # ---- non‑linear fit STD = sqrt(k σ² + B) ----
    if fit_sqrt_sigma2_line:
        sigma_centres = stats[bin_col_name].map(_mid).astype(float).values
        std_vals = std_raw.values.astype(float)
        mask = np.isfinite(sigma_centres) & np.isfinite(std_vals)
        if mask.sum() >= 2:
            σ = sigma_centres[mask]
            y = std_vals[mask]
            # Fit y² = k σ² + B (linear regression)
            k, B = np.polyfit(σ ** 2, y ** 2, 1)
            y_fit = np.sqrt(k * (sigma_centres ** 2) + B)
            eqn = rf"STD = √({k:.1f}·σ² + {B:.1f})"

            fig.add_trace(
                go.Scatter(
                    x=y_series.index.astype(str),
                    y=y_fit,
                    mode="lines",
                    line=dict(color="black", dash="dash"),
                    name="Fit",
                ),
                row=2,
                col=1,
            )
            fig.add_annotation(
                x=y_series.index.astype(str)[-1],
                y=y_fit[-1],
                text=eqn,
                showarrow=False,
                font=dict(size=12),
                row=2,
                col=1,
            )
        else:
            print("[plot_binned_residuals] Warning: insufficient points for sqrt‑sigma² fit.")

    # ---- Row‑3 Violin distributions ----
    for lbl in labels:
        y_data = subset.loc[subset[bin_col_name] == lbl, res_col]
        if y_data.empty:
            continue
        fig.add_trace(
            go.Violin(
                x=[str(lbl)] * len(y_data),
                y=y_data,
                width=1.2,
                scalemode="width",
                scalegroup="all",
                fillcolor=color.replace("1)", "0.5)"),
                line_color=color,
                box_visible=False,
                points="all",
                meanline_visible=True,
                side="positive",
                jitter=0.05,
                pointpos=-0.2,
                marker=dict(size=1.5, opacity=0.8),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # ---- layout ----
    fig.update_layout(
        width=750,
        height=1000,
        template="simple_white",
        margin=dict(l=60, r=50, t=50, b=100),
        font=dict(family="Arial, sans-serif", size=14),
        showlegend=False,
    )

    fig.update_xaxes(showline=False, ticks="", showgrid=True)
    fig.update_xaxes(title_text=x_axis_title, row=3, col=1, showline=True, linecolor="black")

    fig.update_yaxes(title_text="Bias (mm)", row=1, col=1, range=[-2.5, 2.5], showline=True, linecolor="black")
    fig.update_yaxes(title_text=y_axis_title, row=2, col=1, showline=True, linecolor="black")

    res_range = np.quantile(np.abs(df[res_col]), 0.999)
    fig.update_yaxes(title_text="Residual (mm)", row=3, col=1, range=[-res_range, res_range], showline=True, linecolor="black")

    return fig

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trop.plot.plot import _get_us_json  # 你已有的函数

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trop.plot.plot import _get_us_json

def plot_violin_color_geo(
        df,
        value,
        region_gdf=us_region,      # 美国大陆 Polygon,
        lat="lat", lon="lon",
        range_x=(0, 30),
        colorscale="Spectral_r",
        projection="albers usa",
        zoom="scatter",
        background_color="rgb(0.92,0.92,0.92)",
        land_border="#000", land_border_width=0,
        width=750, height=500,
        top_xy_margin=0.15,
        xy_position="bottom",
        show_stats=True,
        title=None,
        template=None,
        midpoint=None,              # ★ 新增：色标中点；None=不指定
        **px_kwargs
    ):
    # ---- 行序与行高（保持一致） ----
    if xy_position not in {"top","bottom"}:
        raise ValueError("xy_position must be 'top' or 'bottom'.")
    if xy_position == "bottom":
        specs = [[{"type":"geo"}],[{"type":"xy"}],[{"type":"xy"}]]
        row_heights = [1.00, 0.08, 0.02]
        geo_row, vio_row, heat_row = 1, 2, 3
    else:
        specs = [[{"type":"xy"}],[{"type":"xy"}],[{"type":"geo"}]]
        row_heights = [0.08, 0.02, 1.00]
        vio_row, heat_row, geo_row = 1, 2, 3

    lo, hi = map(float, range_x)
    if lo == hi: hi = lo + 1e-9
    if lo > hi:  lo, hi = hi, lo
    range_x = (lo, hi)

    # 若用户没显式给 PX 的 midpoint，就帮他同步
    if midpoint is not None and "color_continuous_midpoint" not in px_kwargs:
        px_kwargs["color_continuous_midpoint"] = midpoint

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.01, row_heights=row_heights, specs=specs
    )

    # ---- 1) 地图散点（PX） + 灰底 ----
    px_fig = px.scatter_geo(
        df, lat=lat, lon=lon, color=value,
        range_color=range_x,
        color_continuous_scale=colorscale,   # ★ 与 coloraxis 保持一致
        projection=projection, template=template, **px_kwargs
    )

    region_gdf= us_region.to_crs(4326).dissolve(by="region", as_index=False)
    us_json = _get_us_json(region_gdf)
    land_trace = go.Choropleth(
        geojson=us_json,
        locations=["us"],
        featureidkey="id",
        z=[0],
        showscale=False,
        colorscale=[[0, background_color], [1, background_color]],
        hoverinfo="skip",
        marker_line_color=land_border,
        marker_line_width=land_border_width
    )

    fig.add_trace(land_trace, row=geo_row, col=1)
    for tr in px_fig.data:
        fig.add_trace(tr, row=geo_row, col=1)

    # 统一地理视图
    geo_cfg = dict(projection_type=projection, visible=False)
    if zoom == "us":
        geo_cfg["fitbounds"] = "geojson"
    else:
        pad = 3
        lon0, lon1 = df[lon].min()-pad, df[lon].max()+pad
        lat0, lat1 = df[lat].min()-pad, df[lat].max()+pad
        geo_cfg.update(lonaxis_range=[lon0,lon1], lataxis_range=[lat0,lat1])
    fig.update_geos(row=geo_row, col=1, **geo_cfg)

    # ★★ 关键：用 layout.coloraxis 统一配色（散点都引用它）
    coloraxis = dict(colorscale=colorscale, cmin=lo, cmax=hi, showscale=False)
    if midpoint is not None:
        coloraxis["cmid"] = float(midpoint)
    fig.update_layout(coloraxis=coloraxis)

    # 只设置散点的几何样式（不要再给 colorscale/cmin/cmax）
    fig.update_traces(
        selector=dict(type="scattergeo"),
        marker=dict(size=5, opacity=1, symbol="circle",
                    line=dict(width=0.5, color="black"))
    )

    # ---- 2) Violin ----
    pxv = px.violin(df, x=value, points=False, orientation="h", template=template)
    pxv.update_traces(side="positive", meanline_visible=True, bandwidth=0.5,
                      spanmode="hard", fillcolor=background_color,
                      line_color="black", line_width=0.8, showlegend=False)
    fig.add_trace(pxv.data[0], row=vio_row, col=1)
    fig.update_xaxes(row=vio_row, col=1, showline=False, ticks='',
                     showticklabels=False, range=range_x)
    fig.update_yaxes(row=vio_row, col=1, ticks='', showline=False)

    # ---- 3) Heat-bar（与 coloraxis 同步；在这里使用 zmid）----
    x_band = np.linspace(*range_x, 512)
    z_band = np.linspace(*range_x, x_band.size)[None, :]

    heat_kwargs = dict(
        x=x_band, y=[0], z=z_band, showscale=False,
        zmin=lo, zmax=hi, colorscale=colorscale, name="colorband"
    )
    if midpoint is not None:
        heat_kwargs["zmid"] = float(midpoint)
    heat_trace = go.Heatmap(**heat_kwargs)
    fig.add_trace(heat_trace, row=heat_row, col=1)

    fig.update_xaxes(row=heat_row, col=1, showline=False, ticks='inside',
                     showticklabels=True, range=range_x, title=title)
    fig.update_yaxes(row=heat_row, col=1, ticks='', showline=False, showticklabels=False)

    # ---- 4) 横向留白 ----
    m = float(top_xy_margin)
    fig.update_xaxes(row=vio_row, col=1,  domain=[m, 1-m])
    fig.update_xaxes(row=heat_row, col=1, domain=[m, 1-m])

    # ---- 5) 统计标注（与原一致）----
    if show_stats:
        vals = df[value].dropna()
        stats = {"min":(round(vals.min(),1),4),
                 "mean":(round(vals.mean(),1),28),
                 "max":(round(vals.max(),1),4)}
        for _, (x_val, yshift) in stats.items():
            fig.add_annotation(x=x_val, y=0, text=f"{x_val}",
                               showarrow=False, xanchor="center",
                               yanchor="bottom", yshift=yshift,
                               row=vio_row, col=1)

    # ---- 6) Layout ----
    fig.update_layout(width=width, height=height, template=template,
                      margin=dict(l=0, r=0, t=50, b=0),
                      coloraxis_showscale=False)   # 不显示独立色条
    return fig


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_residuals_with_sigma(df_site, return_stats=False):
    """
    绘制 GNSS 残差散点图（含 ±3σ 区域与 y=0 基准线），
    同时计算 residual 落在 95% 区间(±1.96σ)的比例。
    不在图上做任何标注；若 return_stats=True，返回统计字典。

    参数
    ----
    df_site : pd.DataFrame
        需包含列 ["time", "res_cor", "res", "ztd_nwm_sigma"]
    return_stats : bool, default False
        True -> 返回 (fig, stats_dict)；False -> 仅返回 fig
    """
    # --- 数据准备 ---
    df_site = df_site.copy()
    df_site["time"] = pd.to_datetime(df_site["time"], errors="coerce")

    edge = df_site.loc[df_site["ztd_nwm_sigma"].notna()].sort_values("time")
    edge = edge.loc[edge["time"].notna()]

    t = edge["time"].to_numpy()
    sig = edge["ztd_nwm_sigma"].to_numpy()

    # --- 构造 ±3σ 区域（用于可视化）---
    x_band = np.concatenate([t, t[::-1]])
    y_band = np.concatenate([3 * sig, -3 * sig[::-1]])

    # --- 绘制散点 ---
    fig = px.scatter(
        df_site,
        x="time",
        y=["res_cor", "res"],
        labels={"variable": "Type", "value": "Residual (mm)"},
        template="simple_white",
    ).update_traces(marker=dict(size=4))

    # 图例采用固定命名（不含统计数字）
    fig.for_each_trace(
        lambda tr: tr.update(
            name="Residual (BC)" if tr.name == "res_cor"
            else "Residual (Raw)" if tr.name == "res"
            else tr.name
        )
    )

    # --- 添加 ±3σ 填充带 ---
    fig.add_trace(go.Scatter(
        x=x_band,
        y=y_band,
        fill='toself',
        fillcolor='rgba(0, 0, 0, 0.2)',
        line=dict(width=0, color='rgba(0,0,0,0)'),
        name='±3 RMS spread',
        hoverinfo='skip',
        line_shape='hv',
        showlegend=True
    ))

    # --- 添加 y=0 参考线 ---
    fig.update_yaxes(zeroline=True)

    # --- 布局 ---
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Residual (mm)",
        width=750,
        height=500,
        template="simple_white",
        legend=dict(
            yanchor="top", y=0.98,
            xanchor="left", x=0.02,
            title=None
        )
    )

    # === 统计：95% 区间（±1.96σ）命中率（仅返回，不标注在图上）===
    def _pct_in_band(df, col, sig_col="ztd_nwm_sigma", k=1.96):
        s = df[[col, sig_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if s.empty:
            return np.nan, 0
        within = (s[col].abs() <= k * s[sig_col])
        return float(within.mean() * 100.0), int(within.size)

    stats = {
        "pct_in_95_res_cor": None,
        "n_res_cor": 0,
        "pct_in_95_res": None,
        "n_res": 0,
    }
    pct_cor, n_cor = _pct_in_band(df_site, "res_cor")
    pct_raw, n_raw = _pct_in_band(df_site, "res")
    stats.update({
        "pct_in_95_res_cor": pct_cor,
        "n_res_cor": n_cor,
        "pct_in_95_res": pct_raw,
        "n_res": n_raw,
    })

    return (fig, stats) if return_stats else fig




import numpy as np
import pandas as pd
import plotly.express as px

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def _hex_to_rgba(hex_color, alpha=0.2):
    """#RRGGBB -> rgba(r,g,b,alpha)"""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def _hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def plot_rolling_rmse2(df_site, N, win="30D", min_pts=60,fill=False):
    # --- 数据准备 ---
    df = df_site.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    g = (df.dropna(subset=["time", "res_cor", "res", "ztd_nwm_sigma", "ztd_gnss_sigma"])
           .sort_values("time").set_index("time"))

    # --- 滚动参数 & 基础统计 ---
    roll_kw = dict(window=win, min_periods=min_pts, center=True, closed="both")

    sq_ml = (g["res_cor"] ** 2)
    n_ml, m_ml, v_ml = sq_ml.rolling(**roll_kw).count(), sq_ml.rolling(**roll_kw).mean(), sq_ml.rolling(**roll_kw).var(ddof=1)

    sq_or = (g["res"] ** 2)
    n_or, m_or, v_or = sq_or.rolling(**roll_kw).count(), sq_or.rolling(**roll_kw).mean(), sq_or.rolling(**roll_kw).var(ddof=1)

    sq_sp = (g["ztd_nwm_sigma"] ** 2)
    n_sp0, m_sp0, v_sp0 = sq_sp.rolling(**roll_kw).count(), sq_sp.rolling(**roll_kw).mean(), sq_sp.rolling(**roll_kw).var(ddof=1)

    c = (N + 1) / N
    n_sp, m_sp, v_sp = n_sp0, c * m_sp0, (c ** 2) * v_sp0

    # --- 指标 ---
    roll_df = pd.DataFrame(index=g.index)
    roll_df["RMS error (BC)"]  = np.sqrt(m_ml)
    roll_df["RMS error (Raw)"] = np.sqrt(m_or)
    roll_df["RMS spread"]      = np.sqrt(m_sp)
    roll_df["GNSS sigma"]      = np.sqrt(c * (g["ztd_gnss_sigma"] ** 2).rolling(**roll_kw).mean())
    roll_df = roll_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    # --- CI（Delta Method） ---
    z = 1.96
    def rmse_ci_from_m_v_n(m, v, n):
        se = np.sqrt(v / n / (4.0 * m))
        lower = (np.sqrt(m) - z * se).clip(lower=0)
        upper = np.sqrt(m) + z * se
        mask = (m > 0) & (n >= 2) & (v >= 0)
        return lower.where(mask), upper.where(mask)

    ci = {
        "RMS error (BC)":  rmse_ci_from_m_v_n(m_ml, v_ml, n_ml),
        "RMS error (Raw)": rmse_ci_from_m_v_n(m_or, v_or, n_or),
        "RMS spread":      rmse_ci_from_m_v_n(m_sp, v_sp, n_sp),
    }

    # --- 长表（注意顺序：BC → raw → spread） ---
    order = ["RMS error (BC)", "RMS error (Raw)", "RMS spread"]
    long_df = (roll_df.reset_index(names="time")
               .melt(id_vars="time", value_vars=order, var_name="metric", value_name="value")
               .dropna(subset=["value"]))

    # --- 相关性 ---
    def safe_corr(a, b):
        v = roll_df.dropna(subset=[a, b])
        return v[a].corr(v[b]) if len(v) else np.nan
    corr_ml = safe_corr("RMS error (BC)", "RMS spread")
    corr_or = safe_corr("RMS error (Raw)", "RMS spread")

    # --- 主曲线（图例顺序同上） ---
    fig = px.line(
        long_df, x="time", y="value", color="metric",
        template="simple_white",
        category_orders={"metric": order},
    )
    # 点击图例时让线和带一起隐藏/显示
    for tr in fig.data:
        if tr.name in order:
            tr.legendgroup = tr.name

    # --- 添加 95% CI 区间带：可见但不进图例 ---
    if fill:
        colorway = list(fig.layout.colorway or fig.layout.template.layout.colorway)
        x = roll_df.index
        for i, name in enumerate(order):
            lower, upper = ci[name]
            base_color = colorway[i % len(colorway)]
            fill_color = _hex_to_rgba(base_color, alpha=0.20)

            # 上边界（不进图例、无 hover）
            fig.add_trace(go.Scatter(
                x=x, y=upper.reindex(x), mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
                name=f"{name} upper", legendgroup=name
            ))
            # 下边界（填充到上一个 trace，不进图例）
            fig.add_trace(go.Scatter(
                x=x, y=lower.reindex(x), mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=fill_color,
                showlegend=False, hoverinfo="skip",
                name=f"{name} 95% CI", legendgroup=name
            ))

    # --- 布局与注记 ---
    fig.update_layout(
        width=780, height=520,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                    title=None, traceorder="normal"),
        yaxis_title="Sliding Windows RMS spread/error (mm)",
        xaxis_title="Time",
        title=None
    )
    fig.add_annotation(
        text=f"Corr(RMS spread, RMS error (BC)) = {corr_ml:.3f}",
        xref="paper", yref="paper", x=0.02, y=0.97, showarrow=False, font=dict(size=14)
    )
    fig.add_annotation(
        text=f"Corr(RMS spread, RMS error (Raw)) = {corr_or:.3f}",
        xref="paper", yref="paper", x=0.02, y=0.90, showarrow=False, font=dict(size=14)
    )

    return fig


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors


def _hex_to_rgba(hex_code, alpha=0.2):
    """辅助函数：将Hex颜色转换为RGBA字符串"""
    rgb = mcolors.hex2color(hex_code)
    return f"rgba({rgb[0] * 255},{rgb[1] * 255},{rgb[2] * 255},{alpha})"


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors


def _hex_to_rgba(hex_code, alpha=0.2):
    rgb = mcolors.hex2color(hex_code)
    return f"rgba({rgb[0] * 255},{rgb[1] * 255},{rgb[2] * 255},{alpha})"


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors


def _hex_to_rgba(hex_code, alpha=0.2):
    """辅助函数：将Hex颜色转换为RGBA字符串"""
    rgb = mcolors.hex2color(hex_code)
    return f"rgba({rgb[0] * 255},{rgb[1] * 255},{rgb[2] * 255},{alpha})"


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors


def _hex_to_rgba(hex_code, alpha=1.0):
    """辅助函数：将Hex颜色转换为RGBA字符串"""
    rgb = mcolors.hex2color(hex_code)
    return f"rgba({rgb[0] * 255},{rgb[1] * 255},{rgb[2] * 255},{alpha})"


def plot_rolling_rmse_custom_style(data_pairs, win="30D", min_pts=60, fill=False):
    """
    data_pairs: list of tuples, e.g. [(df1, n1), (df2, n2)]
    样式要求：
    1. 全实线。
    2. 相同变量同色，不同N透明度不同（第一个深，第二个浅）。
    3. 图例排序：Raw(N1)->Raw(N2) -> BC(N1)->BC(N2) -> Spread(N1)->Spread(N2)。
    """

    # --- 1. 基础参数与函数 ---
    roll_kw = dict(window=win, min_periods=min_pts, center=True, closed="both")
    z = 1.96

    def rmse_ci(m, v, n):
        se = np.sqrt(v / n / (4.0 * m))
        lower = (np.sqrt(m) - z * se).clip(lower=0)
        upper = np.sqrt(m) + z * se
        mask = (m > 0) & (n >= 2) & (v >= 0)
        return lower.where(mask), upper.where(mask)

    # --- 2. 数据计算与存储 ---
    # 我们将数据存入一个字典结构，方便后续按特定顺序提取
    # 结构: stored_data[dataset_index] = { 'raw': series, 'bc': series, ... }
    stored_data = []

    # 用于对齐时间轴的总表
    time_index_union = pd.Index([])

    for i, (df_site, N) in enumerate(data_pairs):
        # 清洗
        cols_req = ["time", "res_cor", "res", "ztd_nwm_sigma", "ztd_gnss_sigma"]
        df_clean = df_site.copy()
        df_clean["time"] = pd.to_datetime(df_clean["time"], errors="coerce")
        g = (df_clean.dropna(subset=cols_req)
             .sort_values("time").set_index("time"))

        # 更新总时间索引
        if i == 0:
            time_index_union = g.index
        else:
            time_index_union = time_index_union.union(g.index)

        # A. Spread
        sq_sp = (g["ztd_nwm_sigma"] ** 2)
        n_sp0, m_sp0, v_sp0 = sq_sp.rolling(**roll_kw).count(), sq_sp.rolling(**roll_kw).mean(), sq_sp.rolling(
            **roll_kw).var(ddof=1)
        c = (N + 1) / N
        n_sp, m_sp, v_sp = n_sp0, c * m_sp0, (c ** 2) * v_sp0

        # B. BC Error
        sq_ml = (g["res_cor"] ** 2)
        n_ml, m_ml, v_ml = sq_ml.rolling(**roll_kw).count(), sq_ml.rolling(**roll_kw).mean(), sq_ml.rolling(
            **roll_kw).var(ddof=1)

        # C. Raw Error
        sq_or = (g["res"] ** 2)
        n_or, m_or, v_or = sq_or.rolling(**roll_kw).count(), sq_or.rolling(**roll_kw).mean(), sq_or.rolling(
            **roll_kw).var(ddof=1)

        # 存入字典
        dataset_pack = {
            "meta": {"N": N, "idx": i},
            "RMS error (Raw)": {"val": np.sqrt(m_or), "ci": rmse_ci(m_or, v_or, n_or)},
            "RMS error (BC)": {"val": np.sqrt(m_ml), "ci": rmse_ci(m_ml, v_ml, n_ml)},
            "RMS spread": {"val": np.sqrt(m_sp), "ci": rmse_ci(m_sp, v_sp, n_sp)}
        }
        stored_data.append(dataset_pack)

    # --- 3. 绘图配置 ---
    fig = go.Figure()

    # 定义绘制顺序（决定图例顺序）
    metric_order = ["RMS error (Raw)", "RMS error (BC)", "RMS spread"]

    # 定义基准颜色 (Hex)
    # Raw: 红色系, BC: 蓝色系, Spread: 绿色系
    base_colors = {
        "RMS error (Raw)": "#d62728",  # Red
        "RMS error (BC)": "#1f77b4",  # Blue
        "RMS spread": "#2ca02c"  # Green
    }

    # 定义透明度规则 (索引0深，索引1浅)
    # line_alphas用于线条，fill_alphas用于CI填充(需要更浅)
    line_alphas = [1.0, 0.4, 0.2]
    fill_alphas = [0.2, 0.1, 0.05]

    # --- 4. 绘图循环 (核心逻辑) ---
    # 外层循环：指标类型 (确保图例按 Raw -> BC -> Spread 分块)
    for metric in metric_order:
        base_color = base_colors[metric]

        # 内层循环：数据集 (确保 N1 在 N2 之前)
        for i, pack in enumerate(stored_data):
            N = pack["meta"]["N"]
            data = pack[metric]  # 获取 val 和 ci

            # 对齐时间轴 (reindex)
            # 这样处理是为了防止不同数据集时间不一致导致线条错乱
            series_val = data["val"].reindex(time_index_union)
            ci_lower = data["ci"][0].reindex(time_index_union)
            ci_upper = data["ci"][1].reindex(time_index_union)

            # 生成名称
            # 如果有多个相同N，这里可以加个Set区分，目前只显示N
            trace_name = f"{metric} [N={N}]"

            # 计算颜色
            # 越往后的数据集，透明度越高(颜色越浅)
            alpha_idx = min(i, len(line_alphas) - 1)
            line_color_rgba = _hex_to_rgba(base_color, alpha=line_alphas[alpha_idx])
            fill_color_rgba = _hex_to_rgba(base_color, alpha=fill_alphas[alpha_idx])

            # 设置 Legend Group
            # 使得点击图例时，该指标该N的 线+CI 同时隐藏
            lg_group = trace_name

            # A. 绘制 CI Upper (隐形线)
            if fill:
                fig.add_trace(go.Scatter(
                    x=time_index_union, y=ci_upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                    legendgroup=lg_group,
                    name=f"{trace_name} upper"
                ))

            # B. 绘制 CI Lower (填充)
            if fill:
                fig.add_trace(go.Scatter(
                    x=time_index_union, y=ci_lower,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fill_color_rgba,
                    showlegend=False, hoverinfo="skip",
                    legendgroup=lg_group,
                    name=f"{trace_name} lower"
                ))

            # C. 绘制主曲线
            fig.add_trace(go.Scatter(
                x=time_index_union, y=series_val,
                mode="lines",
                # 强制实线，设置颜色
                line=dict(color=line_color_rgba, width=2, dash="solid"),
                name=trace_name,
                legendgroup=lg_group
            ))

    # --- 5. 计算并添加相关性注记 ---
    # 这一步需要重新把数据取出来对齐计算
    annotations = []

    for i, pack in enumerate(stored_data):
        N = pack["meta"]["N"]
        # 获取原始未 reindex 的数据进行相关性计算 (剔除NaN)
        # 注意：需要让三个指标在时间上对齐
        df_temp = pd.DataFrame({
            "bc": pack["RMS error (BC)"]["val"],
            "raw": pack["RMS error (Raw)"]["val"],
            "sp": pack["RMS spread"]["val"]
        }).dropna()

        if len(df_temp) > 0:
            c_bc = df_temp["bc"].corr(df_temp["sp"])
            c_raw = df_temp["raw"].corr(df_temp["sp"])

            # 使用HTML语法给N加粗，并尝试用颜色区分文本(可选)
            # 这里简单处理，只显示文本
            suffix = f"[N={N}]"
            # if i > 0: suffix += f"(Set {i + 1})"

            annotations.append(f"<b>{suffix}</b>: Corr(Spread, BC)={c_bc:.3f}, Corr(Spread, Raw)={c_raw:.3f}")

    # --- 6. 布局设置 ---
    layout_annotations = []
    start_y = 0.98
    step_y = 0.06

    for k, text in enumerate(annotations):
        layout_annotations.append(dict(
            text=text,
            xref="paper", yref="paper",
            x=0.02, y=start_y - (k * step_y),
            showarrow=False,
            font=dict(size=13, color="black"),  # 统一黑色字体
            align="left",
            bgcolor="rgba(255,255,255,0.6)"  # 增加一点背景让文字在复杂的线上更清楚
        ))

    fig.update_layout(
        width=900, height=600,
        legend=dict(
            yanchor="top", y=0.98, xanchor="right", x=0.9,
            title=None,
            traceorder="normal",  # 严格遵循添加顺序
            # bgcolor="rgba(255,255,255,0.8)",
            # bordercolor="Black",
            # borderwidth=1
        ),
        yaxis_title="Sliding Windows RMS (mm)",
        xaxis_title="Time",
        title=None,
        annotations=layout_annotations,
        template="simple_white"
    )

    return fig
# ---------------- helpers ----------------
def percentile_and_bins(x, bins=20):
    """
    将 x 映射到 [0,100] 的全局百分位，并按百分位等宽分箱。
    返回:
        pct:  每个样本的百分位 (0..100)
        idx:  每个样本的分箱索引 [0..bins-1]
        centers: 每个分箱的中心百分位
        edges:   分箱边界百分位
    """
    pct = pd.Series(x).rank(pct=True).to_numpy() * 100.0
    edges = np.linspace(0, 100, bins + 1)
    idx = np.digitize(pct, edges, right=True) - 1
    idx = np.clip(idx, 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pct, idx, centers, edges

def equalwidth_bins(x, bins=20):
    """
    在 x 的取值范围内做等宽分箱（不做百分位变换）。
    返回:
        x:    原始 x（未变换）
        idx:  每个样本的分箱索引 [0..bins-1]
        centers: 每个分箱的中心值
        edges:   分箱边界
    """
    x = np.asarray(x, float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        # 退化情况：全部 NaN 或常数，强行给一个单箱
        edges = np.array([xmin - 0.5, xmax + 0.5], float)
        idx = np.zeros_like(x, dtype=int)
        centers = np.array([(edges[0] + edges[1]) * 0.5], float)
        return x, idx, centers, edges
    edges = np.linspace(xmin, xmax, bins + 1)
    idx = np.digitize(x, edges, right=True) - 1
    idx = np.clip(idx, 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return x, idx, centers, edges

def ols_fit(x, y):
    """
    简单一元线性回归：y = b0 + b1*x
    返回: beta=(b0,b1), yhat, R^2
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[m], y[m]
    if xx.size < 2:
        return np.array([np.nan, np.nan]), np.full_like(yy, np.nan), np.nan
    X = np.column_stack([np.ones(xx.size), xx])
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    yhat = X @ beta
    sst = np.sum((yy - yy.mean()) ** 2)
    sse = np.sum((yy - yhat) ** 2)
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    # 把 yhat 放回原长度（仅供一致性，这里函数外不直接用回填）
    return beta, yhat, r2

def bin_stats(y, idx, bins):
    """
    对每个分箱计算 (mean, std, se, ci95)。
    返回: shape=(bins, 4) 的数组，列分别为 mean, std, se, ci95。
    """
    out = []
    y = np.asarray(y, float)
    for b in range(bins):
        yy = y[idx == b]
        if yy.size < 2:
            out.append((np.nan, np.nan, np.nan, np.nan))
        else:
            m = float(np.mean(yy))
            s = float(np.std(yy, ddof=1))
            se = s / np.sqrt(yy.size)
            ci = 1.96 * se
            out.append((m, s, se, ci))
    return np.array(out, float)

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- helpers ----------------
def percentile_and_bins(x, bins=20):
    pct = pd.Series(x).rank(pct=True).to_numpy() * 100.0
    edges = np.linspace(0, 100, bins + 1)
    idx = np.digitize(pct, edges, right=True) - 1
    idx = np.clip(idx, 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pct, idx, centers, edges

def equalwidth_bins(x, bins=20):
    x = np.asarray(x, float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        edges = np.array([xmin - 0.5, xmax + 0.5], float)
        idx = np.zeros_like(x, dtype=int)
        centers = np.array([(edges[0] + edges[1]) * 0.5], float)
        return x, idx, centers, edges
    edges = np.linspace(xmin, xmax, bins + 1)
    idx = np.digitize(x, edges, right=True) - 1
    idx = np.clip(idx, 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return x, idx, centers, edges

def ols_fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[m], y[m]
    if xx.size < 2:
        return np.array([np.nan, np.nan]), np.full_like(yy, np.nan), np.nan
    X = np.column_stack([np.ones(xx.size), xx])
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    yhat = X @ beta
    sst = np.sum((yy - yy.mean()) ** 2)
    sse = np.sum((yy - yhat) ** 2)
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return beta, yhat, r2

def bin_stats(y, idx, bins):
    out = []
    y = np.asarray(y, float)
    for b in range(bins):
        yy = y[idx == b]
        if yy.size < 2:
            out.append((np.nan, np.nan, np.nan, np.nan))
        else:
            m = float(np.mean(yy))
            s = float(np.std(yy, ddof=1))
            se = s / np.sqrt(yy.size)
            ci = 1.96 * se
            out.append((m, s, se, ci))
    return np.array(out, float)

# ---------------- main ----------------
def plot_relation(
    df: pd.DataFrame,
    x: str,
    y: str,
    bins: int = 30,
    x_as_percentile: bool = True,
):
    """
    一行三图（Plotly）：
      1) 散点 + OLS
      2) 分箱均值 ±95%CI
      3) 箱线图（按分箱）
    返回 plotly.graph_objects.Figure（不在函数内保存）
    """
    if x not in df.columns or y not in df.columns:
        raise KeyError("指定的 x 或 y 列不存在于 DataFrame 中。")

    x_raw = df[x].to_numpy()
    y_raw = df[y].to_numpy()
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    xv = x_raw[mask]
    yv = y_raw[mask]

    # x 轴与分箱
    if x_as_percentile:
        x_plot, idx, centers, edges = percentile_and_bins(xv, bins=bins)
        x_label = f"{x} percentile (%)"
        x_for_fit = x_plot
    else:
        x_plot, idx, centers, edges = equalwidth_bins(xv, bins=bins)
        x_label = x
        x_for_fit = xv

    # 回归与相关
    beta, yhat, r2 = ols_fit(x_for_fit, yv)
    if np.isfinite(beta).all():
        xp = np.linspace(np.nanmin(x_for_fit), np.nanmax(x_for_fit), 200)
        yp = beta[0] + beta[1] * xp
    else:
        xp = np.array([]); yp = np.array([])

    try:
        pear = float(np.corrcoef(x_for_fit, yv)[0, 1])
    except Exception:
        pear = np.nan
    try:
        spear = float(pd.Series(x_for_fit).rank().corr(pd.Series(yv).rank()))
    except Exception:
        spear = np.nan

    # 分箱统计
    stats = bin_stats(yv, idx, bins)
    means, cis = stats[:, 0], stats[:, 3]
    valid = np.isfinite(centers) & np.isfinite(means) & np.isfinite(cis)
    c = centers[valid]; m = means[valid]; ci = cis[valid]
    upper, lower = m + ci, m - ci

    # 箱线图数据
    data_bins = [yv[idx == b] for b in range(bins)]
    cat_x = [f"{cc:.2f}" for cc in centers]

    # 画布
    fig = make_subplots(
        rows=1, cols=3, shared_xaxes=False, horizontal_spacing=0.07,
        subplot_titles=["Scatter + OLS", "Binned mean ±95% CI", "Box by bin"]
    )

    # (1) 散点 + OLS
    fig.add_trace(
        go.Scatter(
            x=x_for_fit, y=yv,
            mode="markers", marker=dict(size=6, opacity=0.35),
            name="scatter",
            hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}}<extra></extra>"
        ),
        row=1, col=1
    )
    if xp.size:
        fig.add_trace(
            go.Scatter(x=xp, y=yp, mode="lines", name="OLS", hoverinfo="skip"),
            row=1, col=1
        )
    # ✅ 关键修复：第一子图用 'x domain'/'y domain'，不要用 'x1 domain'
    fig.add_annotation(
        xref="x domain", yref="y domain",
        x=0.02, y=0.98, xanchor="left", yanchor="top",
        text=(f"{y} vs " + (f"percentile({x})" if x_as_percentile else f"{x}") +
              f"<br>n={len(yv)}  r={pear:.3f}  ρ={spear:.3f}  R²={r2:.3f}"),
        showarrow=False, font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)"
    )
    fig.update_xaxes(title_text=x_label, row=1, col=1)
    fig.update_yaxes(title_text=y, row=1, col=1)

    # (2) 分箱均值 ±95%CI
    if c.size:
        fig.add_trace(
            go.Scatter(
                x=c, y=m, mode="lines", name="mean",
                hovertemplate="bin center=%{x:.4g}<br>mean=%{y:.4g}<extra></extra>"
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=np.r_[c, c[::-1]], y=np.r_[upper, lower[::-1]],
                fill="toself", line=dict(width=0), opacity=0.25,
                name="95% CI", hoverinfo="skip"
            ),
            row=1, col=2
        )
    fig.update_xaxes(title_text=x_label if x_as_percentile else f"{x} (bin centers)", row=1, col=2)
    fig.update_yaxes(title_text=y, row=1, col=2)

    # (3) 箱线图
    for cc_str, yy in zip(cat_x, data_bins):
        if yy.size == 0:
            continue
        fig.add_trace(
            go.Box(
                y=yy, x=[cc_str] * yy.size, boxpoints=False, showlegend=False,
                hovertemplate=f"bin≈{cc_str}<br>y=%{{y:.4g}}<extra></extra>"
            ),
            row=1, col=3
        )
    fig.update_xaxes(title_text="bin center" + (" (percentile)" if x_as_percentile else ""), row=1, col=3)
    fig.update_yaxes(title_text=y, row=1, col=3)

    # 统一 y 轴范围（1–99%分位）
    y_collect = [yv]
    if xp.size: y_collect.append(yp)
    if m.size: y_collect += [m, upper, lower]
    for yy in data_bins:
        if yy.size: y_collect.append(yy)
    ycat = np.concatenate([np.asarray(v).ravel()
                           for v in y_collect
                           if np.size(v) > 0 and np.isfinite(v).any()], axis=None)
    if ycat.size > 0:
        lo = float(np.nanpercentile(ycat, 1))
        hi = float(np.nanpercentile(ycat, 99))
        for ccol in (1, 2, 3):
            fig.update_yaxes(range=[lo, hi], row=1, col=ccol)

    fig.update_layout(template="simple_white",height=420, width=1200, margin=dict(l=60, r=30, t=60, b=40),
                      title=f"{y} vs {x}" + (" (x→percentile)" if x_as_percentile else ""))
    return fig