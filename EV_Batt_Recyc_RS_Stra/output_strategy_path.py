import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

recycl_list = ['Pyro', 'Hydro', 'Direct']
year_list = [2023, 2033]
df_netprofits_all = pd.DataFrame()
all_countries_table = pd.DataFrame()
all_countries_netprofits = pd.DataFrame()
for year_val in year_list:
    for s in ['Strategy 1', 'Strategy 2', 'Strategy 3']:
        all_countries = pd.read_csv('all_countries.csv', index_col=0)

        for recycl_proce in recycl_list:

            all_countries_table_df = pd.read_csv(f'./trans/{year_val}_net_profit_{recycl_proce}_{s}.csv', index_col=0)
            all_countries_netprofits_df = pd.read_csv(f'./trans/{year_val}_country_net_profit_{recycl_proce}_{s}.csv',
                                                   index_col=0)

            all_countries_table = all_countries_table.append(all_countries_table_df)
            all_countries_netprofits = all_countries_netprofits.append(all_countries_netprofits_df)

        all_countries_table = all_countries_table.reset_index(drop=True)
        all_countries_netprofits = all_countries_netprofits.reset_index(drop=True)

        idx = all_countries_table.iloc[:, [0, 1, 2, 5]].groupby(['country', 'type'])['net_profit'].idxmax()
        all_countries_table_cho = all_countries_table.loc[idx, ['country', 'type', 'recycling_m', 'net_profit']]

        # 使用transform找到每个分组的'值3'的最大值
        all_countries_table_cho['net_profit_max'] = all_countries_table_cho.groupby('country')['net_profit'].transform(max)

        # 选择原始DataFrame中'值3'等于'最大值3'的行
        all_countries_result = all_countries_table_cho[
            all_countries_table_cho['net_profit'] == all_countries_table_cho['net_profit_max']].drop('net_profit_max', axis=1)
        all_countries_result = all_countries_result.drop_duplicates(subset=['country', 'net_profit'])
        all_countries_netprofits_chio = all_countries_result[['country', 'recycling_m']].merge(all_countries_netprofits,
                                                                                               on=['country',
                                                                                                   'recycling_m'],
                                                                                               how='left')
        world_netprofits_value = all_countries_netprofits_chio['total_netprofits'].sum()
        produc_countries = all_countries[all_countries['producer'] == True]['country'].tolist()

        produc_netprofits_value = \
            all_countries_netprofits_chio[all_countries_netprofits_chio['country'].isin(produc_countries)][
                'total_netprofits'].sum()

        # # 选择原始DataFrame中'值3'等于'最大值3'的行
        # all_countries_result = all_countries_table[
        #     all_countries_table['net_profit'] == all_countries_table['net_profit_max']].drop('net_profit_max', axis=1)
        # all_countries_result = all_countries_result.drop_duplicates(subset=['country', 'net_profit'])
        all_countries = all_countries.merge(all_countries_result, on='country', how='left')
        # 修正国家名称
        all_countries.loc[all_countries['country'] == "C?te d'Ivoire", 'country'] = "Côte d'Ivoire"
        all_countries.loc[all_countries['country'] == "Bolivia (Plurinational State of)", 'country'] = "Bolivia"
        all_countries.loc[all_countries['country'] == "Russian Federation", 'country'] = "Russia"
        all_countries.loc[all_countries['country'] == "Lao People's Dem. Rep.", 'country'] = "Laos"
        all_countries.loc[all_countries['country'] == "Viet Nam", 'country'] = "Vietnam"
        all_countries.loc[all_countries['country'] == "USA", 'country'] = "United States of America"
        all_countries.loc[all_countries['country'] == "Korea", 'country'] = "South Korea"
        all_countries.loc[all_countries['country'] == "United Rep. of Tanzania", 'country'] = "Tanzania"
        all_countries.loc[all_countries['country'] == "Solomon Isds", 'country'] = "Solomon Is."
        all_countries.loc[all_countries['country'] == "State of Palestine", 'country'] = "Palestine"
        all_countries.loc[all_countries['country'] == "Rep. of Moldova", 'country'] = "Moldova"
        all_countries.loc[all_countries['country'] == "Eswatini", 'country'] = "eSwatini"
        all_countries.loc[all_countries['country'] == "Bosnia Herzegovina", 'country'] = "Bosnia and Herz."
        all_countries.loc[all_countries['country'] == "North Macedonia", 'country'] = "Macedonia"
        all_countries.loc[all_countries['country'] == "Brunei Darussalam", 'country'] = "Brunei"

        all_countries_copy = all_countries.copy()
        all_countries_copy['Strategy'] = [s] * len(all_countries_copy)
        all_countries_copy['Year'] = [year_val] * len(all_countries_copy)
        df_netprofits_all = df_netprofits_all.append(all_countries_copy, ignore_index=True)

        all_count_list = all_countries['country'].tolist()

        all_countries = all_countries.rename(columns={'country': 'name'})
        all_countries_tab = all_countries[['name', 'kmeans_cluster', 'lon', 'lat']].copy()

        # 加载世界国家边界数据
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        all_name = world['name'].tolist()
        world = world.merge(all_countries_tab, on='name', how='left')

        intersection_2 = list(set(all_count_list) - set(all_name))
        extra_countries_df = all_countries[all_countries['name'].isin(intersection_2)].copy()

        # 过滤出要标记的国家
        highlighted_countries = world[world['name'].isin(all_count_list)].copy()
        highlighted_countries = highlighted_countries.merge(all_countries[['name', 'net_profit']], on='name',
                                                            how='left')

        # 设置地图
        fig, ax = plt.subplots(figsize=(15, 20))
        m = Basemap(
            projection='merc',
            llcrnrlon=-178,  # 左下角的经度
            llcrnrlat=-60,  # 左下角的纬度
            urcrnrlon=180,  # 右上角的经度
            urcrnrlat=70,  # 右上角的纬度
            resolution='l',
            suppress_ticks=True
        )

        m.drawcountries(linewidth=0.3)
        m.drawcoastlines(linewidth=0.1)
        m.fillcontinents(alpha=0.5)

        # 使用颜色映射，范围为大于等于2的部分
        cmap = plt.cm.get_cmap('PRGn')
        norm = Normalize(vmin=-12, vmax=12)

        # 绘制高亮国家和随机值
        for idx, row in highlighted_countries.iterrows():
            poly = row['geometry']
            if poly.geom_type == 'Polygon':
                poly = [poly]
            for polygon in poly:
                x, y = polygon.exterior.xy
                m_x, m_y = m(x, y)
                if row['net_profit'] < -12:
                    # 使用斜杠填充
                    ax.add_patch(Polygon(
                        list(zip(m_x, m_y)),
                        facecolor='none', edgecolor='black', hatch='//'
                    ))
                else:
                    # 使用颜色填充
                    ax.fill(m_x, m_y, color=cmap(norm(row['net_profit'])), edgecolor='black')

        # 标记几何中心的圆点，数值小于2的国家使用灰色填充
        for idx, row in extra_countries_df.iterrows():
            x, y = m(row['lon'], row['lat'])
            if row['net_profit'] < -12:
                ax.scatter(x, y, color='lightgrey', edgecolor='k', s=100, zorder=5)
            else:
                ax.scatter(x, y, color=cmap(norm(row['net_profit'])), edgecolor='k', s=100, zorder=5)

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.015, pad=0.05)
        cbar.set_label('Random Value (>=-12)')
        ax.text(0.5, 0.1, f'Global net profit: {world_netprofits_value}', transform=ax.transAxes, fontsize=16,
                verticalalignment='bottom')
        ax.text(0.5, 0.05, f'Global productor net profit: {produc_netprofits_value}', transform=ax.transAxes,
                fontsize=16,
                verticalalignment='bottom')
        # 调整颜色条的刻度
        cbar.set_ticks(np.arange(-12, 13, 2))
        plt.title(f"{year_val} Waste Flow {s} in the World", fontsize=20)
        plt.tight_layout()
        plt.show()

df_netprofits_all = df_netprofits_all[df_netprofits_all['net_profit'] > 0]

# len(df_netprofits_all['country'].unique())
