import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

# recycl_list = ['Pyro', 'Hydro', 'Direct', 'Optimal']
recycl_list = 'Optimal'
co2_em_data = pd.read_csv(f'./trans/result/CO2_em_trans_{recycl_list}.csv', index_col=0)
co2_em_data_count = co2_em_data.groupby(['year', 'Strategy type'])['CO2'].sum().reset_index()
co2_em_data_count['CO2'] = co2_em_data_count['CO2'] / 1000

year_list = list(range(2023, 2033 + 1))
produ_net_profits_df = pd.DataFrame(columns=['Year', 'Strategy', 'Total_net_profit', 'total_costs'])
net_profits_df = pd.DataFrame(columns=['Year', 'Strategy', 'Total_net_profit', 'total_costs'])
df_netprofits_all = pd.DataFrame()
for year_val in year_list:
    for s in ['Strategy 1', 'Strategy 2', 'Strategy 3']:
        all_countries = pd.read_csv('all_countries.csv', index_col=0)
        all_countries_co2 = pd.read_csv(f'./trans/result/CO2_em_disposal_mass_diff_stra_{recycl_list}.csv', index_col=0)
        world_co2_df = all_countries_co2[
            (all_countries_co2['Strategy type'] == s) & (all_countries_co2['year'] == year_val)].copy()
        world_co2_df_cho = world_co2_df.iloc[:, [0, 1, 7]].groupby('country')['CO2_em'].sum().reset_index()


        all_countries_netprofits = pd.read_csv(f'./trans/result/{year_val}/country_net_profit_{recycl_list}_{s}.csv',
                                               index_col=0)
        all_countries_cost = pd.read_csv(f'./trans/result/{year_val}/country_cost_{recycl_list}_{s}.csv', index_col=0)
        #
        world_netprofits_value = all_countries_netprofits['total_netprofits'].sum()
        world_costs_value = all_countries_cost['total_costs'].sum()
        world_co2_value = world_co2_df_cho['CO2_em'].sum()
        # print(world_co2_value)

        produc_countries = all_countries[all_countries['producer'] == True]['country'].tolist()
        if s != 'Strategy 1':
            co2_em_data_value = \
            co2_em_data_count[(co2_em_data_count['year'] == year_val) & (co2_em_data_count['Strategy type'] == s)][
                'CO2'].tolist()[0]
            world_co2_value = world_co2_value + co2_em_data_value

        net_profits_df = net_profits_df.append(
            {'Year': year_val, 'Strategy': s, 'Total_net_profit': int(world_netprofits_value),
             'total_costs': int(world_costs_value), 'CO2_em': int(world_co2_value)}, ignore_index=True)

net_profits_df = net_profits_df.reset_index(drop=True)


def plot_grouped_waterfall(df, category_col, value_col, label_col, y_min, y_max, num, ylabel):
    categories = df[category_col].unique()
    labels = df[label_col].unique()
    num_categories = len(categories)
    num_values = len(labels)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 5), dpi=300)

    colors = plt.cm.get_cmap('tab20', num_categories)

    fotsize = 16

    for i, category in enumerate(categories):
        values_i = df[df[category_col] == category][value_col].values
        cumulative = np.cumsum(values_i)
        starts = np.zeros_like(values_i)
        starts[1:] = cumulative[:-1]
        color = colors(i)

        for j, (start, value) in enumerate(zip(starts, values_i)):
            ax.bar(j + i * (num_values + 2), value, bottom=start, color=color, label=category if j == 0 else "",
                   zorder=1)
            if abs(value) > 0.5:
                ax.text(j + i * (num_values + 2), start + value + 0.3, f'{value:.1f}', ha='center', va='bottom',
                        color='black', zorder=2, fontsize=fotsize - 2)

        # Add total text annotation for each category
        total = cumulative[-1]
        ax.bar(num_values + i * (num_values + 2), total, bottom=0, color=color, zorder=1)
        ax.text(num_values + i * (num_values + 2), total + 0.3, f'{total:.1f}', ha='center', va='bottom', color='black',
                fontsize=fotsize - 2, zorder=2)

        # Add dashed line between categories
        if i < num_categories - 1:
            ax.axvline(x=(i + 1) * (num_values + 2) - 1, color='black', linestyle='-', linewidth=1, zorder=0)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    # Set labels and title
    all_labels = []
    for i in range(num_categories):
        all_labels.extend(labels.tolist() + [f'Total', ''])

    # Adjust x-ticks to ensure consistent spacing
    x_ticks = []
    for i in range(num_categories):
        x_ticks.extend([x + i * (num_values + 2) for x in range(num_values)] + [num_values + i * (num_values + 2),
                                                                                (num_values + 1) + i * (
                                                                                            num_values + 2)])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(all_labels, rotation=45, ha='center', fontsize=fotsize)
    # ax.set_title(title)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.linspace(y_min, y_max, num))

    ax.set_ylabel(ylabel, fontsize=fotsize)

    # Adjust y-axis tick label size
    ax.tick_params(axis='y', labelsize=fotsize)

    ax.grid(True, linestyle='--')
    handles, _ = ax.get_legend_handles_labels()
    # ax.legend(handles[:num_categories], categories, loc='upper left', bbox_to_anchor=(1, 1))
    if value_col == 'CO2_em':
        ax.legend(handles[:num_categories], categories, loc='upper left', bbox_to_anchor=(0.02, 0.95), borderaxespad=0.,
                  fontsize=fotsize, labelspacing=1.0)

    # Disable vertical grid lines
    ax.grid(axis='x', linestyle='')

    # plt.tight_layout()
    plt.show()


net_profits_df['CO2_em'] = net_profits_df['CO2_em'] / 1e7
df = net_profits_df.copy()
ylabel = "Carbon dioxide emissions \n(unit: 10 million tons)"
# 绘制分组瀑布图
plot_grouped_waterfall(df, 'Strategy', 'CO2_em', 'Year', 0, 13, 6, ylabel)
