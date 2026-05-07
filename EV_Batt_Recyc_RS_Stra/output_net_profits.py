import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

# recycl_list = ['Pyro', 'Hydro', 'Direct', 'Optimal']
recycl_list = 'Optimal'
year_list = list(range(2023, 2033 + 1))
produ_net_profits_df = pd.DataFrame(columns=['Year', 'Strategy', 'Total_net_profit', 'total_costs'])
net_profits_df = pd.DataFrame(columns=['Year', 'Strategy', 'Total_net_profit', 'total_costs'])
df_netprofits_all = pd.DataFrame()
producer_list = ['China', 'India', 'Japan', 'Korea', 'Thailand', 'Viet Nam', 'France', 'Germany', 'Hungary', 'Italy',
                 'Poland', 'Slovakia', 'Spain', 'Sweden', 'United Kingdom', 'Norway', 'Russian Federation', 'Turkey',
                 'USA', 'Canada', 'Brazil', 'Australia', 'Czechia', 'Romania', 'Serbia', 'Belgium', 'Netherlands',
                 'Switzerland', 'Finland']

for year_val in year_list:
    for s in ['Strategy 1', 'Strategy 2', 'Strategy 3']:
        all_countries = pd.read_csv('all_countries.csv', index_col=0)
        all_countries_co2 = pd.read_csv(f'./trans/result/CO2_em_disposal_mass_diff_stra_{recycl_list}.csv', index_col=0)
        world_co2_df = all_countries_co2[
            (all_countries_co2['Strategy type'] == s) & (all_countries_co2['year'] == year_val)].copy()
        world_co2_df_cho = world_co2_df.iloc[:, [0, 1, 7]].groupby('country')['CO2_em'].sum().reset_index()
        all_countries_table_df = pd.read_csv(f'./trans/result/{year_val}/net_profit_{recycl_list}_{s}.csv', index_col=0)
        all_countries_netprofits = pd.read_csv(f'./trans/result/{year_val}/country_net_profit_{recycl_list}_{s}.csv',
                                               index_col=0)
        all_countries_cost = pd.read_csv(f'./trans/result/{year_val}/country_cost_{recycl_list}_{s}.csv', index_col=0)
        #
        world_netprofits_value = all_countries_netprofits['total_netprofits'].sum()
        world_costs_value = all_countries_cost['total_costs'].sum()
        world_co2_value = world_co2_df_cho['CO2_em'].sum()

        all_countries_copy = all_countries_cost[all_countries_cost['country'].isin(producer_list)].iloc[:, [0, 3]].copy()
        all_countries_copy['Year'] = [year_val] * len(all_countries_copy)
        pivot_df1 = all_countries_copy.pivot(index='Year', columns='country', values='unit_cost')

        pivot_df1.reset_index(drop=True, inplace=True)
        pivot_df1['Strategy'] = [s] * len(pivot_df1)
        pivot_df1['Year'] = [year_val] * len(pivot_df1)
        pivot_df1['Total_cost'] = [world_costs_value] * len(pivot_df1)

        df_netprofits_all = df_netprofits_all.append(pivot_df1, ignore_index=True)

df_netprofits_all_cho = df_netprofits_all.reset_index(drop=True)