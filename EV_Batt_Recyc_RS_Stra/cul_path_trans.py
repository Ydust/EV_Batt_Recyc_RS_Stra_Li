import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import fun_class
import os

# %% Load the data
file_path1 = './Scenario result/EV_battery_inuse_scrap.csv'
file_path3 = './Scenario result/EV_battery_inuse_scrap_total.csv'
evb_type_data = pd.read_csv(file_path1)
evb_data = pd.read_csv(file_path3)
evb_type_data = evb_type_data[evb_type_data['region']!='China'].copy()
evb_data = evb_data[evb_data['region']!='China'].copy()


year_val = 2023
year_val1 = 2033

# Load the data from the uploaded files
produced_materials_df = pd.read_csv('./cost/Produced materials for recycling.csv')
materials_value_df = pd.read_csv('./cost/materials value.csv')
# Reco_effi_df = pd.read_csv('./cost/Recovery efficiency.csv')
fenle = produced_materials_df['materials'].unique()
list_va = [0.9] * len(fenle)
fenle_dict = dict(zip(fenle, list_va))
for key in fenle_dict:
    if key == 'Co2+ in product':
        fenle_dict[key] = 0.98
    elif key == 'Ni2+ in product':
        fenle_dict[key] = 0.98
    elif key == 'Mn2+ in product':
        fenle_dict[key] = 0.98
fenle_df = pd.DataFrame(list(fenle_dict.items()), columns=['materials', 'Value'])

# Strip any extra spaces from the column names in the produced materials dataframe
produced_materials_df.columns = produced_materials_df.columns.str.strip()

# Merge the dataframes on the 'materials' column
merged_df = produced_materials_df.merge(materials_value_df, on='materials')
merged_df = merged_df.merge(fenle_df, on='materials', how='left')
merged_df['Value of recycled materials ($/kg)'] = merged_df['Value'] * merged_df['Value of recycled materials ($/kg)']
# Group by recycling method
grouped = merged_df.groupby('recycling_m')

# Initialize a dictionary to store results
results = {}

# List of battery types
battery_types = ['NMC811', 'NMC111', 'NMC523', 'NMC622', 'NCA', 'LFP']

# Calculate the value for each battery type in each recycling method
for name, group in grouped:
    total_values = {}
    for battery in battery_types:
        total_value = (group[battery] * group['Value of recycled materials ($/kg)']).sum()
        total_values[battery] = total_value
    results[name] = total_values
#
# # Convert results to a DataFrame
Recycling_revenue_df = pd.DataFrame(results).T

# Load the data
file_path = './cost/cost_breakdown.csv'
cost_breakdown_data = pd.read_csv(file_path)

# Find unique recycling methods in the data
unique_recycling_methods = cost_breakdown_data['recycling_m'].unique()
# Load the three CSV files
file_1 = pd.read_csv('LAC_4HRL_ECO_CUR_NB_A-20240628T1113.csv')
file_2 = pd.read_csv('LAC_4HRL_ECO_CUR_NB_A-filtered-2024-06-28.csv')
file_3 = pd.read_csv('LAC_4HRL_ECO_CUR_NB_A-20240629T0639.csv')

# Combine the three dataframes, removing duplicate rows
combined_labor_cost_df = pd.concat([file_1, file_2, file_3]).drop_duplicates()

# Save the combined dataframe to a new CSV file
# output_file = './cost/Combined_labor_cost.csv'
# combined_labor_cost_df.to_csv(output_file, index=False)

co_df = combined_labor_cost_df.loc[(combined_labor_cost_df['classif2.label'] == 'Currency: U.S. dollars') & (
        combined_labor_cost_df['classif1.label'] == 'Economic activity (ISIC-Rev.4): C. Manufacturing')].copy()

result_com = co_df.loc[co_df.groupby('ref_area.label')['time'].idxmax()].copy()

combined3 = combined_labor_cost_df.loc[(combined_labor_cost_df['ref_area.label'].isin(['Zimbabwe', 'South Africa']))]
filtered_result2 = combined3[combined3['classif1.label'].str.contains('Manufacturing', case=False, na=False)]
filtered_result3 = filtered_result2[(filtered_result2['ref_area.label'] == 'Zimbabwe') | (
        filtered_result2['classif2.label'] == 'Currency: U.S. dollars') & (filtered_result2['time'] == 2017)]
result_com = result_com.append(filtered_result3).iloc[:, :7]

result_com.loc[result_com[
                   'ref_area.label'] == "United Kingdom of Great Britain and Northern Ireland", 'ref_area.label'] = 'United Kingdom'
result_com.loc[result_com['ref_area.label'] == "Russian Federation", 'ref_area.label'] = 'Russia'
result_com.loc[result_com['ref_area.label'] == "Türkiye", 'ref_area.label'] = 'Turkey'

# https://www.china-briefing.com/news/minimum-wages-china/
# result_com.loc[len(result_com)] = ['China', '', '', '', 'Currency: U.S. dollars', year_val, 3]
# https://www.ceicdata.com/en/countries
result_com.loc[len(result_com)] = ['North Korea', '', '', '', 'Currency: U.S. dollars', year_val, 10]
result_com.loc[len(result_com)] = ['Singapore', '', '', '', 'Currency: U.S. dollars', year_val, 42.4]
result_com.loc[len(result_com)] = ['Japan', '', '', '', 'Currency: U.S. dollars', year_val, 17]
# https://www.stat.gov.rs/en-us/oblasti/trziste-rada/troskovi-rada/
result_com.loc[len(result_com)] = ['Serbia', '', '', '', 'Currency: U.S. dollars', year_val, 7.51]
# https://www.vietnam-briefing.com/news/vietnam-q3-labor-market-update-increase-in-workers-employment.html/
result_com.loc[len(result_com)] = ['Vietnam', '', '', '', 'Currency: U.S. dollars', year_val, 1.81]
# https://remotepad.com/countries/india/minimum-wage/
result_com.loc[len(result_com)] = ['India', '', '', '', 'Currency: U.S. dollars', year_val, 2.16]

result_com.loc[result_com['ref_area.label'] == "Zimbabwe", 'obs_value'] = 2.63
combined1 = result_com['ref_area.label'].unique()

file_4 = './cost/equ_cost.csv'
equ_cost_df = pd.read_csv(file_4)
equ_cost_grouped_sum = equ_cost_df.groupby('recycling_m')['Person-hrs/day'].sum().reset_index()

labor_cost_new = pd.DataFrame()
for lc in unique_recycling_methods:
    number_labor = equ_cost_grouped_sum[equ_cost_grouped_sum['recycling_m'] == lc]['Person-hrs/day'].values[0]
    result_com[f'{lc}'] = result_com['obs_value'] * number_labor * 320 / 1e7

labor_cost_new = result_com[['ref_area.label', 'Direct', 'Hydro', 'Pyro']].reset_index(drop=True).copy()
labor_cost_new.loc[labor_cost_new['ref_area.label'] == "Russia", 'ref_area.label'] = 'Russian Federation'
labor_cost_new.loc[labor_cost_new['ref_area.label'] == "United States of America", 'ref_area.label'] = 'USA'
labor_cost_new.loc[labor_cost_new['ref_area.label'] == "North Korea", 'ref_area.label'] = 'Korea'
labor_cost_new.loc[labor_cost_new['ref_area.label'] == "Vietnam", 'ref_area.label'] = 'Viet Nam'

# unite: US$/tCO2e
carbon_tax_nation_data = pd.read_csv('./cost/carbon_tax_nation.csv')
nation_df = pd.read_csv('nation_list_new.csv', encoding='utf_8_sig')
nation_df_copy = nation_df.copy()
nation_df_copy.rename(columns={'region': 'country'}, inplace=True)
nation_df_copy = nation_df_copy[nation_df_copy['country']!='China'].copy()

# Initialize a dictionary to store grouped data
producer_list = ['India', 'Japan', 'Korea', 'Thailand', 'Viet Nam', 'France', 'Germany', 'Hungary', 'Italy',
                 'Poland', 'Slovakia', 'Spain', 'Sweden', 'United Kingdom', 'Norway', 'Russian Federation', 'Turkey',
                 'USA', 'Canada', 'Brazil', 'Australia', 'Czechia', 'Romania', 'Serbia', 'Belgium', 'Netherlands',
                 'Switzerland', 'Finland']

nation_df_new = nation_df_copy[['iso3', 'country', 'Sub-region Name', 'continent']].copy()

# unite: g / kg spent LiBs
recycling_emission_count = pd.read_csv('./cost/recycling_emission_count.csv')
recycling_emission_count_chiose = recycling_emission_count[['battery_type', 'recycling_m', 'CO2']].copy()
# unite: t / t spent LiBs
recycling_emission_count_chiose['CO2'] = recycling_emission_count_chiose['CO2'] / 1000

carbon_tax_nation_data = carbon_tax_nation_data.rename(columns={'Jurisdiction Covered': 'country'})
carbon_tax_nation_data.loc[carbon_tax_nation_data['country'] == "California", 'country'] = "USA"
carbon_tax_nation_data.loc[carbon_tax_nation_data['country'] == "Korea, Rep.", 'country'] = "Korea"

# panding EU
european_countries = nation_df_new[nation_df_new['continent'] == 'Europe']['country'].tolist()
# 创建一个 DataFrame 并标记是否为欧洲国家
coun_list = list(set(producer_list) - set(carbon_tax_nation_data['country'].unique()))
df_cc = pd.DataFrame(coun_list, columns=['country'])
df_cc['Is_European'] = df_cc['country'].apply(lambda x: x in european_countries)

# 过滤出欧洲国家
european_chiose = df_cc[df_cc['Is_European']]['country'].tolist()

eu_data = carbon_tax_nation_data[carbon_tax_nation_data['country'] == 'EU'].copy()

for ia in european_chiose:
    eu_data_copy = eu_data.copy()
    eu_data_copy.loc[eu_data_copy['country'] == "EU", 'country'] = ia
    carbon_tax_nation_data = carbon_tax_nation_data.append(eu_data_copy)

# unite: US$/tCO2e
carbon_tax_nation = carbon_tax_nation_data[['country', 'Value']].copy()

# carbon cost
# unite: US$/t spent cell
carbon_cost = pd.DataFrame()
for cb in carbon_tax_nation['country'].unique():
    recycling_emission_count_chiose_1 = recycling_emission_count_chiose.copy()
    tax_value = carbon_tax_nation[carbon_tax_nation['country'] == cb]['Value'].tolist()
    recycling_emission_count_chiose_1['CO2'] = recycling_emission_count_chiose_1['CO2'] * tax_value[0]
    recycling_emission_count_chiose_1['country'] = [cb] * len(recycling_emission_count_chiose_1)
    carbon_cost = carbon_cost.append(recycling_emission_count_chiose_1)

carbon_cost = carbon_cost.rename(columns={'CO2': 'carbon_cost'})
# unite: US$/t spent cell
carbon_cost['carbon_cost'] = carbon_cost['carbon_cost'] / 1000

carbon_cost_chiose = carbon_cost.groupby(['country', 'recycling_m'])['carbon_cost'].mean().reset_index()
carbon_cost_chiose = carbon_cost_chiose.merge(nation_df_new, on='country', how='left').dropna()
carbon_cost_chiose = carbon_cost_chiose[['iso3', 'recycling_m', 'carbon_cost']]


# Display the combined data
combined_data_new = cost_breakdown_data[['country', 'recycling_m', 'utility', 'materials', 'general_expenses']].copy()
combined_data_new =combined_data_new[combined_data_new['country'] != 'China'].reset_index(drop=True)
labour_cost_value = combined_data_new.copy()
for i in range(len(labour_cost_value)):
    for cu in labour_cost_value['country'].unique():
        for rec in labour_cost_value['recycling_m'].unique():
            Replace_value = labor_cost_new[(labor_cost_new['ref_area.label'] == cu)][rec].tolist()[0]
            if labour_cost_value.at[i, 'country'] == cu and labour_cost_value.at[i, 'recycling_m'] == rec:
                labour_cost_value.at[i, 'labour'] = Replace_value

other_cost_df = pd.DataFrame()
production_cost_new = nation_df_copy[nation_df_copy['country'].isin(producer_list)].copy()
for index, row in production_cost_new.iterrows():
    if row['country'] not in labour_cost_value['country'].unique():
        if row['continent'] == "Pacific":
            papaa = labour_cost_value[labour_cost_value['country'] == 'Belgium'].iloc[:, :5].copy()
            papab = labor_cost_new[labor_cost_new['ref_area.label'] == row['country']].iloc[:, 1:4].copy()
            papab_melted = pd.melt(papab, var_name='recycling_m', value_name='labour')
            papaa = papaa.merge(papab_melted, on='recycling_m', how='left')
            papaa['country'] = [row['country']] * len(papaa)
            other_cost_df = other_cost_df.append(papaa)
        if row['continent'] == "America":
            papaa = labour_cost_value[labour_cost_value['country'] == 'USA'].iloc[:, :5].copy()
            papab = labor_cost_new[labor_cost_new['ref_area.label'] == row['country']].iloc[:, 1:4].copy()
            papab_melted = pd.melt(papab, var_name='recycling_m', value_name='labour')
            papaa = papaa.merge(papab_melted, on='recycling_m', how='left')
            papaa['country'] = [row['country']] * len(papaa)
            other_cost_df = other_cost_df.append(papaa)
        if row['continent'] == "Asia":
            if row['country'] == 'Japan':
                papaa = labour_cost_value[labour_cost_value['country'] == 'Korea'].iloc[:, :5].copy()
                papab = labor_cost_new[labor_cost_new['ref_area.label'] == row['country']].iloc[:, 1:4].copy()
                papab_melted = pd.melt(papab, var_name='recycling_m', value_name='labour')
                papaa = papaa.merge(papab_melted, on='recycling_m', how='left')
                papaa['country'] = [row['country']] * len(papaa)
                other_cost_df = other_cost_df.append(papaa)
            else:
                papaa = labour_cost_value[labour_cost_value['country'] == 'China'].iloc[:, :5].copy()
                papab = labor_cost_new[labor_cost_new['ref_area.label'] == row['country']].iloc[:, 1:4].copy()
                papab_melted = pd.melt(papab, var_name='recycling_m', value_name='labour')
                papaa = papaa.merge(papab_melted, on='recycling_m', how='left')
                papaa['country'] = [row['country']] * len(papaa)
                other_cost_df = other_cost_df.append(papaa)
        if row['continent'] == "Europe":
            papaa = labour_cost_value[labour_cost_value['country'] == 'United Kingdom'].iloc[:, :5].copy()
            papab = labor_cost_new[labor_cost_new['ref_area.label'] == row['country']].iloc[:, 1:4].copy()
            papab_melted = pd.melt(papab, var_name='recycling_m', value_name='labour')
            papaa = papaa.merge(papab_melted, on='recycling_m', how='left')
            papaa['country'] = [row['country']] * len(papaa)
            other_cost_df = other_cost_df.append(papaa)

labour_cost_value = labour_cost_value.append(other_cost_df)
labour_cost_value['sum'] = labour_cost_value[labour_cost_value.columns[-4:].tolist()].sum(axis=1)

data_grouped = {}
# Loop through each recycling method
for method in unique_recycling_methods:
    # Filter data by recycling method
    data_filtered = labour_cost_value[labour_cost_value['recycling_m'] == method].copy()

    # Check if UK is present in this category, and get its sum
    if 'United Kingdom' in data_filtered['country'].values:
        uk_sum_method = data_filtered[data_filtered['country'] == 'United Kingdom']['sum'].values[0]
    else:
        uk_sum_method = 0  # Default to 0 if UK is not in the group

    # Calculate the difference from UK's sum for each entry
    data_filtered['difference_from_UK'] = data_filtered['sum'] - uk_sum_method

    # Store the modified DataFrame in the dictionary
    data_grouped[method] = data_filtered

# Combine all the dataframes into one
combined_data = pd.concat([data_grouped[method] for method in unique_recycling_methods])

UK_cot_df = pd.read_csv('./cost/default_cot.csv')
combined_data_select = combined_data[['country', 'recycling_m', 'sum']].copy()
pivot_df1 = combined_data_select.pivot(index='country', columns='recycling_m', values='sum')

pivot_df1['country'] = pivot_df1.index.tolist()
pivot_df1.reset_index(drop=True, inplace=True)

UK_cot_df['country'] = ['United Kingdom'] * len(UK_cot_df)
recycl_cap = UK_cot_df['Recycling_capacity'].tolist()

cost_coun_df = pd.DataFrame()
# trans_cun_data = pd.DataFrame()
for cou in combined_data['country'].unique():
    trade_da = pd.DataFrame({
        'Recycling_capacity': recycl_cap,
        'country': [cou] * len(recycl_cap)
    })
    for rec in combined_data['recycling_m'].unique():
        cans = combined_data[(combined_data['country'] == cou) & (combined_data['recycling_m'] == rec)][
            'difference_from_UK'].values
        values_list = trade_da['Recycling_capacity'].iloc[1:4].tolist()
        trade_da[rec] = UK_cot_df[rec] + cans[0]

    cost_coun_df = cost_coun_df.append(trade_da)
cost_coun_df = cost_coun_df.reset_index(drop=True)

# com data
producer_country_df = cost_coun_df.merge(nation_df_new[['country', 'Sub-region Name', 'continent']], on='country',
                                         how='left')

# no-producer country cost
no_producer_country_list = list(set(nation_df_new['country'].tolist()) - set(cost_coun_df['country'].unique()))

grouped_cost_coun_df = producer_country_df.groupby(['Recycling_capacity', 'Sub-region Name']).mean().reset_index()
grouped_cost_coun_df1 = producer_country_df.groupby(['Recycling_capacity', 'continent']).mean().reset_index()
grouped_cost_coun_df2 = producer_country_df.groupby(['Recycling_capacity']).mean().reset_index()

no_producer_country_df = pd.DataFrame()
for np_ in no_producer_country_list:
    aub_name = nation_df_new[nation_df_new['country'] == np_]['Sub-region Name'].tolist()
    con_name = nation_df_new[nation_df_new['country'] == np_]['continent'].tolist()

    choise_data_a = grouped_cost_coun_df[grouped_cost_coun_df['Sub-region Name'] == aub_name[0]].copy()
    choise_data_b = grouped_cost_coun_df1[grouped_cost_coun_df1['continent'] == con_name[0]].copy()
    choise_data_c = grouped_cost_coun_df2.copy()

    if choise_data_a.empty:
        choise_data_d = choise_data_b.copy()
        if 'Sub-region Name' in choise_data_d.columns:
            choise_data_d = choise_data_d.drop(columns=['Sub-region Name'])
        if 'continent' in choise_data_d.columns:
            choise_data_d = choise_data_d.drop(columns=['continent'])

    if choise_data_b.empty | choise_data_a.empty:
        choise_data_d = choise_data_c.copy()
    else:
        choise_data_d = choise_data_a.copy()
        if 'Sub-region Name' in choise_data_d.columns:
            choise_data_d = choise_data_d.drop(columns=['Sub-region Name'])
        if 'continent' in choise_data_d.columns:
            choise_data_d = choise_data_d.drop(columns=['continent'])

    choise_data_d['country'] = [np_] * len(choise_data_d)

    no_producer_country_df = no_producer_country_df.append(choise_data_d)

cost_df = no_producer_country_df.append(producer_country_df.iloc[:, :5]).reset_index(drop=True)


df_cost_fit_data = pd.DataFrame()
for guo in cost_df['country'].unique().tolist():
    chiose_fit_data = cost_df[cost_df['country'] == guo].sort_values(by='Recycling_capacity')
    chiose_fit_data = fun_class.convert_column_to_float(chiose_fit_data, 'Recycling_capacity')
    chiose_fit_data = fun_class.convert_column_to_float(chiose_fit_data, 'Direct')
    chiose_fit_data = fun_class.convert_column_to_float(chiose_fit_data, 'Hydro')
    chiose_fit_data = fun_class.convert_column_to_float(chiose_fit_data, 'Pyro')
    # 对每一对数据进行幂函数拟合
    x_data = chiose_fit_data['Recycling_capacity']
    y_Direct_data = list(chiose_fit_data['Direct'])
    y_Hydro_data = list(chiose_fit_data['Hydro'])
    y_Pyro_data = list(chiose_fit_data['Pyro'])

    params_direct, params_direct_covariance = curve_fit(fun_class.power_func, x_data, y_Direct_data)
    params_hydro, params_hydro_covariance = curve_fit(fun_class.power_func, x_data, y_Hydro_data)
    params_pyro, params_pyro_covariance = curve_fit(fun_class.power_func, x_data, y_Pyro_data)

    canshu = {
        'Direct': list(params_direct),
        'Hydro': list(params_hydro),
        'Pyro': list(params_pyro)
    }

    df_cost_canshu = pd.DataFrame(canshu).T.reset_index()
    df_cost_canshu.columns = ['recycling_m', 'a', 'b']
    df_cost_canshu['country'] = len(df_cost_canshu) * [guo]
    df_cost_fit_data = df_cost_fit_data.append(df_cost_canshu)

ev_battery_selected = evb_data[(evb_data['Year'] <= 2033) & (evb_data['region'].isin(producer_list))][
    ['Year', 'region', evb_data.columns[2]]]

df1 = pd.read_csv('./trade_in_con/850760_Import+new.csv', encoding='ISO-8859-1')
df2 = pd.read_csv('./trade_in_con/870380_Import+new.csv', encoding='ISO-8859-1')
df3 = pd.read_csv('./trade_in_con/870360_Import+new.csv', encoding='ISO-8859-1')
df4 = pd.read_csv('./trade_in_con/850790_Import+new.csv', encoding='ISO-8859-1')

weight_valaue_df = pd.DataFrame()
for year in list(range(year_val, 2034)):
    dfff = fun_class.getDataofWeights(year, df1, df2, df3, df4)
    weight_valaue_df = weight_valaue_df.append(dfff)
weight_valaue_df = weight_valaue_df.reset_index(drop=True)

weight_valaue_df.loc[weight_valaue_df['PartnerDesc'] == "United States of America", 'PartnerDesc'] = 'USA'
weight_valaue_df.loc[weight_valaue_df['PartnerDesc'] == "Vietnam", 'PartnerDesc'] = 'Viet Nam'
weight_valaue_df.loc[weight_valaue_df['PartnerDesc'] == "North Korea", 'PartnerDesc'] = 'Korea'
weight_valaue_df.loc[weight_valaue_df['PartnerDesc'] == "Russia", 'PartnerDesc'] = 'Russian Federation'

weight_valaue_df.loc[weight_valaue_df['ReporterDesc'] == "United States of America", 'ReporterDesc'] = 'USA'
weight_valaue_df.loc[weight_valaue_df['ReporterDesc'] == "Vietnam", 'ReporterDesc'] = 'Viet Nam'
weight_valaue_df.loc[weight_valaue_df['ReporterDesc'] == "North Korea", 'ReporterDesc'] = 'Korea'
weight_valaue_df.loc[weight_valaue_df['ReporterDesc'] == "Russia", 'ReporterDesc'] = 'Russian Federation'

# Convert data from long format to wide format
pivot_df = ev_battery_selected.pivot(index='Year', columns='region', values=evb_data.columns[2])
pivot_df.reset_index(drop=True, inplace=True)
pivot_df['Year'] = ev_battery_selected['Year'].unique().tolist()
pivot_df.loc[:, pivot_df.columns != 'Year'] = pivot_df.loc[:, pivot_df.columns != 'Year'] / 10000
evb_old_data = pivot_df.copy()

# Sorting by 'year' to ensure alignment
evb_old_data.sort_values('Year', inplace=True)
# Setting 'year' as index to facilitate subtraction
evb_old_data.set_index('Year', inplace=True)
pivot_df.to_csv(f'scrap_mass.csv', index=False, sep=',')

#
recycling_cap_data = pd.read_csv('recycling_cap.csv')
recycling_cap_data.sort_values('Year', inplace=True)
recycling_cap_data.set_index('Year', inplace=True)

# CEPII https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6
file_path3 = 'dist_nation.csv'
dist_data_now = pd.read_csv(file_path3)
dist_data = dist_data_now.copy()

# trans cost 0.03 $/ton-mile
dist_data['distcap'] = dist_data['distcap'] * 0.03 / 1e3

coun_list = list(set(nation_df_new['country'].unique()) - set(producer_list))
prod = nation_df_new[nation_df_new['country'].isin(producer_list)]['iso3'].tolist()
no_prod = nation_df_new[~nation_df_new['country'].isin(producer_list)]['iso3'].tolist()

coun_list_chiose1 = dist_data[(dist_data['iso_o'].isin(no_prod)) & (dist_data['iso_d'].isin(prod))]
coun_list_chiose2 = dist_data[(dist_data['iso_o'].isin(prod)) & (dist_data['iso_d'].isin(prod))]

pivot_df4 = fun_class.SplitDistData(coun_list_chiose1)
pivot_df3 = fun_class.SplitDistData(coun_list_chiose2)
result_dist_df = pd.concat([pivot_df4, pivot_df3], ignore_index=False)

recycling_cap_data_copy = recycling_cap_data.copy()
recycling_cap_data_copy['Year'] = recycling_cap_data_copy.index
recycling_cap_data_copy = recycling_cap_data_copy.reset_index(drop=True)
evb_old_data_copy = evb_old_data.copy()
evb_old_data_copy['Year'] = evb_old_data_copy.index
evb_old_data_copy = evb_old_data_copy.reset_index(drop=True)

melt_recycling_cap_df = recycling_cap_data_copy.melt(id_vars='Year', var_name='country', value_name='quantity')
melt_recycling_cap_df = melt_recycling_cap_df[melt_recycling_cap_df['country']!='China'].copy()
melt_evb_old_df = evb_old_data_copy.melt(id_vars='Year', var_name='country', value_name='quantity')
merge_scrap_evb = melt_recycling_cap_df.merge(melt_evb_old_df, on=['Year', 'country'], how='left',
                                              suffixes=('_df1', '_df2'))

pivot_diff_df_cho = pd.DataFrame()
pivot_self_dispos_cho = pd.DataFrame()
# val_list = [2033]
val_list = list(range(year_val, year_val1 + 1))
for year_val in val_list:
    directory = f'./trans/paths/{year_val}'
    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    merge_scrap_evb_cho = merge_scrap_evb[merge_scrap_evb['Year'] == year_val].copy()
    merge_scrap_evb_cho['total_self_dispos_count'] = np.where((merge_scrap_evb_cho['quantity_df1'] == 0.0) |
                                                              (merge_scrap_evb_cho['quantity_df1'] >
                                                               merge_scrap_evb_cho['quantity_df2']),
                                                              merge_scrap_evb_cho['quantity_df2'],
                                                              merge_scrap_evb_cho['quantity_df1'])
    # recycling cap - scrap mass
    Subtracting_df = recycling_cap_data.subtract(evb_old_data).reset_index()
    chiose_da_23 = Subtracting_df[(Subtracting_df['Year'] == year_val)].copy()
    df_chaizhi_cap_scrap = chiose_da_23.copy()
    # pivot_diff_df = df_chaizhi_cap_scrap.pivot(index='Year', columns='region', values='cap_scrap_diff')
    pivot_diff_df = df_chaizhi_cap_scrap.melt(id_vars='Year', var_name='region', value_name='cap_scrap_diff')
    pivot_diff_df.reset_index(drop=True, inplace=True)
    pivot_diff_chain_df = pivot_diff_df[pivot_diff_df['cap_scrap_diff'] < 0].copy()
    pivot_diff_chain_df['cap_scrap_diff'] = pivot_diff_chain_df['cap_scrap_diff'] * -10000
    pivot_diff_df['cap_scrap_diff'] = pivot_diff_df['cap_scrap_diff'] * 10000

    pivot_self_dispos_cho = pivot_self_dispos_cho.append(merge_scrap_evb_cho)
    pivot_diff_df_cho = pivot_diff_df_cho.append(pivot_diff_chain_df)
    # 2023 production weights data
    # weight_valaue_df1 = \
    # weight_valaue_df[(weight_valaue_df['Period'] == year_val) & (weight_valaue_df['Probability'] >= 1)][
    #     ['ReporterDesc', 'PartnerDesc', 'Probability']].copy()
    # chiose_2023_weight, no_recyc_cap, exist_recyc_cap = fun_class.culTransformDta(chiose_da_23, weight_valaue_df1)
    #
    # # 进行运送
    # max_failures1 = 100  # 允许的连续失败次数
    # updated_regions_df1, local_storage1, transport_log1 = fun_class.transport_goods_with_detailed_logging(exist_recyc_cap,
    #                                                                                                       no_recyc_cap,
    #                                                                                                       chiose_2023_weight,
    #                                                                                                       max_failures1)
    # # 创建DataFrame以记录运输日志
    # transport_log_df = pd.DataFrame(transport_log1,
    #                                 columns=['From Region', 'To Region', 'Amount Transported', 'Remaining Storage'])
    #
    # # 合并重复的记录
    # transport_log_df = transport_log_df.groupby(['From Region', 'To Region']).agg({
    #     'Amount Transported': 'sum',
    #     'Remaining Storage': 'last'
    # }).reset_index()
    #
    # # 输出最终各地区的物资量和本地存储的物资量
    # final_regions_df = updated_regions_df1.copy()
    # final_regions_df['Local Storage'] = final_regions_df['Region'].map(local_storage1)
    #
    # file_name = 'transport_log.csv'
    # file_path4 = os.path.join(directory, file_name)
    #
    # transport_log_df.to_csv(file_path4, index=False, sep=',')
    #
    # transport_log_df = pd.read_csv(f'./trans/paths/{year_val}/transport_log.csv')
    # for recycl_proce in df_cost_fit_data['recycling_m'].unique():
    #     no_recyc_cap_group_st_2023, no_recyc_cap_group_en = fun_class.datailTrans(transport_log_df, no_recyc_cap,
    #                                                                               exist_recyc_cap)
    #
    #     exist_recyc_cap_com = exist_recyc_cap.append(no_recyc_cap)
    #
    #     no_recyc_cap_group_st_2023, no_recyc_cap_group_en = fun_class.culRemainingandUndisposed(
    #         no_recyc_cap_group_st_2023,
    #         no_recyc_cap_group_en)
    #
    #     # Supply
    #     # no-producer country data
    #     ev_battery_selected_num = evb_data[
    #                                   (~evb_data['region'].isin(producer_list)) & (evb_data['Year'] == year_val)].iloc[
    #                               :,:3].copy()
    #
    #     ev_battery_selected_num = ev_battery_selected_num.rename(columns={'region': 'country'})
    #     no_recyc_cap_group_st = no_recyc_cap_group_st_2023[no_recyc_cap_group_st_2023['Undisposed_scrap'] != 0][
    #         ['Region', 'Undisposed_scrap']]
    #     no_recyc_cap_group_st = no_recyc_cap_group_st.rename(
    #         columns={'Region': 'country', 'Undisposed_scrap': 'scrap_sum'})
    #     no_recyc_cap_group_st['scrap_sum'] = (no_recyc_cap_group_st['scrap_sum'] * 1e4).apply(np.floor).astype(int)
    #     ev_battery_selected_num = ev_battery_selected_num.append(no_recyc_cap_group_st)
    #
    #     battery_selected_num = ev_battery_selected_num.merge(nation_df_new, on='country', how='left')
    #     battery_selected_num = battery_selected_num[['iso3', 'scrap_sum']].copy()
    #     battery_selected_num.index = battery_selected_num['iso3']
    #     battery_selected_num['scrap_sum'] = battery_selected_num['scrap_sum'].apply(np.floor).astype(int)
    #     battery_selected_num = battery_selected_num.drop('iso3', axis=1)
    #     battery_selected_num = battery_selected_num[battery_selected_num['scrap_sum'] != 0]
    #
    #     # Demand
    #     chois_dat = no_recyc_cap_group_en[no_recyc_cap_group_en['Remaining_capacity'] != 0][
    #         ['Region', 'Remaining_capacity']].copy()
    #     chois_dat = chois_dat.rename(columns={'Remaining_capacity': 'Mass_v'})
    #     chois_dat['Mass_v'] = (chois_dat['Mass_v'] * 1e4).astype(int)
    #     battery_selected_num2 = chois_dat.iloc[:, :2]
    #     battery_selected_num2 = battery_selected_num2.rename(columns={'Region': 'country'})
    #     battery_selected_num2 = battery_selected_num2.merge(nation_df_new, on='country', how='left')
    #     battery_selected_num2 = battery_selected_num2[['iso3', 'Mass_v']].copy()
    #
    #     battery_selected_num2.index = battery_selected_num2['iso3']
    #     battery_selected_num2 = battery_selected_num2.drop('iso3', axis=1)
    #
    #     battery_selected_num = battery_selected_num.drop(index='MNE')
    #
    #     df_cost_fit_data_direct = df_cost_fit_data.merge(nation_df_new, on='country', how='left')
    #
    #     # Differentiate the recycling process
    #     df_cost_fit_data_direct = df_cost_fit_data_direct[df_cost_fit_data_direct['recycling_m'] == recycl_proce]
    #     df_cost_fit_data_direct_new = df_cost_fit_data_direct[['iso3', 'a', 'b']]
    #
    #     df_cost_fit_data_direct_new.index = df_cost_fit_data_direct_new['iso3']
    #     df_cost_fit_data_direct_new = df_cost_fit_data_direct_new.drop('iso3', axis=1)
    #
    #     df_cost_fit_data_direct_new = battery_selected_num2.join(df_cost_fit_data_direct_new, how='left')
    #
    #     df_cost_fit_data_direct_new = df_cost_fit_data_direct_new[['a', 'b']]
    #
    #     pivot_df2 = result_dist_df.loc[battery_selected_num.index.tolist(), battery_selected_num2.index.tolist()]
    #
    #     result_df1 = fun_class.culDistCountryMatrix(carbon_cost_chiose, pivot_df2, battery_selected_num,
    #                                                 battery_selected_num2, df_cost_fit_data_direct_new, year_val,
    #                                                 recycl_proce, 'Strategy 3')
    #
    #     # 导出结果为 CSV 文件
    #     result_df1.to_csv(f'./trans/paths/{year_val}/Strategy 3_{recycl_proce}_optimal_transportation_plan.csv',
    #                       index=True)
    #
    #     # 策略2
    #     exist_recyc_cap_com = exist_recyc_cap_com.rename(columns={'Region': 'country', 'Mass_t': 'scrap_sum'})
    #     exist_recyc_cap_com = exist_recyc_cap_com.merge(nation_df_new, on='country', how='left')
    #     exist_recyc_cap_com = (exist_recyc_cap_com[['iso3', 'Mass_v', 'scrap_sum']]).copy()
    #
    #     exist_recyc_cap_com.index = exist_recyc_cap_com['iso3']
    #     exist_recyc_cap_com = exist_recyc_cap_com.drop('iso3', axis=1)
    #
    #     # Demand
    #     chois_dat = exist_recyc_cap_com[exist_recyc_cap_com['Mass_v'] != 0].copy()
    #
    #     chois_dat['Mass_v'] = (chois_dat['Mass_v'] * 1e4).astype(int)
    #     battery_selected_num3 = chois_dat.iloc[:, :2]
    #     battery_selected_num3 = battery_selected_num3[['Mass_v']].copy()
    #
    #     # Supply
    #     ev_battery_selected_num = ev_battery_selected_num.rename(columns={'region': 'country'})
    #     battery_selected_num4 = ev_battery_selected_num.merge(nation_df_new, on='country', how='left')
    #     battery_selected_num4 = battery_selected_num4[['iso3', 'scrap_sum']].copy()
    #     battery_selected_num4.index = battery_selected_num4['iso3']
    #     battery_selected_num4['scrap_sum'] = battery_selected_num4['scrap_sum'].apply(np.floor).astype(int)
    #     battery_selected_num4 = battery_selected_num4.drop('iso3', axis=1)
    #
    #     chois_buch = exist_recyc_cap_com[exist_recyc_cap_com['scrap_sum'] != 0].copy()
    #
    #     chois_buch['scrap_sum'] = (chois_buch['scrap_sum'] * 1e4).astype(int)
    #     battery_selected_buch = chois_buch.iloc[:, :2]
    #     battery_selected_buch = battery_selected_buch[['scrap_sum']].copy()
    #
    #     battery_selected_num4 = pd.concat([battery_selected_num4, battery_selected_buch], ignore_index=False)
    #
    #     battery_selected_num4 = battery_selected_num4.drop(index='MNE')
    #
    #     battery_selected_num4 = battery_selected_num4[battery_selected_num4['scrap_sum'] != 0]
    #
    #     pivot_df5 = result_dist_df.loc[battery_selected_num4.index.tolist(), battery_selected_num3.index.tolist()]
    #
    #     df_cost_fit_data_direct = df_cost_fit_data_direct[df_cost_fit_data_direct['recycling_m'] == recycl_proce]
    #     df_cost_fit_data_direct_new = df_cost_fit_data_direct[['iso3', 'a', 'b']]
    #
    #     df_cost_fit_data_direct_new.index = df_cost_fit_data_direct_new['iso3']
    #     df_cost_fit_data_direct_new = df_cost_fit_data_direct_new.drop('iso3', axis=1)
    #
    #     df_cost_fit_data_direct_new = battery_selected_num3.join(df_cost_fit_data_direct_new, how='left')
    #
    #     df_cost_fit_data_direct_new = df_cost_fit_data_direct_new[['a', 'b']]
    #
    #     result_df = fun_class.culDistCountryMatrix(carbon_cost_chiose, pivot_df5, battery_selected_num4,
    #                                                battery_selected_num3, df_cost_fit_data_direct_new, year_val,
    #                                                recycl_proce, 'Strategy 2')
    #     # 导出结果为 CSV 文件
    #     result_df.to_csv(f'./trans/paths/{year_val}/Strategy 2_{recycl_proce}_optimal_transportation_plan.csv',
    #                      index=True)

# %% OUTPUT Net Profit of Strategy 2 & 3
# year_val, recycl_proce, Strategy_type = 2023, 'Pyro', 'Strategy 3'
CO2_em_df = pd.DataFrame()
recycl_list = ['Pyro', 'Hydro', 'Direct']

for recycl_proce in recycl_list:
    disposal_mass_diff_stra_df1 = pd.DataFrame()
    CO2_em_disposal_mass_diff_stra = pd.DataFrame()
    for year_val in val_list:
        # Create the directory if it does not exist
        os.makedirs(f'./trans/result/{year_val}', exist_ok=True)
        for Strategy_type in ['Strategy 2', 'Strategy 3']:
            path_file = f'./trans/paths/{year_val}/{Strategy_type}_{recycl_proce}_optimal_transportation_plan.csv'
            pro_tran_vol_pyro_tot = pd.read_csv(path_file, index_col=0)
            pro_tran_vol_pyro_tot['source_point'] = pro_tran_vol_pyro_tot.index.tolist()

            # Converting the wide format back to long format
            original_df = pro_tran_vol_pyro_tot.melt(id_vars='source_point', var_name='dest_point', value_name='quantity')
            original_df = original_df[original_df['quantity'] != 0]
            nation_code_per = original_df['dest_point'].unique()
            nation_code_opt = original_df['source_point'].unique()

            original_df_cho = pd.DataFrame()
            original_df_cho1 = pd.DataFrame()
            if 'Virtual_Supply' in nation_code_opt:
                original_df_cho = original_df[original_df['source_point'] == 'Virtual_Supply']
                original_df = original_df[original_df['source_point'] != 'Virtual_Supply']

            elif 'Virtual_Demand' in nation_code_per:
                original_df_cho = original_df[original_df['dest_point'] == 'Virtual_Demand']
                original_df = original_df[original_df['dest_point'] != 'Virtual_Demand']

                original_df_cho1 = original_df_cho.copy()
                original_df_cho1['Year'] = [year_val] * len(original_df_cho1)
                original_df_cho1['Strategy type'] = [Strategy_type] * len(original_df_cho1)

            if 'Strategy 3' in path_file:
                pro_tran_vol = pd.read_csv(f'./trans/paths/{year_val}/transport_log.csv')
                pro_tran_vol['Amount Transported'] = pro_tran_vol['Amount Transported'] * 10000
                pro_tran_vol = pro_tran_vol.iloc[:, :3].rename(
                    columns={'From Region': 'source_point', 'To Region': 'dest_point', 'Amount Transported': 'quantity'})
                nation_df_dict = nation_df_new.set_index('country')['iso3'].to_dict()
                pro_tran_vol['dest_point'] = pro_tran_vol['dest_point'].replace(nation_df_dict)
                pro_tran_vol['source_point'] = pro_tran_vol['source_point'].replace(nation_df_dict)

                original_df = original_df.append(pro_tran_vol)

            ev_battery_type_selected = \
            evb_type_data[(evb_type_data['Year'] == year_val) & (~evb_type_data['type'].isin(['TLB']))][
                ['region', evb_type_data.columns[2], evb_type_data.columns[3]]].copy()
            prod_scrap_type = ev_battery_type_selected.copy()
            total_export_weight = prod_scrap_type.groupby('region')['scrap_sum'].sum().rename('Totalscrap')

            # Merge the total export weight back into the modified data
            prod_scrap_type = prod_scrap_type.merge(total_export_weight, on='region')

            # Transshipment volume (Tons)
            s_point = original_df['source_point'].unique()
            d_point = original_df['dest_point'].unique()
            # Straight-line distance between two countries (nautical miles)
            coun_list_chio = dist_data_now[(dist_data_now['iso_o'].isin(s_point)) & (dist_data_now['iso_d'].isin(d_point))][['iso_o', 'iso_d', 'distcap']].copy()
            coun_list_chio = coun_list_chio.rename(columns={'iso_o': 'source_point', 'iso_d': 'dest_point'})
            original_df_merge = original_df.merge(coun_list_chio, on=['source_point', 'dest_point'], how='left')
            # CO2 emissions from transport 0.017kg CO2/ ton-mile
            original_df_merge['CO2'] = original_df_merge['distcap'] * original_df_merge['quantity'] * 0.017
            original_df_merge['year'] = [year_val] * len(original_df_merge)
            original_df_merge['Strategy type'] = [Strategy_type] * len(original_df_merge)
            # Unit: kg CO2/ ton-mile
            CO2_em_df = CO2_em_df.append(original_df_merge)

            # Calculate the Probability as the percentage of the total export weight
            prod_scrap_type['weights'] = (prod_scrap_type['scrap_sum'] / prod_scrap_type['Totalscrap'])
            prod_scrap_type_chiose = prod_scrap_type[['region', 'type', 'weights']].copy().rename(
                columns={'region': 'country'})
            prod_scrap_type_chiose = prod_scrap_type_chiose.merge(nation_df_new[['iso3', 'country']], on='country',
                                                                  how='left')

            pivot_self_dispos_cho1 = pivot_self_dispos_cho[pivot_self_dispos_cho['Year'] == year_val][['country', 'total_self_dispos_count']].copy()
            merge_self_dispos_df = pivot_self_dispos_cho1.merge(prod_scrap_type_chiose, on=['country'], how='left')
            merge_self_dispos_df['self_dispos_count'] = merge_self_dispos_df['total_self_dispos_count'] * merge_self_dispos_df['weights'] * 10000

            routes_original_df = fun_class.culWeightAddScrap(original_df, prod_scrap_type_chiose)
            ev_battery_type_selected = ev_battery_type_selected.rename(columns={'region': 'country'})
            ev_battery_type_selected_chios = \
            ev_battery_type_selected.merge(nation_df_new[['iso3', 'country']], on='country', how='left')[
                ['iso3', 'country', 'type', 'scrap_sum']].copy()
            ev_battery_type_selected_chios_copy = ev_battery_type_selected_chios.copy()

            # total trans cost of product country
            dist_cost = fun_class.culConsolidatedDisposalVolume(original_df, result_dist_df)
            dist_cost = dist_cost.rename(columns={'dest_point': 'iso3'})
            dist_cost = dist_cost.merge(nation_df_new[['iso3', 'country']], on='iso3', how='left')

            merged_scrap_df = pd.merge(merge_self_dispos_df, routes_original_df, on=['iso3', 'type'], how='left')
            merged_scrap_df['scrap_sum'] = merged_scrap_df['self_dispos_count'] + merged_scrap_df['scrap_sum'].fillna(0)
            merged_scrap_df = merged_scrap_df[['iso3', 'type', 'scrap_sum']]

            merged_scrap_df_merge = merged_scrap_df.merge(nation_df_new[['iso3', 'country']], on='iso3', how='left')

            merged_scrap_df_merge_new = pd.DataFrame()
            if 'Virtual_Demand' in nation_code_per:
                original_df_cho = original_df_cho.rename(columns={'source_point': 'iso3'})
                original_df_cho = original_df_cho.merge(nation_df_new[['iso3', 'country']], on='iso3', how='left')
                # 需本国处置的国家
                routes_original_df1 = fun_class.culWeightAddScrap1(original_df_cho, prod_scrap_type_chiose)
                routes_original_df1 = routes_original_df1.merge(nation_df_new[['iso3', 'country']], on='iso3',
                                                                how='left')
                merged_scrap_df_merge_new = pd.concat(
                    [routes_original_df1, merged_scrap_df_merge], ignore_index=True)
            else:
                merged_scrap_df_merge_new = merged_scrap_df_merge.copy()

            merged_scrap_df_merge_new['recycling_m'] = [recycl_proce] * len(merged_scrap_df_merge_new)

            ev_battery_type_selected_chios = ev_battery_type_selected_chios.groupby(['country', 'type'])[
                                                 'scrap_sum'].max().reset_index().iloc[:, :2]

            disposal_mass_diff_stra_df = pd.DataFrame()
            disposal_mass_diff_stra_df = disposal_mass_diff_stra_df.append(
                merged_scrap_df_merge_new[['country', 'type', 'scrap_sum']])
            disposal_mass_diff_stra_df['year'] = [year_val] * len(disposal_mass_diff_stra_df)
            disposal_mass_diff_stra_df['Strategy type'] = [Strategy_type] * len(disposal_mass_diff_stra_df)

            recycling_emission_df = recycling_emission_count_chiose[recycling_emission_count_chiose['recycling_m'] == recycl_proce]
            recycling_emission_df = recycling_emission_df.rename(columns={'battery_type': 'type'})
            merged_scrap_recycling_emission = disposal_mass_diff_stra_df.merge(recycling_emission_df, on='type', how='left')
            merged_scrap_recycling_emission['CO2_em'] = merged_scrap_recycling_emission['CO2'] * \
                                                        merged_scrap_recycling_emission['scrap_sum']

            net_profit_strat1_chiose2, net_profit_strat = fun_class.OUTPUTNetProfitofStrategyTwoThree(
                merged_scrap_df_merge_new, Recycling_revenue_df, carbon_cost, df_cost_fit_data, dist_cost)

            colums = net_profit_strat1_chiose2.columns.tolist()
            net_profit_strat1_chiose2 = ev_battery_type_selected_chios[colums[:2]].merge(net_profit_strat1_chiose2,
                                                                                         on=['country', 'type'],
                                                                                         how='left')
            net_profit_strat.fillna(0, inplace=True)
            net_profit_strat1_chiose2.to_csv(f'./trans/result/{year_val}/net_profit_{recycl_proce}_{Strategy_type}.csv', index=True)

            net_profit_strat['total_netprofits'] = net_profit_strat['scrap_sum'] * net_profit_strat['net_profit'] * 1000
            net_profit_strat_country = net_profit_strat.groupby(['country', 'recycling_m'])[
                'total_netprofits'].sum().reset_index()

            net_profit_strat_country.to_csv(f'./trans/result/{year_val}/country_net_profit_{recycl_proce}_{Strategy_type}.csv', index=True)

            net_profit_strat['total_netprofits'] = net_profit_strat['scrap_sum'] * net_profit_strat[
                'net_profit'] * 1000
            net_profit_strat_country = net_profit_strat.groupby('country')[
                ['total_netprofits', 'scrap_sum']].sum().reset_index()
            net_profit_strat_country['net_profit'] = net_profit_strat_country['total_netprofits'] / \
                                                     net_profit_strat_country['scrap_sum'] / 1000

            net_profit_strat['total_costs'] = net_profit_strat['scrap_sum'] * net_profit_strat['unit_cost'] * 1000
            cost_strat_country = net_profit_strat.groupby('country')[
                ['total_costs', 'scrap_sum']].sum().reset_index()
            cost_strat_country['unit_cost'] = cost_strat_country['total_costs'] / cost_strat_country[
                'scrap_sum'] / 1000

            cost_strat_country.to_csv(f'./trans/result/{year_val}/country_cost_{recycl_proce}_{Strategy_type}.csv', index=True)

            net_profit_strat_country.to_csv(f'./trans/result/{year_val}/country_net_profit_{recycl_proce}_{Strategy_type}.csv', index=True)
            disposal_mass_diff_stra_df1 = disposal_mass_diff_stra_df1.append(disposal_mass_diff_stra_df)
            CO2_em_disposal_mass_diff_stra = CO2_em_disposal_mass_diff_stra.append(merged_scrap_recycling_emission)

        # OUTPUT Net Profit of Strategy 1
        disposal_mass_diff_stra_df = disposal_mass_diff_stra_df.append(
            ev_battery_type_selected[['country', 'type', 'scrap_sum']])
        disposal_mass_diff_stra_df['year'] = [year_val] * len(disposal_mass_diff_stra_df)
        disposal_mass_diff_stra_df['Strategy type'] = ['Strategy 1'] * len(disposal_mass_diff_stra_df)

        recycling_emission_df = recycling_emission_count_chiose[
            recycling_emission_count_chiose['recycling_m'] == recycl_proce]
        recycling_emission_df = recycling_emission_df.rename(columns={'battery_type': 'type'})
        merged_scrap_recycling_emission = disposal_mass_diff_stra_df.merge(recycling_emission_df, on='type', how='left')
        merged_scrap_recycling_emission['CO2_em'] = merged_scrap_recycling_emission['CO2'] * \
                                                    merged_scrap_recycling_emission['scrap_sum']

        pivot_diff_df_cho_copy = pivot_diff_df_cho[pivot_diff_df_cho['Year'] == year_val].copy()
        pivot_diff_df_cho_copy = pivot_diff_df_cho_copy.rename(columns={'region': 'country'})
        pivot_diff_df_cho_copy = prod_scrap_type_chiose.merge(pivot_diff_df_cho_copy, on='country', how='left').dropna()
        pivot_diff_df_cho_copy['Expanding production'] = pivot_diff_df_cho_copy['cap_scrap_diff'] * pivot_diff_df_cho_copy['weights']
        pivot_diff_df_cho_copy = pivot_diff_df_cho_copy[['iso3', 'type', 'country', 'Expanding production']].copy()

        net_profit_strat1_chiose1, net_profit_strat1 = fun_class.OUTPUTNetProfitofStrategyOne(ev_battery_type_selected,
                                                                                             pivot_diff_df_cho_copy,
                                                                                             Recycling_revenue_df,
                                                                                             carbon_cost, df_cost_fit_data,
                                                                                             dist_cost, recycl_proce)
        net_profit_strat1_chiose1.to_csv(f'./trans/result/{year_val}/net_profit_{recycl_proce}_Strategy 1.csv', index=True)

        net_profit_strat1['total_netprofits'] = net_profit_strat1['scrap_sum'] * net_profit_strat1['net_profit'] * 1000
        net_profit_strat_country = net_profit_strat1.groupby('country')[
            ['total_netprofits', 'scrap_sum']].sum().reset_index()
        net_profit_strat_country['net_profit'] = net_profit_strat_country['total_netprofits'] / \
                                                 net_profit_strat_country['scrap_sum'] / 1000

        net_profit_strat1['total_costs'] = net_profit_strat1['scrap_sum'] * net_profit_strat1['unit_cost'] * 1000
        cost_strat_country = net_profit_strat1.groupby('country')[['total_costs', 'scrap_sum']].sum().reset_index()
        cost_strat_country['unit_cost'] = cost_strat_country['total_costs'] / cost_strat_country['scrap_sum'] / 1000

        cost_strat_country.to_csv(f'./trans/result/{year_val}/country_cost_{recycl_proce}_Strategy 1.csv', index=True)
        net_profit_strat_country.to_csv(f'./trans/result/{year_val}/country_net_profit_{recycl_proce}_Strategy 1.csv', index=True)
        disposal_mass_diff_stra_df1 = disposal_mass_diff_stra_df1.append(disposal_mass_diff_stra_df)
        CO2_em_disposal_mass_diff_stra = CO2_em_disposal_mass_diff_stra.append(merged_scrap_recycling_emission)

    CO2_em_disposal_mass_diff_stra = CO2_em_disposal_mass_diff_stra.drop_duplicates(
        subset=['country', 'type', 'recycling_m', 'year', 'Strategy type'], keep='first')
    CO2_em_disposal_mass_diff_stra = CO2_em_disposal_mass_diff_stra.reset_index(drop=True)

    # disposal_mass_diff_stra_df1 = disposal_mass_diff_stra_df1.drop_duplicates(
    #     subset=['country', 'type', 'recycling_m', 'year', 'Strategy type'], keep='first')
    disposal_mass_diff_stra_df1 = disposal_mass_diff_stra_df1.reset_index(drop=True)

    disposal_mass_diff_stra_df1.to_csv(f'./trans/result/country_disposal_mass_diff_stra_{recycl_proce}.csv', index=True)
    CO2_em_df.to_csv(f'./trans/result/CO2_em_trans_{recycl_proce}.csv', index=True)
    CO2_em_disposal_mass_diff_stra.to_csv(f'./trans/result/CO2_em_disposal_mass_diff_stra_{recycl_proce}.csv', index=True)

