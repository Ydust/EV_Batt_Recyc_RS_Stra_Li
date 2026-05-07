import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import geopandas as gpd
from math import pi
import fun_class

# %% Load the data
mingan_list = np.linspace(2.2, 3.0, num=9)
for mingan in mingan_list:
    file_path1 = f'D:/文档/Renewable_Lithium_Resources0824/Scenario result/BAU scenario/{mingan:.1f}/EV_battery_inuse_scrap.csv'
    file_path3 = f'D:/文档/Renewable_Lithium_Resources0824/Scenario result/BAU scenario/{mingan:.1f}/EV_battery_inuse_scrap_total.csv'
    evb_type_data = pd.read_csv(file_path1)
    evb_data = pd.read_csv(file_path3)
    evb_type_data['scrap_sum'] = evb_type_data['scrap_sum'] / 1e3
    evb_type_data['in-use'] = evb_type_data['in-use'] / 1e3
    evb_data['inuse_sum'] = evb_data['inuse_sum'] / 1e3
    evb_data['scrap_sum'] = evb_data['scrap_sum'] / 1e3
    # %%
    file_name = '850760_Import'
    # 文件路径
    file_path = f'./trade_in_con/{file_name}.csv'

    # 尝试使用不同的编码读取CSV文件
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    # 指定要标记的国家列表
    countries_to_highlight = ['India', 'Japan', 'North Korea', 'Thailand', 'Vietnam', 'France', 'Germany',
                              'Hungary', 'Italy', 'Poland', 'Slovakia', 'Spain', 'Sweden', 'United Kingdom', 'Norway',
                              'Russia', 'Turkey', 'United States of America', 'Canada', 'Brazil', 'Australia', 'Czechia',
                              'Romania', 'Serbia', 'Belgium', 'Netherlands', 'Switzerland', 'Finland']

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.sort_values(by='name', inplace=True)

    # 选择指定的列
    selected_columns = ["Period", "ReporterISO", "ReporterDesc", "FlowDesc", "PartnerISO", "PartnerDesc", "CmdCode",
                        "NetWgt", "PrimaryValue"]
    selected_data = df[selected_columns].copy()

    # 获取所有唯一的ReporterDesc和PartnerDesc值
    unique_reporter_desc = set(selected_data['ReporterDesc'].unique())
    unique_partner_desc = set(selected_data['PartnerDesc'].unique())

    # 检查每个国家是否存在于这两列中
    missing_in_reporter = [country for country in countries_to_highlight if country not in unique_reporter_desc]
    missing_in_partner = [country for country in countries_to_highlight if country not in unique_partner_desc]

    # 清洗数据：替换'T黵kiye'为'Turkey'
    selected_data.loc[selected_data['ReporterDesc'] == "Türkiye", 'ReporterDesc'] = 'Turkey'
    selected_data.loc[selected_data['ReporterDesc'] == "USA", 'ReporterDesc'] = 'United States of America'
    selected_data.loc[selected_data['ReporterDesc'] == "Viet Nam", 'ReporterDesc'] = 'Vietnam'
    selected_data.loc[selected_data['ReporterDesc'] == "Russian Federation", 'ReporterDesc'] = 'Russia'
    selected_data.loc[selected_data['ReporterDesc'] == "Rep. of Korea", 'ReporterDesc'] = 'North Korea'
    selected_data.loc[selected_data['PartnerDesc'] == "Türkiye", 'PartnerDesc'] = 'Turkey'
    selected_data.loc[selected_data['PartnerDesc'] == "Viet Nam", 'PartnerDesc'] = 'Vietnam'
    selected_data.loc[selected_data['PartnerDesc'] == "Russian Federation", 'PartnerDesc'] = 'Russia'
    selected_data.loc[selected_data['PartnerDesc'] == "Rep. of Korea", 'PartnerDesc'] = 'North Korea'
    selected_data.loc[selected_data['PartnerDesc'] == 'USA', 'PartnerDesc'] = 'United States of America'

    # 筛选数据，只保留ReporterDesc或PartnerDesc列中包含指定国家的数据
    filtered_data = selected_data[
        (selected_data['ReporterDesc'].isin(countries_to_highlight)) &
        (selected_data['PartnerDesc'].isin(countries_to_highlight))
        ]
    # 获取所有唯一的ReporterDesc和PartnerDesc值
    unique_reporter_desc = set(filtered_data['ReporterDesc'].unique())
    unique_partner_desc = set(filtered_data['PartnerDesc'].unique())

    # Apply the function to modify the NetWgt column
    modified_data = fun_class.modify_netwgt(filtered_data.copy())

    # Remove rows where PartnerDesc and ReporterDesc are the same
    filtered_append_data = modified_data[modified_data['PartnerDesc'] != modified_data['ReporterDesc']]

    filtered_append_data = filtered_append_data[filtered_append_data['NetWgt'] >= 100]

    export_list = filtered_append_data['PartnerDesc'].unique()
    colums_name = filtered_append_data.columns.tolist()
    sorted_filtered_data = filtered_append_data.reset_index(drop=True)

    for va in export_list:
        qu_df = sorted_filtered_data[sorted_filtered_data['PartnerDesc'] == va].copy()
        import_list = qu_df['ReporterDesc'].unique()
        for bn in import_list:
            cul_df = qu_df[qu_df['ReporterDesc'] == bn].copy()
            cul_df = cul_df.sort_values(by='Period')
            x = np.array(cul_df[colums_name[0]].tolist())
            y = np.array(cul_df[colums_name[7]].tolist())
            y1 = np.array(cul_df[colums_name[8]].tolist())
            t = 0
            max_year = cul_df['Period'].max()
            if max_year == 2023:
                t = max_year + 10
            else:
                max_num = 2023 - max_year
                t = max_year + 10 + max_num
            future_x = np.arange(max_year + 1, t + 1)
            if len(cul_df) == 1:
                temporary_data = pd.DataFrame({
                    colums_name[0]: list(future_x),
                    colums_name[1]: [cul_df.iloc[0][colums_name[1]]] * len(list(future_x)),
                    colums_name[2]: [cul_df.iloc[0][colums_name[2]]] * len(list(future_x)),
                    colums_name[3]: [cul_df.iloc[0][colums_name[3]]] * len(list(future_x)),
                    colums_name[4]: [cul_df.iloc[0][colums_name[4]]] * len(list(future_x)),
                    colums_name[5]: [cul_df.iloc[0][colums_name[5]]] * len(list(future_x)),
                    colums_name[6]: [cul_df.iloc[0][colums_name[6]]] * len(list(future_x)),
                    colums_name[7]: [cul_df.iloc[0][colums_name[7]]] * len(list(future_x)),
                    colums_name[8]: [cul_df.iloc[0][colums_name[8]]] * len(list(future_x))
                })
                sorted_filtered_data = sorted_filtered_data.append(temporary_data)
            else:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x, y1)

                future_y = slope * future_x + intercept
                future_y1 = slope1 * future_x + intercept1

                first_row_data = df.iloc[0].to_dict()
                temporary_data = pd.DataFrame({
                    colums_name[0]: list(future_x),
                    colums_name[1]: [cul_df.iloc[0][colums_name[1]]] * len(list(future_x)),
                    colums_name[2]: [cul_df.iloc[0][colums_name[2]]] * len(list(future_x)),
                    colums_name[3]: [cul_df.iloc[0][colums_name[3]]] * len(list(future_x)),
                    colums_name[4]: [cul_df.iloc[0][colums_name[4]]] * len(list(future_x)),
                    colums_name[5]: [cul_df.iloc[0][colums_name[5]]] * len(list(future_x)),
                    colums_name[6]: [cul_df.iloc[0][colums_name[6]]] * len(list(future_x)),
                    colums_name[7]: list(future_y),
                    colums_name[8]: list(future_y1)
                })
                sorted_filtered_data = sorted_filtered_data.append(temporary_data)

    sorted_filtered_data = sorted_filtered_data.applymap(lambda c: 0 if isinstance(c, (int, float)) and c < 0 else c)
    sorted_filtered_data = sorted_filtered_data.sort_values(by=['ReporterDesc', 'PartnerDesc'])

    # append_data.to_csv(f'./trade_in_con/{file_name}+new.csv', index=False, sep=',')


    # %%
    year_val = 2023
    year_val1 = 2033

    df1 = pd.read_csv('./trade_in_con/850760_Import+new.csv', encoding='ISO-8859-1')
    df2 = pd.read_csv('./trade_in_con/870380_Import+new.csv', encoding='ISO-8859-1')
    df3 = pd.read_csv('./trade_in_con/870360_Import+new.csv', encoding='ISO-8859-1')
    df4 = pd.read_csv('./trade_in_con/850790_Import+new.csv', encoding='ISO-8859-1')

    dfff = fun_class.getDataofWeights(year_val, df1, df2, df3, df4)
    dfff1 = fun_class.getDataofWeights(year_val1, df1, df2, df3, df4)

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
    result_com.loc[len(result_com)] = ['China', '', '', '', 'Currency: U.S. dollars', year_val, 3]
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


    def power_func(x1, a, b):
        return a * np.power(x1, b)


    def convert_column_to_float(df11, column_name):
        df11[column_name] = df11[column_name].astype(float)
        return df11


    df_cost_fit_data = pd.DataFrame()
    for guo in cost_df['country'].unique().tolist():
        chiose_fit_data = cost_df[cost_df['country'] == guo].sort_values(by='Recycling_capacity')
        chiose_fit_data = convert_column_to_float(chiose_fit_data, 'Recycling_capacity')
        chiose_fit_data = convert_column_to_float(chiose_fit_data, 'Direct')
        chiose_fit_data = convert_column_to_float(chiose_fit_data, 'Hydro')
        chiose_fit_data = convert_column_to_float(chiose_fit_data, 'Pyro')
        # 对每一对数据进行幂函数拟合
        x_data = chiose_fit_data['Recycling_capacity']
        y_Direct_data = list(chiose_fit_data['Direct'])
        y_Hydro_data = list(chiose_fit_data['Hydro'])
        y_Pyro_data = list(chiose_fit_data['Pyro'])

        params_direct, params_direct_covariance = curve_fit(power_func, x_data, y_Direct_data)
        params_hydro, params_hydro_covariance = curve_fit(power_func, x_data, y_Hydro_data)
        params_pyro, params_pyro_covariance = curve_fit(power_func, x_data, y_Pyro_data)

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

    weight_valaue_df = dfff.append(dfff1)
    weight_valaue_df.loc[weight_valaue_df['PartnerDesc'] == "United States of America", 'PartnerDesc'] = 'USA'
    weight_valaue_df.loc[weight_valaue_df['PartnerDesc'] == "Vietnam", 'PartnerDesc'] = 'Viet Nam'
    weight_valaue_df.loc[weight_valaue_df['PartnerDesc'] == "North Korea", 'PartnerDesc'] = 'Korea'
    weight_valaue_df.loc[weight_valaue_df['PartnerDesc'] == "Russia", 'PartnerDesc'] = 'Russian Federation'

    weight_valaue_df.loc[weight_valaue_df['ReporterDesc'] == "United States of America", 'ReporterDesc'] = 'USA'
    weight_valaue_df.loc[weight_valaue_df['ReporterDesc'] == "Vietnam", 'ReporterDesc'] = 'Viet Nam'
    weight_valaue_df.loc[weight_valaue_df['ReporterDesc'] == "North Korea", 'ReporterDesc'] = 'Korea'
    weight_valaue_df.loc[weight_valaue_df['ReporterDesc'] == "Russia", 'ReporterDesc'] = 'Russian Federation'

    # 2023 production weights data
    weight_valaue_df1 = weight_valaue_df[(weight_valaue_df['Period'] == year_val) & (weight_valaue_df['Probability'] >= 1)][
        ['ReporterDesc', 'PartnerDesc', 'Probability']].copy()

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
    # pivot_df.to_csv(f'scrap_mass.csv', index=False, sep=',')

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

    pivot_diff_df_cho = pd.DataFrame()
    val_list = [2023, 2033]
    for year_val in val_list:
        # recycling cap - scrap mass
        Subtracting_df = recycling_cap_data.subtract(evb_old_data).reset_index()
        chiose_da_23 = Subtracting_df[(Subtracting_df['Year'] == year_val)].copy()
        df_chaizhi_cap_scrap = chiose_da_23.copy()
        pivot_diff_df = df_chaizhi_cap_scrap.melt(id_vars='Year', var_name='region', value_name='cap_scrap_diff')
        pivot_diff_df.reset_index(drop=True, inplace=True)
        pivot_diff_chain_df = pivot_diff_df[pivot_diff_df['cap_scrap_diff'] < 0].copy()
        pivot_diff_chain_df['cap_scrap_diff'] = pivot_diff_chain_df['cap_scrap_diff'] * -10000

        pivot_diff_df_cho = pivot_diff_df_cho.append(pivot_diff_chain_df)
        chiose_2023_weight, no_recyc_cap, exist_recyc_cap = fun_class.culTransformDta(chiose_da_23, weight_valaue_df1)

        if chiose_2023_weight is not None:
            # 进行运送
            max_failures1 = 100  # 允许的连续失败次数
            updated_regions_df1, local_storage1, transport_log1 = fun_class.transport_goods_with_detailed_logging(exist_recyc_cap,
                                                                                                                  no_recyc_cap,
                                                                                                                  chiose_2023_weight,
                                                                                                                  max_failures1)
            # 创建DataFrame以记录运输日志
            transport_log_df = pd.DataFrame(transport_log1,
                                            columns=['From Region', 'To Region', 'Amount Transported', 'Remaining Storage'])

            # 合并重复的记录
            transport_log_df = transport_log_df.groupby(['From Region', 'To Region']).agg({
                'Amount Transported': 'sum',
                'Remaining Storage': 'last'
            }).reset_index()

            # 输出最终各地区的物资量和本地存储的物资量
            final_regions_df = updated_regions_df1.copy()
            final_regions_df['Local Storage'] = final_regions_df['Region'].map(local_storage1)

            transport_log_df.to_csv(f'./trans/paths/{mingan:.1f}/transport_log_{year_val}.csv', index=False, sep=',')
        else:
            print('Unable to find a solution')