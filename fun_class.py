import random
import pandas as pd
import itertools
import numpy as np
from scipy.optimize import linprog
from itertools import zip_longest


def power_func(x1, a, b):
    return a * np.power(x1, b)


def convert_column_to_float(df11, column_name):
    df11[column_name] = df11[column_name].astype(float)
    return df11


# 模拟策略三 生产国之间的运送过程
def transport_goods_with_detailed_logging(exist_recyc, no_recyc, probabilities_df, max_failures):
    regions_df = exist_recyc.append(no_recyc)
    regions_df = regions_df.rename(columns={'Mass_v': 'Storage_Limit', 'Mass_t': 'Supply'})

    probabilities_df = probabilities_df.rename(columns={'ReporterDesc': 'From', 'PartnerDesc': 'To'})
    from_regions = probabilities_df['From'].unique()
    failure_count = 0
    local_storage = {region: 0 for region in from_regions}
    transport_log = []

    while True:
        if failure_count >= max_failures:
            print(f"Simulation stopped due to {failure_count} consecutive failures.")
            break

        from_region = random.choice(from_regions)
        available_probabilities = probabilities_df[probabilities_df['From'] == from_region]
        to_regions = available_probabilities['To'].tolist()
        to_region_probabilities = available_probabilities['Probability'].tolist()

        transported = False

        # 按概率大小选择目标地区
        to_regions_weights = sorted(zip(to_regions, to_region_probabilities), key=lambda x: x[1], reverse=True)

        for to_region, weight in to_regions_weights:
            # 运输物资量固定为1
            amount = 1 * 10 ** -3

            from_region_data = regions_df[regions_df['Region'] == from_region].iloc[0]
            to_region_data = regions_df[regions_df['Region'] == to_region].iloc[0]

            # 检查起始地区是否有足够的物资，且目标地区有存储空间
            if from_region_data['Supply'] >= amount and to_region_data['Storage_Limit'] > 0:
                if to_region_data['Supply'] + amount <= to_region_data['Storage_Limit']:
                    regions_df.loc[regions_df['Region'] == from_region, 'Supply'] -= amount
                    regions_df.loc[regions_df['Region'] == to_region, 'Supply'] += amount
                    remaining_storage = to_region_data['Storage_Limit'] - to_region_data['Supply'] - amount
                    transport_log.append((from_region, to_region, amount, remaining_storage))
                    # print(f"Transported {amount} unit from {from_region} to {to_region}.")
                    failure_count = 0  # 重置失败计数
                    transported = True

                    # 如果所有目标地区均达到存储上限，退出循环
                    if all(regions_df[regions_df['Storage_Limit'] > 0]['Supply'] ==
                           regions_df[regions_df['Storage_Limit'] > 0]['Storage_Limit']):
                        return regions_df, local_storage, transport_log
                    # if all(regions[region]['supply'] == regions[region]['storage_limit'] for region in regions if regions[region]['storage_limit'] > 0):
                    #     return regions, local_storage, transport_log
                    break

        if not transported:
            # print(f"Transport from {from_region} failed due to insufficient supply or all targets having zero storage limit.")
            local_storage[from_region] += amount
            failure_count += 1  # 增加失败计数

    return regions_df, local_storage, transport_log


def remove_element_by_key(dictionary, key):
    if key in dictionary:
        del dictionary[key]
    return dictionary


def culComdata(filtered_df):
    append_data = pd.DataFrame()
    for sn in filtered_df['Period'].unique():
        uni_data = filtered_df[filtered_df['Period'] == sn].copy()
        # Calculate the total export weight for each PartnerDesc
        total_export_weight = uni_data.groupby('ReporterDesc')['NetWgt'].sum().rename('TotalExportWeight')

        # Merge the total export weight back into the modified data
        uni_data = uni_data.merge(total_export_weight, on='ReporterDesc')

        # Calculate the Probability as the percentage of the total export weight
        uni_data['Probability'] = (uni_data['NetWgt'] / uni_data['TotalExportWeight']) * 100
        append_data = append_data.append(uni_data)
    # Drop the TotalExportWeight column as it's no longer needed
    append_data = append_data.drop(columns=['TotalExportWeight'])

    return append_data


def getDataofWeights(year_value, df1, df2, df3, df4):
    # Tier-2 share approach: trade share matrix is held at 2033 levels for
    # year_value > 2033 (no reliable bilateral trade projection beyond UN Comtrade
    # horizon). Country-level retired flows entering the LP downstream provide the
    # demographic-driven scaling, consistent with global trade share-approach
    # conventions (IMF 1967; Bekkers 2024 WTO; standard dMFA practice).
    _query_year = min(int(year_value), 2033)
    selected_columns1 = ["ReporterDesc", "PartnerDesc", "NetWgt"]

    selected_data1 = df1[(df1['Period'] == _query_year)][selected_columns1].copy()
    selected_data2 = df2[(df2['Period'] == _query_year)][selected_columns1].copy()
    selected_data3 = df3[(df3['Period'] == _query_year)][selected_columns1].copy()
    selected_data4 = df4[(df4['Period'] == _query_year)][selected_columns1].copy()

    merge_keys = ['ReporterDesc', 'PartnerDesc']

    merged_df = selected_data1.merge(selected_data2, on=merge_keys, suffixes=('_df1', '_df2'), how='outer') \
        .merge(selected_data3, on=merge_keys, suffixes=('', '_df3'), how='outer') \
        .merge(selected_data4, on=merge_keys, suffixes=('', '_df4'), how='outer')

    merged_df.fillna(0, inplace=True)
    merged_df['total_NetWgt'] = merged_df[['NetWgt', 'NetWgt_df1', 'NetWgt_df2', 'NetWgt_df4']].sum(axis=1)

    new_df = merged_df[['ReporterDesc', 'PartnerDesc', 'total_NetWgt']]
    new_df = new_df.rename(columns={'total_NetWgt': 'NetWgt'})
    new_df['Period'] = [year_value] * len(new_df)

    dff = culComdata(new_df)
    return dff


def culTransformDta(chiose_da, weight_valaue):
    # recycling - scrap < 0 : No recycling capacity
    # recycling - scrap > 0 : Recycling capacity exists
    is_negative = chiose_da.iloc[0] < 0
    is_positive = chiose_da.iloc[0] > 0
    positive_values = chiose_da.iloc[0][is_negative].abs()
    positive_values1 = chiose_da.iloc[0][is_positive]

    # Convert to positive number
    output_sat = positive_values.to_dict()
    output_end = positive_values1.to_dict()

    key_to_remove = 'Year'
    output_end = remove_element_by_key(output_end, key_to_remove)
    output_end_df = pd.DataFrame(list(output_end.items()), columns=['PartnerDesc', 'Mass_v'])

    keys_sat = list(output_sat.keys())
    keys_end = list(output_end.keys())

    if len(keys_end) != 0:
        chiose_weight_defult = weight_valaue[
            (weight_valaue['ReporterDesc'].isin(keys_sat)) & (weight_valaue['PartnerDesc'].isin(keys_end))]

        output_end_df['Mass_t'] = [0] * len(output_end_df)

        # Take the top three values
        chiose_weight = chiose_weight_defult.groupby('ReporterDesc').apply(
            lambda x: x.nlargest(3, 'Probability')).reset_index(drop=True)

        group_sum = chiose_weight.groupby('ReporterDesc')['Probability'].transform('sum')

        # 计算各个分组内的权重（占总和的百分比）
        chiose_weight['Weight'] = chiose_weight['Probability'] / group_sum

        export_distribution_df = pd.DataFrame(list(output_sat.items()), columns=['ReporterDesc', 'Mass_t'])
        export_distribution_df['Mass_v'] = [0] * len(export_distribution_df)

        chiose_weight = chiose_weight.merge(export_distribution_df, on='ReporterDesc')
        chiose_weight['Mass'] = chiose_weight['Mass_t'] * chiose_weight['Weight']

        # 按 'group_col' 分组并对 'value_col' 求和
        grouped_sum = chiose_weight.groupby('PartnerDesc')['Mass_t'].sum().reset_index()
        grouped_sum.columns = ['PartnerDesc', 'Mass_e']

        # 将结果合并到原 DataFrame 中
        chiose_weight = pd.merge(chiose_weight, grouped_sum, on='PartnerDesc', how='left')

        exist_recyc_cap_copy = output_end.copy()
        # exist_recyc_cap
        for ke in itertools.islice(output_end, 1, None):
            if ke in chiose_weight['PartnerDesc'].unique():
                shengyu = chiose_weight[chiose_weight['PartnerDesc'] == ke]['Mass_e'].tolist()
                shu = output_end[ke] - shengyu[0]
                exist_recyc_cap_copy.update({ke: shu})

        export_distribution_df = export_distribution_df.rename(columns={'ReporterDesc': 'Region'})
        output_end_df = output_end_df.rename(columns={'PartnerDesc': 'Region'})

        return chiose_weight_defult, export_distribution_df, output_end_df
    else:
        return None, None, None


def datailTrans(transport_log, no_recyc_cap_t, exist_recyc_cap_t):
    no_recyc_cap_group_st = transport_log.groupby('From Region')['Amount Transported'].sum().reset_index()
    no_recyc_cap_group_en = transport_log.groupby('To Region')['Amount Transported'].sum().reset_index()

    no_recyc_cap_group_st = no_recyc_cap_group_st.rename(columns={'From Region': 'Region'})
    no_recyc_cap_group_st_t = no_recyc_cap_group_st.merge(no_recyc_cap_t, on='Region', how='left')

    no_recyc_cap_group_en = no_recyc_cap_group_en.rename(columns={'To Region': 'Region'})
    no_recyc_cap_group_en_t = no_recyc_cap_group_en.merge(exist_recyc_cap_t, on='Region', how='left')

    return no_recyc_cap_group_st_t, no_recyc_cap_group_en_t


def culRemainingandUndisposed(no_recyc_cap_group_st, no_recyc_cap_group_en):
    no_recyc_cap_group_en['Remaining_capacity'] = (
            no_recyc_cap_group_en['Mass_v'] - no_recyc_cap_group_en['Amount Transported']).round(2)
    no_recyc_cap_group_st['Undisposed_scrap'] = (
            no_recyc_cap_group_st['Mass_t'] - no_recyc_cap_group_st['Amount Transported']).round(2)

    return no_recyc_cap_group_st, no_recyc_cap_group_en


def SplitDistData(coun_list_chiose):
    selected_coun_list_chiose_columns = coun_list_chiose.iloc[:, :2].join(coun_list_chiose.iloc[:, -3])
    selected_coun_list_chiose_columns['distcap'] = selected_coun_list_chiose_columns['distcap'].round(4)

    pivot_chiose_df = selected_coun_list_chiose_columns.pivot(index='iso_o', columns='iso_d', values='distcap')

    pivot_chiose_df['Source'] = pivot_chiose_df.index.tolist()
    pivot_chiose_df.reset_index(drop=True, inplace=True)
    pivot_chiose_df.index = pivot_chiose_df['Source'].tolist()
    pivot_chiose_df = pivot_chiose_df.drop(columns=['Source'])

    return pivot_chiose_df


def storage_cost_D1(quantity, a1, a2):
    return a1 * quantity ** a2


# Distance Matrix; battery scrap num; battery recycling number; cost fit data of recycling process;
def culDistCountryMatrix(carbon_cost, pivot_data, battery_scrap_num, battery_recyc_num, df_cost_fit_data_recycl, time,
                         recycl_proce, Strategy_type):
    carbon_cost = carbon_cost[carbon_cost['recycling_m'] == recycl_proce][['iso3', 'carbon_cost']].copy()
    # 转换为 numpy 数组
    transport_cost = pivot_data.values
    supply = battery_scrap_num['scrap'].values
    demand = battery_recyc_num['Mass_v'].values

    carbon_cost.index = carbon_cost['iso3'].tolist()
    carbon_cost = carbon_cost.drop('iso3', axis=1)

    # 定义供应和需求节点
    supply_nodes = battery_scrap_num.index.tolist()
    demand_nodes = battery_recyc_num.index.tolist()
    if list(set(demand_nodes) - set(carbon_cost.index.tolist())):
        df2 = pd.DataFrame({
            'carbon_cost': [0] * len(list(set(demand_nodes) - set(carbon_cost.index.tolist())))
        }, index=list(set(demand_nodes) - set(carbon_cost.index.tolist())))
        carbon_cost = pd.concat([carbon_cost, df2], ignore_index=False)

    carbon_cost_main = carbon_cost.loc[demand_nodes]['carbon_cost'].tolist()

    # 定义存储成本函数列表
    storage_cost_functions = []

    # 使用循环构建存储成本函数列表
    for i, node in enumerate(demand_nodes):
        a = df_cost_fit_data_recycl.loc[node, 'a']
        b = df_cost_fit_data_recycl.loc[node, 'b']

        if pd.notna(a) and pd.notna(b):
            storage_cost_functions.append(lambda quantity, a1=a, a2=b: storage_cost_D1(quantity, a1, a2))

    # 计算每个目标地区的存储成本（假设存储量与需求量相等）
    storage_cost = np.array([func(d) for func, d in zip(storage_cost_functions, demand)])

    storage_cost = np.array([a + b for a, b in zip_longest(storage_cost, carbon_cost_main, fillvalue=0)])

    # 计算总的供需差异
    supply_sum = supply.sum()
    demand_sum = demand.sum()
    difference = demand_sum - supply_sum

    if difference > 0:
        # 添加虚拟供应点
        supply = np.append(supply, difference)
        virtual_supply_cost = np.full((1, len(demand_nodes)), 1e6)  # 虚拟供应点的运输成本设置为一个非常高的值
        transport_cost = np.vstack((transport_cost, virtual_supply_cost))
        supply_nodes.append('Virtual_Supply')
    elif difference < 0:
        # 添加虚拟需求点
        demand = np.append(demand, -difference)
        virtual_demand_cost = np.full((len(supply_nodes), 1), 1e6)  # 虚拟需求点的运输成本设置为一个非常高的值
        transport_cost = np.hstack((transport_cost, virtual_demand_cost))
        demand_nodes.append('Virtual_Demand')

    # 计算总成本矩阵 (运输成本 + 存储成本)
    storage_cost = storage_cost.reshape(1, -1)
    if difference < 0:
        storage_cost = np.append(storage_cost, [1e6])

    total_cost = transport_cost + storage_cost

    # 展平成本矩阵
    c = total_cost.flatten()

    # 约束矩阵和边界
    A_eq = []
    b_eq = []

    # 供应约束
    for i in range(len(supply)):
        A_eq.append([1 if j // len(demand) == i else 0 for j in range(len(supply) * len(demand))])
        b_eq.append(supply[i])

    # 需求约束
    for j in range(len(demand)):
        A_eq.append([1 if j == k % len(demand) else 0 for k in range(len(supply) * len(demand))])
        b_eq.append(demand[j])

    # 转换为 numpy 数组
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # # 打印检查供需平衡
    # print(f"Total Supply: {supply_sum}, Total Demand: {demand_sum}, Difference: {difference}")
    #
    # # 打印检查约束条件和成本矩阵
    # print("A_eq shape:", A_eq.shape)
    # print("b_eq shape:", b_eq.shape)
    # print("Cost vector shape:", c.shape)

    # 使用 linprog 求解线性规划问题
    # result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs-ipm')

    # 检查是否成功找到解
    if result.success:
        # 重塑结果为 len(supply) x len(demand) 矩阵
        x = result.x.reshape(len(supply), len(demand))

        # 创建结果 DataFrame
        result_df = pd.DataFrame(x, columns=demand_nodes, index=supply_nodes)

        print(f"\n Global {recycl_proce} process total cost for {Strategy_type} in {time}:")
        print(result.fun)

        return result_df
    else:
        print("未找到可行解:", result.message)


def OUTPUTNetProfitofStrategyOne(ev_battery_data_selected, pivot_diff, Recycling_revenue_df, carbon_cost,
                                 df_cost_fit_data, dist_cost, recycl_proce):
    net_profit_strat1 = pd.DataFrame()
    for pp in ev_battery_data_selected['country'].unique():
        for pk in ev_battery_data_selected['type'].unique():
            # carbon cost
            carbon_cost_cho = carbon_cost[(carbon_cost['country'] == pp)
                                          & (carbon_cost['battery_type'] == pk)
                                          & (carbon_cost['recycling_m'] == recycl_proce)]
            dist_cost_cho = dist_cost[(dist_cost['country'] == pp)]

            if dist_cost_cho.empty:
                dist_cost_value = 0
            else:
                dist_cost_value = dist_cost_cho['dist_cost'].tolist()[0]

            if carbon_cost_cho.empty:
                carbon_cost_value = 0
            else:
                carbon_cost_value = carbon_cost_cho['carbon_cost'].tolist()[0]

            re_value = Recycling_revenue_df[pk].to_dict()
            canshu_a = \
                df_cost_fit_data[(df_cost_fit_data['recycling_m'] == recycl_proce) & (df_cost_fit_data['country'] == pp)][
                    'a'].tolist()
            canshu_b = \
                df_cost_fit_data[(df_cost_fit_data['recycling_m'] == recycl_proce) & (df_cost_fit_data['country'] == pp)][
                    'b'].tolist()
            scrap_num = ev_battery_data_selected[
                (ev_battery_data_selected['country'] == pp) & (ev_battery_data_selected['type'] == pk)][
                'scrap'].tolist()

            if scrap_num[0] < 0.003:
                scrap_v = 0
                # unit cost
                trade_cost = 0
                # unit net profits
                netprofits = 0
            else:
                scrap_v = scrap_num[0]
                # unit cost
                trade_cost = canshu_a[0] * scrap_v ** canshu_b[0]
                # unit net profits
                netprofits = re_value[recycl_proce] - trade_cost - carbon_cost_value - dist_cost_value

                if len(pivot_diff) != 0:
                    scrap_num_exp = pivot_diff[(pivot_diff['country'] == pp) & (pivot_diff['type'] == pk)][
                        'Expanding production'].tolist()
                    if scrap_num_exp:
                        # unit cost exp
                        trade_cost_exp = canshu_a[0] * scrap_num_exp[0] ** canshu_b[0]
                        if round(scrap_v, 4) - round(scrap_num_exp[0], 4) != 0:
                            # unit cost
                            trade_cost_nei = canshu_a[0] * (scrap_v - scrap_num_exp[0]) ** canshu_b[0]
                            trade_cost = trade_cost_nei + trade_cost_exp
                        # unit net profits
                        netprofits = re_value[recycl_proce] - trade_cost - carbon_cost_value - dist_cost_value

            net_profit_strat1 = net_profit_strat1.append([{
                'country': pp,
                'type': pk,
                'recycling_m': recycl_proce,
                'scrap': scrap_v,
                'unit_cost': trade_cost,
                'net_profit': netprofits
            }])
    net_profit_strat1.reset_index(drop=True, inplace=True)

    for _col in ['scrap', 'unit_cost', 'net_profit']:
        if _col in net_profit_strat1.columns:
            net_profit_strat1[_col] = net_profit_strat1[_col].astype(float)
    idx = net_profit_strat1.iloc[:, [0, 1, 2, 5]].groupby(['country', 'type'])['net_profit'].idxmax()

    net_profit_strat1_chiose = net_profit_strat1.loc[idx, ['country', 'type', 'recycling_m', 'net_profit']]

    return net_profit_strat1_chiose, net_profit_strat1



def OUTPUTNetProfitofOptimalOne(ev_battery_data_selected, pivot_diff, Recycling_revenue_df, carbon_cost,
                                 df_cost_fit_data, dist_cost):
    re_index = Recycling_revenue_df.index.tolist()
    net_profit_strat1 = pd.DataFrame()
    for pp in ev_battery_data_selected['country'].unique():
        for pk in ev_battery_data_selected['type'].unique():
            for pj in re_index:
                # carbon cost
                carbon_cost_cho = carbon_cost[(carbon_cost['country'] == pp)
                                              & (carbon_cost['battery_type'] == pk)
                                              & (carbon_cost['recycling_m'] == pj)]
                dist_cost_cho = dist_cost[(dist_cost['country'] == pp)]

                if dist_cost_cho.empty:
                    dist_cost_value = 0
                else:
                    dist_cost_value = dist_cost_cho['dist_cost'].tolist()[0]

                if carbon_cost_cho.empty:
                    carbon_cost_value = 0
                else:
                    carbon_cost_value = carbon_cost_cho['carbon_cost'].tolist()[0]

                re_value = Recycling_revenue_df[pk].to_dict()
                canshu_a = \
                    df_cost_fit_data[(df_cost_fit_data['recycling_m'] == pj) & (df_cost_fit_data['country'] == pp)][
                        'a'].tolist()
                canshu_b = \
                    df_cost_fit_data[(df_cost_fit_data['recycling_m'] == pj) & (df_cost_fit_data['country'] == pp)][
                        'b'].tolist()
                scrap_num = ev_battery_data_selected[
                    (ev_battery_data_selected['country'] == pp) & (ev_battery_data_selected['type'] == pk)][
                    'scrap'].tolist()

                if scrap_num[0] < 0.003:
                    scrap_v = 0
                    # unit cost
                    trade_cost = 0
                    # unit net profits
                    netprofits = 0
                else:
                    scrap_v = scrap_num[0]
                    # unit cost
                    trade_cost = canshu_a[0] * scrap_v ** canshu_b[0]
                    # unit net profits
                    netprofits = re_value[pj] - trade_cost - carbon_cost_value - dist_cost_value

                    if len(pivot_diff) != 0:
                        scrap_num_exp = pivot_diff[(pivot_diff['country'] == pp) & (pivot_diff['type'] == pk)][
                            'Expanding production'].tolist()
                        if scrap_num_exp:
                            # unit cost exp
                            trade_cost_exp = canshu_a[0] * scrap_num_exp[0] ** canshu_b[0]
                            if round(scrap_v, 4) - round(scrap_num_exp[0], 4) != 0:
                                # unit cost
                                trade_cost_nei = canshu_a[0] * (scrap_v - scrap_num_exp[0]) ** canshu_b[0]
                                trade_cost = trade_cost_nei + trade_cost_exp
                            # unit net profits
                            netprofits = re_value[pj] - trade_cost - carbon_cost_value - dist_cost_value

                net_profit_strat1 = net_profit_strat1.append([{
                    'country': pp,
                    'type': pk,
                    'recycling_m': pj,
                    'scrap': scrap_v,
                    'unit_cost': trade_cost,
                    'net_profit': netprofits
                }])
    net_profit_strat1.reset_index(drop=True, inplace=True)

    for _col in ['scrap', 'unit_cost', 'net_profit']:
        if _col in net_profit_strat1.columns:
            net_profit_strat1[_col] = net_profit_strat1[_col].astype(float)
    idx = net_profit_strat1.iloc[:, [0, 1, 2, 5]].groupby(['country', 'type'])['net_profit'].idxmax()

    net_profit_strat1_chiose = net_profit_strat1.loc[idx, ['country', 'type', 'recycling_m', 'scrap', 'net_profit']]

    # net_profit_strat1_chiose = net_profit_strat1_chiose.groupby('country')['net_profit'].max().reset_index()

    # idx1 = net_profit_strat1_chiose.groupby('country')['net_profit'].idxmax()
    #
    # # Select rows by index, keep 'net_profit' column and the maximum value
    # net_profit_strat1_chiose = net_profit_strat1_chiose.loc[idx1]

    return net_profit_strat1_chiose, net_profit_strat1


def OUTPUTNetProfitofStrategyTwoThree(ev_battery_data_selected, Recycling_revenue_df, carbon_cost, df_cost_fit_data,
                                      dist_cost):
    recycling_fil = ev_battery_data_selected['recycling_m'].tolist()[0]
    net_profit_strat1 = pd.DataFrame()
    for pp in ev_battery_data_selected['country'].unique():
        for pk in ev_battery_data_selected['type'].unique():
            # carbon cost
            carbon_cost_cho = carbon_cost[(carbon_cost['country'] == pp)
                                          & (carbon_cost['battery_type'] == pk)
                                          & (carbon_cost['recycling_m'] == recycling_fil)]
            dist_cost_cho = dist_cost[(dist_cost['country'] == pp)]

            if dist_cost_cho.empty:
                dist_cost_value = 0
            else:
                dist_cost_value = dist_cost_cho['dist_cost'].tolist()[0]

            if carbon_cost_cho.empty:
                carbon_cost_value = 0
            else:
                carbon_cost_value = carbon_cost_cho['carbon_cost'].tolist()[0]

            re_value = Recycling_revenue_df[pk].to_dict()
            canshu_a = \
                df_cost_fit_data[(df_cost_fit_data['recycling_m'] == recycling_fil) & (df_cost_fit_data['country'] == pp)][
                    'a'].tolist()
            canshu_b = \
                df_cost_fit_data[(df_cost_fit_data['recycling_m'] == recycling_fil) & (df_cost_fit_data['country'] == pp)][
                    'b'].tolist()
            scrap_num = ev_battery_data_selected[
                (ev_battery_data_selected['country'] == pp) & (ev_battery_data_selected['type'] == pk)][
                'scrap'].tolist()

            if scrap_num[0] < 0.002:
                scrap_v = 0
                # unit cost
                trade_cost = 0
                # unit net profits
                netprofits = 0
            else:
                # Tone
                scrap_v = scrap_num[0]
                # unit cost $/kg
                trade_cost = canshu_a[0] * scrap_v ** canshu_b[0]
                # unit net profits $/kg
                netprofits = re_value[recycling_fil] - trade_cost - carbon_cost_value - dist_cost_value

            net_profit_strat1 = net_profit_strat1.append([{
                'country': pp,
                'type': pk,
                'recycling_m': recycling_fil,
                'scrap': scrap_v,
                'unit_cost': trade_cost,
                'net_profit': netprofits
            }])
    net_profit_strat1.reset_index(drop=True, inplace=True)

    for _col in ['scrap', 'unit_cost', 'net_profit']:
        if _col in net_profit_strat1.columns:
            net_profit_strat1[_col] = net_profit_strat1[_col].astype(float)
    idx = net_profit_strat1.iloc[:, [0, 1, 2, 5]].groupby(['country', 'type'])['net_profit'].idxmax()

    net_profit_strat1_chiose = net_profit_strat1.loc[idx, ['country', 'type', 'recycling_m', 'net_profit']]

    return net_profit_strat1_chiose, net_profit_strat1


def OUTPUTNetProfitofOptimalTwoThree(ev_battery_data_selected, Recycling_revenue_df, carbon_cost, df_cost_fit_data,
                                      dist_cost):
    re_index = Recycling_revenue_df.index.tolist()
    net_profit_strat1 = pd.DataFrame()
    for pp in ev_battery_data_selected['country'].unique():
        for pk in ev_battery_data_selected['type'].unique():
            for pj in re_index:
                # carbon cost
                carbon_cost_cho = carbon_cost[(carbon_cost['country'] == pp)
                                              & (carbon_cost['battery_type'] == pk)
                                              & (carbon_cost['recycling_m'] == pj)]
                dist_cost_cho = dist_cost[(dist_cost['country'] == pp)]

                if dist_cost_cho.empty:
                    dist_cost_value = 0
                else:
                    dist_cost_value = dist_cost_cho['dist_cost'].tolist()[0]

                if carbon_cost_cho.empty:
                    carbon_cost_value = 0
                else:
                    carbon_cost_value = carbon_cost_cho['carbon_cost'].tolist()[0]

                re_value = Recycling_revenue_df[pk].to_dict()
                canshu_a = \
                    df_cost_fit_data[(df_cost_fit_data['recycling_m'] == pj) & (df_cost_fit_data['country'] == pp)][
                        'a'].tolist()
                canshu_b = \
                    df_cost_fit_data[(df_cost_fit_data['recycling_m'] == pj) & (df_cost_fit_data['country'] == pp)][
                        'b'].tolist()
                scrap_num = ev_battery_data_selected[
                    (ev_battery_data_selected['country'] == pp) & (ev_battery_data_selected['type'] == pk)][
                    'scrap'].tolist()

                if scrap_num[0] < 0.002:
                    scrap_v = 0
                    # unit cost
                    trade_cost = 0
                    # unit net profits
                    netprofits = 0
                else:
                    # Tone
                    scrap_v = scrap_num[0]
                    # unit cost $/kg
                    trade_cost = canshu_a[0] * scrap_v ** canshu_b[0]
                    # unit net profits $/kg
                    netprofits = re_value[pj] - trade_cost - carbon_cost_value - dist_cost_value

                net_profit_strat1 = net_profit_strat1.append([{
                    'country': pp,
                    'type': pk,
                    'recycling_m': pj,
                    'scrap': scrap_v,
                    'unit_cost': trade_cost,
                    'net_profit': netprofits
                }])
    net_profit_strat1.reset_index(drop=True, inplace=True)

    for _col in ['scrap', 'unit_cost', 'net_profit']:
        if _col in net_profit_strat1.columns:
            net_profit_strat1[_col] = net_profit_strat1[_col].astype(float)
    idx = net_profit_strat1.iloc[:, [0, 1, 2, 5]].groupby(['country', 'type'])['net_profit'].idxmax()

    net_profit_strat1_chiose = net_profit_strat1.loc[idx, ['country', 'type', 'recycling_m', 'scrap', 'net_profit']]

    # net_profit_strat1_chiose = net_profit_strat1_chiose.groupby('country')['net_profit'].max().reset_index()

    # idx1 = net_profit_strat1_chiose.groupby('country')['net_profit'].idxmax()
    #
    # # Select rows by index, keep 'net_profit' column and the maximum value
    # net_profit_strat1_chiose = net_profit_strat1_chiose.loc[idx1]

    return net_profit_strat1_chiose, net_profit_strat1

def culWeightAddScrap(routes_1, prod_scrap_type_chiose):
    routes_original_df = pd.DataFrame()
    # Iterate over df
    for index, row in routes_1.iterrows():
        chios_df1 = prod_scrap_type_chiose[prod_scrap_type_chiose['iso3'] == row['source_point']].copy()
        chios_df1['weights'] = chios_df1['weights'] * row['quantity']
        chios_df1['iso3'] = [row['dest_point']] * len(chios_df1)
        routes_original_df = routes_original_df.append(chios_df1, ignore_index=True)
    routes_original_df = routes_original_df.rename(columns={'weights': 'scrap'}).iloc[:, -3:]
    routes_original_df = routes_original_df.groupby(['iso3', 'type'])['scrap'].sum().reset_index()
    return routes_original_df


def culWeightAddScrap1(routes_1, prod_scrap_type_chiose):
    routes_original_df = pd.DataFrame()
    # Iterate over df
    for index, row in routes_1.iterrows():
        chios_df1 = prod_scrap_type_chiose[prod_scrap_type_chiose['iso3'] == row['iso3']].copy()
        chios_df1['weights'] = chios_df1['weights'] * row['quantity']
        if 'Virtual_Demand' not in routes_1.values:
            chios_df1['iso3'] = [row['dest_point']] * len(chios_df1)
        routes_original_df = routes_original_df.append(chios_df1, ignore_index=True)
    routes_original_df = routes_original_df.rename(columns={'weights': 'scrap'}).iloc[:, -3:]
    routes_original_df = routes_original_df.groupby(['iso3', 'type'])['scrap'].sum().reset_index()
    return routes_original_df


def culConsolidatedDisposalVolume(disposal_volume, dist_cost):
    routes_original_df = pd.DataFrame()
    # Iterate over df
    for index, row in disposal_volume.iterrows():
        dist_cost_value = dist_cost.loc[row['source_point'], row['dest_point']]
        path_quantity = row['quantity'] * dist_cost_value
        routes_original_df = routes_original_df.append({
            'dest_point': row['dest_point'],
            'quantity': path_quantity,
        }, ignore_index=True)

    routes_original_df = routes_original_df.groupby('dest_point')['quantity'].sum().reset_index()
    value_drat = disposal_volume.groupby('dest_point')['quantity'].sum().reset_index()
    routes_original_df = routes_original_df.merge(value_drat, on='dest_point', how='left', suffixes=('_df1', '_df2'))
    routes_original_df['dist_cost'] = routes_original_df['quantity_df1'] / routes_original_df['quantity_df2']

    return routes_original_df.iloc[:, [0, 3]]

battery_sub = pd.read_csv('./cost/Metal content.csv').copy()
def cul_diff_type_EV_bat_matel(battery_scrap, metal, type_tab):
    type_list = battery_scrap['type'].unique()
    df_batt = pd.DataFrame()
    li_ll = [x for x in type_list if x != 'TLB']
    for t in li_ll:
        df_data_year = battery_scrap[battery_scrap['type'] == t].copy()
        # Chio Li metal
        substance_val = battery_sub[battery_sub['Type'] == t][metal].tolist()[0]
        df_data_year['Spent battery feed'] = df_data_year[type_tab].map(lambda x: x * substance_val)
        df_batt = df_batt.append(df_data_year)
    df_batt = df_batt[['country', 'type', 'Spent battery feed']].copy()
    df_batt = df_batt.rename(columns={'Spent battery feed': type_tab})

    return df_batt


# Function to modify NetWgt values according to the specified logic
def modify_netwgt(df):
    for index, row in df.iterrows():
        if pd.isna(row['NetWgt']) or row['NetWgt'] == 0:
            partner_iso = row['PartnerISO']
            reporter_desc = row['ReporterDesc']
            primary_value = row['PrimaryValue']

            # Step 1: Find rows with the same PartnerISO and ReporterDesc
            matching_rows = df[
                (df['PartnerISO'] == partner_iso) & (df['ReporterDesc'] == reporter_desc) & (df['NetWgt'] != 0)]

            if not matching_rows.empty:
                # Step 2: Determine if there are multiple matching rows
                if len(matching_rows) > 1:
                    # Select the row with the largest Period value
                    selected_row = matching_rows.loc[matching_rows['Period'].idxmax()]
                else:
                    # Select the single matching row
                    selected_row = matching_rows.iloc[0]

                # Step 3: Calculate the ratio of NetWgt to PrimaryValue in the selected row
                ratio = selected_row['NetWgt'] / selected_row['PrimaryValue']

                # Step 4: Calculate the new NetWgt value and update the dataframe
                new_netwgt = ratio * primary_value
                df.at[index, 'NetWgt'] = new_netwgt

    return df
