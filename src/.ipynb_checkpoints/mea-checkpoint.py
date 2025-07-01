# mea.py module
# Contains functions to implement MEA model.

import pandas as pd
import numpy as np
import sqlite3


def read_basenames(path):
    return [line.lower().strip().split(' ')[0] 
        for line in open(path, 'r') if not line.startswith('#')]


def discount(x, dr, col, ref_year):
    return x[col] / pow(1 + dr, x['year'] - ref_year)  


def import_pricecost_fms(cnn, assid_col='assumption_id', unit_col='tsa_num', tn='pricecost_fms'):
    df = pd.read_sql_query('select * from %s' % tn, cnn)
    df[unit_col] = df[unit_col].astype(int) # hack (should always be integer)
    df.set_index([assid_col, unit_col], inplace=True)
    df = df.astype('float64')
    return df


def import_pricecost_hwp(cnn, assid_col='assumption_id', unit_col='tsa_num', tn='pricecost_hwp'):
    df = pd.read_sql_query('select * from %s' % tn, cnn)
    df[unit_col] = df[unit_col].astype(int) # hack (should always be integer)
    df.set_index([assid_col, unit_col], inplace=True)
    df = df.astype('float64')
    return df


def import_pricecost_dis(cnn, assid_col='assumption_id', unit_col='tsa_num', tn='pricecost_dis'):
    df = pd.read_sql_query('select * from %s' % tn, cnn)
    df[unit_col] = df[unit_col].astype(int) # hack (should always be integer)
    df.set_index([assid_col, unit_col], inplace=True)
    df = df.astype('float64')
    return df


def import_prodconv(cnn, scenario_col='scenario', unit_col='tsa_num', tn='prodconv'):
    #########
    # FIX ME 
    # For production runs, we should be importing conversion parameters from mea.db SQLite database.
    # Should be something like this:
    #  cnn = sqlite3.connect(mea_db_path)
    #  df = pd.read_sql_query('select * from %s' % prodconv_tn, cnn)
    #  df.set_index([scen_col, unit_col], inplace=True)
    #  cnn.close()
    # TO DO: switch to reading from mea.db
    #df = pd.read_csv('../dat/conversion_params.csv').set_index(['scenario', unit_col]) 
    #########
    df = pd.read_sql_query('select * from %s' % tn, cnn).set_index([scenario_col, unit_col])
    return df


def import_mea_params(mea_params_db_path):
    cnn = sqlite3.connect(mea_params_db_path)
    df1 = import_pricecost_fms(cnn)
    df2 = import_pricecost_hwp(cnn)
    df3 = import_pricecost_dis(cnn)
    df4 = import_prodconv(cnn)
    cnn.close()
    return df1, df2, df3, df4
    

def compile_fms_production(cnn, scenario_names, reference_scenario_name,
                           table_name='mea_carbon_flux_by_event',     
                           groupby_cols=['scenario', 'tsa_num', 'year'],   
                           data_cols=['Live_SW_Harvest_Vol', 
                                      'Live_HW_Harvest_Vol', 
                                      'Dead_Wood_harvest_Vol', 
                                      'Harvest_Residue_Volume']):
    """
    Sum FMS production data by scenario, spatial unit, and time step.
    """
    df = pd.read_sql_query("SELECT * FROM %s;" % table_name, cnn)
    ##############
    # HACK #######
    # TSA numbers stored in database as floats (convert to int).
    df['tsa_num'] = df['tsa_num'].astype(int)
    ##############
    # HACK #######
    # Rename time column from 'Year' to 'year'.
    df.rename(columns={'Year':'year'}, inplace=True)
    ##############    
    #############################
    # HACK ######################
    # Some suites apparently include scenarios that simulate no disturbance events at all
    # for some combinations of TSA and year. This seems highly suspect, and is likely
    # symptomatic of a bug in upstream modelling, but there is no time to deal with that
    # right now, so the following hack will make sure that the MultiIndex of the result df
    # will have a valid index and not crash the MEA in the compile_net_revenue function.
    scenarios = df['scenario'].unique()
    basenames = [int(x[-2:]) for x in read_basenames('../dat/basenames.txt')]
    years = [i for i in range(1990, 2071)]
    index = pd.MultiIndex.from_product([scenarios, basenames, years], names=['scenario', 'tsa_num', 'year'])
    _df = pd.DataFrame(index=index)
    try:
        df = _df.join(df.set_index(groupby_cols)).reset_index()
    except:
        print(_df.index)
        print(df.index)
        assert False
    #############################
    result = df[groupby_cols+data_cols].groupby(by=groupby_cols).sum()
    return result


def compile_fms_emissions(cnn,
                          table_name='dfe_dfp_sector_society_rt',     
                          groupby_cols=['scenario', 'tsa_num', 'year'],   
                          data_cols=['forest_mtco2e']):
    """
    Sum FMS emissions data by scenario, spatial unit, and time step.
    """
    df = pd.read_sql_query("SELECT * FROM %s;" % table_name, cnn)
    # HACK #######
    # tsa_num stored in database as float (convert to int).
    df['tsa_num'] = df['tsa_num'].astype(int)
    ##############
        #############################
    # HACK ######################
    # Some suites apparently include scenarios that simulate no disturbance events at all
    # for some combinations of TSA and year. This seems highly suspect, and is likely
    # symptomatic of a bug in upstream modelling, but there is no time to deal with that
    # right now, so the following hack will make sure that the MultiIndex of the result df
    # will have a valid index and not crash the MEA in the compile_net_revenue function.
    scenarios = df['scenario'].unique()
    basenames = [int(x[-2:]) for x in read_basenames('../dat/basenames.txt')]
    years = [i for i in range(1990, 2071)]
    index = pd.MultiIndex.from_product([scenarios, basenames, years], names=['scenario', 'tsa_num', 'year'])
    _df = pd.DataFrame(index=index)
    try:
        df = _df.join(df.set_index(groupby_cols)).reset_index()
    except:
        print(_df.index)
        print(df.index)
        assert False
    #############################
    result = df[groupby_cols+data_cols].groupby(by=groupby_cols).sum()
    return result


def compile_hwp_production(cnn,
                           table_name='production_mea_format',
                           groupby_cols=['scenario', 'tsa_num', 'year', 'is_domestic'],   
                           data_cols=['sawnwood_m3', 'panels_m3', 'otherirw_m3', 'pulppaper_odt']):
    """
    Sum HWP production data by scenario, spatial unit, time step, and location.
    """
    df = pd.read_sql_query("SELECT * FROM %s;" % table_name, cnn)
    #return df
    # HACK #######
    # is_domestic flag stored in database as int (convert to int).
    df['is_domestic'] = df['is_domestic'].astype(int).astype(bool)
    ##############
    # HACK #######
    # tsa_num stored in database as float (convert to int).
    df['tsa_num'] = df['tsa_num'].astype(int)
    ##############
    #############################
    # HACK ######################
    # Some suites apparently include scenarios that simulate no disturbance events at all
    # for some combinations of TSA and year. This seems highly suspect, and is likely
    # symptomatic of a bug in upstream modelling, but there is no time to deal with that
    # right now, so the following hack will make sure that the MultiIndex of the result df
    # will have a valid index and not crash the MEA in the compile_net_revenue function.
    scenarios = df['scenario'].unique()
    basenames = [int(x[-2:]) for x in read_basenames('../dat/basenames.txt')]
    years = [i for i in range(1990, 2071)]
    index = pd.MultiIndex.from_product([scenarios, basenames, years, [False, True]], names=['scenario', 'tsa_num', 'year', 'is_domestic'])
    _df = pd.DataFrame(index=index)
    try:
        df = _df.join(df.set_index(groupby_cols)).reset_index()
    except:
        print(_df.index)
        print(df.index)
        assert False
    #############################
    result = df[groupby_cols+data_cols].groupby(by=groupby_cols).sum()
    return result


def compile_hwp_emissions(cnn,
                          table_name='dfe_dfp_sector_society_rt',
                          groupby_cols=['scenario', 'tsa_num', 'year'],   
                          data_cols=['hwp_emissions_domestic_mtco2e', 'hwp_emissions_foreign_mtco2e']):
    """
    Sum HWP emissions data by scenario, spatial unit, and time step.
    """
    df = pd.read_sql_query("SELECT * FROM %s;" % table_name, cnn)
    # HACK #######
    # TSA numbers stored in database as floats (convert to int).
    df['tsa_num'] = df['tsa_num'].astype(int)
    ##############
    # HACK #######
    # Rename time columen from 'Year' to 'year'.
    df.rename(columns={'Year':'year'}, inplace=True)
    ##############
        #############################
    # HACK ######################
    # Some suites apparently include scenarios that simulate no disturbance events at all
    # for some combinations of TSA and year. This seems highly suspect, and is likely
    # symptomatic of a bug in upstream modelling, but there is no time to deal with that
    # right now, so the following hack will make sure that the MultiIndex of the result df
    # will have a valid index and not crash the MEA in the compile_net_revenue function.
    scenarios = df['scenario'].unique()
    basenames = [int(x[-2:]) for x in read_basenames('../dat/basenames.txt')]
    years = [i for i in range(1990, 2071)]
    index = pd.MultiIndex.from_product([scenarios, basenames, years], names=['scenario', 'tsa_num', 'year'])
    _df = pd.DataFrame(index=index)
    try:
        df = _df.join(df.set_index(groupby_cols)).reset_index()
    except:
        print(_df.index)
        print(df.index)
        assert False
    #############################
    result = df[groupby_cols+data_cols].groupby(by=groupby_cols).sum()
    return result


def compile_bioenergy_production(cnn,
                                 table_name='production_mea_format',
                                 groupby_cols=['scenario', 'tsa_num', 'year', 'is_domestic'],
                                 # HACK ###
                                 # Due to a bug in the upstream biophysical modelling, we need to restrict bioenergy accounting to "harvest residue" stream only
                                 # (not sure exactly why this fixed the bug: see Zach Xu for details).
                                 # Leaving original code commented out (see below), in case we want to revert to the original design. 
                                 #data_cols=['bioenergycommodities_m3', 
                                 #           'bioenergyharvest_m3', 
                                 #           'bioenergyharvestresidue_m3', 
                                 #           'bioenergymillresidue_m3']):
                                 data_cols=['bioenergyharvestresidue_m3']):
    """
    Sum bioenergy production data by scenario, spatial unit, time step, and location.
    """
    df = pd.read_sql_query("SELECT * FROM %s;" % table_name, cnn)
    # HACK #######
    # tsa_num stored in database as float (convert to int).
    df['tsa_num'] = df['tsa_num'].astype(int)
    ##############
    # HACK #######
    # is_domestic flag stored in database as int (convert to int).
    df['is_domestic'] = df['is_domestic'].astype(int).astype(bool)
    ##############
    # HACK #######
    # Patch null data values.
    df = df.fillna(0.)
    ##############
    # HACK #######
    # Rename time columen from 'Year' to 'year'.
    df.rename(columns={'Year':'year'}, inplace=True)
    ##############
    #############################
    # HACK ######################
    # Some suites apparently include scenarios that simulate no disturbance events at all
    # for some combinations of TSA and year. This seems highly suspect, and is likely
    # symptomatic of a bug in upstream modelling, but there is no time to deal with that
    # right now, so the following hack will make sure that the MultiIndex of the result df
    # will have a valid index and not crash the MEA in the compile_net_revenue function.
    scenarios = df['scenario'].unique()
    basenames = [int(x[-2:]) for x in read_basenames('../dat/basenames.txt')]
    years = [i for i in range(1990, 2071)]
    index = pd.MultiIndex.from_product([scenarios, basenames, years, [False, True]], names=['scenario', 'tsa_num', 'year', 'is_domestic'])
    _df = pd.DataFrame(index=index)
    try:
        df = _df.join(df.set_index(groupby_cols)).reset_index()
    except:
        print(_df.index)
        print(df.index)
        assert False
    #############################
    result = df[groupby_cols+data_cols].groupby(by=groupby_cols).sum()
    return result


#def compile_displaced_production_hwp(df, scenario_names, reference_scenario_name, 
#                                     displ_coeffs = {'sawnwood_m3':{'concrete':0.54, 'steel':0.09, 'plastic':0.11},
#                                                     'panels_m3':  {'concrete':1.86, 'steel':0.41, 'plastic':0.10}}):
def compile_displaced_production_hwp(df, scenario_names, reference_scenario_name, displ_coeffs):
    df = df.loc[pd.IndexSlice[:, :, :, True], :].reset_index(level='is_domestic', drop=True)
    result = pd.DataFrame(index=df.index)
    for p1 in displ_coeffs.keys():
        for p2 in displ_coeffs[p1].keys():
            #print(p1, p2, displ_coeffs[p1][p2])
            col = 'displaced_%s' % p2
            if col not in result.columns: result[col] = 0. 
            result[col] += df[p1] * displ_coeffs[p1][p2]
    r = result.loc[reference_scenario_name].copy()
    for sn in scenario_names.keys():
        #print(sn) # debug
        d = result.loc[sn] - r
        result.loc[sn] = d.values
    return result


def compile_displaced_production_bioenergy(df, df_prodconv, 
                                           scenario_names, reference_scenario_name,
                                           index_cols=['scenario', 'tsa_num', 'year']):
    result = df.loc[pd.IndexSlice[:, :, :, True], :].reset_index(level='is_domestic', drop=True)
    nrgm3_cols = result.columns
    nrgm3dis_cols = ['%s_dis' % c for c in nrgm3_cols]
    nrgcp_cols = [c.split('_')[0] for c in nrgm3_cols]
    nrgmwh_cols = ['%s_mwh' % c for c in nrgcp_cols]
    nrgmwhdis_cols = ['%s_dis' % c for c in nrgmwh_cols]
    result_ = result.copy()
    r = result.loc[reference_scenario_name].copy()
    for sn in scenario_names.keys():
        d = result_.loc[sn] - r
        result_.loc[sn] = d.values
    result_.rename(columns=dict(zip(nrgm3_cols, nrgm3dis_cols)), inplace=True)
    result = result.join(result_)
    result = result.reset_index().set_index(index_cols[:2]).join(df_prodconv[nrgcp_cols])
    result = result.reset_index().set_index(index_cols)
    for x in zip(nrgm3_cols, nrgm3dis_cols, nrgmwh_cols, nrgmwhdis_cols, nrgcp_cols):
        cm3, cm3dis, cmwh, cmwhdis, ccp = x
        result[cmwh] = result[cm3] * result[ccp]          # bioenergy production
        result[cmwhdis] = result[cm3dis] * result[ccp]    # displaced bioenergy production
    result.drop(columns=nrgcp_cols, inplace=True)
    return result


def compile_displaced_emissions_hwp(cnn,
                                    table_name='dfe_dfp_sector_society_rt',
                                    groupby_cols=['scenario', 'tsa_num', 'year'],
                                    data_cols=['dfp_mtco2e_domestic']):
    df = pd.read_sql_query("SELECT * FROM %s;" % table_name, cnn)
    # HACK #######
    # tsa_num stored in database as float (convert to int).
    df['tsa_num'] = df['tsa_num'].astype(int).astype(int)
    ##############
    # HACK #######
    # Rename time columen from 'Year' to 'year'.
    df.rename(columns={'Year':'year'}, inplace=True)
    ##############
    result = df[groupby_cols+data_cols].groupby(by=groupby_cols).sum()
    return result


def compile_displaced_emissions_bioenergy(cnn,
                                          table_name='dfe_dfp_sector_society_rt',
                                          groupby_cols=['scenario', 'tsa_num', 'year'],
                                          data_cols=['dfe_domestic_mtco2e']):
    df = pd.read_sql_query("SELECT * FROM %s;" % table_name, cnn)
    # HACK #######
    # tsa_num stored in database as float (convert to int).
    df['tsa_num'] = df['tsa_num'].astype(int).astype(int)
    ##############
    # HACK #######
    # Rename time columen from 'Year' to 'year'.
    df.rename(columns={'Year':'year'}, inplace=True)
    ##############
    result = df[groupby_cols+data_cols].groupby(by=groupby_cols).sum()
    return result
                                          

def compile_nr_data(results, df_pcfms, df_pchwp, df_pcdis,
                    index_cols=['scenario', 'tsa_num', 'year']):
    result = {'fms':{'pv':results['fms_production'], 
                     'up':df_pcfms, 'uc':df_pcfms},
              'hwp':{'pv':results['hwp_production'].copy().reset_index().set_index(index_cols).groupby(index_cols).sum().drop(columns='is_domestic'),
                     'up':df_pchwp, 'uc':df_pchwp},
              'nrg':{'pv':results['displaced_production_bioenergy'], 
                     'up':df_pchwp, 'uc':df_pchwp},
              'ds1':{'pv':results['displaced_production_hwp'], 
                     'up':df_pcdis, 'uc':df_pcdis},
              'ds2':{'pv':results['displaced_production_bioenergy'], 
                     'up':df_pcdis, 'uc':df_pcdis}}
    return result


def compile_net_revenue(d, md, scenario_names, ref_year, unit_col='tsa_num', time_col='year', dr={'soci':0.03, 'priv':0.07}):
    from functools import partial
    discount_priv = partial(discount, dr=dr['priv'], col='sum_tnr', ref_year=ref_year)    
    discount_soci = partial(discount, dr=dr['soci'], col='sum_tnr', ref_year=ref_year)
    result = pd.DataFrame(index=d['hwp']['pv'].index)
    for x in ['fms', 'hwp', 'nrg', 'ds1', 'ds2']:
        for y in ['tr', 'tc', 'nr', 'tnr']:
            for z in md[x][y]:
                result[z] = 0.
    result['sum_tnr'] = 0.
    for sn in scenario_names.keys():
        for x in ['fms', 'hwp', 'nrg', 'ds1', 'ds2']:
            # HACK ####
            # fms_production uses special scenario definitions (yuck)
            _sn = sn if x != 'fms' else scenario_names[sn][0]
            ###########
            for i, pv_col in enumerate(md[x]['pv']):
                #print(sn, x, i, pv_col)
                ff = md[x]['ff']
                assid = scenario_names[sn][1][x]
                pv = d[x]['pv'][[pv_col]].loc[_sn].reset_index().set_index(unit_col)
                up_col = md[x]['up'][i]
                up = d[x]['up'][up_col].loc[assid]
                _pv = pv.join(up).reset_index().set_index([unit_col, time_col])
                tr = _pv[pv_col] * _pv[up_col] * ff
                #print(sn, x, i, pv_col, assid)
                try:
                    result[md[x]['tr'][i]].loc[sn].loc[tr.index] = tr
                except:
                    print(tr)
                    print(result[md[x]['tr'][i]].loc[sn])
                    assert False
                uc_col = md[x]['uc'][i]
                uc = d[x]['uc'][uc_col].loc[assid]
                _pv = pv.join(uc).reset_index().set_index([unit_col, time_col])
                tc = _pv[pv_col] * _pv[uc_col] * ff
                result[md[x]['tc'][i]].loc[sn].loc[tc.index] = tc 
                nr = tr - tc
                result[md[x]['nr'][i]].loc[sn].loc[nr.index] = nr
                result[md[x]['tnr'][0]].loc[sn].loc[nr.index] += nr
            result['sum_tnr'].loc[sn].loc[nr.index] += result[md[x]['tnr'][0]].loc[sn]
    result['sum_dtnr_priv'] = result.reset_index().apply(discount_priv, axis=1).values
    result['sum_dtnr_soci'] = result.reset_index().apply(discount_soci, axis=1).values
    return result


def compile_net_emissions(d, ref_year, dr={'soci':0.01, 'priv':0.00}):
    from functools import partial
    discount_priv = partial(discount, dr=dr['priv'], col='sum_tne', ref_year=ref_year)    
    discount_soci = partial(discount, dr=dr['soci'], col='sum_tne', ref_year=ref_year)
    result = pd.DataFrame(index=d['net_revenue'].index)
    result['sum_tne'] = 0.
    result['sum_tne'] += d['fms_emissions']['forest_mtco2e']
    result['sum_tne'] += d['hwp_emissions'][['hwp_emissions_domestic_mtco2e', 'hwp_emissions_foreign_mtco2e']].sum(axis=1)
    result['sum_tne'] += d['displaced_emissions_hwp']['dfp_mtco2e_domestic']
    result['sum_tne'] += d['displaced_emissions_bioenergy']['dfe_domestic_mtco2e']
    result['sum_dtne_priv'] = result.reset_index().apply(discount_priv, axis=1).values
    result['sum_dtne_soci'] = result.reset_index().apply(discount_soci, axis=1).values
    return result


def compile_portfolios(d, scenario_names, reference_scenario_name, period, emissions_scaling_factor=1e6,
                       goals = {0:{'label':'max_miti_soci', 'col':'sum_dtne_soci', 'sense':'min'}, # minimize net emissions
                                1:{'label':'min_cost_soci', 'col':'sum_dtnr_soci', 'sense':'max'}}, # maximize net revenue,
                       extra_portfolios=None, scen_col='scenario', unit_col='tsa_num'):
    start_year, end_year = period
    tnr = d['net_revenue'].loc[pd.IndexSlice[:, :, start_year:end_year, :], ['sum_tnr', 'sum_dtnr_priv', 'sum_dtnr_soci']].groupby(level=[0, 1]).sum()
    tne = d['net_emissions'].loc[pd.IndexSlice[:, :, start_year:end_year, :], ['sum_tne', 'sum_dtne_priv', 'sum_dtne_soci']].groupby(level=[0, 1]).sum()
    df = pd.concat([tnr, tne], axis=1)
    r = df.loc[reference_scenario_name].copy()
    for sn in scenario_names.keys():
        delta = df.loc[sn] - r
        df.loc[sn].loc[delta.index] = delta 
    df['unit_cost_priv'] = df['sum_dtnr_priv'] / (df['sum_dtne_priv']  * emissions_scaling_factor)
    df['unit_cost_soci'] = df['sum_dtnr_soci'] / (df['sum_dtne_soci']  * emissions_scaling_factor)
    portfolios = {}
    # define single-strategy portfolios
    tmp_sns = list(scenario_names.keys())
    tmp_sns.remove(reference_scenario_name)
    for i, sn in enumerate(tmp_sns):
        #if sn == reference_scenario_name: continue
        portfolios[i] = {'label':sn,
                         'ref_scenario':reference_scenario_name,
                         'alt_scenarios':[sn]}
    # define all-strategy portfolio
    i += 1
    portfolios[i] = {'label':'all',
                     'ref_scenario':reference_scenario_name,
                     'alt_scenarios':tmp_sns}
    if extra_portfolios:
        for portfolio in extra_portfolios:
            i += 1
            portfolios[i] = portfolio
    
    units = d['net_emissions'].index.get_level_values(level=1).unique()
    result = pd.DataFrame(index=pd.MultiIndex.from_product([goals.keys(), portfolios.keys(), units],
                                                           names=['goal_id', 'portfolio_id', unit_col]),
                          columns=['best_strategy']+list(df.columns))
    for gid, goal in goals.items():
        for pid, portfolio in portfolios.items():
            for unit in units: 
                scenarios = [portfolio['ref_scenario']] + portfolio['alt_scenarios']
                _df = df.loc[pd.IndexSlice[scenarios, unit, :]].reset_index().groupby(scen_col).sum().query('sum_dtne_soci <= 0.')
                best_strategy = _df[goal['col']].idxmin() if goal['sense'] == 'min' else _df[goal['col']].idxmax()
                result.loc[gid, pid, unit]['best_strategy'] = best_strategy
                for col in df.loc[best_strategy, unit].index:
                    result.loc[gid, pid, unit][col] = df.loc[best_strategy, unit][col]
    return result, portfolios, goals


def run_suite(suite_name, period, mea_params_db_path, mea_results_db_path, scenario_names1, scenario_names2,
              scenario_names1_ref, scenario_names2_ref, nr_metadata, displ_coeffs):
    #suite_db_path = suite_names[suite_name]
    #cnn = sqlite3.connect(suite_db_path)
    cnn = sqlite3.connect(suite_name)
    df_pcfms, df_pchwp, df_pcdis, df_prodconv = import_mea_params(mea_params_db_path)
    results = {}
    results['fms_production'] = compile_fms_production(cnn, scenario_names1, scenario_names1_ref)
    results['fms_emissions'] = compile_fms_emissions(cnn)
    results['hwp_production'] = compile_hwp_production(cnn)
    results['hwp_emissions'] = compile_hwp_emissions(cnn)
    results['bioenergy_production'] = compile_bioenergy_production(cnn)
    results['displaced_production_hwp'] = compile_displaced_production_hwp(results['hwp_production'],
                                                                           scenario_names2, scenario_names2_ref,
                                                                           displ_coeffs=displ_coeffs)
    results['displaced_production_bioenergy'] = compile_displaced_production_bioenergy(results['bioenergy_production'],
                                                                                       df_prodconv,
                                                                                       scenario_names2, scenario_names2_ref)
    results['displaced_emissions_hwp'] = compile_displaced_emissions_hwp(cnn)
    results['displaced_emissions_bioenergy'] = compile_displaced_emissions_bioenergy(cnn)
    results['net_revenue'] = compile_net_revenue(d=compile_nr_data(results, df_pcfms, df_pchwp, df_pcdis), 
                                                 md=nr_metadata, 
                                                 scenario_names=scenario_names2, 
                                                 ref_year=period[0])
    results['net_emissions'] = compile_net_emissions(d=results, ref_year=period[0])
    results['portfolios'], portfolios, goals = compile_portfolios(results, scenario_names2, scenario_names2_ref, period)
    export_results_to_sqlite(results, mea_results_db_path)
    return results, portfolios, goals


def export_results_to_sqlite(results, db_path):
    cnn = sqlite3.connect(db_path)
    for k in results.keys():
        results[k].to_sql(k, cnn, if_exists='replace', index=True)
    cnn.close()

    
def compile_mcc_figure(mea_results, goals, portfolios, gid, pid, use_plotly=False, fig_ax=None, figsize=(8, 6)):
    df = mea_results['portfolios'].loc[gid, pid].sort_values(by='unit_cost_soci')
    x = -df['sum_tne'].cumsum().values
    x = np.insert(x, 0, 0.)
    y = df['unit_cost_soci'].values
    y = np.insert(y, 0, y[0])
    fig_title = 'goal:%s, portfolio:%s' % (goals[gid]['label'], portfolios[pid]['label'])
    xlabel = 'Cumulative mitigation (MtCO2e)'
    ylabel = 'Unit cost ($ per tCO2e)'
    if use_plotly:
        import plotly.plotly as py
        import plotly.graph_objs as go
        trace = go.Scatter(x=x, y=y, mode='lines+markers', name='MCC', hoverinfo='y',
                           line=dict(shape='vh'), opacity=0.5)
        layout = go.Layout(title=fig_title,
                           xaxis=dict(title=xlabel),
                           yaxis=dict(title=ylabel))
        return dict(data=[trace], layout=layout), None
    else: #use matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=figsize) if not fig_ax else fig_ax
        x = np.concatenate(([0.], -df['sum_tne'].cumsum()))
        y = np.concatenate(([df['unit_cost_soci'].iloc[0]], df['unit_cost_soci']))
        ax.step(x=x, y=y, where='pre')
        ax.set_xlim([0, None])
        ax.set_title(fig_title)
        #ax.axhline(y=0, color='k')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        return fig, ax
