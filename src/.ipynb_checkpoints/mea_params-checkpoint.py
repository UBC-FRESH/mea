import mdbread
import numpy as np
import pandas as pd
import sqlite3
import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact_manual, interactive, fixed
from IPython.display import display

import mea

dat_path = '../dat'
mea_params_db_path = dat_path + '/mea_params'
mea_results_db_path = dat_path + '/mea_results'
#mea_params_db_path = dat_path + '/mea_params.db'
#mea_results_db_path = dat_path + '/mea_results.db'
cat_db_path= dat_path + '/cat_out/20190102_results'

suite_names = {'default_base':'default_base',
               'default_clean':'default_clean',
               'default_woodbuildings':'default_woodbuildings',
               'lo_base':'lo_base',
               'lo_clean':'lo_clean',
               'lo_wood_buildings':'lo_woodbuildings',
               'hi_base':'hi_base',
               'hi_clean':'hi_clean',
               'hi_wood_buildings':'hi_woodbuildings'}

basenames = [int(x[-2:]) for x in mea.read_basenames(dat_path+'/basenames.txt')]

fms_emit_factor = 1. # 0.000001


# TO DO: link this script to discount rate parameter arg in MEA
# discount rates (extracted from db.tblDiscountRates)
#params_dr = {'dr_soci_econ':0.03,
#             'dr_soci_emit':0.01,
#             'dr_priv_econ':0.07,
#             'dr_priv_emit':0.00}

displ_coeffs = {'sawnwood_m3':{'concrete':3.46, 'steel':0.59, 'plastic':0.28},
                'panels_m3':  {'concrete':3.89, 'steel':0.85, 'plastic':0.20}}

scen_col = 'scenario'
unit_col = 'tsa_num'
time_col = 'year'

scenario_names1_ref = 'CBM_BASE'
scenario_names1 = ['CBM_BASE', 'CBM_A', 'CBM_B', 'CBM_C', 'CBM_D', 'CBM_OG']
scenario_names2_ref = 'CBM_Base_HWP_Base'
scenario_names2 = {'CBM_Base_HWP_Base':('CBM_BASE', {'fms':1, 'hwp':1,  'nrg':1,  'ds1':1, 'ds2':0}),
                   'CBM_Base_HWP_LLP': ('CBM_BASE', {'fms':1, 'hwp':7,  'nrg':7,  'ds1':1, 'ds2':0}),
                   'CBM_A_HWP_Base':   ('CBM_A',    {'fms':2, 'hwp':2,  'nrg':2,  'ds1':1, 'ds2':0}),
                   'CBM_A_HWP_LLP':    ('CBM_A',    {'fms':2, 'hwp':8,  'nrg':8,  'ds1':1, 'ds2':0}),
                   'CBM_B_HWP_Base':   ('CBM_B',    {'fms':3, 'hwp':3,  'nrg':3,  'ds1':1, 'ds2':0}),
                   'CBM_B_HWP_LLP':    ('CBM_B',    {'fms':3, 'hwp':9,  'nrg':9,  'ds1':1, 'ds2':0}),
                   'CBM_C_HWP_Base':   ('CBM_C',    {'fms':4, 'hwp':4,  'nrg':4,  'ds1':1, 'ds2':1}),
                   'CBM_C_HWP_LLP':    ('CBM_C',    {'fms':4, 'hwp':10, 'nrg':10, 'ds1':1, 'ds2':1}),
                   'CBM_D_HWP_Base':   ('CBM_D',    {'fms':5, 'hwp':5,  'nrg':5,  'ds1':2, 'ds2':2}),
                   'CBM_D_HWP_LLP':    ('CBM_D',    {'fms':5, 'hwp':11, 'nrg':11, 'ds1':2, 'ds2':2}),
                   'CBM_OG_HWP_Base':  ('CBM_OG',   {'fms':6, 'hwp':6,  'nrg':6,  'ds1':1, 'ds2':0}),
                   'CBM_OG_HWP_LLP':   ('CBM_OG',   {'fms':6, 'hwp':12, 'nrg':12, 'ds1':1, 'ds2':0})}

idx = pd.IndexSlice
#results = {}

nr_metadata = {'fms':{
                   'pv':['Live_SW_Harvest_Vol', 'Live_HW_Harvest_Vol', 'Dead_Wood_harvest_Vol', 'Harvest_Residue_Volume'],
                   'up':['SWLogPrice', 'HWLogPrice', 'SalvageLogPrice', 'ResiduePrice'],
                   'uc':['SWLogCost', 'HWLogCost', 'SalvageLogCost', 'ResidueCost'],
                   'tr':['fms_swh_tr', 'fms_hwh_tr', 'fms_slh_tr', 'fms_rsh_tr'],        
                   'tc':['fms_swh_tc', 'fms_hwh_tc', 'fms_slh_tc', 'fms_rsh_tc'],       
                   'nr':['fms_swh_nr', 'fms_hwh_nr', 'fms_slh_nr', 'fms_rsh_nr'],       
                   'tnr':['fms_tnr'],
                   'ff':1.},
               'hwp':{
                   'pv':['sawnwood_m3', 'panels_m3', 'otherirw_m3', 'pulppaper_odt'], 
                   'up':['SawnwoodPrice', 'PanelPrice', 'OtherIRPrice', 'PulpPrice'],
                   'uc':['SawnwoodCost', 'PanelCost', 'OtherIRCost', 'PulpCost'],
                   'tr':['hwp_swd_tr', 'hwp_pnl_tr', 'hwp_orw_tr', 'hwp_plp_tr'],        
                   'tc':['hwp_swd_tc', 'hwp_pnl_tc', 'hwp_orw_tc', 'hwp_plp_tc'],       
                   'nr':['hwp_swd_nr', 'hwp_pnl_nr', 'hwp_orw_nr', 'hwp_plp_nr'],
                   'tnr':['hwp_tnr'],
                   'ff':1.},
               'nrg':{
                   # HACK ###
                   # Due to a bug in the upstream biophysical modelling, we need to restrict bioenergy 
                   # accounting to "harvest residue" stream only (not sure exactly why this fixed the bug,
                   # see Zach Xu for details). Leaving original code commented out (see below), 
                   # in case we want to revert to the original design. 
                   #'pv':['bioenergycommodities_mwh', 'bioenergyharvest_mwh', 'bioenergyharvestresidue_mwh', 'bioenergymillresidue_mwh'],
                   #'up':['BioenergyFromProductEOLPrice', 'BioenergyFromHarvestPrice', 'BioenergyFromHarvestResiduePrice', 'BioenergyFromMillResiduePrice'],
                   #'uc':['BioenergyFromProductEOLCost', 'BioenergyFromHarvestCost', 'BioenergyFromHarvestResidueCost', 'BioenergyFromMillResidueCost'],
                   #'tr':['nrg_cmd_tr', 'nrg_hvl_tr', 'nrg_hvr_tr', 'nrg_mlr_tr'],        
                   #'tc':['nrg_cmd_tc', 'nrg_hvl_tc', 'nrg_hvr_tc', 'nrg_mlr_tc'],       
                   #'nr':['nrg_cmd_nr', 'nrg_hvl_nr', 'nrg_hvr_nr', 'nrg_mlr_nr'],
                   'pv':['bioenergyharvestresidue_mwh'],
                   'up':['BioenergyFromHarvestResiduePrice'],
                   'uc':['BioenergyFromHarvestResidueCost'],
                   'tr':['nrg_hvr_tr'],        
                   'tc':['nrg_hvr_tc'],       
                   'nr':['nrg_hvr_nr'],
                   ###########
                   'tnr':['nrg_tnr'],
                   'ff':1.},
               'ds1':{
                   'pv':['displaced_concrete', 'displaced_plastic'],
                   'up':['ConcretePrice', 'PlasticPrice'],
                   'uc':['ConcreteCost', 'PlasticCost'],
                   'tr':['ds1_cnc_tr', 'ds1_pls_tr'],        
                   'tc':['ds1_cnc_tc', 'ds1_pls_tc'],       
                   'nr':['ds1_cnc_nr', 'ds1_pls_nr'],
                   'tnr':['ds1_tnr'],
                   'ff':-1.},
                'ds2':{
                    # HACK ###
                    # Due to a bug in the upstream biophysical modelling, we need to restrict bioenergy accounting to "harvest residue" stream only
                    # (not sure exactly why this fixed the bug: see Zach Xu for details).
                    # Leaving original code commented out (see below), in case we want to revert to the original design. 
                    #'pv':['bioenergycommodities_mwh_dis', 'bioenergyharvest_mwh_dis', 'bioenergyharvestresidue_mwh_dis', 'bioenergymillresidue_mwh_dis'],
                    #'up':['EnergyPrice', 'EnergyPrice', 'EnergyPrice', 'EnergyPrice'],
                    #'uc':['EnergyCost', 'EnergyCost', 'EnergyCost', 'EnergyCost'],
                    #'tr':['ds2_cmd_tr', 'ds2_hvl_tr', 'ds2_hvr_tr', 'ds2_mlr_tr'],        
                    #'tc':['ds2_cmd_tc', 'ds2_hvl_tc', 'ds2_hvr_tc', 'ds2_mlr_tc'],       
                    #'nr':['ds2_cmd_nr', 'ds2_hvl_nr', 'ds2_hvr_nr', 'ds2_mlr_nr'],
                    'pv':['bioenergyharvestresidue_mwh_dis'],
                    'up':['EnergyPrice'],
                    'uc':['EnergyCost'],
                    'tr':['ds2_hvr_tr'],        
                    'tc':['ds2_hvr_tc'],       
                    'nr':['ds2_hvr_nr'],
                    ###########
                    'tnr':['ds2_tnr'],
                    'ff':-1.}}

#suite_options = [(v[:-26], v) for v in ss_db_basenames]
suite_select = widgets.Select(options=suite_names.items(), value=list(suite_names.values())[0], description='Suite:', disabled=False)
scenario_select = widgets.SelectMultiple(options=list(scenario_names2.keys()),
                                         value=list(scenario_names2.keys()),
                                         description='Scenarios',
                                         rows=12,
                                         disabled=False)
time_slider = widgets.IntRangeSlider(value=[2020, 2070],
                                     min=1990,
                                     max=2070,
                                     step=1,
                                     description='Period:',
                                     disabled=False,
                                     continuous_update=True,
                                     orientation='horizontal',
                                     readout=True,
                                     readout_format='d',
                                     layout={'width':'500px'})
#run_button1 = widgets.Button(description='Run Selected Suite', disabled=False, button_style='')
#widget_out1 = widgets.Output()
#widget_box1 = widgets.VBox([time_slider, ss_select, scenario_select, run_button1, widget_out1],
#                          layout={'width':'800px'})

#@widget_out1.capture()
def run_mea(period, suite, scenarios):
    global mea_results
    global portfolios
    global goals
    global goal_select
    global portfolio_select
    print('Compiling results for scenario suite: %s' % suite)
    _scenario_names2 = {k:v for k, v in scenario_names2.items() if k in scenarios}
    _scenario_names1 = list({v[0] for v in _scenario_names2.values()})
    mea_results, portfolios, goals = mea.run_suite('%s/%s_future_aggregated_results.db' % (cat_db_path, suite),
                                                   period,
                                                   '%s/mea_params-%s.db' % (mea_params_db_path, suite),
                                                   '%s/mea_results-%s.db' % (mea_results_db_path, suite),
                                                   _scenario_names1, _scenario_names2,
                                                   scenario_names1_ref, scenario_names2_ref,
                                                   nr_metadata,
                                                   displ_coeffs=displ_coeffs)
    goal_select = widgets.Select(options=[(v['label'], k) for k, v in goals.items()], description='Goal')
    portfolio_select = widgets.Select(options=[(v['label'], k) for k, v in portfolios.items()], description='Portfolio')
    #goal_select.options = [(v['label'], k) for k, v in goals.items()]
    #portfolio_select.options = [(v['label'], k) for k, v in portfolios.items()]
    print('Done')
    return mea_results, portfolios, goals, portfolio_select, goal_select

def run_mea_all(period=None):
    if not period: period = time_slider.value
    for suite in suite_names.values():
        print('Running MEA for suite', suite)
        run_mea(period, suite, scenarios=list(scenario_names2.keys()))
    print('Done MEA. See %s for output.' % mea_results_db_path)

run_mea_widget = interactive(run_mea,
                             {'manual':True, 'manual_name':'Run MEA (selected)', 'auto_display':False},
                             period=time_slider, suite=suite_select, scenarios=scenario_select)

savefig_path = widgets.Text(value='', placeholder='path for savefig (e.g., images/mcc01.pdf)', description='Filename')
plotting_backend = widgets.RadioButtons(options=['plotly', 'matplotlib'], value='plotly', description='Backend')

def plot_mcc(gid, pid, savefig_path='', plotting_backend='plotly'):
    use_plotly = plotting_backend in ['plotly']
    fig, ax = mea.compile_mcc_figure(mea_results, goals, portfolios, 
                                      goal_select.value, portfolio_select.value, 
                                      use_plotly=use_plotly)
    if use_plotly:
        from plotly.plotly import iplot
        from plotly.io import write_image
        if savefig_path: write_image(fig, savefig_path)
        return iplot(fig, filename=savefig_path)
    else:
        if savefig_path:
            plt.savefig(savefig_path, format=savefig_path[-3:])
        return# fig, ax

#goal_select = widgets.Select()
#portfolio_select = widgets.Select()

def create_plot_mcc_widget():
    return interactive(plot_mcc,
                       {'manual':True, 'manual_name':'Plot MCC', 'auto_display':True},
                       gid=goal_select,
                       pid=portfolio_select,
                       savefig_path=savefig_path,
                       plotting_backend=plotting_backend)

# @widget_out1.capture()
# def on_button1_clicked(b):
#     #widget_out1.clear_output()
#     print('Compiling results for scenario suite: %s' % ss_select.value)
#     _scenario_names2 = {k:v for k, v in scenario_names2.items() if k in scenario_select.value}
#     _scenario_names1 = list({v[0] for v in _scenario_names2.values()})
#     global mea_results
#     global portfolios
#     global goals
#     mea_results, portfolios, goals = mea.run_scenario_suite('%s/%s.db' % (cat_db_path, ss_select.value),
#                                                             time_slider.value, mea_params_db_path,
#                                                             _scenario_names1, _scenario_names2,
#                                                             scenario_names1_ref, scenario_names2_ref, nr_metadata)
#    print('Done')

#run_button1.on_click(on_button1_clicked)

#goals, portfolios = {}, {}
#goal_select = widgets.Select(options=[(v['label'], k) for k, v in goals.items()], description='Goal')
#portfolio_select = widgets.Select(options=[(v['label'], k) for k, v in portfolios.items()], description='Portfolio')
#run_button2 = widgets.Button(description='Show MCC for selected goal and portfolio', disabled=False, button_style='')
#savefig_path = widgets.Text(value='', placeholder='path for savefig (e.g., mcc01.pdf)', description='Filename')
#widget_box2 = widgets.VBox([goal_select, portfolio_select, run_button2])

#def on_button2_clicked(b):
#    global fig
#    global ax
#    fig, ax = plot_mcc(mea_results, goals, portfolios, goal_select.value, portfolio_select.value, savefig_path=savefig_path.value)

#run_button2.on_click(on_button2_clicked)
