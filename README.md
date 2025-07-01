# Model for Economic Analysis (MEA)
Version `0.1alpha`

This notebook implements the _model for economic analysis_ (MEA), which estimates _marginal cost curves_ (MCC) of  forest sector climate change mitigation scenarios simulated in the context of the phase 2 PICS research project. 

The notebook contains high-level interactive controls for running the model and plotting various scenarios.

Most of the functions implementing the MEA model are in the `mea.py` Python module. Some user-settable parameters are hard-coded into the `mea_params.py` Python file, while other parameters (including price, cost, and product conversion coefficient data tables) are stored in the `mea_params.db` SQLite database. 

The model also allows for interactive exploration of detailed MEA output (including intermediate processing steps) via an _interactive IPython console_. The `mea_results` global variable contains a `dict` of `pandas.DataFrame` objects, keyed on step name. 

The list of keys is as follows 

```python
['fms_production',
 'fms_emissions',
 'hwp_production',
 'hwp_emissions',
 'bioenergy_production',
 'displaced_production_hwp',
 'displaced_production_bioenergy',
 'displaced_emissions_hwp',
 'displaced_emissions_bioenergy',
 'net_revenue',
 'net_emissions',
 'portfolios']
```

To display an interactive console pane, right-click in the notebook and select _New Console for Notebook_.

The overall MEA modelling processNote that the notebooks interface is divided into two parts. 

The first part of the interface allows the user to select one of 9 _suites_ of scenarios to import and compile using the MEA. Each suite contains output from upstream GCBM and HWP simulations for the same 12 scenarios, but using different sets of model parameters to simulate the scenarios. 

The second part of the interface allows the user to display and interactively explore the MCC for one of several available _portfolios_ of scenarios. The default plotting functions use the `plotly` library, which allows interactive graph exploration, as well as interactive editing of graphs one the [plotly](https://plot.ly/#/) web site. Using the plotly functions requires users to sign up for a free plotly account. This can be bypassed by using the default `matplotlib` library for plotting, but the interactive graph exploration and editing functions will no longer be available.