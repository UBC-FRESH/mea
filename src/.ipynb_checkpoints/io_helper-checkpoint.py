# help functions and classes for forest IO model in BC, Canada
# version: 0.1
# date: 22 Aug, 2018
# 
# Author: Jon Duan, Zach Xu


# help function to convert datarame to dictionary
# In[]
import pandas as pd

def multipliers_to_dict(multipliers, name = ""):
    """
    input:
    multipliers: pd.dataframe with all information
    name: industry name

    output:
    a dictionary with industry names as keys and multipliers as values 
    """
    industry_multipliers = multipliers[multipliers["industry"]== name]
    industry_multipliers_dict = dict(zip( industry_multipliers.columns.values,
                                            industry_multipliers.iloc[0,:])) 
    return industry_multipliers_dict

full_industry_list = ["wood", "paper","harvest","bioenergy","residue_extraction",
                 "concrete","plastics","electricity"]

forestry_list = ["wood", "paper","harvest","bioenergy","residue_extraction" ]

other_industry_list = ["concrete","plastics","electricity"]


full_columns = ['year', 'direct_employment', 'indirect_employment',
       'total_employment', 'direct_output',  'indirect_output', 'total_output', 'direct_GDP',
       'indirect_GDP', 'total_GDP', 'government_revenue'] 


def import_industry_input(file, input_sheet_name, industry_list):
    """
    input: file name, sheet name, and a list of industry
    output: a dictionary of pandas dataframe
    """
    import pandas as pd
    input_physical_monetary = pd.read_excel(file, sheet_name= input_sheet_name, usecol = "A:Q") # import data form sheet
    
    industry_input_data = dict.fromkeys(industry_list, 0) # initial a dict, default value is 0
    # dict only references the object, if referenced objects changed, the dict change. 
    for i in industry_list: 
        position_in_list = list(input_physical_monetary.columns).index(i)
        # take two columns
        # put into dict with deep copy, otherwise it is only a view which will change with actoin on original memory address
        # start from row 2
        # first row is the industry names which like "paper	Unnamed: 1	wood	Unnamed: 3	harvest	Unnamed: 5	bioenergy	Unnamed: 7	residue_extraction	Unnamed: 9	concrete	Unnamed: 11	electricity	Unnamed: 13	plastic	Unnamed: 15"
        industry_input_data[i]=input_physical_monetary.iloc[1:,position_in_list:position_in_list+2].copy().reset_index()
        # change the columns names
        industry_input_data[i].columns = ['year','physical_diff','monetary_diff']
        industry_input_data[i][['physical_diff','monetary_diff']]=industry_input_data[i][['physical_diff','monetary_diff']].astype(float)
    return industry_input_data


def import_forestry_input(file, input_sheet_name, forestry_list):
    """
    input: file name, sheet name, and a list of forestry industry
    output: a dictionary of pandas dataframe
    """
    import pandas as pd
    input_physical_monetary = pd.read_excel(file, sheet_name= input_sheet_name, usecol = "A:K") # import data form sheet
    
    forestry_input_data = dict.fromkeys(forestry_list, 0) # initial a dict, default value is 0
    # dict only references the object, if referenced objects changed, the dict change. 
    for i in forestry_list: 
        position_in_list = list(input_physical_monetary.columns).index(i)
        # take two columns
        # put into dict with deep copy
        # start from row 2
        forestry_input_data[i]=input_physical_monetary.iloc[1:,position_in_list:position_in_list+2].copy().reset_index()
        # change the columns names
        forestry_input_data[i].columns = ['year','physical_diff','monetary_diff']
        forestry_input_data[i][['physical_diff','monetary_diff']]=forestry_input_data[i][['physical_diff','monetary_diff']].astype(float)
    return forestry_input_data





class Impact(object):
    """
    The impact of a change in final demand on Output/GDP/Employment/Tax

    multipliers are a dictionary of direct effect, indirect effect, induced effect multipliers  
    examples:
    multipliers = {"direct_impact": 1.2, "indirect_impact": 1.34, "induced_impact": 0.1}
    
    direct_output is the NPV from the difference between base case and strategical case
    direct_output:
        year:
        physical_diff: different biophysical production between base case and strategic scenario. 
        monetary_diff: different monetray outcome between base case and strategic scenario. 
    examples:
        pd.dataframe["year", "physical_diff", "monetary_diff"]
        the direct_output is only in monetary_diff term, biopysical_diff only for forestries to compute unit labors.
        direct_output = pd.dataframe["year", "direct_output"] = pd.dataframe["year", "monetary_diff"]

   """
    import pandas as pd
    def __init__(self, multiplier, direct_output): 
        self.__multipliers = multiplier # one dictionary
        self.__direct_output = pd.DataFrame(direct_output["year"])         # one dataframe
        self.__direct_output["direct_output"] = direct_output.loc[:,"direct_output"]
        self.__direct_impact = pd.DataFrame(direct_output["year"]) # initialize and only has time columns
        self.__indirect_impact = pd.DataFrame(direct_output["year"])
        self.__total_impact = pd.DataFrame(direct_output["year"])
        # self.induced_impact = []
    
    @property
    def multipliers(self):
        return self.__multipliers
    @multipliers.setter
    def multipliers(self, new_multipliers):
        self.__multipliers = new_multipliers

    @property 
    def direct_output(self):
        return self.__direct_output
    @direct_output.setter
    def direct_output(self, new_direct_output):
        self.__direct_output = new_direct_output


    @property
    def direct_impact(self):
        self.__direct_impact["direct_impact"]  = self.__direct_output["direct_output"] * self.__multipliers["direct_impact"] 
        return self.__direct_impact        
    @property
    def indirect_impact(self):

        self.__indirect_impact["indirect_impact"] = self.__direct_output["direct_output"] * self.__multipliers["indirect_impact"]
        return self.__indirect_impact
    @property
    def total_impact(self):
        self.__total_impact["total_impact"] = self.direct_impact["direct_impact"] +  self.indirect_impact["indirect_impact"]
        return self.__total_impact




# In[26]:


class ImpactOnOutput(Impact):
    """
    The same as the impact
    """
    pass


# In[27]:


class ImpactOnGDP(Impact):
    """
    The same as the impact
    """
    pass


# In[ ]:


class ImpactOnEmployment(Impact):
    """
    The same as the impact
    """
    def __init__(self, multiplier, direct_output):
        super().__init__(multiplier, direct_output )
        self.__multipliers = multiplier # one dictionary
        self.__direct_output = pd.DataFrame(direct_output["year"])         # one dataframe
        self.__direct_output["direct_output"] = direct_output.loc[:,"direct_output"] # avoid copy
        self.__direct_impact = pd.DataFrame(direct_output["year"]) # initialize and only has time columns
        self.__indirect_impact = pd.DataFrame(direct_output["year"])
        self.__total_impact = pd.DataFrame(direct_output["year"])
        # still need to init direct output ????
    # the FTE multipliers are per million dollar    
    @property
    def direct_impact(self):
        self.__direct_impact["direct_impact"]  = self.__direct_output["direct_output"]/1000000 * self.__multipliers["direct_impact"] 
        return self.__direct_impact        
    @property
    def indirect_impact(self):

        self.__indirect_impact["indirect_impact"] = self.__direct_output["direct_output"]/1000000 * self.__multipliers["indirect_impact"]
        return self.__indirect_impact
    @property
    def total_impact(self):
        self.__total_impact["total_impact"] = self.direct_impact["direct_impact"] +  self.indirect_impact["indirect_impact"]
        return self.__total_impact

   



# In[27]:


class ImpactOnTax(Impact):
    """
    Only direct impact, indirect is zero
    """
    pass


# In[]
class ImpactOnEmploymentForesty(Impact):
    """
    The employment impact is from unit-labor assumption. More realistic. 
    unit_labor = scalar  represent the number of FTE per difference between base case and strategic scenario m3
    example: unit_labor = 0.2
    
    direct_output is the NPV from the difference between base case and strategical case
    direct_output:
        year:
        physical_diff: different biophysical production between base case and strategic scenario. 
        monetary_diff: different monetray outcome between base case and strategic scenario. 
    examples:
        pd.dataframe["year", "physical_diff", "monetary_diff"]
    the direct_output is only in monetary_diff term, biopysical_diff only for forestries to compute unit labors.
    direct_output = pd.dataframe["year", "direct_output"] = pd.dataframe["year", "monetary_diff"]
        
    direct_production: differet production between base case and strategic scenario
    
    if modify the method and the attributrs need to be modified as well
    """
    def __init__(self, multiplier, direct_output, unit_labor, direct_production):
        super().__init__(multiplier, direct_output )
        self.__multipliers = multiplier # one dictionary
        self.__unit_labor = unit_labor
        self.__direct_production = pd.DataFrame(direct_production["year"])         # one dataframe
        self.__direct_production["direct_production"] = direct_production.loc[:,"direct_production"] # avoid copy error
        self.__direct_impact = pd.DataFrame(direct_output["year"]) # initialize and only has time columns
        self.__indirect_impact = pd.DataFrame(direct_output["year"])
        self.__total_impact = pd.DataFrame(direct_output["year"])
        

    @property 
    def unit_labor(self):
        return self.__unit_labor
    @unit_labor.setter
    def unit_labor(self, new_unit_labor):
        self.__unit_labor = new_unit_labor

    @property 
    def direct_production(self):
        return self.__direct_production
    @direct_production.setter
    def direct_production(self, new_direct_production):
        self.__direct_production = new_direct_production
    

    @property
    def direct_impact(self):
        self.__direct_impact["direct_impact"]  = self.__direct_production["direct_production"] * self.__unit_labor
        return self.__direct_impact  

    @property
    def indirect_impact(self):
        self.__indirect_impact["indirect_impact"] = self.direct_impact["direct_impact"]* (self.__multipliers["direct_impact"]/self.__multipliers["indirect_impact"])
        return self.__indirect_impact
    
    @property
    def total_impact(self):
        self.__total_impact["total_impact"] = self.direct_impact["direct_impact"] +  self.indirect_impact["indirect_impact"]
        return self.__total_impact

     






# class for industry which includes impact on  employment, output, GDP, government revenue, 



class Industry(object):
    """
    One industry can has several Output/GDP/Employment/Tax impacts.
    Forestry industries have different employment computation, which uses unit-labor assumptions. 
    multipliers: pd.dataframe with all information
    direct_output is the NPV from the difference between base case and strategical case
    direct_output:
        year:
        physical_diff: different biophysical production between base case and strategic scenario. 
        monetary_diff: different monetray outcome between base case and strategic scenario. 
    examples:
        pd.dataframe["year", "physical_diff", "monetary_diff"]
        direct_output = pd.dataframe["year", "monetary_diff"]

    """
    import pandas as pd


       
    def __init__(self, name, multiplier, direct_output):
        self.name = name
        self.multipliers =  multipliers_to_dict(multiplier,name) # pd.dataframe to dict
        self.direct_output = pd.DataFrame(direct_output["year"])         # one dataframe in monetary outcome
        self.direct_output["direct_output"] = direct_output.loc[:,"monetary_diff"]
        # output
        self.output_multipliers = {"direct_impact":self.multipliers["Direct Output"], "indirect_impact":self.multipliers["Indirect Output"]}
        self.impact_output = ImpactOnOutput(self.output_multipliers, self.direct_output)
        # GDP
        self.GDP_multipliers = {"direct_impact":self.multipliers["Direct GDP"],
                                    "indirect_impact":self.multipliers["Indirect GDP"]}

        self.impact_GDP = ImpactOnGDP(self.GDP_multipliers, self.direct_output)
        # employment
        self.employment_multipliers = {"direct_impact":self.multipliers["Direct Employment"],
                                    "indirect_impact":self.multipliers["Indirect Employment"]}
        self.__impact_employment = None
        # tax
        self.tax_multipliers = {"direct_impact":self.multipliers["Government Revenue"],
                            "indirect_impact":0}
        self.impact_tax = ImpactOnTax(self.tax_multipliers, self.direct_output)
        # full_table
        self.__full_table = pd.DataFrame(direct_output["year"])         # iniatial a dataframe
        self.__full_table["direct_output"] = direct_output.iloc[:,1]
    
        self.full_columns = ['year', 'direct_employment', 'indirect_employment',
       'total_employment', 'direct_output',  'indirect_output', 'total_output', 'direct_GDP',
       'indirect_GDP', 'total_GDP', 'government_revenue']  

    @property
    def impact_employment(self):
        self.__impact_employment =  ImpactOnEmployment(self.employment_multipliers, self.direct_output)
        return self.__impact_employment
    
    @property
    def full_table(self):
        self.__full_table['direct_employment'] = self.impact_employment.direct_impact["direct_impact"]  
        self.__full_table['indirect_employment'] = self.impact_employment.indirect_impact["indirect_impact"] 
        self.__full_table['total_employment'] = self.impact_employment.total_impact["total_impact"] 
        self.__full_table['indirect_output'] = self.impact_output.indirect_impact["indirect_impact"] 
        self.__full_table['total_output'] = self.impact_output.total_impact["total_impact"]
        self.__full_table['direct_GDP'] = self.impact_GDP.direct_impact["direct_impact"]  
        self.__full_table['indirect_GDP'] = self.impact_GDP.indirect_impact["indirect_impact"] 
        self.__full_table['total_GDP'] = self.impact_GDP.total_impact["total_impact"] 
        self.__full_table['government_revenue'] = self.impact_tax.total_impact["total_impact"] 
        
        self.__full_table = self.__full_table.reindex(columns = self.full_columns)  
        return self.__full_table



# modify the employment calculation for forest industries


class Forestry(Industry):
    """
    One industry can has several Output/GDP/Employment/Tax impacts.
    Forestry industries have different employment computation, which uses unit-labor assumptions. 
    
    The employment impact is from unit-labor assumption. More realistic. 
    unit_labor = scalar  represent the number of FTE per difference between base case and strategic scenario m3
    example: unit_labor = 0.2direct_output
    direct_production: differet production between base case and strategic scenario. Pandas dataframe similar to direct_output
    physical outcome 
    """

    def __init__(self,name, multiplier, direct_output,unit_labor, direct_production ):
        super().__init__(name,multiplier, direct_output)
        self.unit_labor = unit_labor.loc[unit_labor["industry"]== name, "unit_labor"].values[0]
        self.direct_production = pd.DataFrame(direct_production["year"])         # one dataframe
        self.direct_production["direct_production"] = direct_production.loc[:,"physical_diff"]

        # the employment is from the unit labor
    @property
    def impact_employment(self):
        self.__impact_employment = ImpactOnEmploymentForesty(self.employment_multipliers,  self.direct_output, 
                                                            self.unit_labor, self.direct_production)
        return self.__impact_employment


class Harvest(Forestry):
    pass


# In[28]:


class Sawnwood(Forestry):
    pass


# In[29]:


class Panel(Forestry):
    pass


# In[30]:


class PulpPaper(Forestry):
    pass


# In[31]:


class OtherIndustry(Industry):
    pass


# In[32]:


class Steel(OtherIndustry):
    pass


# In[ ]:


class Concrete(OtherIndustry):
    pass


# In[25]:



def to_final_result(industry_list):
    # initial final result, again reference error
    final_result = industry_list[0].full_table.copy()
    for i in industry_list[1:]:
        final_result.iloc[:, 1:] = final_result.iloc[:, 1:].add(i.full_table.iloc[:, 1:])
    # NOT first row/year in which every thing is zero
    OUTPUT_SUMMARY_SUM = final_result.iloc[1:, ].sum()
    OUTPUT_SUMMARY_MEAN = final_result.iloc[1:, ].mean()
    OUTPUT_SUMMARY_STD = final_result.iloc[1:, ].std()
    OUTPUT_SUMMARY_MAX = final_result.iloc[1:, ].max()
    OUTPUT_SUMMARY_MIN = final_result.iloc[1:, ].min()
    final_result.loc["sum"] = OUTPUT_SUMMARY_SUM
    final_result.loc["mean"] = OUTPUT_SUMMARY_MEAN
    final_result.loc["std"] = OUTPUT_SUMMARY_STD
    final_result.loc["max"] = OUTPUT_SUMMARY_MAX
    final_result.loc["min"] = OUTPUT_SUMMARY_MIN
    return final_result


def final_result_to_excel(final_result, output_file ,output_sheet_name):
    from openpyxl import load_workbook
    book = load_workbook(output_file)
    #writer = pd.ExcelWriter(file, engine= 'xlsxwriter')
    with pd.ExcelWriter(output_file, engine= 'openpyxl') as writer:
        writer.book = book
        final_result.to_excel(writer, sheet_name= output_sheet_name)
        writer.save()    



# 
class Strategy(object):
    """
    One strategy has several industries. Each industry has several Output/GDP/Employment/Tax impacts.
    Forestry industries have different employment computation, which uses unit-labor assumptions. 
    
    The employment impact is from unit-labor assumption. More realistic. 
    unit_labor = scalar  represent the number of FTE per difference between base case and strategic scenario m3
    example: unit_labor = 0.2direct_output
    direct_production: differet production between base case and strategic scenario. Pandas dataframe similar to direct_output
    physical outcome 
    
        input:::
    # input data for computation
    input_file = "data_input.xlsx"
    # output data for reporting
    output_file = "output_data.xlsx"



    # specify input data by select the sheets names in the Excel file
    input_sheet_name = "FM2-BASE-INPUT"
    output_sheet_name = 'FM2-BASE-OUTPUT'
    # regular multipliers
    multipliers_sheet = 'multipliers'
    # regular parameters                 
    parameters_sheet = 'parameters'
    # regular unit labor
    unitlabor_sheet = "unit-labor"

    For exampl, FM1 strategy has 3 forest industries and 1 concrete industry:  
    
    One industry can has several Output/GDP/Employment/Tax impacts.
    Forestry industries have different employment computation, which uses unit-labor assumptions. 
    
    Data input from specific sheet from an Excel file
    
    Parameters and multipliers are from another sets of specific sheets from the same Excel file 
    
    The employment impact is from unit-labor assumption. More realistic. 
    unit_labor = scalar  represent the number of FTE per difference between base case and strategic scenario m3
    example: unit_labor = 0.2
    
    direct_output
    
    physical_diff: different biophysical production between base case and strategic scenario. 
    monetary_diff: different monetray outcome between base case and strategic scenario. 
    
    outcome:::
    
    find_final_result() is a function to return a table with the final results. 

    find_strategy_impact() is a function to return a dictionary of all impacts. 
                             'wood', strategy_forestry_list = ["wood", "paper","harvest"]
                                        strategy_industry_list = strategy_forestry_list + ["concrete"]
                                        
    find_concrete_impact() is a function to get the object impact class which has attribute full_table. 
    find_concrete_impact().full_table
    
    find_output_summary():   Return a table only with summary
    to_excel(): put results to excel file
    
    """
    def __init__(self, strategy_name,  input_file, input_sheet_name, output_file, output_sheet_name, multipliers_sheet, 
                 parameters_sheet, unitlabor_sheet):
        
        self.strategy_name = strategy_name

        self.input_file = input_file
        self.output_file = output_file
        
        self.input_sheet_name = input_sheet_name
        #
        self.output_sheet_name = output_sheet_name
        #
        
        
        # strategy FM1 fixed input and output types
        self.strategy_forestry_list = [ "wood", "paper","harvest", "bioenergy", "residue_extraction"]
        #strategy_forestry_list includes ["bioenergy"] etc. add electricity as a total industries list
        self.strategy_other_industry_list = [ "concrete","plastic","electricity"]
        self.strategy_industry_list = self.strategy_forestry_list + self.strategy_other_industry_list
        
        # initial a empty dictionary
        self.__strategy_impact= dict.fromkeys(self.strategy_industry_list, 0)
        
        # import data by function from help.py
#         self.forestry_input_data = import_forestry_input(self.input_file, 
#                                                     self.input_sheet_name,
#                                                     forestry_list =  self.strategy_industry_list)
                # import data by function from help.py
        self.industry_input_data = import_industry_input(self.input_file, 
                                                    self.input_sheet_name,
                                                    industry_list =  self.strategy_industry_list)
        
        # import parameters from file
        self.multipliers = pd.read_excel(self.input_file, sheet_name= multipliers_sheet, usecols= "A:L")
        self.parameters = pd.read_excel(self.input_file, sheet_name= parameters_sheet, usecols = "A:B")
        self.unit_labor = pd.read_excel(self.input_file, sheet_name= unitlabor_sheet , usecols= "A:C")
        
        # initial a data frarme only with year column
        self.__diff_electricity=pd.DataFrame(self.industry_input_data['wood']['year'])
      

    # computer output by using class forestry in help.py 
#     for i in self.strategy_forestry_list:
#         self.__strategy_impact[i] = Forestry(i, self.multipliers, 
#                                       self.forestry_input_data[i][['year','monetary_diff']],
#                                       self.unit_labor, 
#                                       self.forestry_input_data[i][['year','physical_diff']])    
      
#     # computer output by using class forestry in help.py 
#     for i in self.strategy_other_industry_list:
#         self.__strategy_impact[i] = Industry(i, self.multipliers, 
#                                       self.forestry_input_data[i][['year','monetary_diff']],
#                                       self.unit_labor, 
#                                       self.forestry_input_data[i][['year','physical_diff']])   


    #@property
    def find_strategy_impact(self):
        # return a dictionary with industry objects
        #self.__strategy_impact= dict.fromkeys(self.strategy_industry_list, 0)
        print("Return a dictionary with industry objects, and please check out the .['bioenergy'] or . [residue_extraction] for detail...")
#         self.__strategy_impact["electricity"] = self.find_electricity_impact() # a method
        for i in self.strategy_forestry_list:
                self.__strategy_impact[i] = Forestry(i, self.multipliers, 
                                              self.industry_input_data[i][['year','monetary_diff']],
                                              self.unit_labor, 
                                              self.industry_input_data[i][['year','physical_diff']]) 
        # computer output by using class forestry in help.py 
        for i in self.strategy_other_industry_list:
            self.__strategy_impact[i] = Industry(i, self.multipliers, 
                                          self.industry_input_data[i][['year','monetary_diff']])      
            
            
        return self.__strategy_impact

    #@property
    def find_final_result(self):
        print('''Return a table with all impacts and summary, 
                 and please check out the table for detail...''')
        # using function to convert list of impact to a final table
        return to_final_result(list(self.find_strategy_impact().values()))

    #@property
    def find_output_summary(self):
        print('''Return a table only with summary, 
         and please check out the table for detail...''')
        return self.find_final_result().iloc[-5:,1:]


    def to_excel(self):
        # run a function from help.py 
        final_result_to_excel( self.find_final_result(), self.output_file,  self.output_sheet_name)
        print("Export final result table to excel file......")
        return None

class FM1(object):
    """
    FM1 strategy has 3 forest industries and 1 concrete industry:  
    
    One industry can has several Output/GDP/Employment/Tax impacts.
    Forestry industries have different employment computation, which uses unit-labor assumptions. 
    
    Data input from specific sheet from an Excel file
    
    Parameters and multipliers are from another sets of specific sheets from the same Excel file 
    
    The employment impact is from unit-labor assumption. More realistic. 
    unit_labor = scalar  represent the number of FTE per difference between base case and strategic scenario m3
    example: unit_labor = 0.2
    
    direct_output:
        year:
        physical_diff: different biophysical production between base case and strategic scenario. 
        monetary_diff: different monetray outcome between base case and strategic scenario. 
    
    """
    
    def __init__(self, input_file, input_sheet_name, output_file , output_sheet_name, multipliers_sheet, 
                 parameters_sheet, unitlabor_sheet):
        self.strategy_name = "FM1"

        self.input_file = input_file
        self.output_file = output_file
        
        self.input_sheet_name = input_sheet_name
        #self.input_sheet_name= self.strategy_name + "-BASE-INPUT"
        self.output_sheet_name = output_sheet_name
        #self.output_sheet_name = self.strategy_name + '-BASE-OUTPUT'
        
        
        # strategy FM1 fixed input and output types
        self.strategy_forestry_list = ["wood", "paper","harvest"]
        #strategy_forestry_list includes ["harvest"] etc. add concrete as a total industries list
        self.strategy_industry_list = self.strategy_forestry_list + ["concrete"]
        
        # initial a empty dictionary
        self.__strategy_impact= dict.fromkeys(self.strategy_industry_list, 0)
        
        # import data by function from help.py
        self.forestry_input_data = import_forestry_input(self.input_file, 
                                                    self.input_sheet_name,
                                                    forestry_list =  self.strategy_forestry_list)
        # import parameters from file
        self.multipliers = pd.read_excel(self.input_file, sheet_name= multipliers_sheet, usecols= "A:L")
        self.parameters = pd.read_excel(self.input_file, sheet_name= parameters_sheet, usecols = "A:B")
        self.unit_labor = pd.read_excel(self.input_file, sheet_name= unitlabor_sheet , usecols= "A:C")
        
        
        # displacement effect    
        self.panel_subs_concrete= self.parameters.loc[self.parameters.parameter=="subs_panel_concrete","value"].values[0]
        self.weight_per_m3_concrete = self.parameters.loc[self.parameters.parameter=="weight_per_m3_concrete","value"].values[0]
        self.price_per_m3_concrete = self.parameters.loc[self.parameters.parameter=="price_per_m3_concrete","value"].values[0]
        
        # initial a data frarme only with year column
        self.__diff_concrete=pd.DataFrame(self.forestry_input_data['wood']['year'])
        

    # computer output by using class forestry in help.py 
    def find_wood_impact(self):
        print("Return a wood foresty object with all impacts, and please check out the .full_table for detail...")
        return Forestry("wood", self.multipliers, 
                                self.forestry_input_data['wood'][['year','monetary_diff']],
                                self.unit_labor, 
                                self.forestry_input_data['wood'][['year','physical_diff']])

    def find_paper_impact(self):
        print("Return a paper foresty object with all impacts, and please check out the .full_table for detail...")
        return Forestry("paper", self.multipliers,
                            self.forestry_input_data['paper'][['year','monetary_diff']],
                            self.unit_labor, 
                            self.forestry_input_data['paper'][['year','physical_diff']],)
    def find_harvest_impact(self):
        print("Return a harvest foresty object with all impacts, and please check out the .full_table for detail...")
        return Forestry("harvest", 
                                   self.multipliers, 
                                   self.forestry_input_data['harvest'][['year','monetary_diff']], 
                                   self.unit_labor, 
                                   self.forestry_input_data['harvest'][['year','physical_diff']],)

        
    # use a method for computation
    #@property
    def diff_concrete(self):
    # Following literature, we assume that one m3 of wood panel product will decrease 3.46  concrete productions
        # print("Return a foresty object with all impacts, and please check out the .full_table() for detail...")
        self.__diff_concrete["physical_diff"] = self.forestry_input_data['wood']["physical_diff"]*\
                                        self.panel_subs_concrete    
        self.__diff_concrete["monetary_diff"] = self.forestry_input_data['wood']["physical_diff"]*\
                                        self.panel_subs_concrete*(-1/self.weight_per_m3_concrete)*\
                                        self.price_per_m3_concrete
        return self.__diff_concrete

    def find_concrete_impact(self):
        # return an industry object with all impacts
        print("Return a concrete industry object with all impacts, and please check out the .full_table for detail...")
        return Industry("concrete", self.multipliers, self.diff_concrete()[['year','monetary_diff']])


    #@property
    def find_strategy_impact(self):
        # return a dictionary with industry objects
        #self.__strategy_impact= dict.fromkeys(self.strategy_industry_list, 0)
        print("Return a dictionary with industry objects, and please check out the .['concrete'] or . [harvest] for detail...")
        self.__strategy_impact["concrete"] = self.find_concrete_impact() # a method
        self.__strategy_impact["wood"] = self.find_wood_impact()
        self.__strategy_impact["paper"] = self.find_paper_impact()
        self.__strategy_impact["harvest"] = self.find_harvest_impact()
        return self.__strategy_impact




    #@property
    def find_final_result(self):
        print('''Return a table with all impacts and summary, 
                 and please check out the table for detail...''')
        
        return to_final_result(list(self.find_strategy_impact().values()))

    #@property
    def find_output_summary(self):
        print('''Return a table only with summary, 
         and please check out the table for detail...''')
        return self.find_final_result().iloc[-5:,1:]


    def to_excel(self):
        # run a function from help.py 
        final_result_to_excel( self.find_final_result(), self.output_file,  self.output_sheet_name)
        print("Export final result table to excel file......")
        return None


class FM2(object):
    """
    input:::
    # input data for computation
    input_file = "data_input.xlsx"
    # output data for reporting
    output_file = "output_data.xlsx"



    # specify input data by select the sheets names in the Excel file
    input_sheet_name = "FM2-BASE-INPUT"
    output_sheet_name = 'FM2-BASE-OUTPUT'
    # regular multipliers
    multipliers_sheet = 'multipliers'
    # regular parameters                 
    parameters_sheet = 'parameters'
    # regular unit labor
    unitlabor_sheet = "unit-labor"

    FM1 strategy has 3 forest industries and 1 concrete industry:  
    
    One industry can has several Output/GDP/Employment/Tax impacts.
    Forestry industries have different employment computation, which uses unit-labor assumptions. 
    
    Data input from specific sheet from an Excel file
    
    Parameters and multipliers are from another sets of specific sheets from the same Excel file 
    
    The employment impact is from unit-labor assumption. More realistic. 
    unit_labor = scalar  represent the number of FTE per difference between base case and strategic scenario m3
    example: unit_labor = 0.2
    
    direct_output
    
    physical_diff: different biophysical production between base case and strategic scenario. 
    monetary_diff: different monetray outcome between base case and strategic scenario. 
    
    outcome:::
    
    find_final_result() is a function to return a table with the final results. 

    find_strategy_impact() is a function to return a dictionary of all impacts. 
                             'wood', strategy_forestry_list = ["wood", "paper","harvest"]
                                        strategy_industry_list = strategy_forestry_list + ["concrete"]
                                        
    find_concrete_impact() is a function to get the object impact class which has attribute full_table. 
    find_concrete_impact().full_table
    find_wood_impact() ...
    find_paper_impact() ...
    find_harvest_impact() ...
    
    find_output_summary():   Return a table only with summary
    to_excel(): put results to excel file

    """
    
    def __init__(self, input_file, input_sheet_name, output_file, output_sheet_name, multipliers_sheet, 
                 parameters_sheet, unitlabor_sheet):
        
        self.strategy_name = "FM2"

        self.input_file = input_file
        self.output_file = output_file
        
        self.input_sheet_name = input_sheet_name
        #self.input_sheet_name= self.strategy_name + "-BASE-INPUT"
        self.output_sheet_name = output_sheet_name
        #self.output_sheet_name = self.strategy_name + '-BASE-OUTPUT'
        
        
        # strategy FM1 fixed input and output types
        self.strategy_forestry_list = ["wood", "paper","harvest"]
        #strategy_forestry_list includes ["harvest"] etc. add concrete as a total industries list
        self.strategy_industry_list = self.strategy_forestry_list + ["concrete"]
        
        # initial a empty dictionary
        self.__strategy_impact= dict.fromkeys(self.strategy_industry_list, 0)
        
        # import data by function from help.py
        self.forestry_input_data = import_forestry_input(self.input_file, 
                                                    self.input_sheet_name,
                                                    forestry_list =  self.strategy_forestry_list)
        # import parameters from file
        self.multipliers = pd.read_excel(self.input_file, sheet_name= multipliers_sheet, usecols= "A:L")
        self.parameters = pd.read_excel(self.input_file, sheet_name= parameters_sheet, usecols = "A:B")
        self.unit_labor = pd.read_excel(self.input_file, sheet_name= unitlabor_sheet , usecols= "A:C")
        
        
        # displacement effect    
        self.panel_subs_concrete= self.parameters.loc[self.parameters.parameter=="subs_panel_concrete","value"].values[0]
        self.weight_per_m3_concrete = self.parameters.loc[self.parameters.parameter=="weight_per_m3_concrete","value"].values[0]
        self.price_per_m3_concrete = self.parameters.loc[self.parameters.parameter=="price_per_m3_concrete","value"].values[0]
        
        # initial a data frarme only with year column
        self.__diff_concrete=pd.DataFrame(self.forestry_input_data['wood']['year'])
        

    # computer output by using class forestry in help.py 
    def find_wood_impact(self):
        print("Return a wood foresty object with all impacts, and please check out the .full_table for detail...")
        return Forestry("wood", self.multipliers, 
                                self.forestry_input_data['wood'][['year','monetary_diff']],
                                self.unit_labor, 
                                self.forestry_input_data['wood'][['year','physical_diff']])

    def find_paper_impact(self):
        print("Return a paper foresty object with all impacts, and please check out the .full_table for detail...")
        return Forestry("paper", self.multipliers,
                            self.forestry_input_data['paper'][['year','monetary_diff']],
                            self.unit_labor, 
                            self.forestry_input_data['paper'][['year','physical_diff']],)
    def find_harvest_impact(self):
        print("Return a harvest foresty object with all impacts, and please check out the .full_table for detail...")
        return Forestry("harvest", 
                                   self.multipliers, 
                                   self.forestry_input_data['harvest'][['year','monetary_diff']], 
                                   self.unit_labor, 
                                   self.forestry_input_data['harvest'][['year','physical_diff']],)

        
    # use a method for computation
    #@property
    def diff_concrete(self):
    # Following literature, we assume that one m3 of wood panel product will decrease 3.46  concrete productions
        # print("Return a foresty object with all impacts, and please check out the .full_table() for detail...")
        self.__diff_concrete["physical_diff"] = self.forestry_input_data['wood']["physical_diff"]*\
                                        self.panel_subs_concrete    
        self.__diff_concrete["monetary_diff"] = self.forestry_input_data['wood']["physical_diff"]*\
                                        self.panel_subs_concrete*(-1/self.weight_per_m3_concrete)*\
                                        self.price_per_m3_concrete
        return self.__diff_concrete

    def find_concrete_impact(self):
        # return an industry object with all impacts
        print("Return a concrete industry object with all impacts, and please check out the .full_table for detail...")
        return Industry("concrete", self.multipliers, self.diff_concrete())


    #@property
    def find_strategy_impact(self):
        # return a dictionary with industry objects
        #self.__strategy_impact= dict.fromkeys(self.strategy_industry_list, 0)
        print("Return a dictionary with industry objects, and please check out the .['concrete'] or . [harvest] for detail...")
        self.__strategy_impact["concrete"] = self.find_concrete_impact() # a method
        self.__strategy_impact["wood"] = self.find_wood_impact()
        self.__strategy_impact["paper"] = self.find_paper_impact()
        self.__strategy_impact["harvest"] = self.find_harvest_impact()
        return self.__strategy_impact




    #@property
    def find_final_result(self):
        print('''Return a table with all impacts and summary, 
                 and please check out the table for detail...''')
        
        return to_final_result(list(self.find_strategy_impact().values()))

    #@property
    def find_output_summary(self):
        print('''Return a table only with summary, 
         and please check out the table for detail...''')
        return self.find_final_result().iloc[-5:,1:]


    def to_excel(self):
        # run a function from help.py 
        final_result_to_excel( self.find_final_result(), self.output_file,  self.output_sheet_name)
        print("Export final result table to excel file......")
        return None
                                    
class FM3(object):
    """
    input:::
    # input data for computation
    input_file = "data_input.xlsx"
    # output data for reporting
    output_file = "output_data.xlsx"



    # specify input data by select the sheets names in the Excel file
    input_sheet_name = "FM2-BASE-INPUT"
    output_sheet_name = 'FM2-BASE-OUTPUT'
    # regular multipliers
    multipliers_sheet = 'multipliers'
    # regular parameters                 
    parameters_sheet = 'parameters'
    # regular unit labor
    unitlabor_sheet = "unit-labor"

    FM1 strategy has 3 forest industries and 1 concrete industry:  
    
    One industry can has several Output/GDP/Employment/Tax impacts.
    Forestry industries have different employment computation, which uses unit-labor assumptions. 
    
    Data input from specific sheet from an Excel file
    
    Parameters and multipliers are from another sets of specific sheets from the same Excel file 
    
    The employment impact is from unit-labor assumption. More realistic. 
    unit_labor = scalar  represent the number of FTE per difference between base case and strategic scenario m3
    example: unit_labor = 0.2
    
    direct_output
    
    physical_diff: different biophysical production between base case and strategic scenario. 
    monetary_diff: different monetray outcome between base case and strategic scenario. 
    
    outcome:::
    
    find_final_result() is a function to return a table with the final results. 

    find_strategy_impact() is a function to return a dictionary of all impacts. 
                             'wood', strategy_forestry_list = ["wood", "paper","harvest"]
                                        strategy_industry_list = strategy_forestry_list + ["concrete"]
                                        
    find_concrete_impact() is a function to get the object impact class which has attribute full_table. 
    find_concrete_impact().full_table
    
    find_output_summary():   Return a table only with summary
    to_excel(): put results to excel file

    """
    
    def __init__(self, input_file, input_sheet_name, output_file, output_sheet_name, multipliers_sheet, 
                 parameters_sheet, unitlabor_sheet):
        
        self.strategy_name = "FM3"

        self.input_file = input_file
        self.output_file = output_file
        
        self.input_sheet_name = input_sheet_name
        #
        self.output_sheet_name = output_sheet_name
        #
        
        
        # strategy FM1 fixed input and output types
        self.strategy_forestry_list = ["bioenergy", "residue_extraction"]
        #strategy_forestry_list includes ["bioenergy"] etc. add electricity as a total industries list
        self.strategy_industry_list = self.strategy_forestry_list + ["electricity"]
        
        # initial a empty dictionary
        self.__strategy_impact= dict.fromkeys(self.strategy_industry_list, 0)
        
        # import data by function from help.py
        self.forestry_input_data = import_forestry_input(self.input_file, 
                                                    self.input_sheet_name,
                                                    forestry_list =  self.strategy_forestry_list)
        # import parameters from file
        self.multipliers = pd.read_excel(self.input_file, sheet_name= multipliers_sheet, usecols= "A:L")
        self.parameters = pd.read_excel(self.input_file, sheet_name= parameters_sheet, usecols = "A:B")
        self.unit_labor = pd.read_excel(self.input_file, sheet_name= unitlabor_sheet , usecols= "A:C")
        
        # initial a data frarme only with year column
        self.__diff_electricity=pd.DataFrame(self.forestry_input_data['bioenergy']['year'])
      

    # computer output by using class forestry in help.py 
#     for i in self.strategy_forestry_list:
#         self.__strategy_impact[i] = Forestry(i, self.multipliers, 
#                                       self.forestry_input_data[i][['year','monetary_diff']],
#                                       self.unit_labor, 
#                                       self.forestry_input_data[i][['year','physical_diff']])    
      
    # use a method for computation
    #@property
    def find_electricity_impact(self):
    # Following literature, we assume that bioenergy and electricity are one to one substution since electricity is homogeneous.
       
        self.__diff_electricity["monetary_diff"] = (-1)*self.forestry_input_data['bioenergy']["monetary_diff"]       
        self.__diff_electricity["physical_diff"] = (-1)*self.forestry_input_data['bioenergy']["physical_diff"] 
        return Industry("electricity", self.multipliers, self.__diff_electricity)
    


    #@property
    def find_strategy_impact(self):
        # return a dictionary with industry objects
        #self.__strategy_impact= dict.fromkeys(self.strategy_industry_list, 0)
        print("Return a dictionary with industry objects, and please check out the .['bioenergy'] or . [residue_extraction] for detail...")
        self.__strategy_impact["electricity"] = self.find_electricity_impact() # a method
        for i in self.strategy_forestry_list:
                self.__strategy_impact[i] = Forestry(i, self.multipliers, 
                                              self.forestry_input_data[i][['year','monetary_diff']],
                                              self.unit_labor, 
                                              self.forestry_input_data[i][['year','physical_diff']])    
        return self.__strategy_impact

    #@property
    def find_final_result(self):
        print('''Return a table with all impacts and summary, 
                 and please check out the table for detail...''')
        
        return to_final_result(list(self.find_strategy_impact().values()))

    #@property
    def find_output_summary(self):
        print('''Return a table only with summary, 
         and please check out the table for detail...''')
        return self.find_final_result().iloc[-5:,1:]


    def to_excel(self):
        # run a function from help.py 
        final_result_to_excel( self.find_final_result(), self.output_file,  self.output_sheet_name)
        print("Export final result table to excel file......")
        return None
                           
                                   


