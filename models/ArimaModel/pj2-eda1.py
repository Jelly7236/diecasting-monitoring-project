import pandas as pd
import numpy as np
import openpyxl

train = pd.read_csv('train2.csv')

train.info()


num_cols = train.select_dtypes(include=[np.number]).columns.tolist()

tra = pd.read_excel('train_final.xlsx', engine='openpyxl')

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()


tra




#가용 수치형 변수(관리할)

['molten_volume',
 'molten_temp',
 'facility_operation_cycleTime',
 'production_cycletime',
 'low_section_speed',
 'high_section_speed',
 'cast_pressure',
 'biscuit_thickness',
 'upper_mold_temp1',
 'upper_mold_temp2',
 'lower_mold_temp1',
 'lower_mold_temp2',
 'sleeve_temperature',
 'physical_strength',
 'Coolant_temperature']