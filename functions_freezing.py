import scipy.optimize as sp
import numpy as np
import pandas as pd
import math
import csv
from pdb import set_trace as keyboard

import functions_freezing as fn_ref
import constants_freezing as cn_ref
import functions_lyoPronto as fn_lyo

###

def vialData(vialType):
    """
    Liest Vial-Daten aus Excel basierend auf dem vialType aus.
    """
    vial_data = pd.read_excel("LyoPronto_VIAL.xlsx")
    row = vial_data[vial_data.iloc[:,0] == vialType]
    if not row.empty:
        values = row.iloc[0,6:13].tolist()
        
        return tuple(values)
    else:
        raise ValueError(f"Name '{vialType}' not found.")

###

def VIAL(vialType, Vfill):
    """
    Erzeugt ein Dictionary für alle Vial-Parameter und 
    ein Dictionariy für die K Werte (für den Wärmedurchgangskoeffizienten).
    """
    
    vialValues = vialData(vialType)
    vial_name = vialType 
    vial = dict([('Name', vial_name), ('Av',vialValues[6]),('Ap',vialValues[5]),('Vfill',Vfill)])
    
    return vial

###

def productInput(ProductParameter, Tn, Tf, Tpr0):
    """
    Liefert ein Dictionary mit Produkteigenschaften je nach Parameter-Auswahl.
    """
    
    if ProductParameter == 1:
        product_name = 'Mannitol 5%'
        product = dict([('Name', product_name), ('cSolid',0.05),('R0',1.4),('A1',16),('A2',0), ('Tn', Tn), ('Tf', Tf), ('Tpr0', Tpr0)]) # Mannitol 5%
        
    elif ProductParameter == 2:
        product_name = 'Sucrose 5%'
        product = dict([('Name', product_name), ('cSolid',0.05),('R0',0.208),('A1',15.29),('A2',1.6), ('Tn', Tn), ('Tf', Tf), ('Tpr0', Tpr0)]) # Sucrose 5%
        
    return product

###

#weiter