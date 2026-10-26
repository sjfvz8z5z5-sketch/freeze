
import scipy.optimize as sp
import numpy as np
import math
import csv

from pdb import set_trace as keyboard
import copy
#import calc_knownRp as calc
#import functions_k as fn_k
import constants_freezing as cn_lyo

##
def Vapor_pressure(T_sub): # Equation (3.17)
    # Calculates the vapor pressure in Torr
    # Input is sublimation fronttemperature in [degC]
    #
    F = 2.698e10*math.exp(-6144.96/(273.15+T_sub)) # Vapor pressure at the sublimation temperature in Torr (3.17)
    
    return F
##

##
def Lpr0_FUN(Vfill,Ap,cSolid): # Equation (3.1)
    # Calculates the intial fill height of the frozen product in cm
    # Inputs are fillvolume in mL, product area in cm^2, and fractional concentration of the solute in the solution
    #
    F=Vfill/(Ap*cn_lyo.rho_ice)*(1-cSolid*(cn_lyo.rho_solution-cn_lyo.rho_ice)/cn_lyo.rho_solute)  # Fill height in cm; (3.1)

    return F
##

##
def Rp_FUN(l,R0,A1,A2): # Equation (3.15)
    # Calculates product resistance in cm^2-hr-Torr/g. Inputs are cake length 
    # in cm and product resistance parameters R0 in cm^2-hr-Torr/g, 
    # A1 in cm-hr-Torr/g, A2 in 1/cm
    #
    F=R0+A1*l/(1.0+A2*l) # Product resistance in cm^2-hr-Torr/g; (3.15)

    return F
##

##
def Kv_FUN(KC,KP,KD,Pch): # Equation (3.14)
    # Calculates the vial heat transfer coefficeint in cal/s/K/cm^2.
    # Inputs are vial heat transfer parameters KC in cal/s/K/cm^2, KP
    # in cal/s/K/cm^2/Torr, KD in 1/Torr, and chamber pressure in Torr
    #
    F=KC+KP*Pch/(1.0+KD*Pch) # Kv in cal/s/K/cm^2; (3.14)

    return F
##

##
def T_sub_solver_FUN(T_unknown, *data): # Equation (3.18)
    # Tsub is found from solving for T_unknown.
    # Determines the function to calculate the sublimation temperature
    # represented as T_unknown. Other inputs are chamber pressure in Torr, vial
    # area in cm^2, product area in cm^2, vial heat transfer coefficient in
    # cal/s/K/cm^2, initial product length in cm, cake length in cm, product
    # resistance in cm^2-Torr-hr/g, and shelf temperature in degC
    
    Pch, Av, Ap, Kv, Lpr0, Lck, Rp, Tsh = data
    T_unknown = float(np.atleast_1d(T_unknown)[0])
    P_sub = Vapor_pressure(T_unknown)   # Vapor pressure at the sublimation temperature in Torr; (3.17)
    #T = T_unknown[0]
    F = (P_sub-Pch)*(Av/Ap*Kv/cn_lyo.k_ice*(Lpr0-Lck)+1)-Av/Ap*Kv*Rp*cn_lyo.hr_To_s/cn_lyo.dHs*(Tsh-T_unknown)  # Heat and mass transfer energy balance function - Should be zero

    return F
##

##
def sub_rate(Ap,Rp,T_sub,Pch): # Equation (3.19)
    # Calculates the sublimation rate from each vial in kg/hr. Inputs are
    # product area in cm^2, product resistance in cm^2-Torr-hr/g, sublimation
    # front temperature in degC, and chamber pressure in Torr

    P_sub = Vapor_pressure(T_sub)   # Vapor pressure at the sublimation temperature in Torr; (3.17)

    dm_dt = Ap/Rp/cn_lyo.kg_To_g*(P_sub-Pch)  # Sublimation rate in kg/hr

    return dm_dt
##

##
def T_bot_FUN(T_sub,Lpr0,Lck,Pch,Rp): # Equation (3.20)
    # Calculates the temperature at the bottom of the vial in degC. Inputs are
    # sublimation front temperature in degC, initial product length in cm, cake
    # length in cm, chamber pressure in Torr, and product resistance in
    # cm^2-Torr-hr/g

    P_sub = Vapor_pressure(T_sub)   # Vapor pressure at the sublimation temperature in Torr; (3.17)

    Tbot = T_sub + (Lpr0-Lck)*(P_sub-Pch)*cn_lyo.dHs/Rp/cn_lyo.hr_To_s/cn_lyo.k_ice # Vial bottom temperature in degC

    return Tbot
##

##
def Rp_finder(T_sub,Lpr0,Lck,Pch,Tbot): # Equation (3.24)
    # Calculates product resistance in cm^2-hr-Torr/g. Inputs are sublimation
    # temperature in degC, initial product length in cm, cake length in cm,
    # chamber pressure in Torr, and vial bottom temperature in degC

    P_sub = Vapor_pressure(T_sub)   # Vapor pressure at the sublimation temperature in Torr; (3.17)

    F = (Lpr0-Lck)*(P_sub-Pch)*cn_lyo.dHs/(Tbot-T_sub)/cn_lyo.hr_To_s/cn_lyo.k_ice	# Product resistance in cm^2-Torr-hr/g; (3.24)

    return F
##

##
def T_sub_Rp_finder(T_unknown, *data): # Equation (3.23)
    # Tsub is found from solving for T_unknown.
    # Determines the function to calculate the sublimation temperature
    # represented as T_unknown. Other inputs are chamber pressure in Torr, vial
    # area in cm^2, product area in cm^2, vial heat transfer coefficient in
    # cal/s/K/cm^2, initial product length in cm, cake length in cm, vial bottom 
    # temperature in degC, and shelf temperature in degC
    
    Av, Ap, Kv, Lpr0, Lck, Tbot, Tsh = data

    F = (1.0 + Ap/Av/Kv*cn_lyo.k_ice/(Lpr0-Lck))*(Tbot-T_unknown) - (Tsh-T_unknown) # (3.23)

    return F
##

##
def T_sub_fromTpr(T_unknown, *data): # Equation (3.25)
    # Tsub is found from solving for T_unknown.
    # Determines the function to calculate the sublimation temperature
    # represented as T_unknown. Other inputs are vial bottom temperature in degC, 
    # initial product length in cm, cake length in cm, chamber pressure in Torr,
    # and product resistance in cm^2-Torr-hr/g
    
    Tbot, Lpr0, Lck, Pch, Rp = data

    P_sub = Vapor_pressure(T_unknown)   # Vapor pressure at the sublimation temperature in Torr; (3.17)

    F = T_unknown - Tbot + (P_sub-Pch)*(Lpr0-Lck)*cn_lyo.dHs/Rp/cn_lyo.hr_To_s/cn_lyo.k_ice  # Heat and mass transfer energy balance function - Should be zero; (3.25)

    return F
##

##
def Tbot_max_eq_cap(Pch,dm_dt,Lpr0,Lck,Rp,Ap): # Equation (3.26), (3.27), (3.33)
    # Calculates the maximum product temperature (occus at vial bottom) in
    # degC. Inputs are chamber pressure in Torr, sublimation rate based on
    # equipment capability in kg/hr, initial product length in cm, cake length
    # in cm, product resistance in cm^2-Torr-hr/g, and product area in cm^2

    P_sub = dm_dt/Ap*Rp + Pch     # Sublimation front pressure in Torr; (3.26)
    P_sub = ((P_sub)**2)**0.5     # Druck wird negativ
    T_sub = -6144.96/np.log(P_sub/2.698e10) - 273.15    # Sublimation front temperature in degC; (3.33)
    Tbot = T_sub + (Lpr0-Lck)*(P_sub-Pch)*cn_lyo.dHs/Rp/cn_lyo.hr_To_s/cn_lyo.k_ice	#Vial bottom temperature in degC
    Tbot_max = np.max(Tbot)   # Maximum vial bottom temperature in degC; (3.27)

    return Tbot_max
##

##
def Ineq_Constraints(Pch,dm_dt,Tcrit,Tbot,a,b,nVial): # Equation (3.31), (3.32)
    # Defines the inequality constraints for lyophilization optimization within
    # safe operation region inside the desgin space. Inputs are chamber pressure in Torr,
    # sublimation rate in kg/hr, critical product temperature in degC, vial bottom
    # temperature in degC, equipment capability parameters a in kg/hr and b in kg/hr/Torr,
    # and number of vials

    C1 = a + b*Pch - nVial*dm_dt	# Equipment capability limit; (3.31)
   
    C2 = Tcrit - Tbot		# Maximum product temperature limit; (3.32)

    return C1,C2
##

##
def Eq_Constraints(Pch,dmdt,Tbot,Tsh,Psub,Tsub,Kv,Lpr0,Lck,Av,Ap,Rp): # Equations (3.34), (3.35), (3.36(, (3.37)
    # Defines the equality constraints for lyophilization. Inputs are chamber pressure in Torr,
    # sublimation rate in kg/hr, vial bottom temperature in degC, shelf temperature in degC,
    # sublimation front pressure in Torr, sublimation front temperature in degC, vial heat
    # transfer coefficient in cal/s/cm^2/C, initial product length in cm, cake length in cm,
    # vial area in cm^2, product area in cm^2, and product resistance in cm^2-Torr-hr/g 

    C1 = Psub - 2.698e10*math.exp(-6144.96/(273.15+Tsub))   # Vapor pressure at the sublimation temperature in Torr; (3.34)
    
    C2 = dmdt - Ap/Rp/cn_lyo.kg_To_g*(Psub-Pch)  # Sublimation rate in kg/hr; (3.35) 
    # soll der Fehler bei C3 korrigiert werdne? oder ist es doch kein Fehler?!
    C3 = (Tsh-Tbot)*Av*Kv*(Lpr0-Lck) - Ap*(Tbot-Tsub)*cn_lyo.k_ice  # Vial heat transfer balance; (3.36)

    C4 = Tsh - dmdt*cn_lyo.kg_To_g/cn_lyo.hr_To_s*cn_lyo.dHs/Av/Kv - Tbot  # Shelf temperature in C; (3.37)

    return C1,C2,C3,C4
##

##
def lumped_cap_Tpr(t,Tpr0,rho,Cp,V,h,Av,Tsh,Tsh0, Tsh_ramp): # Equation (3.9)
    # Calculates the product temperature in C. Inputs are time in hr, initial product temperature in degC, product density in g/mL, constant pressure specific heat of the product in J/kg/K, product volume in mL, heat transfer coefficient in W/m^2/K, vial area in cm^2, current shelf temperature in degC, initial shelf temperature in degC, shelf temperature ramping rate in degC/min

    F = (Tpr0 + Tsh_ramp/cn_lyo.min_To_s*rho*Cp/cn_lyo.kg_To_g*V/h/Av/cn_lyo.cm_To_m**2 - Tsh0)*math.exp(-h*Av*cn_lyo.cm_To_m**2*t*cn_lyo.hr_To_s/rho/Cp*cn_lyo.kg_To_g/V) - Tsh_ramp/cn_lyo.min_To_s*rho*Cp/cn_lyo.kg_To_g*V/h/Av/cn_lyo.cm_To_m**2 + Tsh

    return F
##

##
def crystallization_time_FUN(V,h,Av,Tf,Tn,Tsh): # Equation (3.13)
    # Calculates the crystallization time in hr. Inputs are fill volume in mL, heat transfer coefficient in W/m^2/K, vial area in cm^2, freezing temperature in degC, nucleation temperature in degC, shelf temperature in degC

    F = cn_lyo.rho_solution*V*(cn_lyo.dHf*cn_lyo.cal_To_J-cn_lyo.Cp_solution/cn_lyo.kg_To_g*(Tf-Tn))/(h*cn_lyo.hr_To_s*Av*(cn_lyo.cm_To_m**2) *(Tf-Tsh))

    return F
##
