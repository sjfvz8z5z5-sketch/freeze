import numpy as np
import pandas as pd

import freezing as ref

##
def run_case(vial, product, h_freezing, temp_setpt, temp_time_setpt, dt):
    # Rechnet einen Fall komplett durch: LyoPRONTO (REF) und energie-korrigierte Rechnung (NEW)
    out_ref = ref.freeze(vial, product, h_freezing, temp_setpt, temp_time_setpt, dt)
    Q_hold, tn, ts = ref.compute_crystallization_energy(out_ref, vial, product, h_freezing)
    tn, ts_star, Q_at_ts, Q_lat, reached = ref.backsolve_ts_star(out_ref, vial, product, h_freezing)
    out_new = ref.build_energy_consistent(out_ref, vial, product, h_freezing, tn, ts_star)

    # Vorzeichen: abgeführte Wärme positiv
    t_ref, q_ref, E_ref = ref.compute_power_and_energy(out_ref, vial, product, h_freezing)
    t_new, q_new, E_new = ref.compute_power_and_energy(out_new, vial, product, h_freezing)

    return dict(
        h=h_freezing, out_ref=out_ref, out_new=out_new,
        tn=tn, ts=ts, ts_star=ts_star, Q_hold=-Q_hold, Q_lat=Q_lat, reached=reached,
        t_ref=t_ref, q_ref=-q_ref, E_ref=-E_ref,
        t_new=t_new, q_new=-q_new, E_new=-E_new,
    )
##

##
def run_k_sweep(vial, product, h_base, temp_setpt, temp_time_setpt, dt,
                factors=np.round(np.arange(0.80, 1.2001, 0.05), 2)):
    # Variation des Wärmedurchgangskoeffizienten h (k-Wert) um h_base, Standard: -20 % ... +20 % in 5 %-Schritten
    results = []
    for f in factors:
        res = run_case(vial, product, h_base*f, temp_setpt, temp_time_setpt, dt)
        res['factor'] = float(f)
        results.append(res)

    return results
##

##
def summary_table(results, n_vials=1):
    # Kennwerte je k-Variation; Energie in J bzw. kWh, Leistung in W bzw. kW (bei n_vials > 1)
    rows = []
    for r in results:
        row = {
            'Δk [%]': round((r['factor'] - 1.0)*100),
            'h [W/m²K]': r['h'],
            'tn [h]': r['tn'],
            'ts LyoPRONTO [h]': r['ts'],
            'ts* korrigiert [h]': r['ts_star'],
            'Δt LyoPRONTO [h]': r['ts'] - r['tn'],
            'Δt korrigiert [h]': r['ts_star'] - r['tn'],
            'Q Tf-Hold LyoPRONTO [J]': r['Q_hold'],
            'Q/Hfreeze LyoPRONTO [-]': r['Q_hold']/r['Q_lat'],
        }
        if n_vials == 1:
            row['P max LyoPRONTO [W]'] = r['q_ref'].max()
            row['P max korrigiert [W]'] = r['q_new'].max()
            row['E ges. LyoPRONTO [J]'] = r['E_ref'][-1]
            row['E ges. korrigiert [J]'] = r['E_new'][-1]
        else:
            row['P max LyoPRONTO [kW]'] = r['q_ref'].max()*n_vials/1e3
            row['P max korrigiert [kW]'] = r['q_new'].max()*n_vials/1e3
            row['E ges. LyoPRONTO [kWh]'] = r['E_ref'][-1]*n_vials/3.6e6
            row['E ges. korrigiert [kWh]'] = r['E_new'][-1]*n_vials/3.6e6
        rows.append(row)

    return pd.DataFrame(rows)
##

##
def energy_budget(vial, product, T0=None, Tend=-44.0):
    # Sollwert der abzuführenden Wärme aus der Energiebilanz (Zustandsgröße, unabhängig von Zeit und k-Wert).
    # Basis sind die LyoPRONTO-Konstanten; Masse = rho_solution * Vfill wie in crystallization_time_FUN.
    cn = ref.constant
    T0 = product['Tpr0'] if T0 is None else T0
    m = cn.rho_solution*vial['Vfill']/cn.kg_To_g   # kg
    Tf = product['Tf']

    Q_liquid = m*cn.Cp_solution*(T0 - Tf)          # J, Flüssigkeit T0 -> Tf
    Q_latent = m*cn.dHf*cn.cal_To_J*cn.kg_To_g     # J, Schmelzwärme
    Q_ice = m*cn.Cp_ice*(Tf - Tend)                # J, Eis Tf -> Tend

    return dict(Q_liquid=Q_liquid, Q_latent=Q_latent, Q_ice=Q_ice,
                Q_total=Q_liquid + Q_latent + Q_ice, T0=T0, Tend=Tend)
##
