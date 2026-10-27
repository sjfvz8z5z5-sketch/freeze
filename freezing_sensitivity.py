import numpy as np

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
