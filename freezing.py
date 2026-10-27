import scipy.optimize as sp
import numpy as np
import math
import os
import csv
import pylab as plt
import time
from datetime import datetime
from pdb import set_trace as keyboard
current_time = time.strftime("%y%m%d_%H%M",time.localtime())
import functions_freezing as fn_ref
import constants_freezing as constant
import functions_lyoPronto as functions

def freeze(vial, product, h_freezing, temp_setpt, temp_time_setpt, dt):

    if len(temp_setpt) < 2:
        raise ValueError("'temp_setpt' benötigt mindestens zwei Werte (Start, Ziel).")
    if len(temp_time_setpt) != len(temp_setpt) - 1:
        raise ValueError("'temp_time_setpt' muss len(temp_setpt)-1 sein.")
    if dt <= 0:
        raise ValueError("dt muss > 0 sein (in Stunden).")

    temps = np.asarray(temp_setpt, dtype=float)              # len = Nseg+1
    durations_hr = np.asarray(temp_time_setpt, dtype=float)  # len = Nseg
    if np.any(durations_hr <= 0):
        raise ValueError("Alle Dauer-Einträge müssen > 0 sein (Stunden).")


    r = [0.0]
    for k in range(1, len(temps)):
        dT = temps[k] - temps[k-1]
        r_k = dT / (durations_hr[k-1] * constant.hr_To_min)  # °C/min
        r.append(r_k)
    r = np.asarray(r, dtype=float)                         


    t_tr = np.concatenate(([0.0], np.cumsum(durations_hr))).astype(float)


    Tsh_tr = temps.astype(float)      


    Lpr0 = functions.Lpr0_FUN(vial['Vfill'], vial['Ap'], product['cSolid'])   # cm
    V_frozen = Lpr0 * vial['Ap']    # mL


    iStep = 0
    t = 0.0
    Tsh = float(Tsh_tr[0])
    Tpr = float(product['Tpr0'])
    Tpr0 = Tpr
    i_prev = 1  # OG-Indexstart

    out = np.array([[t, Tsh, Tpr]], dtype=float)

 
    while Tpr > product['Tn']:
        iStep += 1

        t = iStep * dt

        idxs = np.where(t_tr > t)[0]

        if len(idxs) == 0:
            print("\n" + "!"*78 + "\n"
                  f"WARNUNG (h = {h_freezing:g} W/m²K): Rezept zu kurz – das Produkt ist am Ende des\n"
                  "Temperaturprofils noch nicht durchgefroren. Die Ergebnisse sind unvollständig.\n"
                  "Abhilfe: am Ende des Profils eine längere Haltezeit eingeben.\n" + "!"*78)
            break
        i = int(idxs[0])  # 1..Nseg

        if i != i_prev:
            Tpr0 = Tpr

            i_prev = i


        Tsh = Tsh + r[i] * constant.hr_To_min * dt

        Tpr = functions.lumped_cap_Tpr(t - t_tr[i-1], Tpr0, constant.rho_solution, constant.Cp_solution, vial['Vfill'], h_freezing, vial['Av'], Tsh, Tsh_tr[i-1], r[i],)


        out = np.append(out, [[t, Tsh, Tpr]], axis=0)


    out = np.append(out, [[t, Tsh, product['Tn']]], axis=0)


    tn = t
    dt_crystallization = functions.crystallization_time_FUN(
        vial['Vfill'], h_freezing, vial['Av'], product['Tf'], product['Tn'], Tsh
    )
    ts = tn + dt_crystallization

    while t < ts:
        idxs = np.where(t_tr > t)[0]
        if len(idxs) == 0:
            print("\n" + "!"*78 + "\n"
                  f"WARNUNG (h = {h_freezing:g} W/m²K): Rezept zu kurz – das Produkt ist am Ende des\n"
                  "Temperaturprofils noch nicht durchgefroren. Die Ergebnisse sind unvollständig.\n"
                  "Abhilfe: am Ende des Profils eine längere Haltezeit eingeben.\n" + "!"*78)
            break
        i = int(idxs[0])
        if i != i_prev:
            Tpr0 = Tpr
            i_prev = i
        Tsh = Tsh + r[i] * constant.hr_To_min * dt
        Tpr = product['Tf']
        out = np.append(out, [[t, Tsh, Tpr]], axis=0)

        iStep += 1
        t = iStep * dt



    # Solid cooling: Tsh folgt weiter dem Profil (auch wenn ts noch in einer Rampe liegt),
    # Tpr wird schrittweise mit der exakten Lösung für eine lineare Tsh-Rampe fortgeschrieben.
    Tpr = product['Tf']
    t_prev = ts
    while t < t_tr[-1]:
        i = int(np.where(t_tr > t)[0][0])
        Tsh_prev = Tsh
        Tsh = Tsh + r[i] * constant.hr_To_min * dt
        Tpr = functions.lumped_cap_Tpr(
            t - t_prev,
            Tpr,
            constant.rho_ice,
            constant.Cp_ice,
            V_frozen,
            h_freezing,
            vial['Av'],
            Tsh,
            Tsh_prev,
            r[i],
        )
        t_prev = t

        out = np.append(out, [[t, Tsh, Tpr]], axis=0)

        iStep += 1
        t = iStep * dt

    return out

##################################################################
# Energy calculation of the freezing process

def compute_crystallization_energy(out, vial, product, h_freezing):
 
    t   = out[:, 0]
    Tsh = out[:, 1]
    Tpr = out[:, 2]
    Tf  = float(product['Tf'])

    mask = np.isclose(Tpr, Tf, atol=1e-12)
    idx  = np.where(mask)[0]
    if idx.size < 2:
        return 0.0, np.nan, np.nan

    splits = np.where(np.diff(idx) != 1)[0] + 1
    blocks = np.split(idx, splits)
    block  = max(blocks, key=len)
    i0, i1 = int(block[0]), int(block[-1])

    t_seg   = t[i0:i1+1]
    Tsh_seg = Tsh[i0:i1+1]

    Av_m2 = float(vial['Av']) * 1e-4
    q     = h_freezing * Av_m2 * (Tsh_seg - Tf)  # W

    dt_s  = np.diff(t_seg) * 3600.0
    Q_J = np.sum(0.5 * (q[:-1] + q[1:]) * dt_s)

    tn_hr = float(t_seg[0])
    ts_hr = float(t_seg[-1])
    return float(Q_J), tn_hr, ts_hr

from datetime import datetime

def plot_with_energy(out, vial, product, h_freezing, title="Freezing – Energy over Tf-hold",
                     rho_solution=1000.0, L_f=333_700.0, save_dir='.'):
    t   = out[:, 0]
    Tsh = out[:, 1]
    Tpr = out[:, 2]
    Tf  = float(product['Tf'])
    Tn  = float(product['Tn'])


    Q_J, tn_hr, ts_hr = compute_crystallization_energy(out, vial, product, h_freezing)


    V_m3   = vial['Vfill'] * 1e-6
    m_sol  = rho_solution * V_m3
    #m_w    = (1.0 - float(product.get('cSolid', 0.0))) * m_sol
    Q_lat  = m_sol * L_f
    Q_ratio = -Q_J / Q_lat if Q_lat > 0 else float('nan')


    figwidth = 15
    figheight = 10
    lineWidth = 3
    textFontSize = 30
    gcafontSize = 20
    markerSize = 10
    labelPad = 1
    majorTickWidth = 5
    minorTickWidth = 3
    majorTickLength = 10
    minorTickLength = 5

    fig = plt.figure(0, figsize=(figwidth, figheight))
    ax = fig.add_subplot(1, 1, 1)
    plt.axes(ax)
    plt.minorticks_on()
    plt.setp(ax.get_xticklabels(), fontsize=gcafontSize, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=gcafontSize, fontweight='bold')
    ax.tick_params(axis='x', direction='in', pad=labelPad, width=majorTickWidth, length=majorTickLength, bottom=1, top=0)
    ax.tick_params(axis='y', direction='in', pad=labelPad, width=majorTickWidth, length=majorTickLength, left=1, right=0)
    ax.tick_params(axis='x', which='minor', direction='in', width=minorTickWidth, length=minorTickLength, bottom=1, top=0)
    ax.tick_params(axis='y', which='minor', direction='in', width=minorTickWidth, length=minorTickLength, left=1, right=0)

    # Kurven
    ax.plot(t, Tsh, '-k', linewidth=lineWidth, label="Shelf Temperature")
    ax.plot(t, Tpr, '-b', linewidth=lineWidth, label="Product Temperature")

    # Linien Tf & Tn
    ax.axhline(Tf, linestyle='--', linewidth=2, label='Tf')
    ax.axhline(Tn, linestyle=':',  linewidth=2, label='Tn')

    # Schattierung NUR im Tf-Hold UND nur zwischen tn..ts
    mask_tf = np.isclose(Tpr, Tf, atol=1e-12)
    if np.isfinite(tn_hr) and np.isfinite(ts_hr):
        mask_window = (t >= tn_hr) & (t <= ts_hr)
        mask = mask_tf & mask_window
        if np.any(mask):
            ax.fill_between(t[mask], Tsh[mask], Tf, alpha=0.25, label='∫ hA (Tsh - Tf) dt')
        # vertikale Marker
        ax.axvline(tn_hr, color='gray', ls='--', lw=1)
        ax.axvline(ts_hr, color='gray', ls='--', lw=1)

    ax.set_xlabel("Time [hr]", fontsize=gcafontSize, fontweight='bold')
    ax.set_ylabel("Temperature [C]", fontsize=gcafontSize, fontweight='bold')
    ax.xaxis.labelpad = labelPad
    ax.yaxis.labelpad = labelPad

    # Legende OBEN RECHTS
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, prop={'size': 20}, loc='upper right')

    # Layout
    ll, ul = ax.get_ylim()
    ax.set_ylim([ll, ul + 5.0])
    plt.title(title, fontsize=textFontSize, fontweight='bold')
    plt.tight_layout()


    if np.isfinite(tn_hr) and np.isfinite(ts_hr):
        textstr = (f"Crystallization energy (Tf-hold): Q = {Q_J:.2f} J\n"
                   f"tn = {tn_hr:.4f} h, ts = {ts_hr:.4f} h\n"
                   f"Q/Hfreeze = {Q_ratio:.3f}")
    else:
        textstr = (f"Crystallization energy (Tf-hold): Q = {Q_J:.2f} J\n"
                   f"tn / ts not found\n"
                   f"Q/Hfreeze = {Q_ratio:.3f}")

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
            fontsize=14, va='bottom', ha='right', bbox=props)

    # Save
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    figure_name = 'Freezing_Energy_LyoPronto' + current_time + '.pdf'
    plt.savefig(os.path.join(save_dir, figure_name))
    plt.show()
    plt.close()



##

#################################################################
# Energy optimizer calculation
# Energy optimizer calculation

def _extract_tf_window(out, Tf, tol=1e-8):

    t, Tsh, Tpr = out[:,0], out[:,1], out[:,2]
    mask = np.isclose(Tpr, Tf, atol=tol)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    splits = np.where(np.diff(idx) != 1)[0] + 1
    block = max(np.split(idx, splits), key=len)
    i0, i1 = int(block[0]), int(block[-1])
    return t[i0:i1+1], Tsh[i0:i1+1]

def cumulative_energy_Tf_hold(out, h_freezing, Av_cm2, Tf):
 
    seg = _extract_tf_window(out, Tf)
    if seg is None:
        raise ValueError("Kein Tf-Hold im Output gefunden (Tpr≈Tf).")
    t_seg, Tsh_seg = seg
    if t_seg.size < 2:
        return t_seg, np.zeros_like(t_seg)
    Av_m2 = Av_cm2 * 1e-4
    q = h_freezing * Av_m2 * (Tsh_seg - Tf)  # W
    E = np.zeros_like(t_seg)
    for k in range(1, t_seg.size):
        E[k] = E[k-1] + 0.5*(q[k-1] + q[k]) * ((t_seg[k]-t_seg[k-1]) * 3600.0)
    return t_seg, np.abs(E)

def latent_energy(vial, product, rho_solution=1000.0, L_f=333_700.0):

    V_m3 = vial['Vfill'] * 1e-6
    m_solution = rho_solution * V_m3
    m_water = (1.0 - float(product.get('cSolid', 0.0))) * m_solution
    return m_water * L_f

def backsolve_ts_star(out, vial, product, h_freezing, rho_solution=1000.0, L_f=333_700.0):

    t   = out[:,0]
    Tsh = out[:,1]
    Tpr = out[:,2]
    Tf  = float(product['Tf'])
    Tn  = float(product['Tn'])


    idx_tn = np.where(np.isclose(Tpr, Tn, atol=1e-8))[0]
    if idx_tn.size == 0:

        idx_tn = np.where(np.isclose(Tpr, Tf, atol=1e-8))[0]
        if idx_tn.size == 0:
            raise ValueError("Kein tn/Tf-Fenster gefunden.")
    i0 = int(idx_tn[0])
    tn = float(t[i0])

    # Latentwärme-Soll
    V_m3 = vial['Vfill'] * 1e-6
    m_solution = rho_solution * V_m3
    #m_water    = (1.0 - float(product.get('cSolid', 0.0))) * m_solution
    Q_lat = float(m_solution * L_f) 



    Av_m2 = vial['Av'] * 1e-4
    q_out = h_freezing * Av_m2 * np.maximum(Tf - Tsh, 0.0)  # W


    E = np.zeros_like(t)
    for k in range(i0+1, t.size):
        dt_s = (t[k] - t[k-1]) * 3600.0
        E[k] = E[k-1] + 0.5 * (q_out[k-1] + q_out[k]) * dt_s



    idx = np.where(E >= Q_lat)[0]
    if idx.size == 0:

        return tn, float(t[-1]), float(E[-1]), Q_lat, False

    k = int(idx[0])
    if k == i0:
        ts_star     = float(t[k])
        Q_ref_at_ts = float(E[k])
        return tn, ts_star, Q_ref_at_ts, Q_lat, True


    t0, t1 = float(t[k-1]), float(t[k])
    E0, E1 = float(E[k-1]), float(E[k])
    w = (Q_lat - E0) / max(E1 - E0, 1e-16)
    ts_star = t0 + w * (t1 - t0)


    return tn, float(ts_star), float(Q_lat), Q_lat, True



def build_energy_consistent(out_ref, vial, product, h_freezing, tn, ts_star):

    t_ref = out_ref[:,0]
    Tsh_ref = out_ref[:,1]
    Tpr_ref = out_ref[:,2]
    Tf = float(product['Tf'])


    def Tsh_prog(tk): 
        return float(np.interp(tk, t_ref, Tsh_ref))


    V_m3  = vial['Vfill'] * 1e-6
    Av_m2 = vial['Av']   * 1e-4
    mc_ice = 917.0 * 2050.0 * V_m3
    tau_s = mc_ice / max(h_freezing * Av_m2, 1e-12)


    if t_ref.size >= 2:
        dt = max(1e-12, t_ref[1] - t_ref[0])
    else:
        dt = 1e-3  # Fallback

    out_new = []
    Tpr_prev = Tf  # Zustand genau bei ts*: Produkt sitzt auf Tf
    for tk in t_ref:
        Tsh_k = Tsh_prog(tk)
        if tk < tn:
            out_new.append([tk, Tsh_k, float(np.interp(tk, t_ref, Tpr_ref))])
        elif tk <= ts_star:
            out_new.append([tk, Tsh_k, Tf])
            Tpr_prev = Tf
        else:

            Tpr_k = Tsh_k + (Tpr_prev - Tsh_k)*np.exp(-(dt * 3600.0)/tau_s)
            Tpr_prev = Tpr_k
            out_new.append([tk, Tsh_k, Tpr_k])

    return np.asarray(out_new, float)


def plot_ref_vs_energy(out_ref, out_new, product, tn, ts_star, Q_J, save_dir='.'):
    t_ref, Tsh_ref, Tpr_ref = out_ref[:,0], out_ref[:,1], out_ref[:,2]
    t_new, Tsh_new, Tpr_new = out_new[:,0], out_new[:,1], out_new[:,2]
    Tf = float(product['Tf']); Tn = float(product['Tn'])

    # Plotstyle
    figwidth=15; figheight=10; lineWidth=3
    gcafontSize=20; labelPad=1
    majorTickWidth=5; minorTickWidth=3; majorTickLength=10; minorTickLength=5

    fig = plt.figure(0, figsize=(figwidth, figheight))
    ax  = fig.add_subplot(1, 1, 1)
    plt.axes(ax)
    plt.minorticks_on()
    plt.setp(ax.get_xticklabels(), fontsize=gcafontSize, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=gcafontSize, fontweight='bold')
    ax.tick_params(axis='x', direction='in', pad=labelPad, width=majorTickWidth, length=majorTickLength, bottom=1, top=0)
    ax.tick_params(axis='y', direction='in', pad=labelPad, width=majorTickWidth, length=majorTickLength, left=1, right=0)
    ax.tick_params(axis='x', which='minor', direction='in', width=minorTickWidth, length=minorTickLength, bottom=1, top=0)
    ax.tick_params(axis='y', which='minor', direction='in', width=minorTickWidth, length=minorTickLength, left=1, right=0)

    # Shelf
    ax.plot(t_ref, Tsh_ref, '-k', linewidth=lineWidth, label="Shelf Temperature")


    ax.plot(t_new, Tpr_new, '-b',  linewidth=lineWidth, label="Product Temperature (Energy)")
    ax.plot(t_ref, Tpr_ref, '--g', linewidth=lineWidth+0.5, label="Product Temperature (REF)")  # etwas dicker

    # Tf & Tn
    ax.axhline(Tf, linestyle='--', linewidth=2, label='Tf')
    ax.axhline(Tn, linestyle=':',  linewidth=2, label='Tn')

    # Markierungen tn & ts*
    ax.axvline(tn,      color='gray', ls='--', lw=1)
    ax.axvline(ts_star, color='gray', ls='--', lw=1, label='ts*')


    mask_tf = (np.isclose(Tpr_new, Tf, atol=1e-10)) & (t_new >= tn) & (t_new <= ts_star)
    if np.any(mask_tf):
        ax.fill_between(t_new[mask_tf], np.interp(t_new[mask_tf], t_ref, Tsh_ref), Tf,
                        alpha=0.25, label='∫ hA (Tsh - Tf) dt')

    # Labeling
    ax.set_xlabel("Time [hr]", fontsize=gcafontSize, fontweight='bold')
    ax.set_ylabel("Temperature [C]", fontsize=gcafontSize, fontweight='bold')
    ax.xaxis.labelpad = labelPad
    ax.yaxis.labelpad = labelPad
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, prop={'size': 16}, loc='upper right')
    ll, ul = ax.get_ylim()
    ax.set_ylim([ll, ul + 5.0])
    plt.title("Freezing – Energy-consistent Tf-hold", fontsize=30, fontweight='bold')
    plt.tight_layout()


    textstr = (f"Q (Tf-hold) = {Q_J:.2f} J\n"
               f"tn = {tn:.4f} h\n"
               f"ts* = {ts_star:.4f} h")
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
            fontsize=14, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(save_dir, f"Freezing_Energy_optimized{current_time}.pdf"))
    plt.show(); plt.close()


##




h_freezing = 80 #  in W/m^2/K

vial = dict([
    ('Av',3.9),
    ('Ap', 3.14),
    ('Vfill', 2)
])
##############################
# Energie und Leistung


def compute_power_and_energy(out, vial, product, h_freezing):

    t   = out[:,0]
    Tsh = out[:,1]
    Tpr = out[:,2]

    Av_m2 = vial['Av'] * 1e-4  # cm² → m²
    q = h_freezing * Av_m2 * (Tsh - Tpr)  # W (J/s)

    E = np.zeros_like(t)
    for k in range(1, len(t)):
        dt_s = (t[k] - t[k-1]) * 3600.0  # h → s
        E[k] = E[k-1] + 0.5 * (q[k-1] + q[k]) * dt_s

    return t, q, E



def _apply_plotstyle(ax):
    figwidth = 15
    figheight = 10
    lineWidth = 3
    gcafontSize = 20
    labelPad = 1
    majorTickWidth = 5
    minorTickWidth = 3
    majorTickLength = 10
    minorTickLength = 5

    plt.minorticks_on()
    plt.setp(ax.get_xticklabels(), fontsize=gcafontSize, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=gcafontSize, fontweight='bold')
    ax.tick_params(axis='x', direction='in', pad=labelPad,
                   width=majorTickWidth, length=majorTickLength, bottom=1, top=0)
    ax.tick_params(axis='y', direction='in', pad=labelPad,
                   width=majorTickWidth, length=majorTickLength, left=1, right=0)
    ax.tick_params(axis='x', which='minor', direction='in',
                   width=minorTickWidth, length=minorTickLength, bottom=1, top=0)
    ax.tick_params(axis='y', which='minor', direction='in',
                   width=minorTickWidth, length=minorTickLength, left=1, right=0)

    return lineWidth, gcafontSize


# ref.py
# ref.py

# ref.py

def plot_power_compare(t_ref, q_ref, t_new, q_new, scale=1.0, save_path=None, show_peak_line=True):
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1, 1, 1)
    lineWidth, gcafontSize = _apply_plotstyle(ax)

    is_batch = (scale != 1)
    unit_factor = 1000.0 if is_batch else 1.0
    unit_str = "kW" if is_batch else "W"

    q_ref_plot = (q_ref * scale) / unit_factor
    q_opt_plot = (q_new * scale) / unit_factor

    label_ref = "Power REF × nVials" if is_batch else "Power REF ×1"
    label_opt = "Power OPT × nVials" if is_batch else "Power OPT ×1"

    ax.plot(t_ref, q_ref_plot, '--g', linewidth=lineWidth + 0.5, label=label_ref)
    ax.plot(t_new, q_opt_plot, '-b',  linewidth=lineWidth, label=label_opt)

    ax.set_xlabel("Time [hr]", fontsize=gcafontSize, fontweight='bold')
    ax.set_ylabel(f"Power [{unit_str}]", fontsize=gcafontSize, fontweight='bold')

    if not is_batch:
        title = "Power for one Vial"
    elif float(scale).is_integer() and int(scale) == 90000:
        title = "Power for 90000 Vials"
    else:
        title = f"Power for {int(scale)} Vials" if float(scale).is_integer() else "Power for nVials"
    ax.set_title(title, fontsize=24, fontweight='bold')

    # --- Peak-Linien (warum: exakt bis zur y-Achse, ohne Lücke) ---
    if show_peak_line and len(q_ref_plot) and len(q_opt_plot):
        i_ref = int(np.argmax(q_ref_plot)); y_ref = float(q_ref_plot[i_ref]); x_ref = float(t_ref[i_ref])
        i_opt = int(np.argmax(q_opt_plot)); y_opt = float(q_opt_plot[i_opt]); x_opt = float(t_new[i_opt])

        x_min, x_max = ax.get_xlim()
        denom = max(x_max - x_min, 1e-12)
        x_ref_frac = min(max((x_ref - x_min)/denom, 0.0), 1.0)
        x_opt_frac = min(max((x_opt - x_min)/denom, 0.0), 1.0)

        ax.axhline(y_ref, xmin=0.0, xmax=x_ref_frac, linestyle='--', linewidth=1.5, color='g')
        ax.axhline(y_opt, xmin=0.0, xmax=x_opt_frac, linestyle='--', linewidth=1.5, color='b')

    ax.legend(prop={'size': 20}, loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()



def plot_energy_compare(t_ref, E_ref, t_new, E_new, scale=1.0, save_path=None):
    """Energy: REF vs. OPT. Batch (scale!=1) in kJ, Legenden & Titel wie spezifiziert."""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    lineWidth, gcafontSize = _apply_plotstyle(ax)

    is_batch = (scale != 1)
    unit_factor = 1000.0 if is_batch else 1.0
    unit_str = "kJ" if is_batch else "J"

    E_ref_plot = (E_ref * scale) / unit_factor
    E_opt_plot = (E_new * scale) / unit_factor

    label_ref = "Energy REF × nVials" if is_batch else "Energy REF ×1"
    label_opt = "Energy OPT × nVials" if is_batch else "Energy OPT ×1"

    ax.plot(t_ref, E_ref_plot, '--g', linewidth=lineWidth + 0.5, label=label_ref)
    ax.plot(t_new, E_opt_plot, '-b', linewidth=lineWidth, label=label_opt)

    ax.set_xlabel("Time [hr]", fontsize=gcafontSize, fontweight='bold')
    ax.set_ylabel(f"Energy [{unit_str}]", fontsize=gcafontSize, fontweight='bold')

    if not is_batch:
        ax.set_title("Energy consumption for one Vial", fontsize=24, fontweight='bold')
    elif float(scale).is_integer() and int(scale) == 90000:
        ax.set_title("Energy consumption for 90000 Vials", fontsize=24, fontweight='bold')
    else:
        ax.set_title(f"Energy consumption for {int(scale)} Vials" if float(scale).is_integer() else "Energy consumption for nVials",
                     fontsize=24, fontweight='bold')

    ax.legend(prop={'size': 20}, loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
