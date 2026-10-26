import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Farben: LyoPRONTO = Orange, korrigiert = Blau, Stellfläche = Grau
C_REF = '#eb6834'
C_NEW = '#2a78d6'
C_SHELF = '#52514e'
C_MUTED = '#898781'
C_GRID = '#e1e0d9'
C_INK = '#0b0b0b'

LABEL_REF = 'LyoPRONTO'
LABEL_NEW = 'Corrected (energy balance)'

##
def apply_style():
    # Einheitlicher, ruhiger Plotstil für alle Diagramme
    mpl.rcParams.update({
        'figure.figsize': (12, 7),
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.size': 13,
        'axes.titlesize': 17,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',
        'axes.titlepad': 14,
        'axes.labelsize': 14,
        'axes.labelcolor': C_INK,
        'axes.edgecolor': '#c3c2b7',
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': C_GRID,
        'grid.linewidth': 0.8,
        'xtick.color': C_SHELF,
        'ytick.color': C_SHELF,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'legend.fontsize': 12,
        'lines.linewidth': 2.2,
        'lines.solid_capstyle': 'round',
    })
##

##
def _ramp(cmap_name, n):
    # Sequentielle Farbreihe (hell -> dunkel) für die k-Variation
    cmap = plt.get_cmap(cmap_name)
    return [cmap(x) for x in np.linspace(0.35, 0.95, n)]

def _units(n_vials):
    # Einheiten und Umrechnungsfaktoren für ein Vial bzw. die Charge
    if n_vials == 1:
        return 'W', 1.0, 'J', 1.0
    return 'kW', n_vials/1e3, 'kWh', n_vials/3.6e6

def _scope(n_vials):
    return 'one vial' if n_vials == 1 else f'{n_vials:,} vials'.replace(',', '.')

def _ref_lines(ax, product):
    # Tf / Tn als Hilfslinien mit Direktbeschriftung
    for T, name, ls in [(product['Tf'], 'Tf', '--'), (product['Tn'], 'Tn', ':')]:
        ax.axhline(T, color=C_MUTED, lw=1.0, ls=ls, zorder=1)
        ax.annotate(f'{name} = {T:g} °C', xy=(1.0, T), xycoords=('axes fraction', 'data'),
                    xytext=(4, 0), textcoords='offset points', va='center', fontsize=11, color=C_MUTED)

def _finish(fig, save_path):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()
    plt.close(fig)
##

##############################################################################
# Vergleich LyoPRONTO vs. korrigiert (ein Fall)

def plot_temperature_comparison(res, product, save_path=None):
    apply_style()
    fig, ax = plt.subplots()
    o_ref, o_new = res['out_ref'], res['out_new']
    tn, ts, ts_star = res['tn'], res['ts'], res['ts_star']

    ax.plot(o_ref[:, 0], o_ref[:, 1], color=C_SHELF, lw=1.6, label='Shelf temperature')
    ax.plot(o_ref[:, 0], o_ref[:, 2], color=C_REF, ls='--', label=f'Product – {LABEL_REF}')
    ax.plot(o_new[:, 0], o_new[:, 2], color=C_NEW, label=f'Product – {LABEL_NEW}')
    _ref_lines(ax, product)

    # Kristallisationsdauer wie in der LyoPRONTO-Darstellung als Pfeile
    Tf = product['Tf']
    for t_end, color, name, dy in [(ts, C_REF, 'LyoPRONTO', 7.0), (ts_star, C_NEW, 'corr', 3.5)]:
        y = Tf + dy
        ax.annotate('', xy=(t_end, y), xytext=(tn, y),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=1.6))
        ax.text(t_end + 0.03, y, f'Δt$_{{{name}}}$ = {t_end - tn:.2f} h',
                color=C_INK, va='center', fontsize=12)
        ax.plot([t_end, t_end], [Tf, y], color=color, lw=1.0, ls='-.')
    ax.plot([tn, tn], [product['Tn'], Tf + 7.0], color=C_MUTED, lw=1.0, ls='-.')

    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Temperature [°C]')
    ax.set_title(f'Freezing – {LABEL_REF} vs. corrected crystallization time')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)
    _finish(fig, save_path)


def plot_power_comparison(res, n_vials=1, save_path=None):
    apply_style()
    unit, fac, _, _ = _units(n_vials)
    fig, ax = plt.subplots()

    for t, q, color, ls, label in [(res['t_ref'], res['q_ref'], C_REF, '--', LABEL_REF),
                                   (res['t_new'], res['q_new'], C_NEW, '-', LABEL_NEW)]:
        y = q*fac
        ax.plot(t, y, color=color, ls=ls, label=label)
        k = int(np.argmax(y))
        ax.plot(t[k], y[k], 'o', ms=8, color=color, mec='white', mew=2, zorder=5)
        ax.annotate(f'max {y[k]:.3g} {unit}', xy=(t[k], y[k]), xytext=(8, 6),
                    textcoords='offset points', fontsize=12, color=C_INK)

    ax.set_xlabel('Time [h]')
    ax.set_ylabel(f'Heat removal rate [{unit}]')
    ax.set_title(f'Cooling power – {_scope(n_vials)}')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')
    _finish(fig, save_path)


def plot_energy_comparison(res, n_vials=1, save_path=None):
    apply_style()
    _, _, unit, fac = _units(n_vials)
    fig, ax = plt.subplots()

    for t, E, color, ls, label in [(res['t_ref'], res['E_ref'], C_REF, '--', LABEL_REF),
                                   (res['t_new'], res['E_new'], C_NEW, '-', LABEL_NEW)]:
        y = E*fac
        ax.plot(t, y, color=color, ls=ls, label=label)
        ax.annotate(f'{y[-1]:.4g} {unit}', xy=(t[-1], y[-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=12, color=C_INK)

    ax.set_xlabel('Time [h]')
    ax.set_ylabel(f'Cumulative heat removed [{unit}]')
    ax.set_title(f'Energy demand – {_scope(n_vials)}')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')
    _finish(fig, save_path)
##

##############################################################################
# Kurvenscharen für die k-Variation

def _family(results, quantity, n_vials, title, ylabel, legend_loc, save_path, product=None):
    # Zwei Panels nebeneinander (gleiche y-Achse): links LyoPRONTO, rechts korrigiert
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    n = len(results)

    for ax, model, cmap, label in [(axes[0], 'ref', 'Oranges', LABEL_REF),
                                   (axes[1], 'new', 'Blues', LABEL_NEW)]:
        colors = _ramp(cmap, n)
        if quantity == 'T':
            o = results[n//2]['out_' + model]
            ax.plot(o[:, 0], o[:, 1], color=C_SHELF, lw=1.4, label='Shelf temperature')
            _ref_lines(ax, product)
        for r, c in zip(results, colors):
            base = np.isclose(r['factor'], 1.0)
            if quantity == 'T':
                o = r['out_' + model]
                t, y = o[:, 0], o[:, 2]
            else:
                t = r['t_' + model]
                y = r[quantity + '_' + model]*(_units(n_vials)[1] if quantity == 'q' else _units(n_vials)[3])
            dk = round((r['factor'] - 1.0)*100)
            ax.plot(t, y, color=c, lw=3.0 if base else 1.5, zorder=4 if base else 3,
                    label=f'{dk:+d} %  (h = {r["h"]:.0f})' if dk else f'±0 %  (h = {r["h"]:.0f})')
        ax.set_title(label)
        ax.set_xlabel('Time [h]')
        ax.set_xlim(left=0)
        anchor = {'T': (1.0, 0.10), 'E': (1.0, 0.62)}.get(quantity)
        ax.legend(loc=legend_loc, bbox_to_anchor=anchor, fontsize=10,
                  title='Δk  (h in W/m²K)', title_fontsize=10)
    axes[0].set_ylabel(ylabel)
    if quantity != 'T':
        axes[0].set_ylim(bottom=0)
    fig.suptitle(title, x=0.01, ha='left', fontsize=18, fontweight='bold')
    _finish(fig, save_path)


def plot_k_family_temperature(results, product, save_path=None):
    _family(results, 'T', 1, 'Product temperature – variation of k (±20 %)',
            'Temperature [°C]', 'lower right', save_path, product)


def plot_k_family_power(results, n_vials=1, save_path=None):
    unit = _units(n_vials)[0]
    _family(results, 'q', n_vials, f'Cooling power – variation of k (±20 %), {_scope(n_vials)}',
            f'Heat removal rate [{unit}]', 'upper right', save_path)


def plot_k_family_energy(results, n_vials=1, save_path=None):
    unit = _units(n_vials)[2]
    _family(results, 'E', n_vials, f'Energy demand – variation of k (±20 %), {_scope(n_vials)}',
            f'Cumulative heat removed [{unit}]', 'center right', save_path)


def plot_k_summary(results, n_vials=1, save_path=None):
    # Kennwerte über der k-Variation: Kristallisationsdauer, Spitzenleistung, Gesamtenergie
    apply_style()
    p_unit, p_fac, e_unit, e_fac = _units(n_vials)
    dk = np.array([(r['factor'] - 1.0)*100 for r in results])
    data = [
        ('Crystallization time Δt [h]',
         [r['ts'] - r['tn'] for r in results], [r['ts_star'] - r['tn'] for r in results], '{:.2f} h'),
        (f'Peak cooling power [{p_unit}]',
         [r['q_ref'].max()*p_fac for r in results], [r['q_new'].max()*p_fac for r in results], '{:.3g} ' + p_unit),
        (f'Total energy [{e_unit}]',
         [r['E_ref'][-1]*e_fac for r in results], [r['E_new'][-1]*e_fac for r in results], '{:.4g} ' + e_unit),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (title, y_ref, y_new, fmt) in zip(axes, data):
        for y, color, ls, label in [(y_ref, C_REF, '--', LABEL_REF), (y_new, C_NEW, '-', LABEL_NEW)]:
            ax.plot(dk, y, color=color, ls=ls, marker='o', ms=8, mec='white', mew=2, label=label)
            ax.annotate(fmt.format(y[-1]), xy=(dk[-1], y[-1]), xytext=(8, 0),
                        textcoords='offset points', va='center', fontsize=11, color=C_INK)
            ax.annotate(fmt.format(y[0]), xy=(dk[0], y[0]), xytext=(-8, 0), ha='right',
                        textcoords='offset points', va='center', fontsize=11, color=C_INK)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Δk relative to base [%]')
        ax.set_xticks(dk)
        ax.set_xticklabels([(f'{v:+.0f}' if v else '0') if round(v) % 10 == 0 else '' for v in dk])
        ax.set_xlim(dk[0] - 14, dk[-1] + 14)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc='lower left')
    fig.suptitle(f'Sensitivity to k – {_scope(n_vials)}', x=0.01, ha='left', fontsize=18, fontweight='bold')
    _finish(fig, save_path)
##

##############################################################################
# Einzeldiagramme: jeweils NUR LyoPRONTO oder NUR korrigiert

MODELS = {'ref': (LABEL_REF, C_REF, 'Oranges'), 'new': (LABEL_NEW, C_NEW, 'Blues')}

def plot_temperature_single(res, product, model, save_path=None):
    apply_style()
    label, color, _ = MODELS[model]
    o = res['out_' + model]
    t_end = res['ts'] if model == 'ref' else res['ts_star']
    fig, ax = plt.subplots()

    ax.plot(o[:, 0], o[:, 1], color=C_SHELF, lw=1.6, label='Shelf temperature')
    ax.plot(o[:, 0], o[:, 2], color=color, label=f'Product – {label}')
    _ref_lines(ax, product)
    y = product['Tf'] + 4.0
    ax.annotate('', xy=(t_end, y), xytext=(res['tn'], y), arrowprops=dict(arrowstyle='<->', color=color, lw=1.6))
    ax.text(t_end + 0.03, y, f'Δt = {t_end - res["tn"]:.2f} h', va='center', fontsize=12, color=C_INK)

    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Temperature [°C]')
    ax.set_title(f'Freezing – {label}')
    ax.set_xlim(left=0)
    ax.legend(loc='upper right')
    _finish(fig, save_path)


def plot_power_single(res, model, n_vials=1, save_path=None):
    apply_style()
    label, color, _ = MODELS[model]
    unit, fac, _, _ = _units(n_vials)
    t, y = res['t_' + model], res['q_' + model]*fac
    fig, ax = plt.subplots()

    ax.plot(t, y, color=color, label=label)
    k = int(np.argmax(y))
    ax.plot(t[k], y[k], 'o', ms=8, color=color, mec='white', mew=2, zorder=5)
    ax.annotate(f'max {y[k]:.3g} {unit}', xy=(t[k], y[k]), xytext=(8, 6),
                textcoords='offset points', fontsize=12, color=C_INK)

    ax.set_xlabel('Time [h]')
    ax.set_ylabel(f'Heat removal rate [{unit}]')
    ax.set_title(f'Cooling power – {label}, {_scope(n_vials)}')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    _finish(fig, save_path)


def plot_energy_single(res, model, n_vials=1, save_path=None):
    apply_style()
    label, color, _ = MODELS[model]
    _, _, unit, fac = _units(n_vials)
    t, y = res['t_' + model], res['E_' + model]*fac
    fig, ax = plt.subplots()

    ax.plot(t, y, color=color, label=label)
    ax.annotate(f'{y[-1]:.4g} {unit}', xy=(t[-1], y[-1]), xytext=(-6, 8), ha='right',
                textcoords='offset points', fontsize=12, color=C_INK)

    ax.set_xlabel('Time [h]')
    ax.set_ylabel(f'Cumulative heat removed [{unit}]')
    ax.set_title(f'Energy demand – {label}, {_scope(n_vials)}')
    ax.set_xlim(left=0)
    ax.set_ylim(0, y.max()*1.12)
    ax.legend(loc='upper left')
    _finish(fig, save_path)


def plot_k_family_single(results, model, quantity, n_vials=1, product=None, save_path=None):
    # Kurvenschar der k-Variation für NUR ein Modell; quantity: 'T', 'q' oder 'E'
    apply_style()
    label, _, cmap = MODELS[model]
    p_unit, p_fac, e_unit, e_fac = _units(n_vials)
    fig, ax = plt.subplots()
    n = len(results)

    if quantity == 'T':
        o = results[n//2]['out_' + model]
        ax.plot(o[:, 0], o[:, 1], color=C_SHELF, lw=1.4, label='Shelf temperature')
        _ref_lines(ax, product)
    for r, c in zip(results, _ramp(cmap, n)):
        if quantity == 'T':
            t, y = r['out_' + model][:, 0], r['out_' + model][:, 2]
        else:
            t = r['t_' + model]
            y = r[quantity + '_' + model]*(p_fac if quantity == 'q' else e_fac)
        dk = round((r['factor'] - 1.0)*100)
        base = dk == 0
        ax.plot(t, y, color=c, lw=3.0 if base else 1.5, zorder=4 if base else 3,
                label=(f'{dk:+d} %' if dk else '±0 %') + f'  (h = {r["h"]:.0f})')

    name, ylabel, loc, anchor = {
        'T': ('Product temperature', 'Temperature [°C]', 'lower right', (1.0, 0.10)),
        'q': ('Cooling power', f'Heat removal rate [{p_unit}]', 'upper right', None),
        'E': ('Energy demand', f'Cumulative heat removed [{e_unit}]', 'upper left', None),
    }[quantity]
    if quantity in ('q', 'E'):
        ax.set_ylim(bottom=0)

    ax.set_xlabel('Time [h]')
    ax.set_ylabel(ylabel)
    scope = '' if quantity == 'T' else f', {_scope(n_vials)}'
    ax.set_title(f'{name} – {label}, variation of k (±20 %){scope}')
    ax.set_xlim(left=0)
    ax.legend(loc=loc, bbox_to_anchor=anchor, fontsize=10, title='Δk  (h in W/m²K)', title_fontsize=10)
    _finish(fig, save_path)
##
