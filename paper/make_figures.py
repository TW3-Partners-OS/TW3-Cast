"""Regenerate every figure, table and number of the TW3Cast paper from the data/ folder.

Inputs (all in data/):
  public_scores.csv        per-configuration MASE and WQL of every public GIFT-Eval entry with
                           complete results, snapshot of the leaderboard results/ folder
                           (see SNAPSHOT_DATE.txt); columns dataset, model_dir, MASE, WQL.
  public_configs/*.json    the config.json of each public entry (type, leakage declaration).
  router.csv, experts.json the released frozen router and expert index.
  tw3cast_all_results.csv  OPTIONAL: the submission file of TW3Cast in the GIFT-Eval format
                           (results/TW3Cast/all_results.csv). When present, TW3Cast is inserted
                           into the ranking and the per-configuration figure is produced.

Outputs: figures/*.pdf, tables/*.tex, numbers.tex (macros used by main.tex).
Run:  python3 make_figures.py
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/.claude/skills/arxiv-paper/assets'))
from figstyle import PALETTE, save, setup  # noqa: E402

setup()
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data')
OURS = 'TW3Cast'
POOL = {  # public leaderboard entry -> label used in the paper
    'chronos-2': 'Chronos-2',
    'TiRex': 'TiRex',
    'TiRex-2-Pretrained': 'TiRex-2',
    'Toto-2.0-2.5B-FT': 'Toto 2.0 2.5B FT',
    'TimesFM-2.5': 'TimesFM 2.5',
    'TurkForecast-FM-Chronos2-LoRA-v1': 'TurkForecast LoRA',
}


def esc(s):
    return s.replace('_', '\\_').replace('&', '\\&')


# ----------------------------------------------------------------------------- leaderboard
def load_scores():
    A = pd.read_csv(os.path.join(D, 'public_scores.csv'))
    ours = os.path.join(D, 'tw3cast_all_results.csv')
    have_ours = os.path.exists(ours)
    if have_ours:
        T = pd.read_csv(ours)
        T = T.rename(columns={'eval_metrics/MASE[0.5]': 'MASE',
                              'eval_metrics/mean_weighted_sum_quantile_loss': 'WQL'})
        T['dataset'] = T['dataset'].str.lower()
        T['model_dir'] = OURS
        assert T.dataset.nunique() == 97, T.dataset.nunique()
        A = pd.concat([A, T[['dataset', 'model_dir', 'MASE', 'WQL']]])
    A['mase_rank'] = A.groupby('dataset').MASE.rank(method='min')
    A['wql_rank'] = A.groupby('dataset').WQL.rank(method='min')
    L = A.groupby('model_dir').agg(mase_rank=('mase_rank', 'mean'), wql_rank=('wql_rank', 'mean'),
                                   mase=('MASE', 'mean'), wql=('WQL', 'mean'))
    L = L.sort_values('mase_rank')
    L['position'] = np.arange(1, len(L) + 1)
    cfg = {}
    for f in glob.glob(os.path.join(D, 'public_configs', '*.json')):
        try:
            cfg[os.path.basename(f)[:-5]] = json.load(open(f))
        except Exception:
            pass
    L['type'] = [cfg.get(m, {}).get('model_type', 'router') if m != OURS else 'router'
                 for m in L.index]
    L['leak'] = [cfg.get(m, {}).get('testdata_leakage', 'No') for m in L.index]
    return A, L, have_ours


def table_leaderboard(L, have_ours):
    rows = []
    if have_ours:
        for m, r in L.head(8).iterrows():
            name = OURS + ' (this work)' if m == OURS else esc(m)
            rows.append(f"{int(r.position)} & {name} & {r.mase_rank:.1f} & {r.type} & {r.leak} \\\\")
    else:
        R = pd.read_csv(os.path.join(D, 'reported_standing.csv'))
        for _, r in R.iterrows():
            name = OURS + ' (this work)' if r.entry == OURS else esc(r.entry)
            rows.append(f"{int(r.position)} & {name} & {r.mase_rank:.1f} & {r.type} & {r.leakage} \\\\")
    with open(os.path.join(HERE, 'tables', 'leaderboard_top.tex'), 'w') as f:
        f.write(' \\\\\n'.join(r.rstrip(' \\') for r in rows) + '\n')
    rows = []
    for m, lab in POOL.items():
        if m not in L.index:
            continue
        r = L.loc[m]
        rows.append(f"{lab} & {int(r.position)} & {r.mase_rank:.1f} & {r.wql_rank:.1f} & "
                    f"{r.type} \\\\")
    with open(os.path.join(HERE, 'tables', 'components.tex'), 'w') as f:
        f.write(' \\\\\n'.join(r.rstrip(' \\') for r in rows) + '\n')


def fig_leaderboard(L, have_ours):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(1, len(L) + 1)
    ax.plot(x, L.mase_rank.values, color=PALETTE[5], lw=1.2, label='every public entry')
    marks = ['o', 'v', '^', 'D', 'P', 'X']
    for i, (m, lab) in enumerate(POOL.items()):
        if m not in L.index:
            continue
        r = L.loc[m]
        ax.plot(r.position, r.mase_rank, marks[i], color=PALETTE[i % 6], ms=6, ls='none',
                label=f'{lab} ({int(r.position)})')
    if have_ours:
        pos, rk = L.loc[OURS].position, L.loc[OURS].mase_rank
    else:
        R = pd.read_csv(os.path.join(D, 'reported_standing.csv')).set_index('entry')
        pos, rk = R.loc[OURS].position, R.loc[OURS].mase_rank
    ax.plot(pos, rk, 's', color='black', ms=7, ls='none', label=f'{OURS} ({int(pos)})')
    ax.legend(loc='lower right', ncol=2, fontsize=7.5)
    ax.set_xlabel('position on the leaderboard (sorted by mean MASE rank)')
    ax.set_ylabel('mean per-configuration MASE rank')
    ax.set_xlim(0, len(L) + 1)
    save(fig, os.path.join(HERE, 'figures', 'fig_leaderboard.pdf'))


# ----------------------------------------------------------------------------- router table
def router_modes():
    r = pd.read_csv(os.path.join(D, 'router.csv'))
    ex = json.load(open(os.path.join(D, 'experts.json')))
    assert r.config.nunique() == 97
    assert (r.groupby('config').weight.sum().round(6) == 1).all()
    modes = {}
    members = {}
    for c, g in r.groupby('config'):
        m = list(g.expert)
        members[c] = m
        if m == ['__tournament__']:
            modes[c] = 'tournament'
        elif len(m) == 1:
            modes[c] = 'specialist'
        elif any(x.startswith('E') for x in m):
            modes[c] = 'blend with specialist'
        else:
            modes[c] = 'blend of base models'
    M = pd.DataFrame({'mode': modes, 'members': members})
    M['term'] = [c.split('/')[-1] for c in M.index]
    used = sorted(set(e for e in r.expert if e.startswith('E')))
    fam = pd.Series([ex[e]['type'] for e in used]).value_counts()
    return r, M, used, fam


ORDER = ['specialist', 'blend with specialist', 'blend of base models', 'tournament']
FAM = {'c2': 'Chronos-2', 'tirex': 'TiRex', 'toto': 'Toto 2.5B'}


def fig_router(M, fam):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.2, 2.6), gridspec_kw={'width_ratios': [3, 2], 'wspace': 0.55})
    terms = ['short', 'medium', 'long']
    bottom = np.zeros(3)
    for i, mode in enumerate(ORDER):
        v = np.array([((M.term == t) & (M['mode'] == mode)).sum() for t in terms])
        a.bar(terms, v, bottom=bottom, color=PALETTE[i], label=mode, width=0.6)
        for j, (bb, vv) in enumerate(zip(bottom, v)):
            if vv:
                a.text(j, bb + vv / 2, str(vv), ha='center', va='center', fontsize=7,
                       color='white')
        bottom += v
    a.set_ylabel('configurations')
    a.set_xlabel('horizon class')
    a.legend(fontsize=6.5, loc='upper right')
    labels = [FAM[k] for k in fam.index]
    b.barh(labels, fam.values, color=[PALETTE[3]] * len(fam))
    for i, v in enumerate(fam.values):
        b.text(v + 0.3, i, str(v), va='center', fontsize=7)
    b.set_xlabel('designated specialists')
    b.set_xlim(0, fam.values.max() + 4)
    b.invert_yaxis()
    save(fig, os.path.join(HERE, 'figures', 'fig_router.pdf'))


def table_router(M, r, ex):
    """Appendix table: the full router, one row per configuration."""
    names = {'chronos2': 'C2', 'toto_25b_ft': 'Toto', 'turk': 'Turk', 'timesfm': 'TFM',
             'tirex': 'TiRex', 'tirex2': 'TiRex-2', 'toto_uni': 'Toto-uni'}
    rows = []
    for c in sorted(M.index):
        mem = M.loc[c, 'members']
        if mem == ['__tournament__']:
            s = 'tournament'
        else:
            parts = []
            for e in mem:
                if e.startswith('E'):
                    parts.append(f"{e}\\,({ {'c2': 'C2', 'tirex': 'TiRex', 'toto': 'Toto'}[ex[e]['type']] })")
                else:
                    parts.append(names.get(e, e))
            s = ' + '.join(parts)
        rows.append((esc(c), s))
    half = (len(rows) + 1) // 2
    for part, chunk in (('a', rows[:half]), ('b', rows[half:])):
        with open(os.path.join(HERE, 'tables', f'router_full_{part}.tex'), 'w') as f:
            f.write(' \\\\\n'.join(f"{c} & {m}" for c, m in chunk) + '\n')


# ----------------------------------------------------------------------------- per-config
def fig_perconfig(A, have_ours):
    if not have_ours:
        return
    P = A[A.model_dir.isin(list(POOL) + [OURS])].pivot(index='dataset', columns='model_dir',
                                                        values='MASE')
    best = P[list(POOL)].min(axis=1)
    ratio = (P[OURS] / best).sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(np.arange(len(ratio)), (ratio.values - 1) * 100,
           color=[PALETTE[2] if v < 1 else PALETTE[1] for v in ratio.values], width=0.8)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_yscale('symlog', linthresh=25, linscale=1.2)
    ax.set_yticks([-20, -10, 0, 10, 25, 50, 100, 200, 400])
    ax.set_yticklabels(['-20', '-10', '0', '10', '25', '50', '100', '200', '400'])
    ax.set_xlabel('configuration, sorted by the ratio')
    ax.set_ylabel('MASE vs. best pool member (%)')
    save(fig, os.path.join(HERE, 'figures', 'fig_perconfig.pdf'))
    return ratio


# ----------------------------------------------------------------------------- ablation
def ablation(A, L, have_ours):
    """Mean MASE rank, in the same public field, of each pool member, of the tournament-only
    variant (every configuration served by the tournament mode) and of the full system, on all
    97 configurations and on the configurations the released router serves by tournament."""
    f = os.path.join(D, 'tournament_only_per_config.csv')
    if not (have_ours and os.path.exists(f)):
        return {}
    T = pd.read_csv(f).rename(columns={'config': 'dataset', 'mase': 'MASE', 'wql': 'WQL'})
    T['dataset'] = T['dataset'].str.lower()
    T['model_dir'] = 'tournament-only'
    # the field of Table 1 (public entries plus TW3Cast), into which the tournament-only
    # variant is inserted as one more entry; pool members and the full system keep the
    # ranks of Tables 1 and 2
    B1 = pd.concat([A[['dataset', 'model_dir', 'MASE', 'WQL']],
                    T[['dataset', 'model_dir', 'MASE', 'WQL']]])
    B1['mase_rank'] = B1.groupby('dataset').MASE.rank(method='min')
    B = pd.concat([A[['dataset', 'model_dir', 'MASE', 'WQL', 'mase_rank']],
                   B1[B1.model_dir == 'tournament-only']])
    M = pd.read_csv(os.path.join(HERE, 'tables', 'router_modes.csv'), index_col=0)
    tconf = set(M.index[M['mode'] == 'tournament'].str.lower())
    rows = []
    variants = [(m, POOL[m]) for m in POOL if m in L.index] + \
               [('tournament-only', 'Tournament on every configuration'),
                (OURS, 'Full system (released router)')]
    for key, lab in variants:
        allc = B[B.model_dir == key].mase_rank.mean()
        sub = B[(B.model_dir == key) & (B.dataset.isin(tconf))].mase_rank.mean()
        rows.append((lab, allc, sub))
    with open(os.path.join(HERE, 'tables', 'ablation.tex'), 'w') as fh:
        fh.write(' \\\\\n'.join(f"{lab} & {a:.1f} & {b:.1f}" for lab, a, b in rows) + '\n')
    pd.DataFrame(rows, columns=['variant', 'all97', 'tournament_configs']).to_csv(
        os.path.join(HERE, 'tables', 'ablation.csv'), index=False)
    out = {'tourOnlyRank': f"{rows[-2][1]:.1f}", 'tourOnRankSub': f"{rows[-2][2]:.1f}",
           'bestPoolRankSub': f"{min(r[2] for r in rows[:-2]):.1f}",
           'ourRankSub': f"{rows[-1][2]:.1f}", 'nTournamentConfigs': len(tconf)}
    # how the tournament-only variant differs from the full system on the other configurations
    P = B[B.model_dir.isin(['tournament-only', OURS])].pivot(index='dataset', columns='model_dir',
                                                             values='MASE')
    other = P[~P.index.isin(tconf)]
    out['nSpecBeatsTour'] = int((other[OURS] < other['tournament-only']).sum())
    out['nOther'] = int(len(other))
    out['medGainSpec'] = f"{(1 - (other[OURS] / other['tournament-only']).median()) * 100:.1f}"
    return out


# ----------------------------------------------------------------------------- numbers
def numbers(L, M, used, fam, have_ours, A):  # noqa: C901
    n = len(L)
    cnt = M['mode'].value_counts()
    k = M.members.apply(len)
    macros = {
        'nPublic': n if not have_ours else n - 1,
        'nSpecialist': int(cnt.get('specialist', 0)),
        'nBlendSpec': int(cnt.get('blend with specialist', 0)),
        'nBlendBase': int(cnt.get('blend of base models', 0)),
        'nTournament': int(cnt.get('tournament', 0)),
        'nWithSpecialist': int(cnt.get('specialist', 0) + cnt.get('blend with specialist', 0)),
        'nNotTournament': int(97 - cnt.get('tournament', 0)),
        'nExperts': len(used),
        'nExpC': int(fam.get('c2', 0)), 'nExpTirex': int(fam.get('tirex', 0)),
        'nExpToto': int(fam.get('toto', 0)),
        'nBlendTwo': int(((k == 2)).sum()), 'nBlendThree': int((k == 3).sum()),
        'nBlendFour': int((k == 4).sum()),
        'nBlendAll': int((k >= 2).sum()),
        'nBlendMixed': int(sum(
            len({(json.load(open(os.path.join(D, 'experts.json')))[e]['type'] if e.startswith('E')
                  else {'chronos2': 'c2', 'turk': 'c2', 'toto_25b_ft': 'toto', 'toto_uni': 'toto',
                        'tirex': 'tirex', 'tirex2': 'tirex', 'timesfm': 'tfm'}[e]) for e in mem}) > 1
            for mem in M.members if len(mem) >= 2)),
        'bestPoolName': None, 'bestPoolPos': None, 'bestPoolRank': None,
    }
    pool_in = [m for m in POOL if m in L.index]
    bp = L.loc[pool_in].sort_values('mase_rank').iloc[0]
    macros['bestPoolName'] = POOL[bp.name]
    macros['bestPoolPos'] = int(bp.position)
    macros['bestPoolRank'] = f"{bp.mase_rank:.1f}"
    macros['worstPoolRank'] = f"{L.loc[pool_in].mase_rank.max():.1f}"
    top = L.iloc[0]
    macros['topName'] = esc(top.name)
    macros['topRank'] = f"{top.mase_rank:.1f}"
    if have_ours:
        r = L.loc[OURS]
        macros['ourPos'] = int(r.position)
        macros['ourRank'] = f"{r.mase_rank:.1f}"
        macros['ourWqlRank'] = f"{r.wql_rank:.1f}"
        macros['ourWqlPos'] = int((L.sort_values('wql_rank').index == OURS).argmax() + 1)
        above = L[L.position < r.position]
        macros['nAgentAbove'] = int((above.type == 'agentic').sum())
        macros['nEntries'] = int(len(L))
        P = A[A.model_dir.isin(pool_in + [OURS])].pivot(index='dataset', columns='model_dir',
                                                        values='MASE')
        best = P[pool_in].min(axis=1)
        ratio = P[OURS] / best
        macros['nBeatBest'] = int((ratio < 1).sum())
        macros['nWithinFive'] = int((ratio < 1.05).sum())
        macros['nWorseTen'] = int((ratio > 1.10).sum())
        macros['medRatioBest'] = f"{ratio.median():.3f}"
        macros['nBeatEveryMember'] = int((P[OURS] < P[pool_in].min(axis=1)).sum())
    else:
        R = pd.read_csv(os.path.join(D, 'reported_standing.csv')).set_index('entry')
        macros['ourPos'] = int(R.loc[OURS].position)
        macros['ourRank'] = f"{R.loc[OURS].mase_rank:.1f}"
        macros['nAgentAbove'] = int((R[R.position < R.loc[OURS].position].type == 'agentic').sum())
        macros['nEntries'] = int(macros['nPublic']) + 1
    macros.update(ablation(A, L, have_ours))
    with open(os.path.join(HERE, 'numbers.tex'), 'w') as f:
        f.write('% generated by make_figures.py; do not edit by hand\n')
        for kk, v in macros.items():
            f.write(f"\\newcommand{{\\{kk}}}{{{v}}}\n")
    print(json.dumps(macros, indent=1))


if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    A, L, have_ours = load_scores()
    print('TW3Cast file present:', have_ours, '| entries ranked:', len(L))
    L.to_csv(os.path.join(HERE, 'tables', 'leaderboard_full.csv'))
    table_leaderboard(L, have_ours)
    fig_leaderboard(L, have_ours)
    r, M, used, fam = router_modes()
    ex = json.load(open(os.path.join(D, 'experts.json')))
    fig_router(M, fam)
    table_router(M, r, ex)
    M.to_csv(os.path.join(HERE, 'tables', 'router_modes.csv'))
    fig_perconfig(A, have_ours)
    numbers(L, M, used, fam, have_ours, A)
