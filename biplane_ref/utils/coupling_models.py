# name: coupling_models.py
# description: knee coupling curves used by the constraint-feedback (CF) method.
#
# Added for Darion's coupling study. Lets run_mc10_cf.py / run_eval_cf.py switch
# between the original Walker et al. (1988) curve and curves derived from the
# HAKnee biplane data (Darion-MBL-Work/Coupling/loso_coupling.py).
#
#   'walker'          original IMoveLab curve (default, unchanged behaviour)
#                     (Walker, Rovick & Robertson, J Biomech 1988;21:965-974, eqs 1-2; fitted to cadaver data of Reuben et al. 1986)
#                     'reuben' is accepted as an alias
#   'healthy_global'  one curve fit on ALL 15 healthy HAKnee subjects
#                     (NOTE: includes the subject being evaluated -> optimistic)
#   'healthy_loso'    for each subject, the curve fit WITHOUT that subject
#                     (fair test: the curve never saw the subject it corrects)
#
# All curves take flexion (deg, positive = flexion) and return adduction and
# internal rotation (deg) in the convention used inside correct_nonsagittal_knee.

import os
import numpy as np
import pandas as pd

COUPLING_CHOICES = ['walker', 'healthy_global', 'healthy_loso']

_HERE = os.path.dirname(os.path.abspath(__file__))
COUPLING_DIR = os.environ.get('COUPLING_DIR', os.path.abspath(os.path.join(
    _HERE, '..', '..', '..', 'Darion-MBL-Work', 'Darion-MBL-Work', 'Coupling', 'outputs', 'loso')))


def _walker_add(f):
    return 0.0791*f - 5.733e-4*f**2 - 7.682e-6*f**3 + 5.759e-8*f**4


def _walker_rot(f):
    return 0.3695*f - 2.958e-3*f**2 + 7.666e-6*f**3


def _poly(coefs, lo, hi):
    """c0 + c1 f + c2 f^2 ...; flexion clipped to the range the curve was fit on."""
    c = np.asarray(coefs, dtype=float)
    def fn(f):
        f = np.clip(f, lo, hi)
        return sum(ci * f**k for k, ci in enumerate(c))
    return fn


def _fit_range(group='Healthy'):
    cur = pd.read_csv(os.path.join(COUPLING_DIR, 'curves_bootstrap.csv'))
    cur = cur[(cur.group == group) & cur.in_data_range]
    return float(cur.flexion.min()), float(cur.flexion.max())


def _coefs(df, dof):
    row = df[df.dof == dof]
    cols = sorted([c for c in row.columns if c.startswith('c') and c[1:].isdigit()], key=lambda c: int(c[1:]))
    return row[cols].iloc[0].dropna().to_numpy(float)


def get_coupling(name, subject=None):
    """Return (adduction_fn, rotation_fn) for the chosen coupling."""
    if name in ('walker', 'reuben'):
        return _walker_add, _walker_rot

    lo, hi = _fit_range('Healthy')

    if name == 'healthy_global':
        c = pd.read_csv(os.path.join(COUPLING_DIR, 'coefficients.csv'))
        c = c[c.group == 'Healthy'].pivot(index='dof', columns='coef', values='estimate').reset_index()
        return _poly(_coefs(c, 'adduction'), lo, hi), _poly(_coefs(c, 'rotation'), lo, hi)

    if name == 'healthy_loso':
        subj_id = f'H{int(subject):02d}'
        c = pd.read_csv(os.path.join(COUPLING_DIR, 'fold_coefficients.csv'))
        c = c[(c.group == 'Healthy') & (c.fit_tasks == 'all') & (c.held_out == subj_id)]
        if c.empty:
            raise ValueError(f'No LOSO coupling fold found that holds out {subj_id}')
        return _poly(_coefs(c, 'adduction'), lo, hi), _poly(_coefs(c, 'rotation'), lo, hi)

    raise ValueError(f'Unknown coupling "{name}". Choose from {COUPLING_CHOICES}')


def folder_suffix(name):
    """Output-folder suffix, so different couplings never overwrite each other.
    Walker keeps the original folder name (bm_vqf6d_constrained_90p)."""
    return '' if name in ('walker', 'reuben') else f'_{name}'
