//! Newton / Levenberg–Marquardt solve, decomposition driver and diagnostics.
//!
//! Each step solves the damped minimum-norm least-squares problem
//!
//! ```text
//! minimise ‖r + J δ‖² + μ ‖δ‖²_{W⁻¹}   ⇒   δ = −W Jᵀ (J W Jᵀ + μ I)⁻¹ r
//! ```
//!
//! with the sparse LDLᵀ of `J W Jᵀ + μ I`. Newton mode uses a tiny `μ` and a backtracking line
//! search; when the line search fails the sub-problem switches to Levenberg–Marquardt with
//! Nielsen's damping update. The weight `W` is 1 except for dragged variables, which get a small
//! weight so the step moves them least.

use crate::decompose::{components, dulmage_mendelsohn};
use crate::ids::{ConstraintId, EntityId, VarId};
use crate::linalg::{dense_rank, Ldlt, RowQr, SparseRows};
use crate::report::*;
use crate::residual::{compile, Compiled, Op};
use crate::system::{EntityKind, System, VarFlag};

const NONE: usize = usize::MAX;

/// Variables to pull toward targets during a drag.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DragRequest {
    pub targets: Vec<(VarId, f64)>,
}

impl DragRequest {
    /// Drag a 2D point toward `to`.
    pub fn point(sys: &System, point: EntityId, to: [f64; 2]) -> DragRequest {
        match sys.entity(point).kind {
            EntityKind::Point2 { x, y } => DragRequest { targets: vec![(x, to[0]), (y, to[1])] },
            _ => panic!("{point} is not a point"),
        }
    }

    /// Drag a body's origin toward `to` (orientation follows the constraints).
    pub fn body_position(sys: &System, body: EntityId, to: [f64; 3]) -> DragRequest {
        match sys.entity(body).kind {
            EntityKind::Body { position, .. } => {
                DragRequest { targets: (0..3).map(|i| (position[i], to[i])).collect() }
            }
            _ => panic!("{body} is not a body"),
        }
    }
}

/// Solves the system from its current values (or the warm start) and diagnoses it.
pub fn solve(sys: &System, opts: &SolveOptions) -> SolveReport {
    run(sys, opts, None)
}

/// Solves with a soft drag target: the dragged variables start at their targets and carry a
/// small step weight, so the constraints are met while they stay as close to the targets as the
/// constraints allow.
pub fn drag(sys: &System, request: &DragRequest, opts: &SolveOptions) -> SolveReport {
    run(sys, opts, Some(request))
}

#[derive(Clone, Copy, Debug)]
struct Row {
    c: usize,
    r: usize,
}

struct Problem<'a> {
    compiled: &'a [Compiled],
    ops: &'a [Option<Op>],
    rows: Vec<Row>,
    /// Free variables (global indices) each row reads, ascending.
    row_vars: Vec<Vec<usize>>,
    free: Vec<usize>,
}

struct SubResult {
    converged: bool,
    iterations: usize,
    method: Method,
}

impl<'a> Problem<'a> {
    fn new(compiled: &'a [Compiled], ops: &'a [Option<Op>], is_free: &[bool], x: &[f64], tol: f64) -> Self {
        let mut rows = Vec::new();
        let mut row_vars = Vec::new();
        for (ci, c) in compiled.iter().enumerate() {
            let Some(op) = &ops[ci] else { continue };
            let mut fv: Vec<usize> = c.vars.iter().map(|v| v.index()).filter(|&v| is_free[v]).collect();
            fv.sort_unstable();
            fv.dedup();
            if c.internal && fv.is_empty() {
                // An internal identity on fully fixed geometry: only report it if it is broken.
                let mut vals = vec![0.0; op.rows()];
                let xl: Vec<f64> = c.vars.iter().map(|v| x[v.index()]).collect();
                op.values(&xl, &mut vals);
                if vals.iter().all(|v| v.abs() <= tol) {
                    continue;
                }
            }
            for r in 0..op.rows() {
                rows.push(Row { c: ci, r });
                row_vars.push(fv.clone());
            }
        }
        let free = (0..is_free.len()).filter(|&v| is_free[v]).collect();
        Problem { compiled, ops, rows, row_vars, free }
    }

    fn op(&self, c: usize) -> &Op {
        self.ops[c].as_ref().unwrap()
    }

    fn local_x(&self, c: usize, x: &[f64]) -> Vec<f64> {
        self.compiled[c].vars.iter().map(|v| x[v.index()]).collect()
    }

    /// Residuals of `rows` (indices into `self.rows`, ascending).
    fn residuals(&self, rows: &[usize], x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        let mut cached: Option<(usize, Vec<f64>)> = None;
        for &ri in rows {
            let Row { c, r } = self.rows[ri];
            if cached.as_ref().map(|k| k.0) != Some(c) {
                let op = self.op(c);
                let mut vals = vec![0.0; op.rows()];
                op.values(&self.local_x(c, x), &mut vals);
                cached = Some((c, vals));
            }
            out.push(cached.as_ref().unwrap().1[r]);
        }
    }

    /// Residuals and the Jacobian of `rows` with respect to the variables mapped by `col_of`.
    fn jacobian(&self, rows: &[usize], x: &[f64], col_of: &[usize], ncols: usize) -> (Vec<f64>, SparseRows) {
        let mut res = Vec::with_capacity(rows.len());
        let mut jac = SparseRows::new(ncols);
        let mut cached: Option<(usize, Vec<f64>, Vec<f64>)> = None;
        let mut entries = Vec::new();
        for &ri in rows {
            let Row { c, r } = self.rows[ri];
            if cached.as_ref().map(|k| k.0) != Some(c) {
                let op = self.op(c);
                let xl = self.local_x(c, x);
                let mut vals = vec![0.0; op.rows()];
                let mut block = vec![0.0; op.rows() * xl.len()];
                op.jacobian(&xl, &mut vals, &mut block);
                cached = Some((c, vals, block));
            }
            let (_, vals, block) = cached.as_ref().unwrap();
            let vars = &self.compiled[c].vars;
            let n = vars.len();
            res.push(vals[r]);
            for (p, v) in vars.iter().enumerate() {
                let col = col_of[v.index()];
                let a = block[r * n + p];
                if col != NONE && a != 0.0 && a.is_finite() {
                    entries.push((col, a));
                }
            }
            jac.push_row(&mut entries);
        }
        (res, jac)
    }

    /// Newton with line search, falling back to Levenberg–Marquardt, on a row/variable subset.
    fn solve_sub(
        &self,
        rows: &[usize],
        vars: &[usize],
        x: &mut [f64],
        weights: &[f64],
        col_of: &mut [usize],
        opts: &SolveOptions,
    ) -> SubResult {
        let mut r = Vec::new();
        self.residuals(rows, x, &mut r);
        if max_abs(&r) <= opts.tolerance {
            return SubResult { converged: true, iterations: 0, method: Method::None };
        }
        if vars.is_empty() {
            return SubResult { converged: false, iterations: 0, method: Method::None };
        }
        for (k, &v) in vars.iter().enumerate() {
            col_of[v] = k;
        }
        let w: Vec<f64> = vars.iter().map(|&v| weights[v]).collect();
        let n = vars.len();
        let mut f = norm2(&r);
        let mut lm = false;
        let mut mu = 0.0;
        let mut nu = 2.0;
        let mut method = Method::Newton;
        let mut converged = false;
        let mut iterations = 0;
        let mut trial = x.to_vec();
        let mut r_try = Vec::new();
        let mut jd = vec![0.0; rows.len()];
        while iterations < opts.max_iterations {
            iterations += 1;
            let (_, jac) = self.jacobian(rows, x, col_of, n);
            let mut diag_max = 0.0f64;
            for i in 0..jac.nrows() {
                let (c, v) = jac.row(i);
                let d: f64 = c.iter().zip(v).map(|(&c, &v)| v * v * w[c]).sum();
                diag_max = diag_max.max(d);
            }
            if diag_max == 0.0 {
                break;
            }
            let damping = if lm { mu } else { 1e-12 * diag_max };
            let delta = step(&jac, &r, &w, damping, diag_max);
            if !lm {
                let mut alpha = 1.0;
                let mut accepted = false;
                for _ in 0..12 {
                    for (k, &v) in vars.iter().enumerate() {
                        trial[v] = x[v] + alpha * delta[k];
                    }
                    self.residuals(rows, &trial, &mut r_try);
                    let ft = norm2(&r_try);
                    if ft.is_finite() && ft <= (1.0 - 1e-4 * alpha) * f {
                        accepted = true;
                        break;
                    }
                    alpha *= 0.5;
                }
                if !accepted {
                    lm = true;
                    method = Method::LevenbergMarquardt;
                    mu = 1e-3 * diag_max;
                    continue;
                }
            } else {
                jac.mul(&delta, &mut jd);
                let predicted = f - r.iter().zip(&jd).map(|(a, b)| (a + b) * (a + b)).sum::<f64>();
                for (k, &v) in vars.iter().enumerate() {
                    trial[v] = x[v] + delta[k];
                }
                self.residuals(rows, &trial, &mut r_try);
                let ft = norm2(&r_try);
                let actual = f - ft;
                if predicted > 0.0 && actual > 0.0 && ft.is_finite() {
                    let rho = actual / predicted;
                    mu *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3));
                    nu = 2.0;
                } else {
                    mu *= nu;
                    nu *= 2.0;
                    // Gradient-stationary: a least-squares minimum with a nonzero residual.
                    let mut g = vec![0.0; n];
                    jac.mul_t(&r, &mut g);
                    let gmax = max_abs(&g);
                    if gmax <= 1e-14 * diag_max.sqrt() * (1.0 + f.sqrt()) || mu > 1e16 * diag_max {
                        break;
                    }
                    continue;
                }
            }
            // Accept the trial point.
            let mut moved = 0.0f64;
            for &v in vars {
                moved = moved.max((trial[v] - x[v]).abs() / (1.0 + x[v].abs()));
                x[v] = trial[v];
            }
            std::mem::swap(&mut r, &mut r_try);
            f = norm2(&r);
            if max_abs(&r) <= opts.tolerance {
                converged = true;
                break;
            }
            if moved <= 1e-15 {
                if !lm {
                    lm = true;
                    method = Method::LevenbergMarquardt;
                    mu = 1e-3 * diag_max;
                } else {
                    break;
                }
            }
        }
        for &v in vars {
            col_of[v] = NONE;
        }
        SubResult { converged, iterations, method }
    }

    /// Solves one connected component, by DM blocks when that splits it, else whole.
    fn solve_component(
        &self,
        rows: &[usize],
        vars: &[usize],
        x: &mut [f64],
        weights: &[f64],
        col_of: &mut [usize],
        opts: &SolveOptions,
    ) -> (SubResult, usize) {
        let mut total = 0;
        if opts.decompose && rows.len() > 1 && vars.len() > 1 {
            let pattern = self.structural_pattern(rows, vars, x, col_of);
            let blocks = dulmage_mendelsohn(rows.len(), vars.len(), &pattern);
            if blocks.len() > 1 {
                let saved: Vec<f64> = vars.iter().map(|&v| x[v]).collect();
                let mut ok = true;
                let mut method = Method::None;
                for b in &blocks {
                    let brows: Vec<usize> = b.rows.iter().map(|&i| rows[i]).collect();
                    let bvars: Vec<usize> = b.vars.iter().map(|&i| vars[i]).collect();
                    if brows.is_empty() {
                        continue;
                    }
                    let res = self.solve_sub(&brows, &bvars, x, weights, col_of, opts);
                    total += res.iterations;
                    method = merge_method(method, res.method);
                    if !res.converged {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    let mut r = Vec::new();
                    self.residuals(rows, x, &mut r);
                    if max_abs(&r) <= opts.tolerance {
                        return (SubResult { converged: true, iterations: total, method }, blocks.len());
                    }
                }
                for (k, &v) in vars.iter().enumerate() {
                    x[v] = saved[k];
                }
            }
        }
        let mut res = self.solve_sub(rows, vars, x, weights, col_of, opts);
        res.iterations += total;
        (res, 1)
    }

    /// Which component variables each row really reads: the Jacobian's nonzeros at a
    /// deterministically perturbed point (so axis-aligned geometry shows no false zeros).
    fn structural_pattern(&self, rows: &[usize], vars: &[usize], x: &[f64], col_of: &mut [usize]) -> Vec<Vec<usize>> {
        let mut xp = x.to_vec();
        for &v in vars {
            let h = ((v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f64 / (1u64 << 24) as f64;
            xp[v] += (0.3 + 0.7 * h) * 1e-3 * (1.0 + x[v].abs());
        }
        for (k, &v) in vars.iter().enumerate() {
            col_of[v] = k;
        }
        let (_, jac) = self.jacobian(rows, &xp, col_of, vars.len());
        for &v in vars {
            col_of[v] = NONE;
        }
        (0..rows.len()).map(|i| jac.row(i).0.to_vec()).collect()
    }
}

fn merge_method(a: Method, b: Method) -> Method {
    match (a, b) {
        (Method::LevenbergMarquardt, _) | (_, Method::LevenbergMarquardt) => Method::LevenbergMarquardt,
        (Method::Newton, _) | (_, Method::Newton) => Method::Newton,
        _ => Method::None,
    }
}

/// `δ = −W Jᵀ (J W Jᵀ + μ I)⁻¹ r`.
fn step(jac: &SparseRows, r: &[f64], w: &[f64], mu: f64, diag_max: f64) -> Vec<f64> {
    let m = jac.gram(w, mu);
    let f = Ldlt::factor(&m, 1e-15 * diag_max.max(f64::MIN_POSITIVE));
    let mut y: Vec<f64> = r.iter().map(|v| -v).collect();
    f.solve(&mut y);
    let mut delta = vec![0.0; jac.ncols];
    jac.mul_t(&y, &mut delta);
    for (d, wi) in delta.iter_mut().zip(w) {
        *d *= wi;
    }
    delta
}

fn norm2(r: &[f64]) -> f64 {
    r.iter().map(|v| v * v).sum()
}
fn max_abs(r: &[f64]) -> f64 {
    r.iter().fold(0.0f64, |m, v| if v.is_nan() { f64::INFINITY } else { m.max(v.abs()) })
}

/// Component diagnostics.
struct Diagnosis {
    rank: usize,
    dof: usize,
    /// `(row, rows it depends on, conflicting)` with indices into the component's row list.
    dependent: Vec<(usize, Vec<usize>, bool)>,
    /// Null-space rows per component variable (`None` when skipped for size).
    null_rows: Option<Vec<Vec<f64>>>,
}

fn diagnose(
    p: &Problem<'_>,
    rows: &[usize],
    vars: &[usize],
    x: &[f64],
    converged: bool,
    col_of: &mut [usize],
    opts: &SolveOptions,
) -> Diagnosis {
    let n = vars.len();
    for (k, &v) in vars.iter().enumerate() {
        col_of[v] = k;
    }
    let (r, jac) = p.jacobian(rows, x, col_of, n);
    for &v in vars {
        col_of[v] = NONE;
    }
    let m = rows.len();
    let norms: Vec<f64> = (0..m).map(|i| jac.row_norm(i)).collect();
    let mut jn = SparseRows::new(n);
    let mut entries = Vec::new();
    for i in 0..m {
        let (c, v) = jac.row(i);
        if norms[i] > 1e-14 {
            entries.extend(c.iter().zip(v).map(|(&c, &v)| (c, v / norms[i])));
        }
        jn.push_row(&mut entries);
    }
    let qr = RowQr::factor(&jn, opts.rank_tolerance);
    let rank = qr.rank();
    let dof = n - rank;
    let ctol = (100.0 * opts.tolerance).max(1e-9);

    let indep: Vec<usize> = (0..m).filter(|&i| qr.independent[i]).collect();
    let mut dependent = Vec::new();
    let dep_rows: Vec<usize> = (0..m).filter(|&i| !qr.independent[i]).collect();
    if !dep_rows.is_empty() {
        // Gram matrix of the independent rows, for the dependency coefficients.
        let mut ji = SparseRows::new(n);
        for &i in &indep {
            let (c, v) = jn.row(i);
            entries.extend(c.iter().copied().zip(v.iter().copied()));
            ji.push_row(&mut entries);
        }
        let ones = vec![1.0; n];
        let gram = ji.gram(&ones, 1e-14);
        let f = if indep.is_empty() { None } else { Some(Ldlt::factor(&gram, 1e-14)) };
        let mut dense = vec![0.0; n];
        for &k in &dep_rows {
            if norms[k] <= 1e-14 {
                dependent.push((k, Vec::new(), !converged && r[k].abs() > ctol));
                continue;
            }
            let (kc, kv) = jn.row(k);
            for (&c, &v) in kc.iter().zip(kv) {
                dense[c] = v;
            }
            let mut b: Vec<f64> = (0..indep.len())
                .map(|t| {
                    let (c, v) = ji.row(t);
                    c.iter().zip(v).map(|(&c, &v)| v * dense[c]).sum()
                })
                .collect();
            for &c in kc {
                dense[c] = 0.0;
            }
            if let Some(f) = &f {
                f.solve(&mut b);
            }
            let cmax = max_abs(&b);
            let mut deps = Vec::new();
            let mut g = r[k] / norms[k];
            for (t, &ci) in b.iter().enumerate() {
                if ci.abs() > 1e-6 * cmax.max(1e-300) {
                    deps.push(indep[t]);
                }
                g -= ci * r[indep[t]] / norms[indep[t]];
            }
            dependent.push((k, deps, !converged && g.abs() > ctol));
        }
    }
    if !converged && !dependent.iter().any(|d| d.2) {
        // No dependent row carries the inconsistency (the iterate may sit just off the singular
        // configuration). At a least-squares stationary point the residual itself lies in the
        // left null space of J: its support is the conflicting set.
        let rn: Vec<f64> = (0..m).map(|i| if norms[i] > 1e-14 { r[i] / norms[i] } else { r[i] }).collect();
        let mut g = vec![0.0; n];
        jn.mul_t(&rn, &mut g);
        let rnorm = norm2(&rn).sqrt();
        if rnorm > ctol && norm2(&g).sqrt() <= 1e-6 * rnorm {
            let support: Vec<usize> = (0..m).filter(|&i| rn[i].abs() > ctol).collect();
            if let Some((&last, rest)) = support.split_last() {
                dependent.retain(|d| d.0 != last);
                dependent.push((last, rest.to_vec(), true));
            }
        }
    }
    let null_rows = if dof == 0 {
        Some(Vec::new())
    } else if n.saturating_mul(dof) <= opts.max_null_space_entries {
        let basis = qr.null_space();
        Some((0..n).map(|j| basis.iter().map(|b| b[j]).collect()).collect())
    } else {
        None
    };
    Diagnosis { rank, dof, dependent, null_rows }
}

fn run(sys: &System, opts: &SolveOptions, drag_request: Option<&DragRequest>) -> SolveReport {
    let nv = sys.vars.len();
    let mut x: Vec<f64> = sys.vars.iter().map(|v| v.value).collect();
    let is_free: Vec<bool> = sys.vars.iter().map(|v| v.flag == VarFlag::Free).collect();

    let mut cached = Vec::new();
    if let Some(ws) = &opts.warm_start {
        for &(v, val) in &ws.values {
            if v.index() < nv && is_free[v.index()] && val.is_finite() {
                x[v.index()] = val;
                cached.push((v.index(), val));
            }
        }
    }
    let mut weights = vec![1.0; nv];
    if let Some(d) = drag_request {
        for &(v, t) in &d.targets {
            if v.index() < nv && is_free[v.index()] && t.is_finite() {
                x[v.index()] = t;
                weights[v.index()] = opts.drag_weight;
            }
        }
    }

    let compiled = compile(sys, &x);
    let mut ops: Vec<Option<Op>> =
        compiled.iter().map(|c| (!c.reference && c.range.is_none()).then(|| c.op.clone())).collect();
    let mut active_limits: Vec<(ConstraintId, f64)> = Vec::new();

    let mut col_of = vec![NONE; nv];
    let mut cache_satisfied = false;
    let mut solved: Vec<(Vec<usize>, Vec<usize>, SubResult, usize)>;
    let mut pass = 0;
    loop {
        let problem = Problem::new(&compiled, &ops, &is_free, &x, opts.tolerance);
        if pass == 0 && !cached.is_empty() {
            let all: Vec<usize> = (0..problem.rows.len()).collect();
            let mut r = Vec::new();
            problem.residuals(&all, &x, &mut r);
            cache_satisfied = max_abs(&r) <= opts.tolerance;
        }
        let (comps, _) = split(&problem, nv);
        solved = Vec::with_capacity(comps.len());
        for (rows, vars) in comps {
            let (mut res, blocks) = problem.solve_component(&rows, &vars, &mut x, &weights, &mut col_of, opts);
            // Drag refinement: put the dragged variables back on their targets and solve again.
            // Each round shrinks the remaining offset by about the drag weight, so a reachable
            // target is met exactly and an unreachable one settles at its constrained closest
            // point.
            if let Some(d) = drag_request {
                let dragged: Vec<(usize, f64)> = d
                    .targets
                    .iter()
                    .map(|&(v, t)| (v.index(), t))
                    .filter(|&(v, _)| v < nv && is_free[v] && vars.binary_search(&v).is_ok())
                    .collect();
                for _ in 0..12 {
                    if !res.converged || dragged.is_empty() {
                        break;
                    }
                    let before: Vec<f64> = vars.iter().map(|&v| x[v]).collect();
                    for &(v, t) in &dragged {
                        x[v] = t;
                    }
                    let (again, _) = problem.solve_component(&rows, &vars, &mut x, &weights, &mut col_of, opts);
                    res.iterations += again.iterations;
                    res.method = merge_method(res.method, again.method);
                    res.converged = again.converged;
                    let change = vars.iter().zip(&before).map(|(&v, b)| (x[v] - b).abs()).fold(0.0f64, f64::max);
                    if change <= opts.tolerance {
                        break;
                    }
                }
            }
            solved.push((rows, vars, res, blocks));
        }
        // Range limits: clamp any violated bound and solve again.
        pass += 1;
        if pass >= 8 {
            break;
        }
        let mut changed = false;
        for (ci, c) in compiled.iter().enumerate() {
            let Some((lo, hi)) = c.range else { continue };
            if c.reference || ops[ci].is_some() {
                continue;
            }
            let v = measure(c, &x);
            let bound = if v < lo - opts.tolerance {
                lo
            } else if v > hi + opts.tolerance {
                hi
            } else {
                continue;
            };
            ops[ci] = Some(c.op.with_target(bound));
            active_limits.push((c.id, bound));
            changed = true;
        }
        if !changed {
            break;
        }
    }

    let problem = Problem::new(&compiled, &ops, &is_free, &x, opts.tolerance);
    let (comps, orphan) = split(&problem, nv);
    debug_assert_eq!(comps.len(), solved.len());

    let mut components = Vec::with_capacity(comps.len());
    let mut redundant = Vec::new();
    let mut conflicting = Vec::new();
    let mut determined = is_free.iter().map(|f| !f).collect::<Vec<bool>>();
    // Per free variable: (component index, local column).
    let mut var_comp = vec![(NONE, NONE); nv];
    let mut null_by_comp: Vec<Option<Vec<Vec<f64>>>> = Vec::with_capacity(comps.len());
    let mut comp_dof = Vec::with_capacity(comps.len());
    let mut total_dof = 0;
    let mut all_converged = true;
    let mut total_iterations = 0;
    let mut r = Vec::new();

    let constraint_of = |row: usize| ConstraintId(problem.rows[row].c as u32);
    let push_dependent = |list: &mut Vec<DependentConstraint>, c: ConstraintId, deps: Vec<ConstraintId>| {
        if let Some(d) = list.iter_mut().find(|d| d.constraint == c) {
            d.depends_on.extend(deps);
            d.depends_on.sort();
            d.depends_on.dedup();
        } else {
            let mut deps = deps;
            deps.sort();
            deps.dedup();
            deps.retain(|&d| d != c);
            list.push(DependentConstraint { constraint: c, depends_on: deps });
        }
    };

    for (k, ((rows, vars), (_, _, res, blocks))) in comps.iter().zip(&solved).enumerate() {
        problem.residuals(rows, &x, &mut r);
        let converged = max_abs(&r) <= opts.tolerance;
        all_converged &= converged;
        total_iterations += res.iterations;
        for (j, &v) in vars.iter().enumerate() {
            var_comp[v] = (k, j);
        }
        let (rank, dof, null_rows) = if opts.diagnose {
            let d = diagnose(&problem, rows, vars, &x, converged, &mut col_of, opts);
            for (row, deps, bad) in d.dependent {
                let c = constraint_of(rows[row]);
                let mut deps: Vec<ConstraintId> = deps.iter().map(|&i| constraint_of(rows[i])).collect();
                deps.retain(|&d| d != c);
                if bad {
                    push_dependent(&mut conflicting, c, deps);
                } else {
                    push_dependent(&mut redundant, c, deps);
                }
            }
            (d.rank, d.dof, d.null_rows)
        } else {
            (0, 0, None)
        };
        if opts.diagnose {
            match &null_rows {
                Some(nr) if dof > 0 => {
                    let scale = nr.iter().map(|row| row_norm(row)).fold(0.0f64, f64::max).max(1.0);
                    for (j, &v) in vars.iter().enumerate() {
                        determined[v] = row_norm(&nr[j]) <= 1e-8 * scale;
                    }
                }
                Some(_) => vars.iter().for_each(|&v| determined[v] = true),
                None => {}
            }
        }
        total_dof += dof;
        comp_dof.push(dof);
        null_by_comp.push(null_rows);
        let mut constraints: Vec<ConstraintId> = rows.iter().map(|&i| constraint_of(i)).collect();
        constraints.dedup();
        components.push(ComponentReport {
            vars: vars.iter().map(|&v| VarId(v as u32)).collect(),
            constraints,
            rank,
            dof,
            converged,
            iterations: res.iterations,
            residual_norm: norm2(&r).sqrt(),
            method: res.method,
            blocks: *blocks,
        });
    }
    // Rows that read only fixed values: redundant if satisfied, conflicting if not.
    problem.residuals(&orphan, &x, &mut r);
    for (i, &row) in orphan.iter().enumerate() {
        let c = constraint_of(row);
        if r[i].abs() > opts.tolerance {
            all_converged = false;
            push_dependent(&mut conflicting, c, Vec::new());
        } else if opts.diagnose {
            push_dependent(&mut redundant, c, Vec::new());
        }
    }
    // A constraint reported as conflicting is not also redundant.
    redundant.retain(|d| !conflicting.iter().any(|c| c.constraint == d.constraint));
    redundant.sort_by_key(|d| d.constraint);
    conflicting.sort_by_key(|d| d.constraint);

    let entity_dof = if opts.diagnose {
        (0..sys.entities.len())
            .map(|e| {
                let id = EntityId(e as u32);
                let mut vs: Vec<usize> = sys.entity_vars(id).iter().map(|v| v.index()).filter(|&v| is_free[v]).collect();
                vs.sort_unstable();
                vs.dedup();
                let mut by_comp: Vec<(usize, usize)> = vs.iter().map(|&v| var_comp[v]).collect();
                by_comp.sort_unstable();
                let mut total = Some(0usize);
                let mut i = 0;
                while i < by_comp.len() {
                    let comp = by_comp[i].0;
                    let mut j = i;
                    while j < by_comp.len() && by_comp[j].0 == comp {
                        j += 1;
                    }
                    let add = if comp_dof[comp] == 0 {
                        Some(0)
                    } else {
                        null_by_comp[comp].as_ref().map(|nr| {
                            let sub: Vec<Vec<f64>> = by_comp[i..j].iter().map(|&(_, col)| nr[col].clone()).collect();
                            dense_rank(&sub, 1e-8)
                        })
                    };
                    total = total.zip(add).map(|(a, b)| a + b);
                    i = j;
                }
                EntityDof { entity: id, dof: total }
            })
            .collect()
    } else {
        Vec::new()
    };

    let all: Vec<usize> = (0..problem.rows.len()).collect();
    problem.residuals(&all, &x, &mut r);
    let residual_norm = norm2(&r).sqrt();
    let max_residual = max_abs(&r);

    let measurements = compiled
        .iter()
        .filter(|c| c.reference)
        .map(|c| Measurement { constraint: c.id, value: measure(c, &x) })
        .collect();

    let warm_start = opts.warm_start.as_ref().map(|_| {
        let max_deviation = cached.iter().map(|&(v, c)| (x[v] - c).abs()).fold(0.0f64, f64::max);
        let matches = cached.iter().all(|&(v, c)| (x[v] - c).abs() <= opts.warm_start_tolerance * (1.0 + c.abs()));
        WarmStartReport { applied: cached.len(), cache_satisfied, matches, max_deviation }
    });
    let drag = drag_request.map(|d| DragReport {
        distance_to_target: d
            .targets
            .iter()
            .filter(|(v, _)| v.index() < nv)
            .map(|&(v, t)| (x[v.index()] - t).powi(2))
            .sum::<f64>()
            .sqrt(),
    });

    let status = if !conflicting.is_empty() {
        Status::Conflicting
    } else if !all_converged {
        Status::NotConverged
    } else if !redundant.is_empty() {
        Status::OverDefined
    } else if total_dof == 0 {
        Status::WellDefined
    } else {
        Status::UnderDefined
    };

    SolveReport {
        values: x,
        status,
        converged: all_converged,
        residual_norm,
        max_residual,
        iterations: total_iterations,
        dof: total_dof,
        components,
        entity_dof,
        determined,
        redundant,
        conflicting,
        active_limits,
        measurements,
        warm_start,
        drag,
    }
}

fn row_norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

/// The measured value of a dimension (its residual with a zero target).
fn measure(c: &Compiled, x: &[f64]) -> f64 {
    let op = c.op.with_target(0.0);
    let xl: Vec<f64> = c.vars.iter().map(|v| x[v.index()]).collect();
    let mut out = vec![0.0; op.rows()];
    op.values(&xl, &mut out);
    out[0]
}

type Split = (Vec<(Vec<usize>, Vec<usize>)>, Vec<usize>);

/// Connected components over free variables: `(rows, global vars)` per component, plus rows
/// that read no free variable.
fn split(p: &Problem<'_>, nv: usize) -> Split {
    let mut dense = vec![NONE; nv];
    for (k, &v) in p.free.iter().enumerate() {
        dense[v] = k;
    }
    let row_vars: Vec<Vec<usize>> = p.row_vars.iter().map(|vs| vs.iter().map(|&v| dense[v]).collect()).collect();
    let (comps, orphan) = components(p.free.len(), &row_vars);
    let comps = comps.into_iter().map(|c| (c.rows, c.vars.into_iter().map(|v| p.free[v]).collect())).collect();
    (comps, orphan)
}
