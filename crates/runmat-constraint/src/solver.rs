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
//! Nielsen's damping update. [`Algorithm::DogLeg`] replaces both with Powell's dog leg trust
//! region. The weight `W` is 1 except for dragged variables, which get a small weight so the
//! step moves them least.
//!
//! # Scaling
//!
//! With [`SolveOptions::scaling`], every sub-problem gets a characteristic length `L` (the
//! extent of the coordinates and sizes its constraints read). Length residual rows are divided
//! by `L` and length variables carry the column weight `L²` (they are stepped in units of
//! `L`), so angles, ratios and quaternion components weigh the same as lengths of model size.
//! Convergence is always judged on the unscaled residuals.
//!
//! # Limits
//!
//! Range targets are inequality constraints handled by a primal active set. A pass solves
//! with the active bounds as equalities; then the most violated inactive bound per component
//! (relative to its range width) is activated, or, when none is violated, every active bound
//! whose Lagrange multiplier has the wrong sign is released and the solve restarts from the
//! starting point. The multipliers come from the least-motion stationarity condition
//! `W⁻¹(x - x₀) = Jᵀλ`, solved as `J W Jᵀ λ = J (x - x₀)`.

use crate::decompose::{components, dulmage_mendelsohn};
use crate::ids::{ConstraintId, EntityId, VarId};
use crate::linalg::{dense_rank, Ldlt, RowQr, SparseRows};
use crate::report::*;
use crate::residual::{compile, Compiled, Op, Unit};
use crate::system::{EntityKind, System, VarFlag, VarRole};

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
            EntityKind::Point2 { x, y } => DragRequest {
                targets: vec![(x, to[0]), (y, to[1])],
            },
            _ => panic!("{point} is not a point"),
        }
    }

    /// Drag a body's origin toward `to` (orientation follows the constraints).
    pub fn body_position(sys: &System, body: EntityId, to: [f64; 3]) -> DragRequest {
        match sys.entity(body).kind {
            EntityKind::Body { position, .. } => DragRequest {
                targets: (0..3).map(|i| (position[i], to[i])).collect(),
            },
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

/// How a variable enters the characteristic length of a sub-problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VarClass {
    /// A coordinate along axis 0, 1 or 2 (the extent is its range).
    Coord(u8),
    /// A size (radius, semi-axis): the extent is its magnitude.
    Size,
    /// Dimensionless (angles, quaternions, parameters, weights, user variables).
    Ratio,
}

struct Problem<'a> {
    compiled: &'a [Compiled],
    ops: &'a [Option<Op>],
    rows: Vec<Row>,
    /// Free variables (global indices) each row reads, ascending.
    row_vars: Vec<Vec<usize>>,
    free: Vec<usize>,
    class: Vec<VarClass>,
}

struct SubResult {
    converged: bool,
    iterations: usize,
    method: Method,
}

fn var_class(role: VarRole) -> VarClass {
    match role {
        VarRole::X => VarClass::Coord(0),
        VarRole::Y => VarClass::Coord(1),
        VarRole::Z => VarClass::Coord(2),
        VarRole::Radius | VarRole::SemiMajor | VarRole::SemiMinor => VarClass::Size,
        _ => VarClass::Ratio,
    }
}

/// Weighted norm `sqrt(Σ v² / w)` (the step metric `‖·‖_{W⁻¹}`).
fn wnorm(v: &[f64], w: &[f64]) -> f64 {
    v.iter().zip(w).map(|(a, w)| a * a / w).sum::<f64>().sqrt()
}

impl<'a> Problem<'a> {
    fn new(
        sys: &System,
        compiled: &'a [Compiled],
        ops: &'a [Option<Op>],
        is_free: &[bool],
        x: &[f64],
        tol: f64,
    ) -> Self {
        let mut rows = Vec::new();
        let mut row_vars = Vec::new();
        for (ci, c) in compiled.iter().enumerate() {
            let Some(op) = &ops[ci] else { continue };
            let mut fv: Vec<usize> = c
                .vars
                .iter()
                .map(|v| v.index())
                .filter(|&v| is_free[v])
                .collect();
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
        let class = sys.vars.iter().map(|v| var_class(v.role)).collect();
        Problem {
            compiled,
            ops,
            rows,
            row_vars,
            free,
            class,
        }
    }

    /// Row scales and column weight factors for a sub-problem (`1` without scaling).
    fn scaling(
        &self,
        rows: &[usize],
        vars: &[usize],
        x: &[f64],
        opts: &SolveOptions,
    ) -> (Vec<f64>, Vec<f64>) {
        if !opts.scaling {
            return (vec![1.0; rows.len()], vec![1.0; vars.len()]);
        }
        // Extent of every coordinate and size the rows read (fixed values included).
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut size = 0.0f64;
        let mut last = usize::MAX;
        for &ri in rows {
            let c = self.rows[ri].c;
            if c == last {
                continue;
            }
            last = c;
            for v in &self.compiled[c].vars {
                let val = x[v.index()];
                match self.class[v.index()] {
                    VarClass::Coord(k) => {
                        lo[k as usize] = lo[k as usize].min(val);
                        hi[k as usize] = hi[k as usize].max(val);
                    }
                    VarClass::Size => size = size.max(val.abs()),
                    VarClass::Ratio => {}
                }
            }
        }
        let mut l = size;
        for k in 0..3 {
            if hi[k] >= lo[k] {
                l = l.max(hi[k] - lo[k]);
            }
        }
        if !(l.is_finite() && l > 1e-12) {
            l = 1.0;
        }
        let rs = rows
            .iter()
            .map(|&ri| {
                let Row { c, r } = self.rows[ri];
                match self.compiled[c].units[r] {
                    Unit::Length => 1.0 / l,
                    Unit::Ratio => 1.0,
                }
            })
            .collect();
        let cw = vars
            .iter()
            .map(|&v| match self.class[v] {
                VarClass::Coord(_) | VarClass::Size => l * l,
                VarClass::Ratio => 1.0,
            })
            .collect();
        (rs, cw)
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
    fn jacobian(
        &self,
        rows: &[usize],
        x: &[f64],
        col_of: &[usize],
        ncols: usize,
    ) -> (Vec<f64>, SparseRows) {
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

    /// Solves a row/variable subset with the selected algorithm.
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
            return SubResult {
                converged: true,
                iterations: 0,
                method: Method::None,
            };
        }
        if vars.is_empty() {
            return SubResult {
                converged: false,
                iterations: 0,
                method: Method::None,
            };
        }
        for (k, &v) in vars.iter().enumerate() {
            col_of[v] = k;
        }
        let (rs, cw) = self.scaling(rows, vars, x, opts);
        let w: Vec<f64> = vars.iter().zip(&cw).map(|(&v, c)| weights[v] * c).collect();
        let sub = Sub {
            p: self,
            rows,
            vars,
            w: &w,
            rs: &rs,
            col_of,
            opts,
        };
        let res = match opts.algorithm {
            Algorithm::NewtonLevenbergMarquardt => sub.newton_lm(x, r),
            Algorithm::DogLeg => sub.dogleg(x, r),
        };
        for &v in vars {
            col_of[v] = NONE;
        }
        res
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
                        return (
                            SubResult {
                                converged: true,
                                iterations: total,
                                method,
                            },
                            blocks.len(),
                        );
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
    fn structural_pattern(
        &self,
        rows: &[usize],
        vars: &[usize],
        x: &[f64],
        col_of: &mut [usize],
    ) -> Vec<Vec<usize>> {
        let mut xp = x.to_vec();
        for &v in vars {
            let h =
                ((v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f64 / (1u64 << 24) as f64;
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

/// One sub-problem in scaled form: `rs` scales the rows, `w` is the column step weight.
struct Sub<'p, 'a, 'c> {
    p: &'p Problem<'a>,
    rows: &'p [usize],
    vars: &'p [usize],
    w: &'p [f64],
    rs: &'p [f64],
    col_of: &'c [usize],
    opts: &'p SolveOptions,
}

impl Sub<'_, '_, '_> {
    fn scaled(&self, r: &[f64]) -> Vec<f64> {
        r.iter().zip(self.rs).map(|(a, s)| a * s).collect()
    }

    /// The row-scaled Jacobian.
    fn jac(&self, x: &[f64]) -> SparseRows {
        let (_, mut jac) = self.p.jacobian(self.rows, x, self.col_of, self.vars.len());
        jac.scale_rows(self.rs);
        jac
    }

    fn diag_max(&self, jac: &SparseRows) -> f64 {
        let mut diag_max = 0.0f64;
        for i in 0..jac.nrows() {
            let (c, v) = jac.row(i);
            let d: f64 = c.iter().zip(v).map(|(&c, &v)| v * v * self.w[c]).sum();
            diag_max = diag_max.max(d);
        }
        diag_max
    }

    /// Newton with line search, falling back to Levenberg–Marquardt. `r` holds the unscaled
    /// residuals at `x`.
    fn newton_lm(&self, x: &mut [f64], mut r: Vec<f64>) -> SubResult {
        let (rows, vars, w, opts) = (self.rows, self.vars, self.w, self.opts);
        let n = vars.len();
        let mut rsc = self.scaled(&r);
        let mut f = norm2(&rsc);
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
            let jac = self.jac(x);
            let diag_max = self.diag_max(&jac);
            if diag_max == 0.0 {
                break;
            }
            let damping = if lm { mu } else { 1e-12 * diag_max };
            let delta = step(&jac, &rsc, w, damping, diag_max);
            let mut rsc_try;
            if !lm {
                let mut alpha = 1.0;
                let mut accepted = false;
                rsc_try = Vec::new();
                for _ in 0..12 {
                    for (k, &v) in vars.iter().enumerate() {
                        trial[v] = x[v] + alpha * delta[k];
                    }
                    self.p.residuals(rows, &trial, &mut r_try);
                    rsc_try = self.scaled(&r_try);
                    let ft = norm2(&rsc_try);
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
                let predicted = f - rsc
                    .iter()
                    .zip(&jd)
                    .map(|(a, b)| (a + b) * (a + b))
                    .sum::<f64>();
                for (k, &v) in vars.iter().enumerate() {
                    trial[v] = x[v] + delta[k];
                }
                self.p.residuals(rows, &trial, &mut r_try);
                rsc_try = self.scaled(&r_try);
                let ft = norm2(&rsc_try);
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
                    jac.mul_t(&rsc, &mut g);
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
            rsc = rsc_try;
            f = norm2(&rsc);
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
        SubResult {
            converged,
            iterations,
            method,
        }
    }

    /// Powell's dog leg. The Gauss–Newton step is the minimum-norm step of [`step`]; the
    /// Cauchy step minimises the model along `-W g`; distances are in the `W⁻¹` metric.
    fn dogleg(&self, x: &mut [f64], mut r: Vec<f64>) -> SubResult {
        let (rows, vars, w, opts) = (self.rows, self.vars, self.w, self.opts);
        let n = vars.len();
        let mut rsc = self.scaled(&r);
        let mut f = norm2(&rsc);
        let mut radius: Option<f64> = None;
        let mut converged = false;
        let mut iterations = 0;
        let mut trial = x.to_vec();
        let mut r_try = Vec::new();
        let mut jd = vec![0.0; rows.len()];
        let mut g = vec![0.0; n];
        let mut need_jac = true;
        let mut jac = SparseRows::new(n);
        let mut gn = Vec::new();
        let mut sd = Vec::new();
        while iterations < opts.max_iterations {
            iterations += 1;
            if need_jac {
                jac = self.jac(x);
                let diag_max = self.diag_max(&jac);
                if diag_max == 0.0 {
                    break;
                }
                jac.mul_t(&rsc, &mut g);
                if max_abs(&g) <= 1e-14 * diag_max.sqrt() * (1.0 + f.sqrt()) {
                    break; // stationary: a least-squares minimum with a nonzero residual
                }
                gn = step(&jac, &rsc, w, 1e-12 * diag_max, diag_max);
                let wg: Vec<f64> = g.iter().zip(w).map(|(a, b)| a * b).collect();
                jac.mul(&wg, &mut jd);
                let num: f64 = g.iter().zip(&wg).map(|(a, b)| a * b).sum();
                let den = norm2(&jd);
                let alpha = if den > 0.0 { num / den } else { 0.0 };
                sd = wg.iter().map(|v| -alpha * v).collect();
                need_jac = false;
            }
            let ngn = wnorm(&gn, w);
            let tr = *radius.get_or_insert(ngn.max(1e-300));
            let delta: Vec<f64> = if ngn <= tr {
                gn.clone()
            } else {
                let nsd = wnorm(&sd, w);
                if nsd >= tr {
                    sd.iter().map(|v| v * tr / nsd).collect()
                } else {
                    // δ = sd + β (gn − sd) with ‖δ‖ = Δ.
                    let d: Vec<f64> = gn.iter().zip(&sd).map(|(a, b)| a - b).collect();
                    let a: f64 = d.iter().zip(w).map(|(v, w)| v * v / w).sum();
                    let b: f64 = 2.0
                        * sd.iter()
                            .zip(&d)
                            .zip(w)
                            .map(|((s, d), w)| s * d / w)
                            .sum::<f64>();
                    let c = nsd * nsd - tr * tr;
                    let beta = if a > 0.0 {
                        (-b + (b * b - 4.0 * a * c).max(0.0).sqrt()) / (2.0 * a)
                    } else {
                        0.0
                    };
                    sd.iter().zip(&d).map(|(s, d)| s + beta * d).collect()
                }
            };
            let nd = wnorm(&delta, w);
            jac.mul(&delta, &mut jd);
            let predicted = f - rsc
                .iter()
                .zip(&jd)
                .map(|(a, b)| (a + b) * (a + b))
                .sum::<f64>();
            for (k, &v) in vars.iter().enumerate() {
                trial[v] = x[v] + delta[k];
            }
            self.p.residuals(rows, &trial, &mut r_try);
            let rsc_try = self.scaled(&r_try);
            let ft = norm2(&rsc_try);
            let actual = f - ft;
            let rho = if predicted > 0.0 && ft.is_finite() {
                actual / predicted
            } else {
                -1.0
            };
            let tr = if rho > 0.75 {
                tr.max(3.0 * nd)
            } else if rho < 0.25 {
                0.25 * nd.min(tr)
            } else {
                tr
            };
            radius = Some(tr);
            if rho > 0.0 && actual > 0.0 {
                let mut moved = 0.0f64;
                for &v in vars {
                    moved = moved.max((trial[v] - x[v]).abs() / (1.0 + x[v].abs()));
                    x[v] = trial[v];
                }
                std::mem::swap(&mut r, &mut r_try);
                rsc = rsc_try;
                f = ft;
                need_jac = true;
                if max_abs(&r) <= opts.tolerance {
                    converged = true;
                    break;
                }
                if moved <= 1e-15 {
                    break;
                }
            } else {
                let xs: f64 = vars.iter().map(|&v| x[v].abs()).fold(0.0, f64::max);
                if tr <= 1e-15 * (1.0 + xs) {
                    break;
                }
            }
        }
        SubResult {
            converged,
            iterations,
            method: Method::DogLeg,
        }
    }
}

fn merge_method(a: Method, b: Method) -> Method {
    match (a, b) {
        (Method::DogLeg, _) | (_, Method::DogLeg) => Method::DogLeg,
        (Method::LevenbergMarquardt, _) | (_, Method::LevenbergMarquardt) => {
            Method::LevenbergMarquardt
        }
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
    r.iter().fold(0.0f64, |m, v| {
        if v.is_nan() {
            f64::INFINITY
        } else {
            m.max(v.abs())
        }
    })
}

/// Component diagnostics.
/// Sparse null-space rows per component variable: `(basis vector, value)`.
type NullRows = Vec<Vec<(usize, f64)>>;

struct Diagnosis {
    rank: usize,
    dof: usize,
    /// `(row, rows it depends on, conflicting)` with indices into the component's row list.
    dependent: Vec<(usize, Vec<usize>, bool)>,
    /// Sparse null-space rows per component variable, `(basis vector, value)` (`None` when
    /// skipped for size).
    null_rows: Option<NullRows>,
    /// The row-normalised Jacobian (for per-entity DOF of large entities).
    jn: SparseRows,
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
    for (i, &norm) in norms.iter().enumerate() {
        let (c, v) = jac.row(i);
        if norm > 1e-14 {
            entries.extend(c.iter().zip(v).map(|(&c, &v)| (c, v / norm)));
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
        let f = if indep.is_empty() {
            None
        } else {
            Some(Ldlt::factor(&gram, 1e-14))
        };
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
        let rn: Vec<f64> = (0..m)
            .map(|i| {
                if norms[i] > 1e-14 {
                    r[i] / norms[i]
                } else {
                    r[i]
                }
            })
            .collect();
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
    } else {
        qr.null_rows_sparse(opts.max_null_space_entries)
    };
    Diagnosis {
        rank,
        dof,
        dependent,
        null_rows,
        jn,
    }
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

    // A fasten more than a quarter turn from its start begins where it lands (a drag follows
    // its own continuation from the current pose).
    if drag_request.is_none() {
        crate::seed::seed_fastens(sys, &mut x, &is_free);
    }

    let compiled = compile(sys, &x);
    let mut ops: Vec<Option<Op>> = compiled
        .iter()
        .map(|c| (!c.reference && c.range.is_none()).then(|| c.op.clone()))
        .collect();
    let mut active_limits: Vec<(ConstraintId, f64)> = Vec::new();
    let mut released_limits: Vec<ConstraintId> = Vec::new();
    let mut releases = vec![0usize; compiled.len()];
    let limit_count = compiled
        .iter()
        .filter(|c| c.range.is_some() && !c.reference)
        .count();
    let max_passes = opts.max_limit_passes.min(8 + 4 * limit_count).max(1);
    let x_start = x.clone();

    let mut col_of = vec![NONE; nv];
    let mut cache_satisfied = false;
    let mut solved: Vec<(Vec<usize>, Vec<usize>, SubResult, usize)>;
    // (first variable of the component, clusters, constructed points).
    let mut construction: Vec<(usize, usize, usize)> = Vec::new();
    let mut pass = 0;
    loop {
        let problem = Problem::new(sys, &compiled, &ops, &is_free, &x, opts.tolerance);
        if pass == 0 && !cached.is_empty() {
            let all: Vec<usize> = (0..problem.rows.len()).collect();
            let mut r = Vec::new();
            problem.residuals(&all, &x, &mut r);
            cache_satisfied = max_abs(&r) <= opts.tolerance;
        }
        let (comps, _) = split(&problem, nv);
        solved = Vec::with_capacity(comps.len());
        let mut in_comp = vec![false; nv];
        for (rows, vars) in comps {
            // Constructive pre-pass on the first pass of a plain solve (a drag follows its
            // own continuation; a satisfied component needs nothing).
            if pass == 0 && opts.clusters && drag_request.is_none() && !rows.is_empty() {
                let mut r = Vec::new();
                problem.residuals(&rows, &x, &mut r);
                if max_abs(&r) > opts.tolerance {
                    vars.iter().for_each(|&v| in_comp[v] = true);
                    let mut cons: Vec<usize> = rows.iter().map(|&i| problem.rows[i].c).collect();
                    cons.dedup();
                    let built = crate::cluster::construct(
                        sys, &compiled, &ops, &cons, &is_free, &in_comp, &x,
                    );
                    vars.iter().for_each(|&v| in_comp[v] = false);
                    if !built.values.is_empty() {
                        let before = norm2(&r);
                        let saved: Vec<(usize, f64)> =
                            built.values.iter().map(|&(v, _)| (v, x[v])).collect();
                        for &(v, val) in &built.values {
                            x[v] = val;
                        }
                        problem.residuals(&rows, &x, &mut r);
                        if norm2(&r) <= before {
                            construction.push((vars[0], built.clusters, built.points));
                        } else {
                            for (v, val) in saved {
                                x[v] = val;
                            }
                        }
                    }
                }
            }
            let (mut res, blocks) =
                problem.solve_component(&rows, &vars, &mut x, &weights, &mut col_of, opts);
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
                    let (again, _) =
                        problem.solve_component(&rows, &vars, &mut x, &weights, &mut col_of, opts);
                    res.iterations += again.iterations;
                    res.method = merge_method(res.method, again.method);
                    res.converged = again.converged;
                    let change = vars
                        .iter()
                        .zip(&before)
                        .map(|(&v, b)| (x[v] - b).abs())
                        .fold(0.0f64, f64::max);
                    if change <= opts.tolerance {
                        break;
                    }
                }
            }
            solved.push((rows, vars, res, blocks));
        }
        pass += 1;
        if pass >= max_passes || limit_count == 0 {
            break;
        }
        // Range limits, a primal active set. First activate the most violated inactive bound
        // of each component (relative to its range width).
        let mut changed = false;
        let mut best: Vec<Option<(f64, usize, f64)>> = vec![None; solved.len()];
        let comp_of_var = |v: usize| solved.iter().position(|s| s.1.binary_search(&v).is_ok());
        for (ci, c) in compiled.iter().enumerate() {
            let Some((lo, hi)) = c.range else { continue };
            if c.reference || ops[ci].is_some() {
                continue;
            }
            let v = measure(c, &x);
            let (bound, excess) = if v < lo - opts.tolerance {
                (lo, lo - v)
            } else if v > hi + opts.tolerance {
                (hi, v - hi)
            } else {
                continue;
            };
            let width = hi - lo;
            let rel = if width > 0.0 {
                excess / width
            } else {
                f64::INFINITY
            };
            let comp = c
                .vars
                .iter()
                .find_map(|v| is_free[v.index()].then(|| comp_of_var(v.index())).flatten());
            // A limit on fixed geometry only is its own "component".
            let slot = comp.unwrap_or(usize::MAX);
            if slot == usize::MAX {
                ops[ci] = Some(c.op.with_target(bound));
                active_limits.push((c.id, bound));
                changed = true;
                continue;
            }
            if best[slot].is_none_or(|b| rel > b.0) {
                best[slot] = Some((rel, ci, bound));
            }
        }
        for (_, ci, bound) in best.into_iter().flatten() {
            ops[ci] = Some(compiled[ci].op.with_target(bound));
            active_limits.push((compiled[ci].id, bound));
            changed = true;
        }
        if !changed && !active_limits.is_empty() {
            // Then release every active bound whose multiplier says it holds the solution back.
            let problem = Problem::new(sys, &compiled, &ops, &is_free, &x, opts.tolerance);
            let release = wrong_sign_limits(
                &problem,
                &compiled,
                &active_limits,
                &x,
                &x_start,
                &weights,
                &mut col_of,
                opts,
            );
            for ci in release {
                if releases[ci] >= 2 {
                    continue; // anti-cycling: a bound is released at most twice
                }
                releases[ci] += 1;
                ops[ci] = None;
                active_limits.retain(|(id, _)| id.index() != ci);
                released_limits.push(compiled[ci].id);
                changed = true;
            }
            if changed {
                // Restart the free variables from the starting point: least motion.
                for v in 0..nv {
                    if is_free[v] {
                        x[v] = x_start[v];
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    released_limits.sort();
    released_limits.dedup();
    released_limits.retain(|c| !active_limits.iter().any(|(a, _)| a == c));

    let problem = Problem::new(sys, &compiled, &ops, &is_free, &x, opts.tolerance);
    let (comps, orphan) = split(&problem, nv);
    debug_assert_eq!(comps.len(), solved.len());

    let mut components = Vec::with_capacity(comps.len());
    let mut redundant = Vec::new();
    let mut conflicting = Vec::new();
    let mut determined = is_free.iter().map(|f| !f).collect::<Vec<bool>>();
    // Per free variable: (component index, local column).
    let mut var_comp = vec![(NONE, NONE); nv];
    let mut null_by_comp: Vec<Option<NullRows>> = Vec::with_capacity(comps.len());
    let mut jn_by_comp: Vec<Option<(SparseRows, usize)>> = Vec::with_capacity(comps.len());
    let mut comp_dof = Vec::with_capacity(comps.len());
    let mut total_dof = 0;
    let mut all_converged = true;
    let mut total_iterations = 0;
    let mut r = Vec::new();

    let constraint_of = |row: usize| ConstraintId(problem.rows[row].c as u32);
    let push_dependent =
        |list: &mut Vec<DependentConstraint>, c: ConstraintId, deps: Vec<ConstraintId>| {
            if let Some(d) = list.iter_mut().find(|d| d.constraint == c) {
                d.depends_on.extend(deps);
                d.depends_on.sort();
                d.depends_on.dedup();
            } else {
                let mut deps = deps;
                deps.sort();
                deps.dedup();
                deps.retain(|&d| d != c);
                list.push(DependentConstraint {
                    constraint: c,
                    depends_on: deps,
                });
            }
        };

    for (k, ((rows, vars), (_, _, res, blocks))) in comps.iter().zip(&solved).enumerate() {
        problem.residuals(rows, &x, &mut r);
        let converged = max_abs(&r) <= opts.tolerance;
        all_converged &= converged;
        // Components only merge as limits activate: credit each construction to the final
        // component holding its first variable.
        let built = construction
            .iter()
            .filter(|c| vars.binary_search(&c.0).is_ok())
            .fold((0, 0), |a, c| (a.0 + c.1, a.1 + c.2));
        total_iterations += res.iterations;
        for (j, &v) in vars.iter().enumerate() {
            var_comp[v] = (k, j);
        }
        let mut jn = None;
        let (rank, dof, null_rows) = if opts.diagnose {
            let d = diagnose(&problem, rows, vars, &x, converged, &mut col_of, opts);
            for (row, deps, bad) in d.dependent {
                let c = constraint_of(rows[row]);
                let mut deps: Vec<ConstraintId> =
                    deps.iter().map(|&i| constraint_of(rows[i])).collect();
                deps.retain(|&d| d != c);
                if bad {
                    push_dependent(&mut conflicting, c, deps);
                } else {
                    push_dependent(&mut redundant, c, deps);
                }
            }
            jn = Some((d.jn, d.rank));
            (d.rank, d.dof, d.null_rows)
        } else {
            (0, 0, None)
        };
        if opts.diagnose {
            match &null_rows {
                Some(nr) if dof > 0 => {
                    let scale = nr
                        .iter()
                        .map(|row| sparse_norm(row))
                        .fold(0.0f64, f64::max)
                        .max(1.0);
                    for (j, &v) in vars.iter().enumerate() {
                        determined[v] = sparse_norm(&nr[j]) <= 1e-8 * scale;
                    }
                }
                Some(_) => vars.iter().for_each(|&v| determined[v] = true),
                None => {}
            }
        }
        total_dof += dof;
        comp_dof.push(dof);
        null_by_comp.push(null_rows);
        jn_by_comp.push(jn);
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
            clusters: built.0,
            constructed_points: built.1,
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
                let mut vs: Vec<usize> = sys
                    .entity_vars(id)
                    .iter()
                    .map(|v| v.index())
                    .filter(|&v| is_free[v])
                    .collect();
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
                    let cols: Vec<usize> = by_comp[i..j].iter().map(|&(_, col)| col).collect();
                    // A small entity takes the rank of its null-space rows; a large one (a
                    // spline with hundreds of control points) the exact identity
                    // dof(E) = |E| − rank(J) + rank(J without E's columns) on the sparse QR.
                    let large = cols.len().saturating_mul(comp_dof[comp]) > 40_000;
                    let add = if comp_dof[comp] == 0 {
                        Some(0)
                    } else {
                        match (&null_by_comp[comp], large) {
                            (Some(nr), false) => {
                                // Densify over the basis vectors these columns touch only.
                                let mut touched: Vec<usize> = cols
                                    .iter()
                                    .flat_map(|&col| nr[col].iter().map(|e| e.0))
                                    .collect();
                                touched.sort_unstable();
                                touched.dedup();
                                let sub: Vec<Vec<f64>> = cols
                                    .iter()
                                    .map(|&col| {
                                        let mut row = vec![0.0; touched.len()];
                                        for &(b, v) in &nr[col] {
                                            row[touched.binary_search(&b).expect("touched")] = v;
                                        }
                                        row
                                    })
                                    .collect();
                                Some(dense_rank(&sub, 1e-8))
                            }
                            (_, true) => jn_by_comp[comp].as_ref().map(|(jn, rank)| {
                                (cols.len() + rank_without(jn, &cols, opts.rank_tolerance))
                                    .saturating_sub(*rank)
                            }),
                            (None, false) => None,
                        }
                    };
                    total = total.zip(add).map(|(a, b)| a + b);
                    i = j;
                }
                EntityDof {
                    entity: id,
                    dof: total,
                }
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
        .map(|c| Measurement {
            constraint: c.id,
            value: measure(c, &x),
        })
        .collect();

    let warm_start = opts.warm_start.as_ref().map(|_| {
        let max_deviation = cached
            .iter()
            .map(|&(v, c)| (x[v] - c).abs())
            .fold(0.0f64, f64::max);
        let matches = cached
            .iter()
            .all(|&(v, c)| (x[v] - c).abs() <= opts.warm_start_tolerance * (1.0 + c.abs()));
        WarmStartReport {
            applied: cached.len(),
            cache_satisfied,
            matches,
            max_deviation,
        }
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
        released_limits,
        measurements,
        warm_start,
        drag,
    }
}

/// Active range limits whose least-motion multiplier has the wrong sign (the bound pulls the
/// solution instead of pushing it), as constraint indices.
#[allow(clippy::too_many_arguments)]
fn wrong_sign_limits(
    p: &Problem<'_>,
    compiled: &[Compiled],
    active: &[(ConstraintId, f64)],
    x: &[f64],
    x_start: &[f64],
    weights: &[f64],
    col_of: &mut [usize],
    opts: &SolveOptions,
) -> Vec<usize> {
    let (comps, _) = split(p, x.len());
    let mut out = Vec::new();
    for (rows, vars) in comps {
        let limit_rows: Vec<(usize, f64)> = rows
            .iter()
            .enumerate()
            .filter_map(|(k, &ri)| {
                let c = p.rows[ri].c;
                let (lo, hi) = compiled[c].range?;
                let &(_, bound) = active.iter().find(|(id, _)| id.index() == c)?;
                if lo == hi {
                    return None; // an equality in disguise: never released
                }
                Some((k, if bound == lo { 1.0 } else { -1.0 }))
            })
            .collect();
        if limit_rows.is_empty() {
            continue;
        }
        for (k, &v) in vars.iter().enumerate() {
            col_of[v] = k;
        }
        let (rs, cw) = p.scaling(&rows, &vars, x, opts);
        let w: Vec<f64> = vars.iter().zip(&cw).map(|(&v, c)| weights[v] * c).collect();
        let (_, mut jac) = p.jacobian(&rows, x, col_of, vars.len());
        for &v in &vars {
            col_of[v] = NONE;
        }
        jac.scale_rows(&rs);
        let d: Vec<f64> = vars.iter().map(|&v| x[v] - x_start[v]).collect();
        // J W Jᵀ λ = J (x − x₀).
        let mut b = vec![0.0; rows.len()];
        jac.mul(&d, &mut b);
        let diag_max = (0..jac.nrows())
            .map(|i| {
                let (c, v) = jac.row(i);
                c.iter().zip(v).map(|(&c, &v)| v * v * w[c]).sum::<f64>()
            })
            .fold(0.0f64, f64::max);
        if diag_max == 0.0 {
            continue;
        }
        let gram = jac.gram(&w, 1e-12 * diag_max);
        let f = Ldlt::factor(&gram, 1e-15 * diag_max);
        f.solve(&mut b);
        let scale = max_abs(&b).max(f64::MIN_POSITIVE);
        for (k, sense) in limit_rows {
            let nu = sense * b[k];
            if nu < -1e-6 * scale {
                out.push(p.rows[rows[k]].c);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// The rank of `jn` with the columns `drop` removed (sorted ascending not required).
fn rank_without(jn: &SparseRows, drop: &[usize], tol: f64) -> usize {
    let mut gone = vec![false; jn.ncols];
    for &c in drop {
        gone[c] = true;
    }
    let mut out = SparseRows::new(jn.ncols);
    let mut entries = Vec::new();
    for i in 0..jn.nrows() {
        let (c, v) = jn.row(i);
        entries.extend(
            c.iter()
                .zip(v)
                .filter(|(c, _)| !gone[**c])
                .map(|(&c, &v)| (c, v)),
        );
        out.push_row(&mut entries);
    }
    RowQr::factor(&out, tol).rank()
}

fn sparse_norm(v: &[(usize, f64)]) -> f64 {
    v.iter().map(|a| a.1 * a.1).sum::<f64>().sqrt()
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
    let row_vars: Vec<Vec<usize>> = p
        .row_vars
        .iter()
        .map(|vs| vs.iter().map(|&v| dense[v]).collect())
        .collect();
    let (comps, orphan) = components(p.free.len(), &row_vars);
    let comps = comps
        .into_iter()
        .map(|c| (c.rows, c.vars.into_iter().map(|v| p.free[v]).collect()))
        .collect();
    (comps, orphan)
}
