//! The enforced constraint equations `Φ(x) = 0` and their Jacobian `Φ_x` at given variable
//! values, for a caller that integrates the system through time (a multibody dynamics study)
//! rather than solving it once.
//!
//! The rows are the ones [`crate::solve`] enforces: reference constraints (measurements) and
//! range limits (inequalities, active only when a bound is hit) are left out; the limits are
//! listed so the caller can say they are not enforced. Each row's Jacobian is exact (the same
//! forward-mode differentiation the solver uses).

use crate::residual::compile;
use crate::{ConstraintId, System};

/// `Φ(x)` and `Φ_x` over every variable of the system.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConstraintRows {
    /// The constraint each row belongs to (a mate has several rows).
    pub owners: Vec<ConstraintId>,
    /// `Φ(x)`, one entry per row.
    pub residual: Vec<f64>,
    /// `Φ_x`, row-major, `owners.len() × var_count`.
    pub jacobian: Vec<f64>,
    pub var_count: usize,
    /// Range-limited constraints, which these rows leave out.
    pub limits: Vec<ConstraintId>,
}

impl ConstraintRows {
    pub fn rows(&self) -> usize {
        self.owners.len()
    }

    /// Entry `(row, var)` of `Φ_x`.
    pub fn at(&self, row: usize, var: usize) -> f64 {
        self.jacobian[row * self.var_count + var]
    }
}

/// The enforced rows of `sys` at `values` (one per variable, as [`System::vars`] orders them).
pub fn constraint_rows(sys: &System, values: &[f64]) -> ConstraintRows {
    let nv = sys.var_count();
    assert_eq!(values.len(), nv, "one value per variable");
    let mut out = ConstraintRows {
        var_count: nv,
        ..Default::default()
    };
    for c in compile(sys, values) {
        if c.reference {
            continue;
        }
        if c.range.is_some() {
            out.limits.push(c.id);
            continue;
        }
        let rows = c.op.rows();
        let x: Vec<f64> = c.vars.iter().map(|v| values[v.index()]).collect();
        let mut vals = vec![0.0; rows];
        let mut jac = vec![0.0; rows * x.len()];
        c.op.jacobian(&x, &mut vals, &mut jac);
        for r in 0..rows {
            out.owners.push(c.id);
            out.residual.push(vals[r]);
            let mut row = vec![0.0; nv];
            // A variable read twice by one op adds both partial derivatives.
            for (k, v) in c.vars.iter().enumerate() {
                row[v.index()] += jac[r * x.len() + k];
            }
            out.jacobian.extend(row);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Constraint, Frame, Mate};

    /// A pendulum: a body hinged to ground. The rows' Jacobian matches central differences of
    /// the residual.
    #[test]
    fn hinge_rows_match_finite_differences() {
        let mut s = System::new();
        let g = s.ground(Frame::IDENTITY);
        let b = s.body(Frame::axis_angle([0.3, -0.1, 0.05], [0.2, 1.0, 0.1], 0.4));
        let ca = s.connector(g, Frame::IDENTITY).unwrap();
        let cb = s.connector(b, Frame::at([0.0, 0.0, 0.5])).unwrap();
        s.constrain(Constraint::Mate(Mate::Revolute { a: ca, b: cb }))
            .unwrap();
        let x: Vec<f64> = s.vars().iter().map(|v| v.value).collect();
        let r = constraint_rows(&s, &x);
        // Quaternion norm for each of the two bodies, five hinge rows.
        assert_eq!(r.rows(), 7);
        let h = 1e-6;
        for v in 0..r.var_count {
            let mut up = x.clone();
            let mut down = x.clone();
            up[v] += h;
            down[v] -= h;
            let (ru, rd) = (constraint_rows(&s, &up), constraint_rows(&s, &down));
            for row in 0..r.rows() {
                let fd = (ru.residual[row] - rd.residual[row]) / (2.0 * h);
                assert!((fd - r.at(row, v)).abs() < 1e-6, "row {row} var {v}: {fd} vs {}", r.at(row, v));
            }
        }
    }
}
