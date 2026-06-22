use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatorSystem {
    pub dof_count: usize,
    pub constrained: Vec<bool>,
    #[serde(default)]
    pub stiffness_dense: Option<Vec<f64>>,
    pub stiffness_diag: Vec<f64>,
    pub stiffness_upper: Vec<f64>,
    pub mass_diag: Vec<f64>,
    pub damping_diag: Vec<f64>,
    pub rhs: Vec<f64>,
}

pub fn apply_k(system: &OperatorSystem, x: &[f64]) -> Vec<f64> {
    if let Some(dense) = dense_stiffness(system) {
        let mut y = vec![0.0; x.len()];
        for i in 0..x.len() {
            if system.constrained[i] {
                y[i] = x[i];
                continue;
            }
            y[i] = (0..x.len())
                .filter(|j| !system.constrained[*j])
                .map(|j| dense[i * system.dof_count + j] * x[j])
                .sum();
        }
        return y;
    }

    let mut y = vec![0.0; x.len()];
    for i in 0..x.len() {
        if system.constrained[i] {
            y[i] = x[i];
            continue;
        }

        let mut value = system.stiffness_diag[i] * x[i];

        if i > 0 && !system.constrained[i - 1] {
            value -= system.stiffness_upper[i - 1] * x[i - 1];
        }
        if i + 1 < x.len() && !system.constrained[i + 1] {
            value -= system.stiffness_upper[i] * x[i + 1];
        }

        y[i] = value;
    }
    y
}

pub fn apply_k_unconstrained(system: &OperatorSystem, x: &[f64]) -> Vec<f64> {
    if let Some(dense) = dense_stiffness(system) {
        let mut y = vec![0.0; x.len()];
        for i in 0..x.len() {
            y[i] = (0..x.len())
                .map(|j| dense[i * system.dof_count + j] * x[j])
                .sum();
        }
        return y;
    }

    let mut y = vec![0.0; x.len()];
    for i in 0..x.len() {
        let mut value = system.stiffness_diag[i] * x[i];

        if i > 0 {
            value -= system.stiffness_upper[i - 1] * x[i - 1];
        }
        if i + 1 < x.len() {
            value -= system.stiffness_upper[i] * x[i + 1];
        }

        y[i] = value;
    }
    y
}

pub fn dense_stiffness(system: &OperatorSystem) -> Option<&[f64]> {
    system
        .stiffness_dense
        .as_deref()
        .filter(|dense| dense.len() == system.dof_count.saturating_mul(system.dof_count))
}

pub fn apply_m(system: &OperatorSystem, x: &[f64]) -> Vec<f64> {
    apply_diag_with_constraints(&system.mass_diag, &system.constrained, x)
}

pub fn apply_c(system: &OperatorSystem, x: &[f64]) -> Vec<f64> {
    apply_diag_with_constraints(&system.damping_diag, &system.constrained, x)
}

fn apply_diag_with_constraints(diag: &[f64], constrained: &[bool], x: &[f64]) -> Vec<f64> {
    diag.iter()
        .zip(constrained.iter())
        .zip(x.iter())
        .map(
            |((&d, &is_constrained), &value)| {
                if is_constrained {
                    value
                } else {
                    d * value
                }
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operator_apply_respects_constraint_identity() {
        let system = OperatorSystem {
            dof_count: 3,
            constrained: vec![true, false, false],
            stiffness_dense: None,
            stiffness_diag: vec![100.0, 10.0, 5.0],
            stiffness_upper: vec![1.0, 0.5],
            mass_diag: vec![1.0, 2.0, 3.0],
            damping_diag: vec![0.1, 0.2, 0.3],
            rhs: vec![0.0, -100.0, 0.0],
        };
        let x = vec![4.0, 5.0, 6.0];

        assert_eq!(apply_k(&system, &x), vec![4.0, 47.0, 27.5]);
        assert_eq!(apply_k_unconstrained(&system, &x), vec![395.0, 43.0, 27.5]);
        assert_eq!(apply_m(&system, &x), vec![4.0, 10.0, 18.0]);
        let c = apply_c(&system, &x);
        assert!((c[0] - 4.0).abs() <= 1.0e-12);
        assert!((c[1] - 1.0).abs() <= 1.0e-12);
        assert!((c[2] - 1.8).abs() <= 1.0e-12);
    }

    #[test]
    fn operator_apply_uses_dense_stiffness_when_present() {
        let system = OperatorSystem {
            dof_count: 3,
            constrained: vec![true, false, false],
            stiffness_dense: Some(vec![100.0, 2.0, 3.0, 2.0, 10.0, 4.0, 3.0, 4.0, 5.0]),
            stiffness_diag: vec![100.0, 10.0, 5.0],
            stiffness_upper: vec![1.0, 0.5],
            mass_diag: vec![1.0, 2.0, 3.0],
            damping_diag: vec![0.1, 0.2, 0.3],
            rhs: vec![0.0, -100.0, 0.0],
        };
        let x = vec![4.0, 5.0, 6.0];

        assert_eq!(apply_k(&system, &x), vec![4.0, 74.0, 50.0]);
        assert_eq!(apply_k_unconstrained(&system, &x), vec![428.0, 82.0, 62.0]);
    }
}
