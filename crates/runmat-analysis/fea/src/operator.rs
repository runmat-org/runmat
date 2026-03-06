use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatorSystem {
    pub dof_count: usize,
    pub constrained: Vec<bool>,
    pub stiffness_diag: Vec<f64>,
    pub mass_diag: Vec<f64>,
    pub damping_diag: Vec<f64>,
    pub rhs: Vec<f64>,
}

pub fn apply_k(system: &OperatorSystem, x: &[f64]) -> Vec<f64> {
    apply_diag_with_constraints(&system.stiffness_diag, &system.constrained, x)
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
            stiffness_diag: vec![100.0, 10.0, 5.0],
            mass_diag: vec![1.0, 2.0, 3.0],
            damping_diag: vec![0.1, 0.2, 0.3],
            rhs: vec![0.0, -100.0, 0.0],
        };
        let x = vec![4.0, 5.0, 6.0];

        assert_eq!(apply_k(&system, &x), vec![4.0, 50.0, 30.0]);
        assert_eq!(apply_m(&system, &x), vec![4.0, 10.0, 18.0]);
        let c = apply_c(&system, &x);
        assert!((c[0] - 4.0).abs() <= 1.0e-12);
        assert!((c[1] - 1.0).abs() <= 1.0e-12);
        assert!((c[2] - 1.8).abs() <= 1.0e-12);
    }
}
