use crate::operator::OperatorSystem;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpdPreconditionerKind {
    Jacobi,
    Ilu0,
}

impl SpdPreconditionerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SpdPreconditionerKind::Jacobi => "jacobi",
            SpdPreconditionerKind::Ilu0 => "ilu0",
        }
    }
}

pub trait SpdPreconditioner {
    fn kind(&self) -> SpdPreconditionerKind;
    fn apply(&self, residual: &[f64]) -> Vec<f64>;
}

pub struct JacobiPreconditioner {
    inv_diag: Vec<f64>,
    constrained: Vec<bool>,
}

impl JacobiPreconditioner {
    pub fn from_operator(operator: &OperatorSystem) -> Self {
        let inv_diag = operator
            .stiffness_diag
            .iter()
            .map(|value| 1.0 / value.abs().max(1.0e-12))
            .collect();

        Self {
            inv_diag,
            constrained: operator.constrained.clone(),
        }
    }
}

pub struct Ilu0TridiagonalPreconditioner {
    constrained: Vec<bool>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    u_diag: Vec<f64>,
}

impl Ilu0TridiagonalPreconditioner {
    pub fn from_operator(operator: &OperatorSystem) -> Self {
        let n = operator.dof_count;
        let constrained = operator.constrained.clone();

        let mut lower = vec![0.0; n.saturating_sub(1)];
        let mut upper = vec![0.0; n.saturating_sub(1)];
        for i in 0..n.saturating_sub(1) {
            if constrained[i] || constrained[i + 1] {
                continue;
            }
            let coupling = operator.stiffness_upper[i];
            lower[i] = -coupling;
            upper[i] = -coupling;
        }

        let mut u_diag = vec![1.0; n];
        if n > 0 {
            u_diag[0] = if constrained[0] {
                1.0
            } else {
                operator.stiffness_diag[0].max(1.0e-12)
            };
        }

        for i in 1..n {
            if constrained[i] {
                u_diag[i] = 1.0;
                continue;
            }
            let prev_u = u_diag[i - 1].abs().max(1.0e-12);
            let l = if constrained[i - 1] {
                0.0
            } else {
                lower[i - 1] / prev_u
            };
            let mut value = operator.stiffness_diag[i] - l * upper[i - 1];
            if value.abs() < 1.0e-12 {
                value = value.signum() * 1.0e-12;
                if value == 0.0 {
                    value = 1.0e-12;
                }
            }
            u_diag[i] = value;
        }

        Self {
            constrained,
            lower,
            upper,
            u_diag,
        }
    }
}

impl SpdPreconditioner for Ilu0TridiagonalPreconditioner {
    fn kind(&self) -> SpdPreconditionerKind {
        SpdPreconditionerKind::Ilu0
    }

    fn apply(&self, residual: &[f64]) -> Vec<f64> {
        let n = residual.len();
        let mut y = vec![0.0; n];
        for i in 0..n {
            if self.constrained[i] {
                y[i] = residual[i];
                continue;
            }
            let mut value = residual[i];
            if i > 0 && !self.constrained[i - 1] {
                let prev_u = self.u_diag[i - 1].abs().max(1.0e-12);
                let l = self.lower[i - 1] / prev_u;
                value -= l * y[i - 1];
            }
            y[i] = value;
        }

        let mut z = vec![0.0; n];
        for i in (0..n).rev() {
            if self.constrained[i] {
                z[i] = y[i];
                continue;
            }
            let mut value = y[i];
            if i + 1 < n && !self.constrained[i + 1] {
                value -= self.upper[i] * z[i + 1];
            }
            z[i] = value / self.u_diag[i].abs().max(1.0e-12);
        }

        z
    }
}

impl SpdPreconditioner for JacobiPreconditioner {
    fn kind(&self) -> SpdPreconditionerKind {
        SpdPreconditionerKind::Jacobi
    }

    fn apply(&self, residual: &[f64]) -> Vec<f64> {
        residual
            .iter()
            .enumerate()
            .map(|(i, &ri)| {
                if self.constrained[i] {
                    ri
                } else {
                    ri * self.inv_diag[i]
                }
            })
            .collect()
    }
}

pub fn build_spd_preconditioner(
    operator: &OperatorSystem,
    kind: SpdPreconditionerKind,
) -> Box<dyn SpdPreconditioner> {
    match kind {
        SpdPreconditionerKind::Jacobi => Box::new(JacobiPreconditioner::from_operator(operator)),
        SpdPreconditionerKind::Ilu0 => {
            Box::new(Ilu0TridiagonalPreconditioner::from_operator(operator))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::assembly::{assemble_structural_system, StructuralFormulationRequirement};
    use crate::fixtures::{fixture_model, FixtureId};

    use super::*;

    fn model() -> runmat_analysis_core::AnalysisModel {
        fixture_model(FixtureId::CantileverLinearStatic)
    }

    #[test]
    fn ilu0_preconditioner_is_buildable_and_finite() {
        let summary = assemble_structural_system(
            &model(),
            None,
            None,
            None,
            StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        )
        .expect("test system should assemble");
        let pc = build_spd_preconditioner(&summary.operator, SpdPreconditionerKind::Ilu0);
        let z = pc.apply(&summary.operator.rhs);
        assert_eq!(z.len(), summary.dof_count);
        assert!(z.iter().all(|v| v.is_finite()));
    }
}
