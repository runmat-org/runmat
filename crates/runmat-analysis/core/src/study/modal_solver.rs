//! The eigensolver of a structural modal extraction and the step's default algebraic measure.
//!
//! `StructuralModalBasisSettings` also lists inertia residual vectors: for each global
//! direction, the static response to a unit acceleration load in that direction with the
//! retained modes removed.

use super::{
    accuracy::AccuracyPlan,
    sequence::{ProcedureSettings, ProcedureStep},
    AnalysisProcedure, PhysicsFamily,
};
use serde::{Deserialize, Serialize};

/// The eigensolver of a structural modal extraction (`StructuralModalBasisSettings::solver_method`;
/// `Auto` lets the resource plan choose).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalSolverMethod {
    /// The resource plan chooses: the shift-invert factorization when its plan fits the
    /// admitted memory, otherwise preconditioned LOBPCG when that plan fits.
    #[default]
    Auto,
    /// Shift-invert block Lanczos on a sparse Cholesky factor. Refused when its plan does not
    /// fit the admitted memory.
    Factorization,
    /// Block LOBPCG preconditioned by multigrid, with no factor. Refused when its plan does not
    /// fit the admitted memory.
    Iterative,
}

impl ModalSolverMethod {
    /// The `.fea` spelling.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Factorization => "factorization",
            Self::Iterative => "iterative",
        }
    }
}

impl ProcedureStep {
    /// The baseline accuracy plan of a new step, on the backward error when its producer
    /// publishes one.
    pub(super) fn default_accuracy(
        step_id: &str,
        family: PhysicsFamily,
        procedure: AnalysisProcedure,
    ) -> AccuracyPlan {
        let mut accuracy = AccuracyPlan::baseline(format!("{step_id}.accuracy"));
        if Self::qualifies_on_backward_error(family, procedure) {
            accuracy.set_algebraic_measure(crate::AlgebraicAccuracyMeasure::BackwardError);
        }
        accuracy
    }

    /// The eigensolver an extraction step requires; every other step is `Auto`.
    pub fn modal_solver_method(&self) -> ModalSolverMethod {
        match &self.settings {
            ProcedureSettings::StructuralModalBasis(settings) => settings.solver_method,
            _ => ModalSolverMethod::Auto,
        }
    }

    /// Whether the step's producer publishes the normwise backward error, so its default
    /// algebraic criterion is the backward error. Structural modal extraction publishes it per
    /// mode; the other producers still publish the relative residual their receipts document.
    pub const fn qualifies_on_backward_error(
        family: PhysicsFamily,
        procedure: AnalysisProcedure,
    ) -> bool {
        matches!(family, PhysicsFamily::Structural)
            && matches!(
                procedure,
                AnalysisProcedure::ModalExtraction | AnalysisProcedure::PrestressedModalExtraction
            )
    }
}
