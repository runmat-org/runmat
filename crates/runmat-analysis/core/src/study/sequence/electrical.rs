//! Explicit static and harmonic electrical procedure templates.
use super::{
    state, AnalysisProcedure, IndependentDomain, PhysicsFamily, ProcedureStep, SolutionSpace,
    StateKind,
};

impl ProcedureStep {
    pub fn electromagnetic_static(step_id: impl Into<String>) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Electromagnetic,
            AnalysisProcedure::Static,
            SolutionSpace::Direct,
            IndependentDomain::Static,
            vec![state(
                "electromagnetic_static",
                StateKind::ElectromagneticStatic,
            )],
        )
    }

    pub fn electromagnetic_harmonic(step_id: impl Into<String>) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Electromagnetic,
            AnalysisProcedure::HarmonicResponse,
            SolutionSpace::Direct,
            IndependentDomain::FrequencyHertz(vec![60.0]),
            vec![state(
                "electromagnetic_complex",
                StateKind::ElectromagneticComplex,
            )],
        )
    }
}
