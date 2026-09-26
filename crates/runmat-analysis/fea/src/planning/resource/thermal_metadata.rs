//! Producer-owned initial-state, physical, assignment and successful-solver metadata projections.
//! Backend evidence and transport lifetimes remain separate bounds.
use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_canonical_codec::{profile_serialized, CanonicalLimits, SerializedCanonicalProfile};

use super::{ByteCount, ResourcePlanningError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThermalMetadataResourceFacts {
    pub initial_state: SerializedCanonicalProfile,
    pub physical_diagnostics: SerializedCanonicalProfile,
    pub assignment_diagnostics: SerializedCanonicalProfile,
    pub solver_diagnostics: SerializedCanonicalProfile,
}

impl ThermalMetadataResourceFacts {
    pub fn derive(
        model: &AnalysisModel,
        step: &ProcedureStep,
    ) -> Result<Self, ResourcePlanningError> {
        Ok(Self {
            solver_diagnostics: crate::solve::linear::successful_diagnostics_profile()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            assignment_diagnostics: crate::diagnostics::material_assignment::profile(
                &model.material_assignments,
            )
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            initial_state: crate::physics::thermal::initial_state::evidence_profile(model, step)
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            physical_diagnostics: crate::diagnostics::thermal::thermal_diagnostic_profile(
                step.step_id.as_str(),
            )
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        self.initial_state
            .materialization_resident_bytes_estimate
            .checked_add(
                self.physical_diagnostics
                    .materialization_resident_bytes_estimate,
            )
            .and_then(|bytes| {
                bytes.checked_add(
                    self.assignment_diagnostics
                        .materialization_resident_bytes_estimate,
                )
            })
            // Previous/current solves overlap, and final reporting clones the last list.
            .and_then(|bytes| {
                self.solver_diagnostics
                    .materialization_resident_bytes_estimate
                    .checked_mul(2)
                    .and_then(|solver| bytes.checked_add(solver))
            })
            .map(ByteCount)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal metadata materialization",
            ))
    }

    pub(crate) fn verify_initial_state(
        &self,
        evidence: &crate::ThermalInitialStateEvidence,
    ) -> Result<(), ResourcePlanningError> {
        verify(evidence, self.initial_state)
    }

    pub(crate) fn verify_assignment_diagnostics(
        &self,
        diagnostics: &[crate::diagnostics::FeaDiagnostic],
    ) -> Result<(), ResourcePlanningError> {
        verify(diagnostics, self.assignment_diagnostics)
    }

    pub(crate) fn verify_solver_diagnostics(
        &self,
        diagnostics: &[crate::diagnostics::FeaDiagnostic],
    ) -> Result<(), ResourcePlanningError> {
        verify(diagnostics, self.solver_diagnostics)
    }

    pub(crate) fn verify_physical_diagnostics(
        &self,
        diagnostics: &[crate::diagnostics::FeaDiagnostic],
    ) -> Result<(), ResourcePlanningError> {
        verify(diagnostics, self.physical_diagnostics)
    }
}

fn verify(
    value: &(impl serde::Serialize + ?Sized),
    expected: SerializedCanonicalProfile,
) -> Result<(), ResourcePlanningError> {
    let actual = profile_serialized(
        value,
        CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
    )
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    if actual.encoded_bytes_upper_bound > expected.encoded_bytes_upper_bound
        || actual.materialization_resident_bytes_estimate
            > expected.materialization_resident_bytes_estimate
    {
        return Err(ResourcePlanningError::Invalid(
            "thermal evidence exceeds producer metadata projection".into(),
        ));
    }
    Ok(())
}
