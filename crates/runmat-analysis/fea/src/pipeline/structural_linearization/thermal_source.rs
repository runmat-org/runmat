//! Retained thermal transition and structural result authority for a preload.

use super::*;
use crate::execution::{CanonicalStateTransition, SolverResultIdentity};
use runmat_analysis_core::{CouplingImplementation, StateTransitionMethod};

mod result;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalLinearizationSource {
    transition: CanonicalStateTransition,
    solver_model_digest: Digest,
    /// Content digest of the thermoelastic structural result, not the upstream heat result.
    structural_result_content: Digest,
    structural_result: SolverResultIdentity,
}

impl ThermalLinearizationSource {
    pub(super) fn from_result(
        model: &runmat_analysis_core::AnalysisModel,
        transition: CanonicalStateTransition,
        run: &crate::FeaRunResult,
    ) -> Result<Self, StructuralLinearizationError> {
        let solver_model_digest = result::model_digest(model)?;
        let structural_result_content =
            result::content_digest(solver_model_digest, &transition, run)?;
        let structural_result =
            SolverResultIdentity::from_step_result(transition.consumer, structural_result_content)
                .map_err(|error| StructuralLinearizationError::Identity(error.to_string()))?;
        Ok(Self {
            transition,
            solver_model_digest,
            structural_result_content,
            structural_result,
        })
    }

    pub fn transition(&self) -> &CanonicalStateTransition {
        &self.transition
    }

    pub fn structural_result(&self) -> SolverResultIdentity {
        self.structural_result
    }

    pub(crate) fn structural_result_content(&self) -> Digest {
        self.structural_result_content
    }

    /// Bind the retained physical result to the full authored solver contract, not a
    /// transport projection with its state inputs removed.
    pub fn validate_model(
        &self,
        model: &runmat_analysis_core::AnalysisModel,
    ) -> Result<(), StructuralLinearizationError> {
        if result::model_digest(model)? != self.solver_model_digest {
            return Err(StructuralLinearizationError::InvalidArtifact(
                "thermal result solver model differs",
            ));
        }
        Ok(())
    }

    pub fn validate_result(
        &self,
        run: &crate::FeaRunResult,
    ) -> Result<(), StructuralLinearizationError> {
        if result::content_digest(self.solver_model_digest, &self.transition, run)?
            != self.structural_result_content
        {
            return Err(StructuralLinearizationError::InvalidArtifact(
                "thermal result fields differ from retained physical source",
            ));
        }
        Ok(())
    }
}

pub(super) fn validate_context(
    run: &AnalysisRunIdentity,
    topology: Digest,
    source: Option<&ThermalLinearizationSource>,
) -> Result<(), StructuralLinearizationError> {
    if (run.realization().coupling == CouplingImplementation::OneWaySolvedThermoelasticState)
        != source.is_some()
    {
        return Err(StructuralLinearizationError::InvalidArtifact(
            "thermal source and coupling differ",
        ));
    }
    if let Some(source) = source {
        source
            .transition
            .validate()
            .map_err(|error| StructuralLinearizationError::Identity(error.to_string()))?;
        let expected = SolverResultIdentity::from_step_result(
            source.transition.consumer,
            source.structural_result_content,
        )
        .map_err(|error| StructuralLinearizationError::Identity(error.to_string()))?;
        if source.transition.method != StateTransitionMethod::Reuse
            || source.transition.target_topology_digest != topology
            || source.structural_result != expected
        {
            return Err(StructuralLinearizationError::InvalidArtifact(
                "thermal transition and structural result differ",
            ));
        }
    }
    Ok(())
}
