//! Retained thermal transition and structural result authority for a preload.

use super::*;
use crate::execution::{CanonicalStateTransition, SolverResultIdentity};
use runmat_analysis_core::{CouplingImplementation, StateTransitionMethod};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalLinearizationSource {
    transition: CanonicalStateTransition,
    /// Content digest of the thermoelastic structural result, not the upstream heat result.
    structural_result_content: Digest,
    structural_result: SolverResultIdentity,
}

impl ThermalLinearizationSource {
    pub(super) fn new(
        transition: CanonicalStateTransition,
        structural_result_content: Digest,
    ) -> Result<Self, StructuralLinearizationError> {
        let structural_result =
            SolverResultIdentity::from_step_result(transition.consumer, structural_result_content)
                .map_err(|error| StructuralLinearizationError::Identity(error.to_string()))?;
        Ok(Self {
            transition,
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
