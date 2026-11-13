//! Authenticated thermal lifecycle input to the shared structural prestress producer.
use runmat_analysis_core::{StateId, StateSource, StateUse};
use runmat_analysis_fea::{
    execution::{AnalysisTaskIdentity, ThermalWorkload},
    results::accuracy::AccuracyStatus,
};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;

use super::{invalid, StructuralPrestressProducerPreparation};
use crate::{
    bind_upstream_state_transition, import_thermal_study_completion, AnalysisBridgeResult,
    CompletedThermalStudy, PreparedAnalysisObjects, ThermalStudyPreparation,
    ThermalStudyResultClosure, ThermalStudyResultPhase, ThermalStudyScope,
};

/// Sealed, replay-authenticated thermal completion. Public mutable fields on a
/// `CompletedThermalStudy` are not themselves evidence: import reconstructs every phase
/// from the retained response/accounting closure without executing a numerical solve.
pub struct ThermalPrestressSource {
    completed: CompletedThermalStudy,
}

pub(super) struct BoundThermalPrestressSource {
    pub objects: PreparedAnalysisObjects,
    pub dependency: AnalysisTaskIdentity,
}

impl ThermalPrestressSource {
    pub fn import(
        source: &impl CacheImport,
        preparation: ThermalStudyPreparation,
        closures: Vec<ThermalStudyResultClosure>,
    ) -> AnalysisBridgeResult<Self> {
        if preparation.scope != ThermalStudyScope::SelectedStep {
            return Err(invalid(
                "thermal prestress requires selected-step lifecycle scope",
            ));
        }
        let completed = import_thermal_study_completion(source, preparation, closures)?;
        if completed.accuracy.artifact.gate().status != AccuracyStatus::Satisfied {
            return Err(invalid(
                "thermal prestress requires satisfied thermal accuracy evidence",
            ));
        }
        Ok(Self { completed })
    }

    pub fn completed(&self) -> &CompletedThermalStudy {
        &self.completed
    }

    /// Deterministic dependency in the serial thermal lifecycle graph. This binds
    /// source lineage; it is not an independent attestation of scheduler execution.
    pub fn state_publication_dependency(
        &self,
        state_id: &StateId,
    ) -> AnalysisBridgeResult<AnalysisTaskIdentity> {
        let ordinal = self
            .completed
            .states
            .iter()
            .position(|state| &state.state.state_id == state_id)
            .ok_or_else(|| invalid("named thermal state was not published"))?;
        let mut preceding = None;
        for closure in self.completed.result_closures() {
            let identity = AnalysisTaskIdentity::new(
                closure.workload.identity,
                preceding.into_iter().collect(),
            )?;
            if closure.phase == (ThermalStudyResultPhase::State { ordinal }) {
                return Ok(identity);
            }
            preceding = Some(identity);
        }
        Err(invalid("thermal state publication has no retained phase"))
    }

    pub(super) fn bind(
        &self,
        expected: &ThermalWorkload,
        preparation: &StructuralPrestressProducerPreparation,
    ) -> AnalysisBridgeResult<BoundThermalPrestressSource> {
        let completed = &self.completed;
        let original = completed.preparation();
        if completed.workload != *expected
            || completed.compiled.model != preparation.preload_workload.model
            || completed
                .compiled
                .definition_digest()
                .map_err(|error| invalid(error.to_string()))?
                != preparation.authored_study_digest
            || original.mesh != preparation.mesh
            || original.frozen_project_digest != preparation.frozen_project_digest
            || original.artifact_access != preparation.artifact_access
            || original.program_revision != preparation.program_revision
        {
            return Err(invalid(
                "thermal prestress source differs from the exact authored execution context",
            ));
        }
        let step = preparation.preload_workload.procedure_step()?;
        let [input] = step.state_inputs.as_slice() else {
            return Err(invalid(
                "thermal prestress requires exactly one named state input",
            ));
        };
        let StateSource::Upstream {
            producer_step,
            state_id,
        } = &input.source
        else {
            return Err(invalid("thermal prestress requires an upstream state"));
        };
        if producer_step != &expected.step_id || input.use_as != StateUse::Prestress {
            return Err(invalid(
                "thermal prestress input names a different producer or use",
            ));
        }
        let state = completed
            .states
            .iter()
            .find(|state| &state.state.state_id == state_id)
            .ok_or_else(|| invalid("thermal producer did not publish the selected state"))?;
        bind_upstream_state_transition(
            &preparation.preload_workload.model,
            &preparation.preload_workload.step_id,
            &state.state,
            StateUse::Prestress,
            Digest::from_bytes(*preparation.mesh.canonical_digest.bytes()),
        )?;
        Ok(BoundThermalPrestressSource {
            objects: state.objects.clone(),
            dependency: self.state_publication_dependency(state_id)?,
        })
    }
}
