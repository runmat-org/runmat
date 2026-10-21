//! Placement-neutral authored load blocks and shared terminal publication.
//! This coordinator dispatches no kernels and retains earlier physics only by immutable closure.
use crate::steady_study::{SteadyStudyPhase, SteadyStudyTail};
use crate::AnalysisPhaseExecutionEvidence;
use crate::*;
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind as Kind, NonlinearSolidBlockRequest},
    study::{plan_nonlinear_solid_study_product, NonlinearSolidStudyProduct},
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, ArtifactAccess, ProgramExecutionResponse};
use runmat_meshing_core::SolverMeshArtifact;
mod blocks;
mod completion;
mod observations;
mod recovery;
mod retention;
mod root;
mod terminal;
pub use completion::{
    import_nonlinear_solid_study_completion, NonlinearSolidStudyResultClosure,
    NonlinearSolidStudyResultPhase, OriginalNonlinearSolidBlockProjection,
};
pub use root::{
    import_nonlinear_solid_study_root, prepare_nonlinear_solid_study_root,
    PreparedNonlinearSolidStudyRoot,
};

pub struct NonlinearSolidStudyPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub product: NonlinearSolidStudyProduct,
    pub mesh: SolverMeshArtifact,
    /// Limits one task's new outputs; it never replaces the original global accepted cap.
    pub accepted_block_limit: usize,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub output_policy: AnalysisOutputPolicy,
}

/// Task output granularity only; the original global accepted-increment cap is unchanged.
pub const DEFAULT_NONLINEAR_SOLID_ACCEPTED_BLOCK_LIMIT: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NonlinearSolidStudyPhase {
    ReadyForBlock { ordinal: u64 },
    AwaitingBlock { ordinal: u64 },
    ReadyForFieldRecovery,
    AwaitingFieldRecovery,
    ReadyForObservationFanout,
    AwaitingObservationPartitions,
    ReadyForObservationReduction { ordinal: usize },
    AwaitingObservationReduction { ordinal: usize },
    ReadyForObservationAssembly,
    AwaitingObservationAssembly,
    ReadyForAccuracy,
    AwaitingAccuracy,
    ReadyForState { ordinal: usize },
    AwaitingState { ordinal: usize },
    ReadyForBaseline,
    ReadyForPublication,
    AwaitingPublication,
    Complete,
}

pub struct NonlinearSolidStudyCoordinator {
    preparation: NonlinearSolidStudyPreparation,
    definition: Digest,
    phase: NonlinearSolidStudyPhase,
    block_task: Option<PreparedNonlinearSolidBlockTask>,
    block: Option<ImportedNonlinearSolidBlock>,
    closures: Vec<NonlinearSolidStudyResultClosure>,
    attempts: Vec<PlannedResourceUsageAttempt>,
    tail: Option<SteadyStudyTail>,
    source_host: Option<AnalysisHostWorkload>,
    source_plan: Option<runmat_analysis_fea::planning::resource::FeaResourcePlan>,
    recovery_task: Option<PreparedNonlinearFieldRecovery>,
    fields: Option<ImportedRecoveredFields>,
    observation_state: crate::sampled_study_observations::SampledStudyObservations,
    external_retained_bytes: u64,
    /// Private cold-consumption mode: original immutable objects, never Newton execution.
    /// Derived from the current original phase before preparing its replay task.
    import_workspace_bytes: Option<u64>,
}

pub struct CompletedNonlinearSolidStudy {
    replay: root::ReplayContext,
    pub compiled: runmat_analysis_fea::study::CompiledAnalysisStudy,
    pub workload: runmat_analysis_fea::execution::NonlinearSolidWorkload,
    pub terminal_block: ImportedNonlinearSolidBlock,
    pub accuracy: ImportedAccuracy,
    pub fields: Option<ImportedRecoveredFields>,
    pub observations: Option<ImportedObservations>,
    pub observation_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    pub states: Vec<ImportedSemanticState>,
    pub baseline_usage: PreparedResourceUsage,
    pub terminal_usage: PreparedResourceUsage,
    pub publication: ImportedPublishedResult,
    /// Original scheduler response/accounting associations, not scheduler attestation.
    pub closures: Vec<NonlinearSolidStudyResultClosure>,
}

impl NonlinearSolidStudyCoordinator {
    pub fn new(preparation: NonlinearSolidStudyPreparation) -> AnalysisBridgeResult<Self> {
        let expected = plan_nonlinear_solid_study_product(
            preparation.product.compiled.clone(),
            &preparation.mesh,
            preparation.product.workload.controls.clone(),
        )
        .map_err(invalid)?;
        if expected != preparation.product {
            return Err(invalid(
                "nonlinear workload differs from original compiled study",
            ));
        }
        NonlinearSolidBlockRequest::initial(
            &expected.workload,
            Digest::from_bytes(*preparation.mesh.canonical_digest.bytes()),
            preparation.accepted_block_limit,
        )?;
        let step = &expected.workload.model.sequence.steps[0];
        runmat_analysis_fea::pipeline::nonlinear::require_nonlinear_terminal_field_outputs(step)
            .map_err(invalid)?;
        if step.state_outputs.iter().any(|d| {
            !matches!(
                d.kind,
                runmat_analysis_core::StateKind::StructuralPrimary
                    | runmat_analysis_core::StateKind::StructuralStressStrain
                    | runmat_analysis_core::StateKind::MaterialHistory
            ) || d.completeness != runmat_analysis_core::StateCompleteness::Complete
                || d.persistence == runmat_analysis_core::StatePersistence::CheckpointOnly
        }) {
            return Err(invalid(
                "authored nonlinear state declarations need their separately admitted publisher",
            ));
        }
        let definition = expected.compiled.definition_digest().map_err(invalid)?;
        let coordinator = Self {
            preparation,
            definition,
            phase: NonlinearSolidStudyPhase::ReadyForBlock { ordinal: 0 },
            block_task: None,
            block: None,
            closures: vec![],
            attempts: vec![],
            tail: None,
            source_host: None,
            source_plan: None,
            recovery_task: None,
            fields: None,
            observation_state: Default::default(),
            external_retained_bytes: 0,
            import_workspace_bytes: None,
        };
        coordinator.admit_retention(None, None)?;
        Ok(coordinator)
    }

    pub fn phase(&self) -> NonlinearSolidStudyPhase {
        let Some(tail) = &self.tail else {
            return self.phase;
        };
        match tail.phase {
            SteadyStudyPhase::ReadyForAccuracy => NonlinearSolidStudyPhase::ReadyForAccuracy,
            SteadyStudyPhase::AwaitingAccuracy => NonlinearSolidStudyPhase::AwaitingAccuracy,
            SteadyStudyPhase::ReadyForState { ordinal } => {
                NonlinearSolidStudyPhase::ReadyForState { ordinal }
            }
            SteadyStudyPhase::AwaitingState { ordinal } => {
                NonlinearSolidStudyPhase::AwaitingState { ordinal }
            }
            SteadyStudyPhase::ReadyForBaseline => NonlinearSolidStudyPhase::ReadyForBaseline,
            SteadyStudyPhase::ReadyForPublication => NonlinearSolidStudyPhase::ReadyForPublication,
            SteadyStudyPhase::AwaitingPublication => NonlinearSolidStudyPhase::AwaitingPublication,
            SteadyStudyPhase::Complete => NonlinearSolidStudyPhase::Complete,
        }
    }
    pub fn terminal_block(&self) -> Option<&ImportedNonlinearSolidBlock> {
        self.block.as_ref().filter(|b| b.block.complete)
    }
    pub fn accuracy(&self) -> Option<&ImportedAccuracy> {
        self.tail.as_ref().and_then(|t| t.accuracy.as_ref())
    }
    pub fn states(&self) -> &[ImportedSemanticState] {
        self.tail.as_ref().map_or(&[], |t| t.states.as_slice())
    }
    pub fn result_closures(&self) -> &[NonlinearSolidStudyResultClosure] {
        &self.closures
    }
    pub fn finish(self) -> AnalysisBridgeResult<CompletedNonlinearSolidStudy> {
        if self.phase() != NonlinearSolidStudyPhase::Complete {
            return Err(invalid("nonlinear study is incomplete"));
        }
        let tail = self.tail.expect("completed tail");
        Ok(CompletedNonlinearSolidStudy {
            replay: root::ReplayContext {
                local_execution: self.preparation.local_execution,
                mesh: self.preparation.mesh,
                accepted_block_limit: self.preparation.accepted_block_limit,
                frozen_project_digest: self.preparation.frozen_project_digest,
                program_revision: self.preparation.program_revision,
                artifact_access: self.preparation.artifact_access,
                output_policy: self.preparation.output_policy,
            },
            compiled: self.preparation.product.compiled,
            workload: self.preparation.product.workload,
            terminal_block: self.block.expect("terminal block"),
            accuracy: tail.accuracy.expect("completed accuracy"),
            fields: self.fields,
            observations: self.observation_state.completed,
            observation_catalog: self.observation_state.catalog,
            states: tail.states,
            baseline_usage: tail.baseline.expect("frozen baseline"),
            terminal_usage: tail.terminal.expect("terminal usage"),
            publication: tail.publication.expect("completed publication"),
            closures: self.closures,
        })
    }
    fn tail_mut(&mut self) -> AnalysisBridgeResult<&mut SteadyStudyTail> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("original nonlinear schedule has not completed"))
    }
}
fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}

/// Capacity is distinct from malformed or substituted original history.
fn limit(message: &'static str) -> AnalysisBridgeError {
    runmat_analysis_fea::execution::AnalysisExecutionError::Limit(message).into()
}
