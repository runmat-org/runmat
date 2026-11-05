use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    validate_model_sequence, ProcedureStepId, StateId, StateSource, StateUse,
    ValidatedAnalysisSequence,
};
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, SemanticStatePublicationSource, StructuralLinearWorkload,
};
use runmat_analysis_fea::planning::resource::{
    recorded_static_solve_path, LocalExecutionPlan, StaticSolvePath, StaticSolveSelection,
};
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ProgramExecutionResponse};
use runmat_execution_runner::AttemptSuccess;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use crate::{
    import_semantic_state_completion, import_structural_linear_completion,
    prepare_semantic_state_publication, prepare_structural_sequence_step, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    PlannedResourceUsageAttempt, PreparedAnalysisObjects, PreparedResourceUsage,
    PreparedSemanticStatePublication, PreparedStructuralLinearProgram,
    SemanticStatePublicationPreparation, StructuralLinearProgramPreparation,
};

/// Stable semantic node names used to bind generic scheduler dependencies to an authored
/// structural sequence. They identify meaning, not placement or an execution attempt.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralSequenceNode {
    Solve(ProcedureStepId),
    PublishState {
        producer_step: ProcedureStepId,
        state_id: StateId,
    },
}

pub struct StructuralSequencePreparation {
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workloads: BTreeMap<ProcedureStepId, StructuralLinearWorkload>,
    pub mesh: Option<SolverMeshArtifact>,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
    pub accelerator_requests: Vec<AcceleratorRequest>,
    pub local_execution: LocalExecutionPlan,
    /// How each step's static solve path is chosen.
    pub solve_selection: StructuralSequenceSolveSelection,
}

/// How a structural sequence chooses each step's static solve path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuralSequenceSolveSelection {
    /// A fresh solve: every step chooses within min(authored ceiling, this machine's admitted
    /// memory when known).
    WithinCeiling { machine_memory_bytes: Option<u64> },
    /// An imported completion: every solve step reproduces the path its admitted plan recorded,
    /// whatever machine performs the import.
    Recorded(BTreeMap<ProcedureStepId, StaticSolvePath>),
}

impl StructuralSequenceSolveSelection {
    /// The paths recorded by a completed sequence's solve closures.
    pub fn recorded(closures: &[StructuralSequenceResultClosure]) -> AnalysisBridgeResult<Self> {
        let mut paths = BTreeMap::new();
        for closure in closures {
            match (&closure.node, closure.static_solve_path) {
                (StructuralSequenceNode::Solve(step), Some(path)) => {
                    if paths.insert(step.clone(), path).is_some() {
                        return Err(AnalysisBridgeError::Invalid(
                            "structural sequence closure records a solve path twice".into(),
                        ));
                    }
                }
                (StructuralSequenceNode::PublishState { .. }, None) => {}
                (StructuralSequenceNode::Solve(_), None) => {
                    return Err(AnalysisBridgeError::Invalid(
                        "structural sequence solve closure records no static solve path".into(),
                    ))
                }
                (StructuralSequenceNode::PublishState { .. }, Some(_)) => {
                    return Err(AnalysisBridgeError::Invalid(
                        "structural state publication closure records a solve path".into(),
                    ))
                }
            }
        }
        Ok(Self::Recorded(paths))
    }

    fn for_step(&self, step: &ProcedureStepId) -> AnalysisBridgeResult<StaticSolveSelection> {
        match self {
            Self::WithinCeiling {
                machine_memory_bytes,
            } => Ok(StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: *machine_memory_bytes,
            }),
            Self::Recorded(paths) => paths
                .get(step)
                .copied()
                .map(StaticSolveSelection::Recorded)
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid(format!(
                        "imported structural sequence records no solve path for step {step:?}"
                    ))
                }),
        }
    }
}

pub struct PreparedStructuralSolveTask {
    pub node: StructuralSequenceNode,
    pub dependencies: Vec<StructuralSequenceNode>,
    pub prepared: PreparedStructuralLinearProgram,
}

pub struct PreparedStructuralSolveWave {
    level: usize,
    pub tasks: Vec<PreparedStructuralSolveTask>,
}

impl PreparedStructuralSolveWave {
    pub fn level(&self) -> usize {
        self.level
    }
}

pub struct PreparedStructuralStateTask {
    pub node: StructuralSequenceNode,
    pub dependencies: Vec<StructuralSequenceNode>,
    pub prepared: PreparedSemanticStatePublication,
}

pub struct PreparedStructuralStateWave {
    level: usize,
    pub tasks: Vec<PreparedStructuralStateTask>,
}

impl PreparedStructuralStateWave {
    pub fn level(&self) -> usize {
        self.level
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralSequenceStep {
    pub step_id: ProcedureStepId,
    pub host: AnalysisHostWorkload,
    pub result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub result: runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
    pub objects: PreparedAnalysisObjects,
    pub numerical_receipt: runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub numerical_receipt_objects: PreparedAnalysisObjects,
    pub resource_usage: Option<PreparedResourceUsage>,
    pub response: ProgramExecutionResponse,
    /// The static solve path the step's admitted plan recorded.
    pub static_solve_path: StaticSolvePath,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralSequenceState {
    pub producer_step: ProcedureStepId,
    pub state_id: StateId,
    pub state: CanonicalSemanticState,
    pub objects: PreparedAnalysisObjects,
    pub resource_attempt: StructuralSequenceStateResourceAttempt,
    pub response: ProgramExecutionResponse,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuralSequenceStateResourceAttempt {
    /// Completion imported from durable closures without a live scheduler attempt.
    UnmeteredImport,
    /// Exact resource plan and scheduler accounting for the publication attempt.
    Metered(Box<PlannedResourceUsageAttempt>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralSequence {
    pub steps: Vec<CompletedStructuralSequenceStep>,
    pub states: Vec<CompletedStructuralSequenceState>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralSequenceResultClosure {
    pub node: StructuralSequenceNode,
    pub response: ProgramExecutionResponse,
    /// The recorded static solve path of a solve node (`None` for state publications), so an
    /// import re-prepares exactly the admitted plan instead of re-selecting on its own machine.
    pub static_solve_path: Option<StaticSolvePath>,
}

impl CompletedStructuralSequence {
    pub fn result_closures(&self) -> AnalysisBridgeResult<Vec<StructuralSequenceResultClosure>> {
        let steps = self
            .steps
            .iter()
            .map(|step| StructuralSequenceResultClosure {
                node: StructuralSequenceNode::Solve(step.step_id.clone()),
                response: step.response.clone(),
                static_solve_path: Some(step.static_solve_path),
            });
        let states = self
            .states
            .iter()
            .map(|state| StructuralSequenceResultClosure {
                node: StructuralSequenceNode::PublishState {
                    producer_step: state.producer_step.clone(),
                    state_id: state.state_id.clone(),
                },
                response: state.response.clone(),
                static_solve_path: None,
            });
        Ok(steps.chain(states).collect())
    }
}

pub fn import_structural_sequence_completion(
    source: &impl CacheImport,
    preparation: StructuralSequencePreparation,
    closures: Vec<StructuralSequenceResultClosure>,
) -> AnalysisBridgeResult<CompletedStructuralSequence> {
    let mut responses = BTreeMap::new();
    for closure in closures {
        if responses.insert(closure.node, closure.response).is_some() {
            return Err(AnalysisBridgeError::Invalid(
                "structural sequence closure contains a duplicate node".into(),
            ));
        }
    }
    let mut coordinator = StructuralSequenceCoordinator::new(preparation)?;
    while !coordinator.is_complete() {
        let solve_wave = coordinator.prepare_next_solve_wave()?;
        let solve_responses = take_wave_responses(
            solve_wave.tasks.iter().map(|task| &task.node),
            &mut responses,
        )?;
        let state_wave = coordinator.complete_solve_wave(source, solve_wave, solve_responses)?;
        let state_responses = take_wave_responses(
            state_wave.tasks.iter().map(|task| &task.node),
            &mut responses,
        )?;
        coordinator.complete_unmetered_state_wave(source, state_wave, state_responses)?;
    }
    if !responses.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "structural sequence closure contains an unauthored node".into(),
        ));
    }
    coordinator.finish()
}

fn take_wave_responses<'a>(
    nodes: impl IntoIterator<Item = &'a StructuralSequenceNode>,
    responses: &mut BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>,
) -> AnalysisBridgeResult<BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>> {
    nodes
        .into_iter()
        .map(|node| {
            responses
                .remove(node)
                .map(|response| (node.clone(), response))
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid(format!(
                        "structural sequence closure is missing node {node:?}"
                    ))
                })
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CoordinatorPhase {
    ReadyForSolve,
    AwaitingSolve,
    AwaitingStatePublication,
    Complete,
}

/// Advances a validated semantic sequence through immutable solve and state-publication waves.
///
/// The coordinator owns no scheduler, worker, or transport. A placement adapter executes each
/// returned wave through the generic runner and returns the exact closed responses. A level is
/// committed only after every response in that level has been admitted, so partial or stale work
/// cannot unlock a dependent procedure.
pub struct StructuralSequenceCoordinator {
    preparation: StructuralSequencePreparation,
    sequence: ValidatedAnalysisSequence,
    next_level: usize,
    phase: CoordinatorPhase,
    steps: BTreeMap<ProcedureStepId, CompletedStructuralSequenceStep>,
    states: BTreeMap<(ProcedureStepId, StateId), CompletedStructuralSequenceState>,
}

impl StructuralSequenceCoordinator {
    pub fn new(preparation: StructuralSequencePreparation) -> AnalysisBridgeResult<Self> {
        let model = validated_structural_sequence_model(&preparation.workloads)?;
        let sequence = validate_model_sequence(&model)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        validate_accelerator_requests(&preparation)?;
        if let Some(mesh) = &preparation.mesh {
            mesh.validate()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        }
        if preparation.chunk_bytes == 0 {
            return Err(AnalysisBridgeError::Invalid(
                "structural sequence chunk size must be non-zero".into(),
            ));
        }
        Ok(Self {
            preparation,
            sequence,
            next_level: 0,
            phase: CoordinatorPhase::ReadyForSolve,
            steps: BTreeMap::new(),
            states: BTreeMap::new(),
        })
    }

    pub fn sequence(&self) -> &ValidatedAnalysisSequence {
        &self.sequence
    }

    pub fn is_complete(&self) -> bool {
        self.phase == CoordinatorPhase::Complete
    }

    fn next_solve_level(&self) -> AnalysisBridgeResult<&[ProcedureStepId]> {
        if self.phase != CoordinatorPhase::ReadyForSolve {
            return Err(AnalysisBridgeError::Invalid(
                "structural sequence is not ready to prepare a solve wave".into(),
            ));
        }
        self.sequence
            .parallel_levels()
            .get(self.next_level)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "structural sequence has no remaining solve wave".into(),
                )
            })
    }

    /// Shared metered topology work for each task in the next synchronous
    /// preparation wave. The preparation kernels record completed work.
    pub fn next_solve_wave_work_upper_bound(
        &self,
    ) -> AnalysisBridgeResult<runmat_simulation_usage::OperationCount> {
        let count = u64::try_from(self.next_solve_level()?.len()).map_err(|_| {
            AnalysisBridgeError::FeaRun(runmat_analysis_fea::FeaRunError::Usage(
                runmat_simulation_usage::LedgerError::CountOverflow,
            ))
        })?;
        runmat_analysis_fea::structural_linear_preparation_work_upper_bound(
            self.preparation.mesh.as_ref(),
        )?
        .checked_mul(count)
        .map_err(|cause| {
            AnalysisBridgeError::FeaRun(runmat_analysis_fea::FeaRunError::Usage(cause.into()))
        })
    }

    pub fn prepare_next_solve_wave(&mut self) -> AnalysisBridgeResult<PreparedStructuralSolveWave> {
        let level = self.next_solve_level()?;
        let mut tasks = Vec::with_capacity(level.len());
        for step_id in level {
            let step = self
                .sequence
                .sequence()
                .steps
                .iter()
                .find(|step| step.step_id == *step_id)
                .ok_or(AnalysisBridgeError::Identity(
                    "validated sequence level references an absent step",
                ))?;
            let state_input = match step.state_inputs.as_slice() {
                [] => None,
                [input] if matches!(input.use_as, StateUse::InitialGuess | StateUse::Prestress) => {
                    let StateSource::Upstream {
                        producer_step,
                        state_id,
                    } = &input.source
                    else {
                        return Err(AnalysisBridgeError::Invalid(
                            "structural state input must name an upstream state".into(),
                        ));
                    };
                    Some(
                        self.states
                            .get(&(producer_step.clone(), state_id.clone()))
                            .ok_or_else(|| {
                                AnalysisBridgeError::Invalid(format!(
                                    "step `{}` is missing completed state `{}` from `{}`",
                                    step_id.as_str(),
                                    state_id.0,
                                    producer_step.as_str()
                                ))
                            })?
                            .objects
                            .clone(),
                    )
                }
                _ => None,
            };
            let workload = self
                .preparation
                .workloads
                .get(step_id)
                .ok_or(AnalysisBridgeError::Identity(
                    "validated sequence step has no exact solver workload",
                ))?
                .clone();
            let prepared = prepare_structural_sequence_step(StructuralLinearProgramPreparation {
                authored_study_digest: self.preparation.authored_study_digest,
                frozen_project_digest: self.preparation.frozen_project_digest,
                program_revision: self.preparation.program_revision.clone(),
                artifact_access: self.preparation.artifact_access.clone(),
                workload,
                mesh: self.preparation.mesh.clone(),
                state_input,
                chunk_bytes: self.preparation.chunk_bytes,
                inventory_limits: self.preparation.inventory_limits,
                required_capabilities: self.preparation.required_capabilities.clone(),
                accelerator_requests: self.preparation.accelerator_requests.clone(),
                local_execution: self.preparation.local_execution,
                solve_selection: self.preparation.solve_selection.for_step(step_id)?,
            })?;
            tasks.push(PreparedStructuralSolveTask {
                node: StructuralSequenceNode::Solve(step_id.clone()),
                dependencies: self.solve_dependencies(step_id)?,
                prepared,
            });
        }
        self.phase = CoordinatorPhase::AwaitingSolve;
        Ok(PreparedStructuralSolveWave {
            level: self.next_level,
            tasks,
        })
    }

    pub fn complete_solve_wave(
        &mut self,
        source: &impl CacheImport,
        wave: PreparedStructuralSolveWave,
        responses: BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>,
    ) -> AnalysisBridgeResult<PreparedStructuralStateWave> {
        self.complete_solve_wave_internal(source, wave, responses, None)
    }

    pub fn complete_metered_solve_wave(
        &mut self,
        source: &impl CacheImport,
        wave: PreparedStructuralSolveWave,
        responses: BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>,
        attempts: BTreeMap<StructuralSequenceNode, AttemptSuccess>,
    ) -> AnalysisBridgeResult<PreparedStructuralStateWave> {
        self.complete_solve_wave_internal(source, wave, responses, Some(attempts))
    }

    fn complete_solve_wave_internal(
        &mut self,
        source: &impl CacheImport,
        wave: PreparedStructuralSolveWave,
        mut responses: BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>,
        mut attempts: Option<BTreeMap<StructuralSequenceNode, AttemptSuccess>>,
    ) -> AnalysisBridgeResult<PreparedStructuralStateWave> {
        if self.phase != CoordinatorPhase::AwaitingSolve || wave.level != self.next_level {
            return Err(AnalysisBridgeError::Invalid(
                "completed solve wave does not match coordinator state".into(),
            ));
        }
        let expected = wave
            .tasks
            .iter()
            .map(|task| task.node.clone())
            .collect::<BTreeSet<_>>();
        if responses.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(AnalysisBridgeError::Invalid(
                "solve responses do not exactly cover the prepared wave".into(),
            ));
        }
        if attempts
            .as_ref()
            .is_some_and(|attempts| attempts.keys().cloned().collect::<BTreeSet<_>>() != expected)
        {
            return Err(AnalysisBridgeError::Invalid(
                "solve resource accounting does not exactly cover the prepared wave".into(),
            ));
        }

        let mut completed_steps = Vec::with_capacity(wave.tasks.len());
        let mut state_tasks = Vec::new();
        for task in wave.tasks {
            let response = responses.remove(&task.node).ok_or_else(|| {
                AnalysisBridgeError::Invalid("prepared solve response is absent".into())
            })?;
            let retained_response = response.clone();
            let StructuralSequenceNode::Solve(step_id) = &task.node else {
                return Err(AnalysisBridgeError::Identity(
                    "solve wave contains a non-solve node",
                ));
            };
            let imported = import_structural_linear_completion(
                source,
                response,
                &task.prepared,
                self.preparation.inventory_limits,
            )?;
            let resource_usage = attempts
                .as_mut()
                .map(|attempts| {
                    let attempt = attempts.remove(&task.node).ok_or_else(|| {
                        AnalysisBridgeError::Invalid(
                            "prepared solve has no scheduler resource accounting".into(),
                        )
                    })?;
                    crate::prepare_structural_resource_usage(
                        &task.prepared,
                        &imported,
                        &attempt,
                        self.preparation.chunk_bytes,
                        self.preparation.inventory_limits,
                    )
                })
                .transpose()?;
            let step = self
                .sequence
                .sequence()
                .steps
                .iter()
                .find(|step| step.step_id == *step_id)
                .ok_or(AnalysisBridgeError::Identity(
                    "completed solve references an absent authored step",
                ))?;
            for declaration in &step.state_outputs {
                let prepared =
                    prepare_semantic_state_publication(SemanticStatePublicationPreparation {
                        source_host: task.prepared.host.clone(),
                        source_result: imported.objects.clone(),
                        source_profiles: imported.state_publication_profile_objects.clone(),
                        source: match declaration.kind {
                            runmat_analysis_core::StateKind::StructuralPrimary => {
                                SemanticStatePublicationSource::StructuralPrimaryFields
                            }
                            runmat_analysis_core::StateKind::GeometricStiffness => {
                                SemanticStatePublicationSource::StructuralLinearization
                            }
                            _ => {
                                return Err(AnalysisBridgeError::Invalid(format!(
                                    "linear-static state output `{}` has unsupported kind {:?}",
                                    declaration.state_id.0, declaration.kind
                                )))
                            }
                        },
                        declaration: declaration.clone(),
                        topology_digest: imported
                            .result
                            .constraint_result
                            .topology
                            .discretization_digest(),
                        program_revision: self.preparation.program_revision.clone(),
                        chunk_bytes: self.preparation.chunk_bytes,
                        inventory_limits: self.preparation.inventory_limits,
                    })?;
                state_tasks.push(PreparedStructuralStateTask {
                    node: StructuralSequenceNode::PublishState {
                        producer_step: step_id.clone(),
                        state_id: declaration.state_id.clone(),
                    },
                    dependencies: vec![task.node.clone()],
                    prepared,
                });
            }
            let static_solve_path = recorded_static_solve_path(&task.prepared.resource_plan)
                .ok_or(AnalysisBridgeError::Identity(
                    "structural solve plan records no static solve path",
                ))?;
            completed_steps.push(CompletedStructuralSequenceStep {
                static_solve_path,
                step_id: step_id.clone(),
                host: task.prepared.host,
                result_identity: imported.identity,
                result: imported.result,
                objects: imported.objects,
                numerical_receipt: imported.numerical_receipt,
                numerical_receipt_objects: imported.numerical_receipt_objects,
                resource_usage,
                response: retained_response,
            });
        }
        if completed_steps
            .iter()
            .any(|step| self.steps.contains_key(&step.step_id))
        {
            return Err(AnalysisBridgeError::Invalid(
                "structural sequence step was completed more than once".into(),
            ));
        }
        for step in completed_steps {
            self.steps.insert(step.step_id.clone(), step);
        }
        self.phase = CoordinatorPhase::AwaitingStatePublication;
        Ok(PreparedStructuralStateWave {
            level: self.next_level,
            tasks: state_tasks,
        })
    }

    pub(crate) fn complete_unmetered_state_wave(
        &mut self,
        source: &impl CacheImport,
        wave: PreparedStructuralStateWave,
        responses: BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>,
    ) -> AnalysisBridgeResult<()> {
        self.complete_state_wave_internal(source, wave, responses, None)
    }

    pub fn complete_metered_state_wave(
        &mut self,
        source: &impl CacheImport,
        wave: PreparedStructuralStateWave,
        responses: BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>,
        attempts: BTreeMap<StructuralSequenceNode, PlannedResourceUsageAttempt>,
    ) -> AnalysisBridgeResult<()> {
        self.complete_state_wave_internal(source, wave, responses, Some(attempts))
    }

    fn complete_state_wave_internal(
        &mut self,
        source: &impl CacheImport,
        wave: PreparedStructuralStateWave,
        mut responses: BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>,
        mut attempts: Option<BTreeMap<StructuralSequenceNode, PlannedResourceUsageAttempt>>,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != CoordinatorPhase::AwaitingStatePublication || wave.level != self.next_level
        {
            return Err(AnalysisBridgeError::Invalid(
                "completed state wave does not match coordinator state".into(),
            ));
        }
        let expected = wave
            .tasks
            .iter()
            .map(|task| task.node.clone())
            .collect::<BTreeSet<_>>();
        if responses.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(AnalysisBridgeError::Invalid(
                "state responses do not exactly cover the prepared wave".into(),
            ));
        }
        if attempts
            .as_ref()
            .is_some_and(|attempts| attempts.keys().cloned().collect::<BTreeSet<_>>() != expected)
        {
            return Err(AnalysisBridgeError::Invalid(
                "state resource accounting does not exactly cover the prepared wave".into(),
            ));
        }
        let mut completed_states = Vec::with_capacity(wave.tasks.len());
        for task in wave.tasks {
            let response = responses.remove(&task.node).ok_or_else(|| {
                AnalysisBridgeError::Invalid("prepared state response is absent".into())
            })?;
            let retained_response = response.clone();
            let StructuralSequenceNode::PublishState {
                producer_step,
                state_id,
            } = &task.node
            else {
                return Err(AnalysisBridgeError::Identity(
                    "state wave contains a non-publication node",
                ));
            };
            let imported = import_semantic_state_completion(
                source,
                response,
                &task.prepared,
                self.preparation.inventory_limits,
            )?;
            if imported.state.state_id != *state_id {
                return Err(AnalysisBridgeError::Identity(
                    "published state identity differs from its sequence node",
                ));
            }
            let resource_attempt = match attempts.as_mut() {
                Some(attempts) => {
                    let attempt = attempts.remove(&task.node).ok_or_else(|| {
                        AnalysisBridgeError::Invalid(
                            "prepared state has no scheduler resource accounting".into(),
                        )
                    })?;
                    if attempt.plan != task.prepared.resource_plan {
                        return Err(AnalysisBridgeError::Identity(
                            "state scheduler accounting names another resource plan",
                        ));
                    }
                    StructuralSequenceStateResourceAttempt::Metered(Box::new(attempt))
                }
                None => StructuralSequenceStateResourceAttempt::UnmeteredImport,
            };
            completed_states.push(CompletedStructuralSequenceState {
                producer_step: producer_step.clone(),
                state_id: state_id.clone(),
                state: imported.state,
                objects: imported.objects,
                resource_attempt,
                response: retained_response,
            });
        }
        if completed_states.iter().any(|state| {
            self.states
                .contains_key(&(state.producer_step.clone(), state.state_id.clone()))
        }) {
            return Err(AnalysisBridgeError::Invalid(
                "structural sequence state was published more than once".into(),
            ));
        }
        for state in completed_states {
            self.states
                .insert((state.producer_step.clone(), state.state_id.clone()), state);
        }
        self.next_level = self.next_level.checked_add(1).ok_or_else(|| {
            AnalysisBridgeError::Invalid("structural sequence level overflowed".into())
        })?;
        self.phase = if self.next_level == self.sequence.parallel_levels().len() {
            CoordinatorPhase::Complete
        } else {
            CoordinatorPhase::ReadyForSolve
        };
        Ok(())
    }

    pub fn finish(mut self) -> AnalysisBridgeResult<CompletedStructuralSequence> {
        if !self.is_complete() {
            return Err(AnalysisBridgeError::Invalid(
                "structural sequence cannot finish before every wave completes".into(),
            ));
        }
        let mut steps = Vec::with_capacity(self.steps.len());
        let mut states = Vec::with_capacity(self.states.len());
        for step in &self.sequence.sequence().steps {
            steps.push(self.steps.remove(&step.step_id).ok_or_else(|| {
                AnalysisBridgeError::Invalid(format!(
                    "completed sequence is missing step `{}`",
                    step.step_id.as_str()
                ))
            })?);
            for declaration in &step.state_outputs {
                states.push(
                    self.states
                        .remove(&(step.step_id.clone(), declaration.state_id.clone()))
                        .ok_or_else(|| {
                            AnalysisBridgeError::Invalid(format!(
                                "completed sequence is missing state `{}` from `{}`",
                                declaration.state_id.0,
                                step.step_id.as_str()
                            ))
                        })?,
                );
            }
        }
        if !self.steps.is_empty() || !self.states.is_empty() {
            return Err(AnalysisBridgeError::Identity(
                "completed sequence contains unauthored steps or states",
            ));
        }
        Ok(CompletedStructuralSequence { steps, states })
    }

    fn solve_dependencies(
        &self,
        step_id: &ProcedureStepId,
    ) -> AnalysisBridgeResult<Vec<StructuralSequenceNode>> {
        let step = self
            .sequence
            .sequence()
            .steps
            .iter()
            .find(|step| step.step_id == *step_id)
            .ok_or(AnalysisBridgeError::Identity(
                "validated sequence dependency references an absent step",
            ))?;
        self.sequence
            .direct_predecessors()
            .get(step_id)
            .ok_or(AnalysisBridgeError::Identity(
                "validated sequence has no predecessor inventory for a step",
            ))?
            .iter()
            .map(|predecessor| {
                let matching_states = step
                    .state_inputs
                    .iter()
                    .filter_map(|input| match &input.source {
                        StateSource::Upstream {
                            producer_step,
                            state_id,
                        } if producer_step == predecessor => Some(state_id.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                match matching_states.as_slice() {
                    [] => Ok(StructuralSequenceNode::Solve(predecessor.clone())),
                    [state_id] => Ok(StructuralSequenceNode::PublishState {
                        producer_step: predecessor.clone(),
                        state_id: state_id.clone(),
                    }),
                    _ => Err(AnalysisBridgeError::Invalid(
                        "one predecessor supplies multiple unsupported state inputs".into(),
                    )),
                }
            })
            .collect()
    }
}

fn validate_accelerator_requests(
    preparation: &StructuralSequencePreparation,
) -> AnalysisBridgeResult<()> {
    if preparation.accelerator_requests.is_empty() {
        Ok(())
    } else {
        Err(AnalysisBridgeError::Invalid(
            "CPU-only structural sequences cannot publish an accelerator allocation".into(),
        ))
    }
}

pub fn validate_structural_sequence_workloads(
    workloads: &BTreeMap<ProcedureStepId, StructuralLinearWorkload>,
) -> AnalysisBridgeResult<()> {
    runmat_analysis_fea::execution::StructuralStudyWorkloads::validate_linear_sequence_workloads(
        workloads,
    )
    .map_err(Into::into)
}

fn validated_structural_sequence_model(
    workloads: &BTreeMap<ProcedureStepId, StructuralLinearWorkload>,
) -> AnalysisBridgeResult<runmat_analysis_core::AnalysisModel> {
    validate_structural_sequence_workloads(workloads)?;
    let model = workloads
        .values()
        .next()
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "structural sequence preparation has no step workloads".into(),
            )
        })?
        .model
        .clone();
    Ok(model)
}
