//! Checked-source serial execution. Hosts supply geometry, capacity and a clock;
//! the shared authoring, meshing, numerical and publication owners retain meaning.
use super::CheckedFeaSessionSource;
use runmat_analysis_execution::{
    execute_serial_structural_sequence_with_capacity, AsyncCapacityOwner, GeometryProviderProfile,
    PreparedGeometryMaterializationProgram, PreparedGeometrySource, SerialAnalysisHost,
    StructuralSequencePreparation,
};
use runmat_analysis_fea::{
    execution::{StructuralStudyWorkloads, DEFAULT_ANALYSIS_CHUNK_BYTES},
    planning::resource::{FeaResourcePlan, LocalExecutionPlan},
    study::{
        compile_analysis_study, plan_structural_study, AnalysisStudyCompilationInput,
        CompiledAnalysisStudy, StructuralNumericalPlanningPolicy, StructuralStudyPlanningPolicy,
    },
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::MemoryObjectStore, object::ObjectInventoryLimits, ArtifactAccess,
};
use runmat_meshing_core::{MeshingEvidence, MeshingRequestSettings, SolverMeshArtifact};
use runmat_runtime::analysis::{
    ResolvedFeaRoot, StructuralStudyNumericalResult, StructuralStudyResult,
    StructuralStudySequenceNumericalResult, StructuralStudyStepNumericalResult,
};
use std::{collections::BTreeSet, future::Future};

#[derive(Debug)]
pub enum FeaStudyPreparationError {
    RequiresDesktop(&'static str),
    Invalid(String),
    Meshing(Box<runmat_meshing_core::MeshingFailure>),
    Admission(runmat_runtime::analysis::fea_run_admission::FeaRunDenied),
    /// No solve path fits the browser budget the host admitted.
    ResourceLimit(super::BrowserResourceLimit),
}
impl std::fmt::Display for FeaStudyPreparationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequiresDesktop(reason) => f.write_str(reason),
            Self::Invalid(reason) => f.write_str(reason),
            Self::Meshing(failure) => write!(f, "Mesh preparation failed: {failure}"),
            Self::Admission(denied) => denied.fmt(f),
            Self::ResourceLimit(limit) => f.write_str(&limit.message()),
        }
    }
}
impl std::error::Error for FeaStudyPreparationError {}
impl FeaStudyPreparationError {
    pub fn diagnostic(&self) -> serde_json::Value {
        if let Self::ResourceLimit(limit) = self {
            return limit.diagnostic();
        }
        let mut value = serde_json::json!({"kind":"runtime","message":self.to_string(),"diagnostic":self.to_string(),"callstack":[]});
        if let Self::Meshing(failure) = self {
            value["identifier"] = "RM.FEA.PREPARATION.MESH_FAILED".into();
            value["meshingFailure"] =
                serde_json::to_value(failure).expect("canonical meshing failure is serializable");
        }
        if let Self::Admission(denied) = self {
            value["identifier"] = denied.error_code().into();
            value["admissionDenial"] =
                serde_json::to_value(denied).expect("typed admission denial is serializable");
        }
        value
    }
}
impl From<runmat_analysis_execution::SerialGeometryMeshingError> for FeaStudyPreparationError {
    fn from(error: runmat_analysis_execution::SerialGeometryMeshingError) -> Self {
        match error.meshing_failure() {
            Some(failure) => Self::Meshing(Box::new(failure.clone())),
            None => Self::Invalid(error.to_string()),
        }
    }
}

impl From<String> for FeaStudyPreparationError {
    fn from(value: String) -> Self {
        Self::Invalid(value)
    }
}
impl From<&str> for FeaStudyPreparationError {
    fn from(value: &str) -> Self {
        Self::Invalid(value.into())
    }
}

pub trait FeaGeometryExecutionHost {
    fn ensure_preparation_capacity(
        &self,
        additional: runmat_simulation_usage::OperationCount,
    ) -> impl Future<Output = Result<(), runmat_runtime::analysis::fea_run_admission::FeaRunDenied>>;
    fn provider_profile(&self) -> impl Future<Output = Result<GeometryProviderProfile, String>>;
    fn materialize_and_mesh(
        &self,
        program: PreparedGeometryMaterializationProgram,
        source: PreparedGeometrySource,
        settings: MeshingRequestSettings,
    ) -> impl Future<
        Output = Result<runmat_analysis_execution::PreparedFeaGeometry, FeaStudyPreparationError>,
    >;
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct PreparedSerialFeaStudy {
    pub compiled: CompiledAnalysisStudy,
    pub mesh: SolverMeshArtifact,
    pub evidence: MeshingEvidence,
    pub workloads: StructuralStudyWorkloads,
    pub resource_plan: FeaResourcePlan,
    pub revision: ProgramRevision,
    pub checked_revision: [u8; 32],
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    /// The resident memory the solve is planned within (`None`: the authored ceiling
    /// alone). For a browser worker this is its budget less this study's request text,
    /// which the worker holds while it reads the study. Preparation and execution choose
    /// the solve path within it.
    pub machine_memory_bytes: Option<u64>,
}
impl CheckedFeaSessionSource {
    /// Prepares one serial study. `worker_budget_bytes` is the resident budget of the
    /// worker that receives the study as text (`None`: the authored ceiling alone); the
    /// solve is planned within it less that text.
    pub async fn prepare_serial_fea_study(
        &self,
        host: &impl FeaGeometryExecutionHost,
        authority: &str,
        worker_budget_bytes: Option<u64>,
    ) -> Result<PreparedSerialFeaStudy, FeaStudyPreparationError> {
        let ResolvedFeaRoot::Study(resolved) =
            self.resolve_root(None).map_err(|e| e.to_string())?
        else {
            return Err(FeaStudyPreparationError::RequiresDesktop(
                "Run this simulation group in the desktop app.",
            ));
        };
        if !resolved.study.inputs.is_empty()
            || resolved.study.assessments.iter().any(|assessment| {
                matches!(
                    assessment.rule,
                    runmat_analysis_core::AssessmentRule::Function { .. }
                )
            })
        {
            return Err(FeaStudyPreparationError::RequiresDesktop(
                "Run simulations with custom input or assessment functions in the desktop app.",
            ));
        }
        let realized = resolved
            .study
            .realize_with_models(&resolved.models, &resolved.geometry, &[])
            .map_err(|e| e.to_string())?;
        if realized.sequence.steps.len() != 1
            || realized.sequence.steps.iter().any(|step| {
                step.family != runmat_analysis_core::PhysicsFamily::Structural
                    || step.procedure != runmat_analysis_core::AnalysisProcedure::Static
            })
        {
            return Err(FeaStudyPreparationError::RequiresDesktop(
                "Run this type of simulation in the desktop app.",
            ));
        }
        let revision = self.program_revision()?;
        let access = runmat_analysis_execution::checked_fea_artifact_access(authority);
        let super::fea_geometry_program::SerialGeometryProgram {
            request: geometry_request,
            program,
            source: geometry_source,
        } = self
            .serial_geometry_program(
                &resolved.geometry,
                realized.geometry.bindings.clone(),
                host.provider_profile().await?,
                &revision,
                &access,
            )
            .await?;
        runmat_analysis_fea::progress::emit_phase(
            "fea.prepare",
            runmat_analysis_fea::FeaProgressPhase::GeometryLoad,
            runmat_analysis_fea::FeaProgressStatus::Started,
            "Preparing simulation geometry",
            None,
            None,
        );
        let geometry = host
            .materialize_and_mesh(program, geometry_source, resolved.mesh.settings)
            .await?;
        geometry
            .geometry
            .validate_against(&geometry_request)
            .map_err(|e| e.to_string())?;
        geometry
            .meshing_evidence
            .validate(&geometry.mesh)
            .map_err(|e| e.to_string())?;
        let authored_study_digest = runmat_analysis_fea::study::structural_study_source_digest(
            &realized,
            &resolved.material_catalog,
        )
        .map_err(|e| e.to_string())?;
        runmat_analysis_fea::progress::emit_phase(
            "fea.prepare",
            runmat_analysis_fea::FeaProgressPhase::ModelAssembly,
            runmat_analysis_fea::FeaProgressStatus::Started,
            "Preparing simulation operators",
            None,
            None,
        );
        let compiled = compile_analysis_study(AnalysisStudyCompilationInput {
            study: &realized,
            geometry: &geometry.geometry.document,
            regions: &geometry.geometry.regions,
            materials: &resolved.material_catalog,
            mesh: &geometry.mesh,
        })
        .map_err(|e| e.to_string())?;
        let workloads = plan_structural_study(
            &compiled,
            StructuralStudyPlanningPolicy {
                backend: runmat_analysis_fea::ComputeBackend::Cpu,
                numerics: StructuralNumericalPlanningPolicy::production(),
            },
        )
        .map_err(|e| e.to_string())?;
        let frozen_project_digest = self
            .frozen_project()
            .map(|project| Digest::from_bytes(*project.graph_digest().bytes()));
        let resource_plan = plan_admitted_solve(
            host,
            SolvePlanInputs {
                workloads: &workloads,
                mesh: &geometry.mesh,
                revision: &revision,
                access: &access,
                authored_study_digest,
                frozen_project_digest,
            },
            worker_budget_bytes,
        )
        .await?;
        let mut study = PreparedSerialFeaStudy {
            compiled,
            mesh: geometry.mesh,
            evidence: geometry.meshing_evidence,
            workloads,
            resource_plan,
            revision,
            checked_revision: self.check_result().revision.0,
            authored_study_digest,
            frozen_project_digest,
            artifact_access: access,
            machine_memory_bytes: worker_budget_bytes,
        };
        if let Some(budget) = worker_budget_bytes {
            fit_request_within(host, &mut study, budget).await?;
        }
        Ok(study)
    }
}

/// Rounds of re-planning against the measured request text. The text changes only by the
/// plan it carries, so the second round normally settles it.
const REQUEST_FIT_ROUNDS: usize = 3;

/// The owned worker holds the request text while it reads the study, so the solve must fit
/// the worker budget less that text. A plan chosen against the whole budget is re-derived
/// within the budget less the measured text: the planner then takes a path that fits (the
/// elastic multigrid path where the factorization does not), or refuses with its own
/// numbers. The study keeps the ceiling its final plan used, so execution re-derives it.
async fn fit_request_within(
    host: &impl FeaGeometryExecutionHost,
    study: &mut PreparedSerialFeaStudy,
    budget: u64,
) -> Result<(), FeaStudyPreparationError> {
    for _ in 0..REQUEST_FIT_ROUNDS {
        let request = study.request_bytes()?;
        let Some(ceiling) = replanning_ceiling(
            study.resource_plan.request.memory_bytes,
            study.machine_memory_bytes,
            request,
            budget,
        ) else {
            break;
        };
        let plan = plan_admitted_solve(host, study.plan_inputs(), Some(ceiling))
            .await
            .map_err(|error| match error {
                FeaStudyPreparationError::ResourceLimit(limit) => {
                    FeaStudyPreparationError::ResourceLimit(limit.with_request(request, budget))
                }
                error => error,
            })?;
        study.resource_plan = plan;
        study.machine_memory_bytes = Some(ceiling);
    }
    let request = study.request_bytes()?;
    super::admit_browser_solve(&study.resource_plan, request, budget)
        .map_err(FeaStudyPreparationError::ResourceLimit)
}

/// The ceiling to re-plan within when the plan and its request text overrun the worker
/// budget, or `None` when the plan fits or was already made within that ceiling (the gate
/// then refuses with the planner's numbers).
fn replanning_ceiling(
    plan_memory_bytes: u64,
    planned_ceiling_bytes: Option<u64>,
    request_bytes: u64,
    budget_bytes: u64,
) -> Option<u64> {
    let ceiling = budget_bytes.saturating_sub(request_bytes);
    (plan_memory_bytes > ceiling && planned_ceiling_bytes.is_none_or(|planned| planned > ceiling))
        .then_some(ceiling)
}

struct SolvePlanInputs<'a> {
    workloads: &'a StructuralStudyWorkloads,
    mesh: &'a SolverMeshArtifact,
    revision: &'a ProgramRevision,
    access: &'a ArtifactAccess,
    authored_study_digest: Digest,
    frozen_project_digest: Option<Digest>,
}

/// Plans the study's one solve within `machine_memory_bytes`, after the host admits the
/// preparation work.
async fn plan_admitted_solve(
    host: &impl FeaGeometryExecutionHost,
    inputs: SolvePlanInputs<'_>,
    machine_memory_bytes: Option<u64>,
) -> Result<FeaResourcePlan, FeaStudyPreparationError> {
    let preparation = sequence_preparation(
        inputs.workloads,
        inputs.mesh,
        inputs.revision.clone(),
        inputs.access.clone(),
        inputs.authored_study_digest,
        inputs.frozen_project_digest,
        Placement {
            local_execution: LocalExecutionPlan::serial(),
            machine_memory_bytes,
        },
    )?;
    let mut coordinator =
        runmat_analysis_execution::StructuralSequenceCoordinator::new(preparation)
            .map_err(|e| e.to_string())?;
    let work = coordinator
        .next_solve_wave_work_upper_bound()
        .map_err(preparation_bridge_error)?;
    host.ensure_preparation_capacity(work)
        .await
        .map_err(FeaStudyPreparationError::Admission)?;
    let wave = coordinator
        .prepare_next_solve_wave()
        .map_err(preparation_bridge_error)?;
    let [task] = wave.tasks.as_slice() else {
        return Err("This execution host requires one admitted solve.".into());
    };
    Ok(task.prepared.resource_plan.clone())
}
fn preparation_bridge_error(
    error: runmat_analysis_execution::AnalysisBridgeError,
) -> FeaStudyPreparationError {
    if let Some(cause) = error.accounting_cause() {
        return FeaStudyPreparationError::Admission(
            runmat_runtime::analysis::fea_run_admission::FeaRunDenied::Accounting(*cause),
        );
    }
    if let runmat_analysis_execution::AnalysisBridgeError::Resource(planning) = &error {
        if let Some(limit) = super::BrowserResourceLimit::from_planning(planning) {
            return FeaStudyPreparationError::ResourceLimit(limit);
        }
    }
    FeaStudyPreparationError::Invalid(error.to_string())
}
/// Where a solve runs: its thread allocation and the resident memory the host
/// admits one worker (`None`: the authored ceiling alone).
struct Placement {
    local_execution: LocalExecutionPlan,
    machine_memory_bytes: Option<u64>,
}
fn sequence_preparation(
    workloads: &StructuralStudyWorkloads,
    mesh: &SolverMeshArtifact,
    revision: ProgramRevision,
    access: ArtifactAccess,
    authored_study_digest: Digest,
    frozen_project_digest: Option<Digest>,
    placement: Placement,
) -> Result<StructuralSequencePreparation, String> {
    let Placement {
        local_execution,
        machine_memory_bytes,
    } = placement;
    let StructuralStudyWorkloads::LinearSequence { workloads } = workloads else {
        return Err("This procedure requires another execution host.".into());
    };
    Ok(StructuralSequencePreparation {
        authored_study_digest,
        frozen_project_digest,
        program_revision: revision,
        artifact_access: access,
        workloads: workloads.clone(),
        mesh: Some(mesh.clone()),
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution,
        solve_selection:
            runmat_analysis_execution::StructuralSequenceSolveSelection::WithinCeiling {
                machine_memory_bytes,
            },
    })
}
#[derive(Debug)]
pub enum PreparedSerialFeaExecutionError {
    Preparation(String),
    Bridge(runmat_analysis_execution::AnalysisBridgeError),
    Result(String),
}
impl std::fmt::Display for PreparedSerialFeaExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preparation(message) | Self::Result(message) => formatter.write_str(message),
            Self::Bridge(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}
impl std::error::Error for PreparedSerialFeaExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bridge(error) => Some(error),
            _ => None,
        }
    }
}
impl From<String> for PreparedSerialFeaExecutionError {
    fn from(message: String) -> Self {
        Self::Preparation(message)
    }
}
impl From<runmat_analysis_execution::AnalysisBridgeError> for PreparedSerialFeaExecutionError {
    fn from(error: runmat_analysis_execution::AnalysisBridgeError) -> Self {
        Self::Bridge(error)
    }
}

impl PreparedSerialFeaStudy {
    /// Length of the request text (this study as JSON), counted without building it.
    pub fn request_bytes(&self) -> Result<u64, String> {
        struct Count(u64);
        impl std::io::Write for Count {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self.0.saturating_add(bytes.len() as u64);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut count = Count(0);
        serde_json::to_writer(&mut count, self).map_err(|error| error.to_string())?;
        Ok(count.0)
    }
    fn plan_inputs(&self) -> SolvePlanInputs<'_> {
        SolvePlanInputs {
            workloads: &self.workloads,
            mesh: &self.mesh,
            revision: &self.revision,
            access: &self.artifact_access,
            authored_study_digest: self.authored_study_digest,
            frozen_project_digest: self.frozen_project_digest,
        }
    }
    /// Recompute canonical resource admission for a different thread allocation.
    /// This does not infer speedup or change the physical analysis definition.
    pub fn resource_plan_for_threads(
        &self,
        local_execution: LocalExecutionPlan,
    ) -> Result<FeaResourcePlan, String> {
        let preparation = sequence_preparation(
            &self.workloads,
            &self.mesh,
            self.revision.clone(),
            self.artifact_access.clone(),
            self.authored_study_digest,
            self.frozen_project_digest,
            Placement {
                local_execution,
                machine_memory_bytes: self.machine_memory_bytes,
            },
        )?;
        let mut coordinator =
            runmat_analysis_execution::StructuralSequenceCoordinator::new(preparation)
                .map_err(|e| e.to_string())?;
        let wave = coordinator
            .prepare_next_solve_wave()
            .map_err(|e| e.to_string())?;
        let [task] = wave.tasks.as_slice() else {
            return Err("Execution planning requires one admitted solve.".into());
        };
        Ok(task.prepared.resource_plan.clone())
    }
    pub fn execute(
        self,
        host: &mut SerialAnalysisHost,
        store: &mut MemoryObjectStore,
        clock: &dyn Fn() -> u64,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<StructuralStudyResult, String> {
        use futures::FutureExt;
        self.execute_with_capacity(host, store, clock, cancelled, None)
            .now_or_never()
            .ok_or_else(|| "Synchronous simulation requires an immediately ready host.".to_owned())?
            .map_err(|error| error.to_string())
    }

    pub async fn execute_with_capacity(
        self,
        host: &mut SerialAnalysisHost,
        store: &mut MemoryObjectStore,
        clock: &dyn Fn() -> u64,
        cancelled: &dyn Fn() -> bool,
        capacity: Option<&dyn AsyncCapacityOwner>,
    ) -> Result<StructuralStudyResult, PreparedSerialFeaExecutionError> {
        let sequence = sequence_preparation(
            &self.workloads,
            &self.mesh,
            self.revision,
            self.artifact_access,
            self.authored_study_digest,
            self.frozen_project_digest,
            Placement {
                local_execution: LocalExecutionPlan::serial(),
                machine_memory_bytes: self.machine_memory_bytes,
            },
        )?;
        runmat_analysis_fea::progress::emit_phase(
            "fea.run",
            runmat_analysis_fea::FeaProgressPhase::Solve,
            runmat_analysis_fea::FeaProgressStatus::Started,
            "Running simulation",
            None,
            None,
        );
        let completed = execute_serial_structural_sequence_with_capacity(
            sequence, store, host, clock, cancelled, capacity,
        )
        .await?;
        runmat_analysis_fea::progress::emit_phase(
            "fea.run",
            runmat_analysis_fea::FeaProgressPhase::Solve,
            runmat_analysis_fea::FeaProgressStatus::Completed,
            "Simulation solve completed",
            Some(1),
            Some(1),
        );
        let numerical = StructuralStudyNumericalResult::LinearSequence(
            StructuralStudySequenceNumericalResult {
                frozen_project_digest: self.frozen_project_digest,
                effective_memory_ceiling_bytes: self.machine_memory_bytes,
                steps: completed
                    .steps
                    .into_iter()
                    .map(|step| StructuralStudyStepNumericalResult {
                        step_id: step.step_id,
                        solver_result_identity: step.result_identity,
                        result: step.result,
                        numerical_receipt: step.numerical_receipt,
                        static_solve_path: Some(step.static_solve_path),
                    })
                    .collect(),
                states: completed
                    .states
                    .into_iter()
                    .map(|state| state.state)
                    .collect(),
            },
        );
        StructuralStudyResult::new(
            self.compiled,
            self.mesh,
            self.evidence,
            self.workloads,
            numerical,
        )
        .map_err(PreparedSerialFeaExecutionError::Result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_fea::planning::resource::ResourcePlanningError;

    #[test]
    fn a_solve_that_fits_no_path_in_the_browser_budget_is_a_coded_preparation_refusal() {
        let error =
            preparation_bridge_error(runmat_analysis_execution::AnalysisBridgeError::Resource(
                ResourcePlanningError::StaticSolveCeiling {
                    factorization_resident_bytes: Some(6_000_000_000),
                    multigrid_resident_bytes: 4_500_000_000,
                    multigrid_scratch_bytes: Some(1_000),
                    maximum: 4_290_000_000,
                },
            ));
        let FeaStudyPreparationError::ResourceLimit(limit) = &error else {
            panic!("expected a resource limit, got {error:?}");
        };
        assert_eq!(limit.term, super::super::BrowserResourceTerm::SolveMemory);
        assert_eq!(limit.required_bytes, Some(4_500_000_000));
        assert_eq!(limit.maximum_bytes, 4_290_000_000);
        let diagnostic = error.diagnostic();
        assert_eq!(
            diagnostic["identifier"],
            super::super::BROWSER_RESOURCE_LIMIT_CODE
        );
        assert_eq!(
            diagnostic["resourceLimit"]["requiredBytes"],
            4_500_000_000_u64
        );
        assert_eq!(error.to_string(), limit.message());
    }

    /// AGENTBUILD5: the 1 mm bracket's plan fit the 4.27 GB budget alone and the gate then
    /// refused it with the request text (4,328,600,066 > 4,267,253,392).
    #[test]
    fn a_plan_that_fits_only_without_its_request_is_replanned_within_the_rest() {
        let budget = 4_267_253_392;
        let request = 61_346_674 + 1_000;
        let factorization = 4_267_253_392;
        assert_eq!(
            replanning_ceiling(factorization, Some(budget), request, budget),
            Some(budget - request)
        );
        // A plan that fits with its request, or one already made within that ceiling.
        assert_eq!(
            replanning_ceiling(1_000_000_000, Some(budget), request, budget),
            None
        );
        assert_eq!(
            replanning_ceiling(factorization, Some(budget - request), request, budget),
            None
        );
        // The planner's refusal within the reduced ceiling is restated as the gate counts it.
        let limit = super::super::BrowserResourceLimit {
            term: super::super::BrowserResourceTerm::SolveMemory,
            required_bytes: Some(4_300_000_000),
            maximum_bytes: budget - request,
        }
        .with_request(request, budget);
        assert_eq!(limit.required_bytes, Some(4_300_000_000 + request));
        assert_eq!(limit.maximum_bytes, budget);
    }

    #[test]
    fn other_planning_failures_keep_their_message() {
        let error =
            preparation_bridge_error(runmat_analysis_execution::AnalysisBridgeError::Resource(
                ResourcePlanningError::Overflow("rows"),
            ));
        assert!(matches!(error, FeaStudyPreparationError::Invalid(_)));
    }
}
