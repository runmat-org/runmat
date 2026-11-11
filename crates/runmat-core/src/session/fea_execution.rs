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
}
impl std::fmt::Display for FeaStudyPreparationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequiresDesktop(reason) => f.write_str(reason),
            Self::Invalid(reason) => f.write_str(reason),
            Self::Meshing(failure) => write!(f, "Mesh preparation failed: {failure}"),
            Self::Admission(denied) => denied.fmt(f),
        }
    }
}
impl std::error::Error for FeaStudyPreparationError {}
impl FeaStudyPreparationError {
    pub fn diagnostic(&self) -> serde_json::Value {
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
}
impl CheckedFeaSessionSource {
    pub async fn prepare_serial_fea_study(
        &self,
        host: &impl FeaGeometryExecutionHost,
        authority: &str,
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
        let geometry_source = match &resolved.geometry.source {
            runmat_geometry_core::GeometryDefinitionSource::Constructed(_) => {
                PreparedGeometrySource::constructed()
            }
            runmat_geometry_core::GeometryDefinitionSource::Imported(imported) => {
                let logical =
                    runmat_runtime::analysis::LogicalSourcePath::new(imported.logical_path.clone())
                        .map_err(|e| e.to_string())?;
                let path =
                    runmat_runtime::analysis::physical_source_path(&self.source_root, &logical)
                        .map_err(|e| e.to_string())?;
                let bytes = self.read_source(&path).await?;
                if Digest::sha256(&bytes).bytes() != imported.content_digest.bytes() {
                    return Err("The geometry changed after checking. Refresh the simulation before running.".into());
                }
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| "Geometry filename is not valid UTF-8.".to_owned())?;
                PreparedGeometrySource::imported(name, &bytes, 4 << 20, &access)
                    .map_err(|e| e.to_string())?
            }
            _ => {
                return Err(FeaStudyPreparationError::RequiresDesktop(
                    "Run this generated geometry in the desktop app.",
                ))
            }
        };
        let geometry_request = runmat_analysis_execution::prepare_geometry_materialization_request(
            resolved.geometry.clone(),
            realized.geometry.bindings.clone(),
            &geometry_source,
            host.provider_profile().await?,
            Default::default(),
        )
        .map_err(|e| e.to_string())?;
        let program = runmat_analysis_execution::prepare_geometry_materialization_program(
            geometry_request.clone(),
            revision.clone(),
            &access,
            &geometry_source.roots,
        )
        .map_err(|e| e.to_string())?;
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
        let preparation = sequence_preparation(
            &workloads,
            &geometry.mesh,
            revision.clone(),
            access.clone(),
            authored_study_digest,
            frozen_project_digest,
            LocalExecutionPlan::serial(),
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
        Ok(PreparedSerialFeaStudy {
            compiled,
            mesh: geometry.mesh,
            evidence: geometry.meshing_evidence,
            workloads,
            resource_plan: task.prepared.resource_plan.clone(),
            revision,
            checked_revision: self.check_result().revision.0,
            authored_study_digest,
            frozen_project_digest,
            artifact_access: access,
        })
    }
}
fn preparation_bridge_error(
    error: runmat_analysis_execution::AnalysisBridgeError,
) -> FeaStudyPreparationError {
    match error.accounting_cause() {
        Some(cause) => FeaStudyPreparationError::Admission(
            runmat_runtime::analysis::fea_run_admission::FeaRunDenied::Accounting(*cause),
        ),
        None => FeaStudyPreparationError::Invalid(error.to_string()),
    }
}
fn sequence_preparation(
    workloads: &StructuralStudyWorkloads,
    mesh: &SolverMeshArtifact,
    revision: ProgramRevision,
    access: ArtifactAccess,
    authored_study_digest: Digest,
    frozen_project_digest: Option<Digest>,
    local_execution: LocalExecutionPlan,
) -> Result<StructuralSequencePreparation, String> {
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
                machine_memory_bytes: None,
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
            local_execution,
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
            LocalExecutionPlan::serial(),
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
                effective_memory_ceiling_bytes: None,
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
