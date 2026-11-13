//! Narrow adapter from deterministic meshing artifacts to shared execution infrastructure.
//!
//! Meshing owns logical records, stage identity, and manifest closure. This crate only maps that
//! immutable closure onto execution artifact objects and, in later modules, execution workloads.

/// Actual package build configuration used for internal artifact reuse.
#[doc(hidden)]
pub const BUILD_CONFIGURATION: &str = env!("RUNMAT_BUILD_CONFIGURATION");

pub(crate) use runmat_execution_artifact::ArtifactAccess;

mod accounting;
mod budget;
mod checkpoint_clock;
mod checkpoint_diagnostic;
mod controlled_streams;
mod curve_join_kernel;
mod curve_kernel;
mod curve_refinement_kernel;
mod diagnostic;
mod dispatcher;
mod error;
mod evaluator_provider;
mod evidence;
mod evidence_objects;
mod exact_dag;
mod exact_dag_run;
mod faceted_geometry_objects;
mod geometry_control;
mod geometry_failure;
mod geometry_metric;
mod geometry_objects;
mod host;
mod object_support;
mod objects;
mod publication;
mod publication_digest;
mod response;
mod reuse;
pub use reuse::{
    meshing_build_configuration, meshing_stage_reuse_key, rebind_committed_meshing_stage,
    AcceptedMeshingStageReceipt,
};
mod serial;
mod solver_projection_kernel;
mod solver_terminal_kernel;
mod stage_evidence_object;
mod subtask_cache;
mod surface_join_kernel;
mod surface_kernel;
pub use subtask_cache::diagnostic_chunks::{read_volume_diagnostic, VolumeDiagnosticRecord};
pub use subtask_cache::{
    MeshingSubtaskCacheEntry, MeshingSubtaskCacheError, MeshingSubtaskCacheStore,
};
mod task;
mod volume_kernel;

#[cfg(test)]
mod faceted_geometry_object_tests;
#[cfg(test)]
mod geometry_object_tests;
#[cfg(test)]
mod host_tests;
#[cfg(test)]
mod publication_tests;
#[cfg(test)]
mod serial_tests;
#[cfg(test)]
mod task_tests;
#[cfg(test)]
mod tests;

pub use budget::{
    MeshingProgressSink, MeshingStageCheckpoint, MeshingStageControl, NoopMeshingProgress,
};
pub use curve_join_kernel::ExactCurveJoinKernel;
pub use curve_kernel::ExactCurveStageKernel;
pub use curve_refinement_kernel::ExactCurveRefinementKernel;
pub use dispatcher::MeshingKernelDispatcher;
pub use error::{MeshingExecutionError, MeshingExecutionResult};
pub use evaluator_provider::{
    ExactMeshingEvaluatorProvider, ExactMeshingGeometryEvaluation, PortableMeshingEvaluatorProvider,
};
pub use evidence::{assemble_meshing_evidence, MeshingEvidenceContext};
pub use evidence_objects::{
    import_evidence_input, import_evidence_objects, prepare_evidence_input,
    prepare_evidence_objects, EvidenceObjectRoot, PreparedEvidenceInput, PreparedEvidenceObjects,
};
pub use exact_dag::{
    ExactCurvePassPlan, ExactMeshingDagPlanner, ExactSurfacePassPlan, PlannedMeshingStage,
};
pub use exact_dag_run::{
    execute_exact_meshing_dag, execute_exact_meshing_dag_async, import_exact_meshing_result,
    ExactMeshingDagExecutor, ExactMeshingDagRun, ExactMeshingDagRunError, ExactMeshingDagRunResult,
    MeshingRunEvidenceContext, MeshingStageExecutionError, SerialExactMeshingExecutor,
};
pub use faceted_geometry_objects::{
    import_faceted_geometry_input, import_faceted_geometry_objects, prepare_faceted_geometry_input,
    prepare_faceted_geometry_objects, FacetedGeometryObjectRoot, PreparedFacetedGeometryInput,
    PreparedFacetedGeometryObjects,
};
pub use geometry_control::{GeometryEvaluationUsage, MeshingGeometryEvaluationControl};
pub use geometry_objects::{
    import_exact_geometry_input, import_exact_geometry_objects, prepare_exact_geometry_input,
    prepare_exact_geometry_objects, prepare_exact_geometry_result, ExactGeometryObjectRoot,
    PreparedExactGeometryInput, PreparedExactGeometryObjects,
};
pub use host::{
    prepare_meshing_host_program, MeshingHostProgram, MeshingHostWorkload,
    PreparedMeshingHostInvocation, MESHING_HOST_ABI, MESHING_HOST_EXECUTION_MODE,
    MESHING_HOST_INVOCATION_MEDIA_TYPE, MESHING_HOST_INVOCATION_VALUE_SCHEMA,
    MESHING_HOST_TARGET_PROFILE, MESHING_HOST_WORKLOAD_SCHEMA_VERSION,
};
pub use objects::{
    import_stage_objects, prepare_stage_objects, MeshingStageObjectRoot,
    PreparedMeshingStageObjects, MESHING_RESULT_IDENTITY_MEDIA_TYPE,
    MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};
pub use publication::{
    import_result_publication, prepare_result_publication, PreparedMeshingResultPublication,
};
pub use response::{MeshingHostResponse, MESHING_HOST_RESPONSE_SCHEMA_VERSION};
pub use serial::{
    execute_serial_stage, CompletedMeshingStage, MeshingSerialExecutionError,
    MeshingStageInvocation, MeshingStageKernel, PreparedMeshingInput, ValidatedMeshingStageOutput,
};
pub use solver_projection_kernel::ExactSolverProjectionKernel;
pub use solver_terminal_kernel::{
    SolverPublicationKernel, SolverSerializationKernel, SolverValidationKernel,
};
pub use stage_evidence_object::{
    import_stage_evidence_observation, STAGE_EVIDENCE_MEDIA_TYPE, STAGE_EVIDENCE_VALUE_SCHEMA,
};
pub use surface_join_kernel::ExactSurfaceJoinKernel;
pub use surface_kernel::ExactSurfacePartitionKernel;
pub use task::{
    build_task_submission, exact_meshing_worker_capabilities,
    exact_meshing_worker_capability_inventory, meshing_task_id, MeshingExecutionContext,
    MeshingTaskEffectPolicy, EXACT_CAD_ADMISSION_CAPABILITY, MESHING_EXECUTION_CALLABLE_OWNER,
};
pub use volume_kernel::ExactVolumeKernel;

mod codec_control;
