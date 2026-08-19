//! Narrow adapter from deterministic meshing artifacts to shared execution infrastructure.
//!
//! Meshing owns logical records, stage identity, and manifest closure. This crate only maps that
//! immutable closure onto execution artifact objects and, in later modules, execution workloads.

mod accounting;
mod budget;
mod curve_join_kernel;
mod curve_kernel;
mod curve_refinement_kernel;
mod diagnostic;
mod dispatcher;
mod error;
mod faceted_geometry_objects;
mod geometry_control;
mod geometry_objects;
mod host;
mod object_support;
mod objects;
mod publication;
mod response;
mod serial;
mod surface_dag;
mod surface_join_kernel;
mod surface_kernel;
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
pub use curve_kernel::{
    ExactCurveEvaluatorProvider, ExactCurveGeometryEvaluation, ExactCurveStageKernel,
    PortableCurveEvaluatorProvider,
};
pub use curve_refinement_kernel::ExactCurveRefinementKernel;
pub use dispatcher::MeshingKernelDispatcher;
pub use error::{MeshingExecutionError, MeshingExecutionResult};
pub use faceted_geometry_objects::{
    import_faceted_geometry_input, import_faceted_geometry_objects, prepare_faceted_geometry_input,
    prepare_faceted_geometry_objects, FacetedGeometryObjectRoot, PreparedFacetedGeometryInput,
    PreparedFacetedGeometryObjects,
};
pub use geometry_control::{GeometryEvaluationUsage, MeshingGeometryEvaluationControl};
pub use geometry_objects::{
    import_exact_geometry_input, import_exact_geometry_objects, prepare_exact_geometry_input,
    prepare_exact_geometry_objects, ExactGeometryObjectRoot, PreparedExactGeometryInput,
    PreparedExactGeometryObjects,
};
pub use host::{
    MeshingHostWorkload, MESHING_HOST_EXECUTION_MODE, MESHING_HOST_TARGET_PROFILE,
    MESHING_HOST_WORKLOAD_SCHEMA_VERSION,
};
pub use objects::{
    import_stage_objects, prepare_stage_objects, MeshingStageObjectRoot,
    PreparedMeshingStageObjects, MESHING_RESULT_IDENTITY_MEDIA_TYPE,
    MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};
pub use publication::{
    import_result_publication, prepare_result_publication, MeshingArtifactAccess,
    PreparedMeshingResultPublication,
};
pub use response::{MeshingHostResponse, MESHING_HOST_RESPONSE_SCHEMA_VERSION};
pub use serial::{
    execute_serial_stage, CompletedMeshingStage, MeshingSerialExecutionError,
    MeshingStageInvocation, MeshingStageKernel, PreparedMeshingInput, ValidatedMeshingStageOutput,
};
pub use surface_dag::{ExactSurfaceDagPlanner, ExactSurfacePassPlan, PlannedMeshingStage};
pub use surface_join_kernel::ExactSurfaceJoinKernel;
pub use surface_kernel::ExactSurfacePartitionKernel;
pub use task::{
    build_task_submission, MeshingExecutionContext, MeshingTaskEffectPolicy,
    MESHING_EXECUTION_CALLABLE_OWNER,
};
pub use volume_kernel::ExactVolumeKernel;
