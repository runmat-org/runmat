//! Narrow adapter from deterministic meshing artifacts to shared execution infrastructure.
//!
//! Meshing owns logical records, stage identity, and manifest closure. This crate only maps that
//! immutable closure onto execution artifact objects and, in later modules, execution workloads.

mod error;
mod objects;
mod publication;
mod task;

#[cfg(test)]
mod publication_tests;
#[cfg(test)]
mod task_tests;
#[cfg(test)]
mod tests;

pub use error::{MeshingExecutionError, MeshingExecutionResult};
pub use objects::{
    import_stage_objects, prepare_stage_objects, MeshingStageObjectRoot,
    PreparedMeshingStageObjects, MESHING_RESULT_IDENTITY_MEDIA_TYPE,
    MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};
pub use publication::{
    import_result_publication, prepare_result_publication, MeshingArtifactAccess,
    PreparedMeshingResultPublication,
};
pub use task::{
    build_task_submission, MeshingExecutionContext, MeshingTaskEffectPolicy,
    MESHING_EXECUTION_CALLABLE_OWNER,
};
