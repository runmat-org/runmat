//! Canonical logical execution bundles and exact program artifacts.

pub mod archive;
pub mod bundle;
pub mod cache;
pub mod encryption;
mod error;
#[cfg(not(target_arch = "wasm32"))]
mod materialized;
pub mod object;
pub mod program;
pub mod target_pack;

pub use bundle::{
    BuildResourceDeclaration, BundleCallable, BundleCodeClosure, BundleManifest,
    CompiledPackageClosure, ExecutionBundle, ExecutionBundleBuilder, ForeignArtifactClosure,
    PreparedForeignParentView, ProjectRevisionRecord, SelectedForeignProjection,
    SelectedPreparedForeignMaterial, SourceReader, EXECUTION_BUNDLE_SCHEMA_VERSION,
};
pub use error::{ArtifactError, ArtifactResult};
#[cfg(not(target_arch = "wasm32"))]
pub use materialized::{
    PrivateArtifactLease, PrivateArtifactPath, PrivateArtifactRoot,
    SELECTED_GRAPH_ATTEMPT_SCRATCH_PREFIX,
};
pub use object::{
    capture_typed_foreign_policy, decode_typed_foreign_policy, foreign_policy_descriptor,
    foreign_policy_object, foreign_policy_object_from_json_value, foreign_policy_ref,
    LogicalObject, ObjectDescriptor, ObjectNamespace, MAX_FOREIGN_POLICY_BYTES,
};
pub use program::{
    admit_program_execution_request_bytes, admit_remote_program_object_bytes,
    admit_worker_program_request_bytes, AdmittedGraphUnitBundle,
    ArchiveAdmittedGraphExecutionRequest, ExecutableForm, GraphExecutionRequest,
    GraphInvocationRoots, GraphResponseAuthority, InterpreterRegionSourcesV1, NativeObjectPayload,
    NativeTargetIdentity, ProgramArtifact, ProgramArtifactId, ProgramBuildRecipe,
    ProgramExecutionDescriptor, ProgramExecutionInputs, ProgramExecutionRequest,
    ProgramExecutionResponse, ProgramRecipeId, ProgramTarget, ProgramTargetCohort,
    RemoteProgramObject, ValidatedGraphExecutionRequest, WorkerProgramRequest,
    GRAPH_EXECUTION_REQUEST_SCHEMA_V10, GRAPH_EXECUTION_REQUEST_SCHEMA_V11,
    MAX_PROGRAM_EXECUTION_ARGUMENTS, NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION,
    PROGRAM_ARTIFACT_SCHEMA_VERSION, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V5, PROGRAM_EXECUTION_REQUEST_SCHEMA_V6,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V7, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
    PROGRAM_TARGET_SCHEMA_VERSION,
};
