//! Canonical logical execution bundles and exact program artifacts.

pub mod archive;
pub mod bundle;
pub mod cache;
pub mod encryption;
mod environment;
mod error;
#[cfg(not(target_arch = "wasm32"))]
mod materialized;
pub mod object;
pub mod program;
pub mod target_pack;
pub mod target_test;

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
    ArtifactAccess, ArtifactPublishedObjectSource, LogicalObject, ObjectDescriptor,
    ObjectNamespace, MAX_FOREIGN_POLICY_BYTES,
};
pub use program::{
    admit_program_execution_request_bytes, admit_remote_program_object_bytes,
    admit_worker_program_request_bytes, AdmittedGraphUnitBundle,
    ArchiveAdmittedGraphExecutionRequest, BorrowedProgramExecutionRequest, ExecutableForm,
    GraphExecutionRequest, GraphInvocationRoots, GraphResponseAuthority,
    InterpreterRegionSourcesV1, NativeObjectPayload, NativeTargetIdentity, ProgramArtifact,
    ProgramArtifactId, ProgramBuildRecipe, ProgramExecutionDescriptor, ProgramExecutionInputs,
    ProgramExecutionObject, ProgramExecutionObjectClosure, ProgramExecutionObjectLocation,
    ProgramExecutionOutputs, ProgramExecutionRequest, ProgramExecutionResponse, ProgramProductMode,
    ProgramRecipeId, ProgramTarget, ProgramTargetCohort, RemoteProgramObject,
    ReservedProgramExecutionRequest, ValidatedGraphExecutionRequest, WorkerProgramRequest,
    EXTERNALIZED_VALUE_MEDIA_TYPE, EXTERNALIZED_VALUE_SCHEMA, GRAPH_EXECUTION_REQUEST_SCHEMA_V10,
    GRAPH_EXECUTION_REQUEST_SCHEMA_V11, GRAPH_EXECUTION_REQUEST_SCHEMA_V12,
    MAX_PROGRAM_EXECUTION_ARGUMENTS, MAX_PROGRAM_EXECUTION_OBJECT_BYTES,
    MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES, MAX_PROGRAM_EXECUTION_RESULT_OBJECTS,
    NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION, PROGRAM_ARTIFACT_SCHEMA_VERSION,
    PROGRAM_BUILD_RECIPE_SCHEMA_VERSION, PROGRAM_EXECUTION_OBJECT_CLOSURE_MEDIA_TYPE,
    PROGRAM_EXECUTION_OBJECT_CLOSURE_SCHEMA_VERSION, PROGRAM_EXECUTION_OBJECT_CLOSURE_VALUE_SCHEMA,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V5, PROGRAM_EXECUTION_REQUEST_SCHEMA_V6,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V7, PROGRAM_EXECUTION_REQUEST_SCHEMA_V8,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION, PROGRAM_TARGET_SCHEMA_VERSION,
    RESERVED_PROGRAM_REQUEST_SCHEMA_VERSION,
};
pub use runmat_execution::value::PublishedObjectCatalog;
