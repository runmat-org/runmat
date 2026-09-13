//! Canonical logical execution bundles and exact program artifacts.

pub mod archive;
pub mod bundle;
pub mod cache;
pub mod encryption;
mod error;
pub mod object;
pub mod program;

pub use bundle::{
    BuildResourceDeclaration, BundleCallable, BundleCodeClosure, BundleManifest,
    CompiledPackageClosure, ExecutionBundle, ExecutionBundleBuilder, ProjectRevisionRecord,
    SourceReader, EXECUTION_BUNDLE_SCHEMA_VERSION,
};
pub use error::{ArtifactError, ArtifactResult};
pub use object::{ArtifactAccess, LogicalObject, ObjectDescriptor, ObjectNamespace};
pub use program::{
    ExecutableForm, NativeObjectPayload, NativeTargetIdentity, ProgramArtifact, ProgramArtifactId,
    ProgramBuildRecipe, ProgramExecutionDescriptor, ProgramExecutionInputs, ProgramExecutionObject,
    ProgramExecutionObjectClosure, ProgramExecutionObjectLocation, ProgramExecutionOutputs,
    ProgramExecutionRequest, ProgramExecutionResponse, ProgramRecipeId, ProgramTarget,
    ProgramTargetCohort, EXTERNALIZED_VALUE_MEDIA_TYPE, EXTERNALIZED_VALUE_SCHEMA,
    MAX_PROGRAM_EXECUTION_ARGUMENTS, MAX_PROGRAM_EXECUTION_OBJECT_BYTES,
    MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES, MAX_PROGRAM_EXECUTION_RESULT_OBJECTS,
    NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION, PROGRAM_ARTIFACT_SCHEMA_VERSION,
    PROGRAM_BUILD_RECIPE_SCHEMA_VERSION, PROGRAM_EXECUTION_OBJECT_CLOSURE_MEDIA_TYPE,
    PROGRAM_EXECUTION_OBJECT_CLOSURE_SCHEMA_VERSION, PROGRAM_EXECUTION_OBJECT_CLOSURE_VALUE_SCHEMA,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1, PROGRAM_TARGET_SCHEMA_VERSION,
};
