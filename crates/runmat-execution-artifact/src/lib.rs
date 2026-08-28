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
pub use object::{LogicalObject, ObjectDescriptor, ObjectNamespace};
pub use program::{
    ExecutableForm, NativeObjectPayload, NativeTargetIdentity, ProgramArtifact, ProgramArtifactId,
    ProgramBuildRecipe, ProgramExecutionDescriptor, ProgramExecutionInputs,
    ProgramExecutionRequest, ProgramExecutionResponse, ProgramRecipeId, ProgramTarget,
    ProgramTargetCohort, MAX_PROGRAM_EXECUTION_ARGUMENTS, NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION,
    PROGRAM_ARTIFACT_SCHEMA_VERSION, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1, PROGRAM_TARGET_SCHEMA_VERSION,
};
