//! Canonical logical execution bundles and exact program artifacts.

pub mod archive;
pub mod bundle;
pub mod cache;
pub mod encryption;
mod error;
pub mod object;
pub mod program;

pub use bundle::{
    BuildResourceDeclaration, BundleCallable, BundleManifest, ExecutionBundle,
    ExecutionBundleBuilder, ProjectRevisionRecord, SourceReader,
};
pub use error::{ArtifactError, ArtifactResult};
pub use object::{LogicalObject, ObjectDescriptor, ObjectNamespace};
pub use program::{
    ExecutableForm, ProgramArtifact, ProgramArtifactId, ProgramBuildRecipe,
    ProgramExecutionDescriptor, ProgramExecutionInputs, ProgramExecutionRequest,
    ProgramExecutionResponse, ProgramRecipeId, MAX_PROGRAM_EXECUTION_ARGUMENTS,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
