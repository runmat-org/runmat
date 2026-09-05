mod artifact;
mod identity;
mod native_object;
mod recipe;
mod request;
mod target;
mod transport;

pub use artifact::{ExecutableForm, ProgramArtifact, PROGRAM_ARTIFACT_SCHEMA_VERSION};
pub use identity::{ProgramArtifactId, ProgramRecipeId};
pub use native_object::{NativeObjectPayload, NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION};
pub use recipe::{ProgramBuildRecipe, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION};
pub use request::{
    ProgramExecutionDescriptor, ProgramExecutionRequest, ProgramExecutionResponse,
    MAX_PROGRAM_EXECUTION_ARGUMENTS, MAX_PROGRAM_EXECUTION_RESULT_OBJECTS,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
pub use target::{
    NativeTargetIdentity, ProgramTarget, ProgramTargetCohort, PROGRAM_TARGET_SCHEMA_VERSION,
};
pub use transport::{
    ProgramExecutionInputs, ProgramExecutionObject, ProgramExecutionOutputs,
    MAX_PROGRAM_EXECUTION_OBJECT_BYTES, MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES,
};
