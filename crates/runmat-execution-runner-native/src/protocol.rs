use serde::{Deserialize, Serialize};

pub use runmat_execution_artifact::{
    ProgramExecutionRequest as WorkerRequest, ProgramExecutionResponse as WorkerResponse,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredProgram {
    pub recipe: runmat_execution_artifact::ProgramBuildRecipe,
    pub artifact: runmat_execution_artifact::ProgramArtifact,
}
