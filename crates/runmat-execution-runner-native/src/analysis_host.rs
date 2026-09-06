//! Analysis-capable native worker seam; scheduling and object movement remain generic.

use runmat_analysis_execution::{execute_serial_analysis, AnalysisHostResponse, AnalysisKernel};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};

use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeAnalysisHostLimits {
    pub chunk_bytes: usize,
    pub inventory: ObjectInventoryLimits,
}

impl NativeAnalysisHostLimits {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        if self.chunk_bytes == 0
            || self.inventory.max_objects == 0
            || self.inventory.max_object_bytes == 0
            || self.inventory.max_total_bytes == 0
        {
            return Err(NativeExecutionError::Configuration(
                "native analysis host limits must all be non-zero".into(),
            ));
        }
        Ok(())
    }
}

impl Default for NativeAnalysisHostLimits {
    fn default() -> Self {
        Self {
            chunk_bytes: runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory: ObjectInventoryLimits::default(),
        }
    }
}

pub fn execute_analysis_program_request<
    S: runmat_execution_artifact::cache::CacheImport + runmat_execution_artifact::cache::CacheExport,
>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    kernel: &dyn AnalysisKernel,
    limits: NativeAnalysisHostLimits,
) -> ProgramExecutionResponse {
    if let Err(error) = limits.validate() {
        return failure(error.to_string());
    }
    let host =
        match runmat_analysis_execution::AnalysisHostWorkload::from_program_request(request, store)
        {
            Ok(host) => host,
            Err(error) => return failure(error.to_string()),
        };
    match execute_serial_analysis(request, store, kernel, limits.chunk_bytes, limits.inventory)
        .and_then(|completed| AnalysisHostResponse::completed(&host, &completed))
    {
        Ok(response) => {
            let response = response.program_response();
            match response.validate_against(request) {
                Ok(()) => response,
                Err(error) => failure(error.to_string()),
            }
        }
        Err(error) => failure(error.to_string()),
    }
}

fn failure(message: impl Into<String>) -> ProgramExecutionResponse {
    ProgramExecutionResponse::Failure {
        message: message.into(),
    }
}
