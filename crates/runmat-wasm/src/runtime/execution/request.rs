//! Browser-worker transport composition. Native Direct/Graph protocols keep
//! their existing sealed two-variant wire; the third form is confined to an
//! explicitly admitted browser worker and never masquerades as an artifact.

use runmat_execution_artifact::{GraphExecutionRequest, ProgramExecutionRequest};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum BrowserWorkerRequest {
    Direct(ProgramExecutionRequest),
    Graph(GraphExecutionRequest),
    OwnedBuiltin(runmat_execution::OwnedBuiltinWorkRequest),
}

impl BrowserWorkerRequest {
    pub(crate) fn validate(&self) -> Result<(), String> {
        match self {
            Self::Direct(request) => request.validate().map_err(|error| error.to_string()),
            Self::Graph(request) => {
                if request.project_runtime_policy.is_some() {
                    return Err("project-policy Graph V13 executes on native cluster workers; browser workers support local Graph execution only".into());
                }
                if request
                    .invocation_controls
                    .as_ref()
                    .is_some_and(|controls| controls.console.is_some())
                {
                    // This transport has no ordered console/receipt channel.
                    // Preserve the same explicit restriction as Direct work;
                    // schema decoding is not authority to discard a capture.
                    return Err("requested console capture requires an owning transport".into());
                }
                request.validate().map_err(|error| error.to_string())
            }
            Self::OwnedBuiltin(request) => {
                request.validate_wire().map_err(|error| error.to_string())
            }
        }
    }
}
