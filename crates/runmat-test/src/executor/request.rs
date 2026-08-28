use serde::{Deserialize, Serialize};

use crate::context::TestExecutionContext;
use crate::descriptor::ProcedureDescriptor;
use crate::lifecycle::{ExecutionPhase, FixtureScopeKey};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionRequest {
    pub context: TestExecutionContext,
    pub phase: ExecutionPhase,
    pub scope: FixtureScopeKey,
    pub procedure: ProcedureDescriptor,
}
