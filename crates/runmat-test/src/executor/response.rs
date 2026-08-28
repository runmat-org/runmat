use serde::{Deserialize, Serialize};

use crate::context::TestCommand;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionResponse {
    #[serde(default)]
    pub commands: Vec<TestCommand>,
    #[serde(default)]
    pub output: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "message")]
pub enum ExecutionFault {
    Uncaught(String),
    TimedOut(String),
    Cancelled(String),
    WorkerCrashed(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionFailure {
    pub fault: ExecutionFault,
    pub partial: ExecutionResponse,
}

impl From<ExecutionFault> for ExecutionFailure {
    fn from(fault: ExecutionFault) -> Self {
        Self {
            fault,
            partial: ExecutionResponse::default(),
        }
    }
}
