use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    RunSetup,
    SuiteSetup,
    ClassSetup,
    TestSetup,
    TestBody,
    DynamicTeardown,
    TestTeardown,
    ClassTeardown,
    SuiteTeardown,
    RunTeardown,
}

impl ExecutionPhase {
    pub fn is_teardown(self) -> bool {
        matches!(
            self,
            Self::DynamicTeardown
                | Self::TestTeardown
                | Self::ClassTeardown
                | Self::SuiteTeardown
                | Self::RunTeardown
        )
    }
}
