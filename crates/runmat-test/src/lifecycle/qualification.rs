use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationKind {
    VerificationFailed,
    AssumptionFailed,
    AssertionFailed,
    FatalAssertionFailed,
}

impl QualificationKind {
    pub fn aborts_test(self) -> bool {
        !matches!(self, Self::VerificationFailed)
    }

    pub fn aborts_run(self) -> bool {
        matches!(self, Self::FatalAssertionFailed)
    }
}
