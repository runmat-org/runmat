use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalDisposition {
    Passed,
    Failed,
    Filtered,
    Cancelled,
    TimedOut,
    Crashed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResultState {
    pub failed: bool,
    pub incomplete: bool,
    pub disposition: TerminalDisposition,
}

impl ResultState {
    pub const PASSED: Self = Self {
        failed: false,
        incomplete: false,
        disposition: TerminalDisposition::Passed,
    };

    pub fn is_success(self) -> bool {
        self == Self::PASSED
    }
}
