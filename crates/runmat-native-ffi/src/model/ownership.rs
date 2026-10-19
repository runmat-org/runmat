use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterDirection {
    Input,
    Output,
    InputOutput,
}

impl Default for ParameterDirection {
    fn default() -> Self {
        Self::Input
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerOwnership {
    Borrowed,
    CallerOwned,
    LibraryOwned,
    Shared,
}

impl Default for PointerOwnership {
    fn default() -> Self {
        Self::Borrowed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerMutability {
    Const,
    Mutable,
}
