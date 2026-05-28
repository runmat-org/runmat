use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CompatMode {
    #[serde(rename = "runmat")]
    RunMat,
    #[default]
    #[serde(rename = "matlab")]
    Matlab,
    #[serde(rename = "strict")]
    Strict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParserOptions {
    #[serde(default)]
    pub compat_mode: CompatMode,
}

impl Default for ParserOptions {
    fn default() -> Self {
        Self {
            compat_mode: CompatMode::Matlab,
        }
    }
}

impl ParserOptions {
    pub fn new(compat_mode: CompatMode) -> Self {
        Self { compat_mode }
    }
}

impl CompatMode {
    /// Whether semantic lowering should allow RunMat-only extension syntax.
    pub fn allows_runmat_extensions(self) -> bool {
        !matches!(self, Self::Strict)
    }
}
