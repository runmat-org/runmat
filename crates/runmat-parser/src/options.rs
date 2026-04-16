use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompatMode {
    #[default]
    Matlab,
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
