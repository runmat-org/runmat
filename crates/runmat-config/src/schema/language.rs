use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LanguageConfig {
    /// Compatibility mode for MATLAB command syntax and legacy behaviors.
    /// Default: "runmat" (RunMat identifiers with MATLAB-compatible command syntax).
    /// "strict" disables command syntax; require `hold(\"on\")` style.
    #[serde(default)]
    pub compat: LanguageCompatMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum LanguageCompatMode {
    #[serde(rename = "runmat", alias = "run-mat")]
    #[value(name = "runmat")]
    RunMat,
    Matlab,
    Strict,
}

impl Default for LanguageCompatMode {
    fn default() -> Self {
        Self::RunMat
    }
}

pub fn error_namespace_for_language_compat(mode: LanguageCompatMode) -> &'static str {
    match mode {
        LanguageCompatMode::Matlab => "MATLAB",
        LanguageCompatMode::RunMat | LanguageCompatMode::Strict => "RunMat",
    }
}
