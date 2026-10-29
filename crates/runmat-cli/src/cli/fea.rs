use clap::Args;

/// Selects one runnable root of a `.fea` document for `check`, `plan` and `run`.
#[derive(Args, Clone, Debug, Default)]
pub struct FeaRootArgs {
    /// Select a study declared by the root .fea module
    #[arg(long, conflicts_with = "sweep")]
    pub study: Option<String>,
    /// Select a sweep declared by the root .fea module
    #[arg(long, conflicts_with = "study")]
    pub sweep: Option<String>,
}

impl FeaRootArgs {
    pub fn is_empty(&self) -> bool {
        self.study.is_none() && self.sweep.is_none()
    }
}
