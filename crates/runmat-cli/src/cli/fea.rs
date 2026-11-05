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

/// How `snapshot` and `render-snapshot` draw their posters and scenes. Context geometry is drawn
/// in every view's scene, so the live viewer shows it too.
#[derive(Args, Clone, Debug, Default)]
pub struct SnapshotPresentationArgs {
    /// Draw this CAD file around the result as translucent context geometry (repeatable).
    /// Append `@OPACITY` (0 to 1, default 0.12) to set its surface opacity. Context geometry
    /// is drawn as shape only and carries no result; each view records its files.
    #[arg(long = "context", value_name = "FILE[@OPACITY]")]
    pub context: Vec<String>,
    /// Length unit of the context files' coordinates
    #[arg(long, default_value = "m", value_name = "m|mm|in")]
    pub context_units: String,
    /// Caption drawn as each view's title, stating what was solved
    #[arg(long)]
    pub caption: Option<String>,
    /// Poster size in pixels
    #[arg(long, default_value = "1200x750", value_name = "WIDTHxHEIGHT")]
    pub poster_size: String,
    /// Poster camera direction as MATLAB `view(az, el)` in degrees (default: the standard
    /// poster view, about view(19, 23))
    #[arg(long, value_name = "AZ,EL", allow_hyphen_values = true)]
    pub view_angle: Option<String>,
}
