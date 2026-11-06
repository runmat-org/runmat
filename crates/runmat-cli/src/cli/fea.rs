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
#[derive(Args, Clone, Debug)]
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
    /// What the poster camera frames: `all` (the result and its context geometry) or `result`
    /// (the solved mesh alone, with context geometry reaching past the frame)
    #[arg(long, default_value = "all", value_name = "all|result")]
    pub frame: String,
    /// Frame this multiple of the framed bounds (1 or more), centered on them, so a `result`
    /// frame keeps some of the surrounding context in view
    #[arg(long, default_value_t = 1.0, value_name = "FACTOR")]
    pub frame_padding: f32,
    /// Leave the floor grid out of posters and scenes
    #[arg(long)]
    pub no_grid: bool,
    /// Leave the corner orientation triad out of posters and scenes
    #[arg(long)]
    pub no_triad: bool,
    /// Draw poster text (caption, legend, triad) this many times its standard size; a poster
    /// shown small, such as a website hero on a phone, uses about 2
    #[arg(long, default_value_t = 1.0, value_name = "FACTOR")]
    pub text_scale: f32,
    /// Poster theme: `dark` or `light`, the app's dark or light plot theme for the background,
    /// grid, text, legend and triad. Scenes are the same in both; only the posters differ.
    #[arg(long, default_value = "dark", value_name = "dark|light")]
    pub theme: String,
    /// Largest scene file a view may write, in bytes (default 16 MiB, the app's scene file
    /// limit). A larger view is drawn lighter until it fits: corner triangles for a quadratic
    /// surface, a coarser section cap. Each view records what that leaves out.
    #[arg(long, default_value_t = 16 * 1024 * 1024, value_name = "BYTES")]
    pub max_scene_bytes: usize,
}
