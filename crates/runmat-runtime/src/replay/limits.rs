#[derive(Debug, Clone, Copy)]
pub struct ReplayLimits {
    pub max_scene_payload_bytes: usize,
    pub max_scene_plots: usize,
    pub max_workspace_payload_bytes: usize,
    pub max_workspace_mat_bytes: usize,
    pub max_workspace_variables: usize,
}

impl ReplayLimits {
    pub const DEFAULT: Self = Self {
        max_scene_payload_bytes: 4 * 1024 * 1024,
        max_scene_plots: 4096,
        max_workspace_payload_bytes: 32 * 1024 * 1024,
        max_workspace_mat_bytes: 24 * 1024 * 1024,
        max_workspace_variables: 2048,
    };
}

impl Default for ReplayLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}
