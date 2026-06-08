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
        // Figure scene payloads can be large for dense 3D subplots (e.g. surf + scatter3 in a
        // 2x1 grid). Keep this aligned with desktop replay object budgeting defaults so valid
        // scenes persist/rehydrate instead of being dropped at export time.
        max_scene_payload_bytes: 16 * 1024 * 1024,
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
