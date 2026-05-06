use runmat_hir::{EnvironmentEffect, WorkspaceEffect};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MirEffects {
    pub workspace: Vec<WorkspaceEffect>,
    pub environment: Vec<EnvironmentEffect>,
}
