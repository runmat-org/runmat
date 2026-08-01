use serde::{Deserialize, Serialize};

use crate::identity::{FixtureGroupId, FixtureId};

use super::ProcedureDescriptor;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FixtureDescriptor {
    pub id: FixtureId,
    pub group_id: FixtureGroupId,
    pub display_name: String,
    pub scope: FixtureScope,
    pub setup: Option<ProcedureDescriptor>,
    pub teardown: Option<ProcedureDescriptor>,
    #[serde(default)]
    pub dependencies: Vec<FixtureId>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureScope {
    Run,
    Suite,
    Class,
    Test,
}
