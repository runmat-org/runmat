mod fixture;
mod parameter;
mod procedure;
mod requirement;
mod selector;
mod source;

pub use fixture::{FixtureDescriptor, FixtureScope};
pub use parameter::ParameterDescriptor;
pub use procedure::{ProcedureDescriptor, ProcedureKind};
pub use requirement::{ResourceRequirement, TestCapability, TestRequirements};
pub use selector::TestSelector;
pub use source::{SourceDescriptor, SourceSpan};

use serde::{Deserialize, Serialize};

use crate::identity::{FixtureGroupId, SuiteId, TestId};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TestDescriptor {
    pub id: TestId,
    pub suite_id: SuiteId,
    pub fixture_group_id: FixtureGroupId,
    pub display_name: String,
    pub procedure: ProcedureDescriptor,
    #[serde(default)]
    pub parameters: Vec<ParameterDescriptor>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub requirements: TestRequirements,
}
