use serde::{Deserialize, Serialize};

use crate::descriptor::{FixtureDescriptor, TestDescriptor};
use crate::identity::{FixtureGroupId, RunId, SuiteId};
use crate::version::TEST_PLAN_SCHEMA_VERSION;

use super::ProgramRevision;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TestPlan {
    pub schema_version: u16,
    pub run_id: RunId,
    pub program_revision: ProgramRevision,
    pub suites: Vec<SuitePlan>,
}

impl TestPlan {
    pub fn new(run_id: RunId, program_revision: ProgramRevision, suites: Vec<SuitePlan>) -> Self {
        Self {
            schema_version: TEST_PLAN_SCHEMA_VERSION,
            run_id,
            program_revision,
            suites,
        }
    }

    pub fn tests(&self) -> impl Iterator<Item = &TestDescriptor> {
        self.suites
            .iter()
            .flat_map(|suite| suite.fixture_groups.iter())
            .flat_map(|group| group.tests.iter())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SuitePlan {
    pub id: SuiteId,
    pub display_name: String,
    pub fixture_groups: Vec<FixtureGroupPlan>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FixtureGroupPlan {
    pub id: FixtureGroupId,
    pub fixtures: Vec<FixtureDescriptor>,
    pub tests: Vec<TestDescriptor>,
}
