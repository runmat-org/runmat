use runmat_test::identity::{FixtureGroupId, TestId};
use runmat_test::plan::TestPlan;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixtureGroupJob {
    pub group_id: FixtureGroupId,
    pub tests: Vec<TestId>,
}

pub fn fixture_group_jobs(plan: &TestPlan) -> Vec<FixtureGroupJob> {
    plan.suites
        .iter()
        .flat_map(|suite| suite.fixture_groups.iter())
        .filter(|group| !group.tests.is_empty())
        .map(|group| FixtureGroupJob {
            group_id: group.id.clone(),
            tests: group.tests.iter().map(|test| test.id.clone()).collect(),
        })
        .collect()
}
