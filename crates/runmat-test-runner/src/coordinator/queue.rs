use runmat_test::identity::{FixtureGroupId, TestId};
use runmat_test::plan::TestPlan;

use crate::schedule::{fixture_group_jobs, selected_for_shard};
use crate::RunnerResult;

use super::run::CoordinatorConfig;

#[derive(Clone, Debug)]
pub(super) struct GroupQueue {
    pub group_id: FixtureGroupId,
    pub tests: Vec<TestId>,
}

pub(super) fn build_queue(
    plan: &TestPlan,
    config: &CoordinatorConfig,
) -> RunnerResult<Vec<GroupQueue>> {
    fixture_group_jobs(plan)
        .into_iter()
        .map(|job| {
            let tests = job
                .tests
                .into_iter()
                .map(|test_id| {
                    selected_for_shard(&test_id, config.shard_index, config.shard_count)
                        .map(|selected| (test_id, selected))
                })
                .collect::<RunnerResult<Vec<_>>>()?
                .into_iter()
                .filter_map(|(test_id, selected)| selected.then_some(test_id))
                .collect::<Vec<_>>();
            Ok(GroupQueue {
                group_id: job.group_id,
                tests,
            })
        })
        .filter(|group| group.as_ref().is_ok_and(|group| !group.tests.is_empty()))
        .collect()
}
