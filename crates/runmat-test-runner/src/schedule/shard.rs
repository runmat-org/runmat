use runmat_test::identity::TestId;
use runmat_test::plan::shard_for;

use crate::{RunnerError, RunnerResult};

pub fn selected_for_shard(
    test_id: &TestId,
    shard_index: Option<u32>,
    shard_count: Option<u32>,
) -> RunnerResult<bool> {
    match (shard_index, shard_count) {
        (None, None) => Ok(true),
        (Some(index), Some(count)) if count > 0 && index < count => {
            Ok(shard_for(test_id, count) == Some(index))
        }
        _ => Err(RunnerError::InvalidConfiguration(
            "shard_index and shard_count must be provided together with index < count".into(),
        )),
    }
}
