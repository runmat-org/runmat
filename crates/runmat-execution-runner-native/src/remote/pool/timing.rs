//! Host-observed task intervals retained across exact committed-task replay.
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use runmat_execution::TaskId;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) enum TaskWallInterval {
    Started(Instant),
    /// Start is node-signed and checkpointed; only that same live owner may end it.
    ObservedPending([u8; 16]),
    Completed(u64),
}

impl TaskWallInterval {
    pub(super) fn finish(&mut self) -> NativeExecutionResult<u64> {
        match self {
            Self::Started(started) => {
                let elapsed = u64::try_from(started.elapsed().as_nanos().div_ceil(1_000_000))
                    .map_err(|_| {
                        NativeExecutionError::Protocol(
                            "original task wall interval overflowed".into(),
                        )
                    })?;
                if elapsed == 0 {
                    return Err(NativeExecutionError::Protocol(
                        "original task wall interval was not observed".into(),
                    ));
                }
                *self = Self::Completed(elapsed);
                Ok(elapsed)
            }
            Self::Completed(elapsed) => Ok(*elapsed),
            Self::ObservedPending(_) => Err(NativeExecutionError::Protocol(
                "original node observer has not closed this task wall interval".into(),
            )),
        }
    }
}

pub(super) fn completed(intervals: &HashMap<TaskId, TaskWallInterval>) -> BTreeMap<TaskId, u64> {
    intervals
        .iter()
        .filter_map(|(task, interval)| match interval {
            TaskWallInterval::Completed(elapsed) => Some((*task, *elapsed)),
            TaskWallInterval::Started(_) => None,
            TaskWallInterval::ObservedPending(_) => None,
        })
        .collect()
}
