use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use runmat_execution::{
    CollectiveRequest, CollectiveResponse, CollectiveSequence, GangHandle, GangId,
};
use runmat_execution_runner::{CollectiveCompletion, CollectiveCoordinator};
use runmat_types::{CollectiveId, LabRank, ParallelRegionId};

use super::TaskCompletion;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CompletionKey {
    gang: GangId,
    generation: u64,
    region: ParallelRegionId,
    collective: CollectiveId,
    sequence: CollectiveSequence,
    rank: LabRank,
}

impl CompletionKey {
    fn for_request(request: &CollectiveRequest) -> Self {
        Self {
            gang: request.context.gang.id,
            generation: request.context.gang.generation,
            region: request.context.region,
            collective: request.id,
            sequence: request.sequence,
            rank: request.context.rank,
        }
    }

    fn for_completion(completion: &CollectiveCompletion) -> Self {
        Self {
            gang: completion.context.gang.id,
            generation: completion.context.gang.generation,
            region: completion.context.region,
            collective: completion.id,
            sequence: completion.sequence,
            rank: completion.context.rank,
        }
    }
}

#[derive(Default)]
struct BrokerState {
    coordinator: CollectiveCoordinator,
    completions: BTreeMap<CompletionKey, Result<CollectiveResponse, String>>,
    completed_ranks: BTreeMap<(GangId, u64), BTreeSet<LabRank>>,
    failed_gangs: BTreeMap<(GangId, u64), String>,
}

#[derive(Default)]
pub(crate) struct ProcessCollectiveBroker {
    state: Mutex<BrokerState>,
    ready: Condvar,
}

impl ProcessCollectiveBroker {
    pub(super) fn execute(
        &self,
        request: CollectiveRequest,
        task: &TaskCompletion,
    ) -> Result<CollectiveResponse, String> {
        self.execute_with(request, || task.is_cancelled())
    }

    pub(crate) fn execute_remote(
        &self,
        request: CollectiveRequest,
    ) -> Result<CollectiveResponse, String> {
        self.execute_with(request, || false)
    }

    fn execute_with(
        &self,
        request: CollectiveRequest,
        is_cancelled: impl Fn() -> bool,
    ) -> Result<CollectiveResponse, String> {
        let key = CompletionKey::for_request(&request);
        let gang = request.context.gang.clone();
        let mut state = self.state.lock().expect("collective broker poisoned");
        if let Some(reason) = state.failed_gangs.get(&(gang.id, gang.generation)).cloned() {
            return Err(reason);
        }
        match state.coordinator.submit(request) {
            Ok(completions) => Self::record(&mut state, completions),
            Err(error) => {
                let reason = error.to_string();
                state
                    .failed_gangs
                    .insert((gang.id, gang.generation), reason.clone());
                let completions = state.coordinator.fail_gang(&gang, reason.clone());
                Self::record(&mut state, completions);
                self.ready.notify_all();
                return Err(reason);
            }
        }
        Self::fail_deadlock(&mut state, &gang);
        self.ready.notify_all();
        loop {
            if let Some(result) = state.completions.remove(&key) {
                return result;
            }
            if is_cancelled() {
                state
                    .failed_gangs
                    .insert((gang.id, gang.generation), "execution was cancelled".into());
                let completions = state
                    .coordinator
                    .fail_gang(&gang, "execution was cancelled");
                Self::record(&mut state, completions);
                self.ready.notify_all();
                if let Some(result) = state.completions.remove(&key) {
                    return result;
                }
                return Err("execution was cancelled".into());
            }
            let (next, _) = self
                .ready
                .wait_timeout(state, Duration::from_millis(10))
                .expect("collective broker poisoned while waiting");
            state = next;
        }
    }

    pub(crate) fn fail_gang(&self, gang: &GangHandle, reason: impl Into<String>) {
        let mut state = self.state.lock().expect("collective broker poisoned");
        let reason = reason.into();
        state
            .failed_gangs
            .insert((gang.id, gang.generation), reason.clone());
        let completions = state.coordinator.fail_gang(gang, reason);
        Self::record(&mut state, completions);
        state.completed_ranks.remove(&(gang.id, gang.generation));
        self.ready.notify_all();
    }

    pub(crate) fn rank_finished(&self, gang: &GangHandle, rank: LabRank) {
        let mut state = self.state.lock().expect("collective broker poisoned");
        let key = (gang.id, gang.generation);
        let completed = state.completed_ranks.entry(key).or_default();
        completed.insert(rank);
        let completed_ranks = completed.iter().copied().collect::<Vec<_>>();
        if let Some(deadlock) = state.coordinator.deadlock(gang, &completed_ranks) {
            let reason = deadlock.to_string();
            state
                .failed_gangs
                .insert((gang.id, gang.generation), reason.clone());
            let completions = state.coordinator.fail_gang(gang, reason);
            Self::record(&mut state, completions);
            state.completed_ranks.remove(&key);
            self.ready.notify_all();
        } else if completed_ranks.len() == gang.labs.0 as usize {
            state.completed_ranks.remove(&key);
        }
    }

    fn record(state: &mut BrokerState, completions: Vec<CollectiveCompletion>) {
        for completion in completions {
            state.completions.insert(
                CompletionKey::for_completion(&completion),
                completion.result.map_err(|error| error.to_string()),
            );
        }
    }

    fn fail_deadlock(state: &mut BrokerState, gang: &GangHandle) {
        let completed_ranks = state
            .completed_ranks
            .get(&(gang.id, gang.generation))
            .map(|ranks| ranks.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        if let Some(deadlock) = state.coordinator.deadlock(gang, &completed_ranks) {
            let reason = deadlock.to_string();
            state
                .failed_gangs
                .insert((gang.id, gang.generation), reason.clone());
            let completions = state.coordinator.fail_gang(gang, reason);
            Self::record(state, completions);
            state.completed_ranks.remove(&(gang.id, gang.generation));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use runmat_execution::{
        CollectiveInvocation, CollectiveMessageTag, CollectiveRequest, ExecutionScopeId,
        GangHandle, PoolHandle, PoolId, ReceiveSelection, SpmdTaskContext,
    };
    use runmat_types::{CollectiveId, LabCount, ProgramFunctionId, RegionId};

    use super::*;

    fn gang() -> GangHandle {
        let scope_id = ExecutionScopeId::derive(&[b"native-collective-deadlock"]);
        GangHandle {
            id: GangId::derive(&[b"gang"]),
            scope_id,
            generation: 1,
            pool: PoolHandle {
                id: PoolId::derive(&[b"pool"]),
                scope_id,
                generation: 1,
            },
            labs: LabCount(2),
        }
    }

    fn request(
        gang: &GangHandle,
        rank: u32,
        ordinal: u32,
        invocation: CollectiveInvocation,
    ) -> CollectiveRequest {
        let region = ParallelRegionId(RegionId {
            function: ProgramFunctionId(1),
            ordinal: 1,
        });
        CollectiveRequest {
            context: SpmdTaskContext {
                gang: gang.clone(),
                region,
                rank: LabRank(rank),
            },
            id: CollectiveId { region, ordinal },
            sequence: CollectiveSequence(1),
            invocation,
        }
    }

    #[test]
    fn mismatched_rounds_fail_without_a_wall_clock_timeout() {
        let broker = Arc::new(ProcessCollectiveBroker::default());
        let gang = gang();
        let first_broker = Arc::clone(&broker);
        let first_gang = gang.clone();
        let first = std::thread::spawn(move || {
            first_broker.execute(
                request(
                    &first_gang,
                    1,
                    1,
                    CollectiveInvocation::Receive {
                        selection: ReceiveSelection {
                            source: Some(LabRank(2)),
                            tag: Some(CollectiveMessageTag(7)),
                        },
                    },
                ),
                &TaskCompletion::new(),
            )
        });
        let second = broker.execute(
            request(&gang, 2, 2, CollectiveInvocation::Barrier),
            &TaskCompletion::new(),
        );
        assert!(second.unwrap_err().contains("SPMD collective deadlock"));
        assert!(first
            .join()
            .expect("blocked rank thread completes")
            .unwrap_err()
            .contains("SPMD collective deadlock"));
    }

    #[test]
    fn completed_peer_fences_a_receive_that_can_no_longer_progress() {
        let broker = Arc::new(ProcessCollectiveBroker::default());
        let gang = gang();
        let waiting_broker = Arc::clone(&broker);
        let waiting_gang = gang.clone();
        let waiting = std::thread::spawn(move || {
            waiting_broker.execute(
                request(
                    &waiting_gang,
                    1,
                    1,
                    CollectiveInvocation::Receive {
                        selection: ReceiveSelection {
                            source: Some(LabRank(2)),
                            tag: None,
                        },
                    },
                ),
                &TaskCompletion::new(),
            )
        });
        broker.rank_finished(&gang, LabRank(2));
        assert!(waiting
            .join()
            .expect("blocked rank thread completes")
            .unwrap_err()
            .contains("SPMD collective deadlock"));
    }

    #[test]
    fn failure_before_submission_fences_a_late_collective_request() {
        let broker = ProcessCollectiveBroker::default();
        let gang = gang();
        broker.fail_gang(&gang, "rank failed before its peer entered the collective");

        let failure = broker
            .execute_remote(request(&gang, 2, 1, CollectiveInvocation::Barrier))
            .unwrap_err();
        assert_eq!(
            failure,
            "rank failed before its peer entered the collective"
        );
    }
}
