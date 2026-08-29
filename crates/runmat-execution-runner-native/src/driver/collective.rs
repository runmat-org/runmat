use std::collections::BTreeMap;
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
}

#[derive(Default)]
pub(super) struct ProcessCollectiveBroker {
    state: Mutex<BrokerState>,
    ready: Condvar,
}

impl ProcessCollectiveBroker {
    pub(super) fn execute(
        &self,
        request: CollectiveRequest,
        task: &TaskCompletion,
    ) -> Result<CollectiveResponse, String> {
        let key = CompletionKey::for_request(&request);
        let gang = request.context.gang.clone();
        let mut state = self.state.lock().expect("collective broker poisoned");
        match state.coordinator.submit(request) {
            Ok(completions) => Self::record(&mut state, completions),
            Err(error) => {
                let completions = state.coordinator.fail_gang(&gang, error.to_string());
                Self::record(&mut state, completions);
                self.ready.notify_all();
                return Err(error.to_string());
            }
        }
        self.ready.notify_all();
        loop {
            if let Some(result) = state.completions.remove(&key) {
                return result;
            }
            if task.is_cancelled() {
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

    pub(super) fn fail_gang(&self, gang: &GangHandle, reason: impl Into<String>) {
        let mut state = self.state.lock().expect("collective broker poisoned");
        let completions = state.coordinator.fail_gang(gang, reason);
        Self::record(&mut state, completions);
        self.ready.notify_all();
    }

    fn record(state: &mut BrokerState, completions: Vec<CollectiveCompletion>) {
        for completion in completions {
            state.completions.insert(
                CompletionKey::for_completion(&completion),
                completion.result.map_err(|error| error.to_string()),
            );
        }
    }
}
