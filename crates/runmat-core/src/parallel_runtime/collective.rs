use futures::channel::oneshot;
use runmat_execution::{
    CollectiveRequest, CollectiveResponse, CollectiveSequence, GangId, SpmdTaskContext,
};
use runmat_execution_runner::{CollectiveCompletion, CollectiveCoordinator, RunnerError};
use runmat_runtime::context::{RuntimeCollectiveService, RuntimeServiceFuture};
use runmat_runtime::RuntimeError;
use runmat_types::{CollectiveId, LabRank, ParallelRegionId};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PendingKey {
    gang: GangId,
    generation: u64,
    region: ParallelRegionId,
    collective: CollectiveId,
    sequence: CollectiveSequence,
    rank: LabRank,
}

impl PendingKey {
    fn from_request(request: &CollectiveRequest) -> Self {
        Self {
            gang: request.context.gang.id,
            generation: request.context.gang.generation,
            region: request.context.region,
            collective: request.id,
            sequence: request.sequence,
            rank: request.context.rank,
        }
    }

    fn from_completion(completion: &CollectiveCompletion) -> Self {
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

type PendingSender = oneshot::Sender<Result<CollectiveResponse, RuntimeError>>;

#[derive(Default)]
pub(super) struct SharedCollectives {
    coordinator: RefCell<CollectiveCoordinator>,
    pending: RefCell<BTreeMap<PendingKey, PendingSender>>,
    completed_ranks: RefCell<BTreeMap<(GangId, u64), std::collections::BTreeSet<LabRank>>>,
}

impl SharedCollectives {
    fn submit(
        &self,
        request: CollectiveRequest,
    ) -> Result<oneshot::Receiver<Result<CollectiveResponse, RuntimeError>>, RuntimeError> {
        let key = PendingKey::from_request(&request);
        let gang = request.context.gang.clone();
        let (sender, receiver) = oneshot::channel();
        if self.pending.borrow_mut().insert(key, sender).is_some() {
            return Err(runtime_error(
                "a collective request is already pending for this lab and sequence",
            ));
        }
        let completions = match self.coordinator.borrow_mut().submit(request) {
            Ok(completions) => completions,
            Err(error) => {
                self.pending.borrow_mut().remove(&key);
                return Err(runner_error(error));
            }
        };
        self.complete(completions);
        self.fail_deadlock(&gang);
        Ok(receiver)
    }

    pub(super) fn rank_finished(&self, gang: &runmat_execution::GangHandle, rank: LabRank) {
        let key = (gang.id, gang.generation);
        let completed_count = {
            let mut ranks = self.completed_ranks.borrow_mut();
            let completed = ranks.entry(key).or_default();
            completed.insert(rank);
            completed.len()
        };
        self.fail_deadlock(gang);
        if completed_count == gang.labs.0 as usize {
            self.completed_ranks.borrow_mut().remove(&key);
        }
    }

    pub(super) fn fail_gang(&self, gang: &runmat_execution::GangHandle, reason: &str) {
        let completions = self.coordinator.borrow_mut().fail_gang(gang, reason);
        self.completed_ranks
            .borrow_mut()
            .remove(&(gang.id, gang.generation));
        self.complete(completions);
    }

    fn fail_deadlock(&self, gang: &runmat_execution::GangHandle) {
        let completed = self
            .completed_ranks
            .borrow()
            .get(&(gang.id, gang.generation))
            .map(|ranks| ranks.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        let deadlock = self.coordinator.borrow().deadlock(gang, &completed);
        if let Some(deadlock) = deadlock {
            let completions = self
                .coordinator
                .borrow_mut()
                .fail_gang(&deadlock.gang, deadlock.to_string());
            self.completed_ranks
                .borrow_mut()
                .remove(&(gang.id, gang.generation));
            self.complete(completions);
        }
    }

    fn complete(&self, completions: Vec<CollectiveCompletion>) {
        let mut pending = self.pending.borrow_mut();
        for completion in completions {
            let key = PendingKey::from_completion(&completion);
            if let Some(sender) = pending.remove(&key) {
                let _ = sender.send(completion.result.map_err(runner_error));
            }
        }
    }
}

pub(super) struct LabCollectiveService {
    context: SpmdTaskContext,
    shared: Rc<SharedCollectives>,
    sequences: RefCell<BTreeMap<CollectiveId, u64>>,
}

impl LabCollectiveService {
    pub(super) fn new(context: SpmdTaskContext, shared: Rc<SharedCollectives>) -> Self {
        Self {
            context,
            shared,
            sequences: RefCell::new(BTreeMap::new()),
        }
    }
}

impl RuntimeCollectiveService for LabCollectiveService {
    fn context(&self) -> &SpmdTaskContext {
        &self.context
    }

    fn next_sequence(&self, id: CollectiveId) -> Result<CollectiveSequence, RuntimeError> {
        if id.region != self.context.region {
            return Err(runtime_error(
                "collective identity does not belong to this lab execution context",
            ));
        }
        let mut sequences = self.sequences.borrow_mut();
        let sequence = sequences.entry(id).or_default();
        *sequence = sequence
            .checked_add(1)
            .ok_or_else(|| runtime_error("collective invocation sequence overflowed"))?;
        Ok(CollectiveSequence(*sequence))
    }

    fn execute(
        &self,
        request: CollectiveRequest,
    ) -> RuntimeServiceFuture<Result<CollectiveResponse, RuntimeError>> {
        if request.context != self.context {
            return Box::pin(async {
                Err(runtime_error(
                    "collective request does not belong to this lab execution context",
                ))
            });
        }
        let receiver = self.shared.submit(request);
        Box::pin(async move {
            receiver?
                .await
                .map_err(|_| runtime_error("collective coordination ended before completion"))?
        })
    }
}

fn runner_error(error: RunnerError) -> RuntimeError {
    runtime_error(error.to_string())
}

fn runtime_error(message: impl Into<String>) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error("RunMat:parallel:Collective", message.into())
}
