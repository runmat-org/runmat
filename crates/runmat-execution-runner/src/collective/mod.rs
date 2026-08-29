use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_execution::value::ValuePayload;
use runmat_execution::{
    CollectiveInvocation, CollectiveMessageTag, CollectiveRequest, CollectiveResponse,
    CollectiveSequence, GangHandle, GangId, ReceiveSelection, SpmdTaskContext,
};
use runmat_types::{CollectiveId, LabRank, OperatorKind, ParallelRegionId};

use crate::{RunnerError, RunnerResult};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectiveCompletion {
    pub context: SpmdTaskContext,
    pub id: CollectiveId,
    pub sequence: CollectiveSequence,
    pub result: RunnerResult<CollectiveResponse>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockedCollective {
    pub gang: GangHandle,
    pub region: ParallelRegionId,
    pub id: CollectiveId,
    pub sequence: CollectiveSequence,
    pub waiting_ranks: Vec<LabRank>,
}

/// A deterministic SPMD wait cycle. Every lab is either blocked in a
/// collective or has completed its region, so no later submission can make
/// progress.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectiveDeadlock {
    pub gang: GangHandle,
    pub blocked: Vec<BlockedCollective>,
    pub completed_ranks: Vec<LabRank>,
}

impl std::fmt::Display for CollectiveDeadlock {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "SPMD collective deadlock")?;
        if !self.completed_ranks.is_empty() {
            write!(formatter, "; completed labs")?;
            for rank in &self.completed_ranks {
                write!(formatter, " {}", rank.0)?;
            }
        }
        for blocked in &self.blocked {
            write!(
                formatter,
                "; region {} collective {} sequence {} waits for labs",
                blocked.region.0.ordinal, blocked.id.ordinal, blocked.sequence.0
            )?;
            for rank in &blocked.waiting_ranks {
                write!(formatter, " {}", rank.0)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RoundKey {
    gang: GangId,
    generation: u64,
    region: ParallelRegionId,
    id: CollectiveId,
    sequence: CollectiveSequence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RoundSignature {
    Barrier,
    Broadcast(LabRank),
    Gather(LabRank),
    Scatter(LabRank),
    AllGather,
    Reduce(Option<LabRank>, OperatorKind),
    Cat(Option<LabRank>, u32),
    FunctionalReduce(Option<LabRank>),
}

#[derive(Clone, Debug)]
struct Round {
    gang: GangHandle,
    signature: RoundSignature,
    submissions: BTreeMap<LabRank, CollectiveRequest>,
}

#[derive(Clone, Debug)]
struct Message {
    source: LabRank,
    destination: LabRank,
    tag: CollectiveMessageTag,
    value: ValuePayload,
}

#[derive(Clone, Debug)]
struct WaitingReceive {
    request: CollectiveRequest,
    selection: ReceiveSelection,
}

/// Deterministic, transport-independent SPMD coordination. It orders values by
/// stable lab rank, but never evaluates language operations or owns workers.
#[derive(Default)]
pub struct CollectiveCoordinator {
    rounds: BTreeMap<RoundKey, Round>,
    messages: BTreeMap<(GangId, u64), VecDeque<Message>>,
    receivers: BTreeMap<(GangId, u64), VecDeque<WaitingReceive>>,
    submitted: BTreeSet<(RoundKey, LabRank)>,
}

impl CollectiveCoordinator {
    pub fn submit(
        &mut self,
        request: CollectiveRequest,
    ) -> RunnerResult<Vec<CollectiveCompletion>> {
        request.validate().map_err(invalid)?;
        let key = round_key(&request);
        if !self.submitted.insert((key, request.context.rank)) {
            return Err(RunnerError::Invalid(
                "a lab submitted the same collective identity and sequence twice".into(),
            ));
        }
        match &request.invocation {
            CollectiveInvocation::Send { .. } => self.submit_send(request),
            CollectiveInvocation::SendReceive { .. } => self.submit_send_receive(request),
            CollectiveInvocation::Receive { .. } => self.submit_receive(request),
            CollectiveInvocation::Probe { .. } => self.submit_probe(request),
            _ => self.submit_round(request),
        }
    }

    pub fn blocked(&self, gang: &GangHandle) -> Vec<BlockedCollective> {
        let mut blocked = self
            .rounds
            .iter()
            .filter(|(key, _)| key.gang == gang.id && key.generation == gang.generation)
            .map(|(key, round)| BlockedCollective {
                gang: round.gang.clone(),
                region: key.region,
                id: key.id,
                sequence: key.sequence,
                waiting_ranks: round.submissions.keys().copied().collect(),
            })
            .collect::<Vec<_>>();
        if let Some(receivers) = self.receivers.get(&(gang.id, gang.generation)) {
            blocked.extend(receivers.iter().map(|receiver| BlockedCollective {
                gang: receiver.request.context.gang.clone(),
                region: receiver.request.context.region,
                id: receiver.request.id,
                sequence: receiver.request.sequence,
                waiting_ranks: vec![receiver.request.context.rank],
            }));
        }
        blocked.sort_by_key(|item| (item.region, item.id, item.sequence));
        blocked
    }

    pub fn deadlock(
        &self,
        gang: &GangHandle,
        completed_ranks: &[LabRank],
    ) -> Option<CollectiveDeadlock> {
        let blocked = self.blocked(gang);
        if blocked.is_empty() {
            return None;
        }
        let mut accounted = completed_ranks.iter().copied().collect::<BTreeSet<_>>();
        accounted.extend(
            blocked
                .iter()
                .flat_map(|collective| collective.waiting_ranks.iter().copied()),
        );
        let expected = (1..=gang.labs.0).map(LabRank).collect::<BTreeSet<_>>();
        if accounted != expected {
            return None;
        }
        let mut completed_ranks = completed_ranks.to_vec();
        completed_ranks.sort_unstable();
        completed_ranks.dedup();
        Some(CollectiveDeadlock {
            gang: gang.clone(),
            blocked,
            completed_ranks,
        })
    }

    pub fn fail_gang(
        &mut self,
        gang: &GangHandle,
        reason: impl Into<String>,
    ) -> Vec<CollectiveCompletion> {
        let reason = reason.into();
        let keys = self
            .rounds
            .keys()
            .filter(|key| key.gang == gang.id && key.generation == gang.generation)
            .copied()
            .collect::<Vec<_>>();
        let mut completions = Vec::new();
        for key in keys {
            if let Some(round) = self.rounds.remove(&key) {
                completions.extend(round.submissions.into_values().map(|request| {
                    completion(
                        request,
                        Err(RunnerError::Backend(format!(
                            "SPMD gang terminated: {reason}"
                        ))),
                    )
                }));
            }
        }
        if let Some(receivers) = self.receivers.remove(&(gang.id, gang.generation)) {
            completions.extend(receivers.into_iter().map(|receiver| {
                completion(
                    receiver.request,
                    Err(RunnerError::Backend(format!(
                        "SPMD gang terminated: {reason}"
                    ))),
                )
            }));
        }
        self.messages.remove(&(gang.id, gang.generation));
        self.submitted
            .retain(|(key, _)| key.gang != gang.id || key.generation != gang.generation);
        completions
    }

    fn submit_round(
        &mut self,
        request: CollectiveRequest,
    ) -> RunnerResult<Vec<CollectiveCompletion>> {
        let key = round_key(&request);
        let signature = signature(&request.invocation)?;
        let round = self.rounds.entry(key).or_insert_with(|| Round {
            gang: request.context.gang.clone(),
            signature,
            submissions: BTreeMap::new(),
        });
        if round.gang != request.context.gang || round.signature != signature {
            return Err(RunnerError::Invalid(
                "labs disagreed on a collective operation, root, or reduction operator".into(),
            ));
        }
        round.submissions.insert(request.context.rank, request);
        if round.submissions.len() != round.gang.labs.0 as usize {
            return Ok(Vec::new());
        }
        let round = self.rounds.remove(&key).expect("completed round exists");
        complete_round(round)
    }

    fn submit_send(
        &mut self,
        request: CollectiveRequest,
    ) -> RunnerResult<Vec<CollectiveCompletion>> {
        let CollectiveInvocation::Send {
            destination,
            tag,
            value,
        } = request.invocation.clone()
        else {
            unreachable!()
        };
        let source = request.context.rank;
        let queue_key = gang_key(&request.context.gang);
        let waiting = self.receivers.entry(queue_key).or_default();
        if let Some(index) = waiting.iter().position(|receiver| {
            receiver.request.context.rank == destination
                && matches_selection(&receiver.selection, source, tag)
        }) {
            let receiver = waiting
                .remove(index)
                .expect("matching receiver index remains valid");
            return Ok(vec![
                completion(request, Ok(CollectiveResponse::Complete)),
                completion(
                    receiver.request,
                    Ok(CollectiveResponse::Received { value, source, tag }),
                ),
            ]);
        }
        self.messages
            .entry(queue_key)
            .or_default()
            .push_back(Message {
                source,
                destination,
                tag,
                value,
            });
        Ok(vec![completion(request, Ok(CollectiveResponse::Complete))])
    }

    fn submit_receive(
        &mut self,
        request: CollectiveRequest,
    ) -> RunnerResult<Vec<CollectiveCompletion>> {
        let CollectiveInvocation::Receive { selection } = request.invocation else {
            unreachable!()
        };
        let queue_key = gang_key(&request.context.gang);
        let messages = self.messages.entry(queue_key).or_default();
        if let Some(index) = messages.iter().position(|message| {
            message.destination == request.context.rank
                && matches_selection(&selection, message.source, message.tag)
        }) {
            let message = messages
                .remove(index)
                .expect("matching message index remains valid");
            return Ok(vec![completion(
                request,
                Ok(CollectiveResponse::Received {
                    value: message.value,
                    source: message.source,
                    tag: message.tag,
                }),
            )]);
        }
        self.receivers
            .entry(queue_key)
            .or_default()
            .push_back(WaitingReceive { request, selection });
        Ok(Vec::new())
    }

    fn submit_send_receive(
        &mut self,
        request: CollectiveRequest,
    ) -> RunnerResult<Vec<CollectiveCompletion>> {
        let CollectiveInvocation::SendReceive {
            destination,
            source: expected_source,
            tag,
            value,
        } = request.invocation.clone()
        else {
            unreachable!()
        };
        let sender = request.context.rank;
        let queue_key = gang_key(&request.context.gang);
        let mut completions = Vec::new();
        if let Some(destination) = destination {
            let waiting = self.receivers.entry(queue_key).or_default();
            if let Some(index) = waiting.iter().position(|receiver| {
                receiver.request.context.rank == destination
                    && matches_selection(&receiver.selection, sender, tag)
            }) {
                let receiver = waiting
                    .remove(index)
                    .expect("matching receiver index remains valid");
                completions.push(completion(
                    receiver.request,
                    Ok(CollectiveResponse::Received {
                        value: value.clone(),
                        source: sender,
                        tag,
                    }),
                ));
            } else {
                self.messages
                    .entry(queue_key)
                    .or_default()
                    .push_back(Message {
                        source: sender,
                        destination,
                        tag,
                        value,
                    });
            }
        }
        let Some(expected_source) = expected_source else {
            completions.push(completion(request, Ok(CollectiveResponse::Complete)));
            return Ok(completions);
        };
        let selection = ReceiveSelection {
            source: Some(expected_source),
            tag: Some(tag),
        };
        let messages = self.messages.entry(queue_key).or_default();
        if let Some(index) = messages.iter().position(|message| {
            message.destination == request.context.rank
                && matches_selection(&selection, message.source, message.tag)
        }) {
            let message = messages
                .remove(index)
                .expect("matching message index remains valid");
            completions.push(completion(
                request,
                Ok(CollectiveResponse::Received {
                    value: message.value,
                    source: message.source,
                    tag: message.tag,
                }),
            ));
        } else {
            self.receivers
                .entry(queue_key)
                .or_default()
                .push_back(WaitingReceive { request, selection });
        }
        Ok(completions)
    }

    fn submit_probe(
        &mut self,
        request: CollectiveRequest,
    ) -> RunnerResult<Vec<CollectiveCompletion>> {
        let CollectiveInvocation::Probe { selection } = &request.invocation else {
            unreachable!()
        };
        let available = self
            .messages
            .get(&gang_key(&request.context.gang))
            .is_some_and(|messages| {
                messages.iter().any(|message| {
                    message.destination == request.context.rank
                        && matches_selection(selection, message.source, message.tag)
                })
            });
        Ok(vec![completion(
            request,
            Ok(CollectiveResponse::Probe { available }),
        )])
    }
}

fn complete_round(round: Round) -> RunnerResult<Vec<CollectiveCompletion>> {
    let submissions = round.submissions.into_values().collect::<Vec<_>>();
    let values = submissions
        .iter()
        .filter_map(|request| match &request.invocation {
            CollectiveInvocation::Gather { value, .. }
            | CollectiveInvocation::AllGather { value }
            | CollectiveInvocation::Reduce { value, .. }
            | CollectiveInvocation::Cat { value, .. }
            | CollectiveInvocation::FunctionalReduce { value, .. } => Some(value.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let broadcast = submissions
        .iter()
        .find_map(|request| match &request.invocation {
            CollectiveInvocation::Broadcast {
                root,
                value: Some(value),
            } if *root == request.context.rank => Some(value.clone()),
            _ => None,
        });
    let scatter = submissions
        .iter()
        .find_map(|request| match &request.invocation {
            CollectiveInvocation::Scatter {
                root,
                values: Some(values),
            } if *root == request.context.rank => Some(values.clone()),
            _ => None,
        });
    let reducer = submissions
        .iter()
        .find_map(|request| match &request.invocation {
            CollectiveInvocation::FunctionalReduce { reducer, .. } => Some(reducer.clone()),
            _ => None,
        });
    if matches!(round.signature, RoundSignature::FunctionalReduce(_))
        && submissions.iter().any(|request| match &request.invocation {
            CollectiveInvocation::FunctionalReduce {
                reducer: candidate, ..
            } => Some(candidate) != reducer.as_ref(),
            _ => false,
        })
    {
        return Err(RunnerError::Invalid(
            "labs disagreed on the callable used by a functional reduction".into(),
        ));
    }
    if matches!(round.signature, RoundSignature::Broadcast(_)) && broadcast.is_none() {
        return Err(RunnerError::Invalid(
            "broadcast root did not provide a value".into(),
        ));
    }
    if matches!(round.signature, RoundSignature::Scatter(_))
        && scatter
            .as_ref()
            .is_none_or(|values| values.len() != round.gang.labs.0 as usize)
    {
        return Err(RunnerError::Invalid(
            "scatter root must provide exactly one value per lab".into(),
        ));
    }
    submissions
        .into_iter()
        .map(|request| {
            let response = match round.signature {
                RoundSignature::Barrier => CollectiveResponse::Complete,
                RoundSignature::Broadcast(_) => CollectiveResponse::Value {
                    value: broadcast.clone().expect("validated broadcast value"),
                },
                RoundSignature::Gather(root) if request.context.rank == root => {
                    CollectiveResponse::Values {
                        values: values.clone(),
                    }
                }
                RoundSignature::Gather(_) => CollectiveResponse::Complete,
                RoundSignature::Scatter(_) => CollectiveResponse::Value {
                    value: scatter.as_ref().expect("validated scatter values")
                        [(request.context.rank.0 - 1) as usize]
                        .clone(),
                },
                RoundSignature::AllGather => CollectiveResponse::Values {
                    values: values.clone(),
                },
                RoundSignature::Reduce(Some(root), operator) if request.context.rank == root => {
                    CollectiveResponse::ReductionInputs {
                        operator,
                        values: values.clone(),
                    }
                }
                RoundSignature::Reduce(Some(_), _) => CollectiveResponse::Complete,
                RoundSignature::Reduce(None, operator) => CollectiveResponse::ReductionInputs {
                    operator,
                    values: values.clone(),
                },
                RoundSignature::Cat(Some(root), dimension) if request.context.rank == root => {
                    CollectiveResponse::ConcatenationInputs {
                        dimension,
                        values: values.clone(),
                    }
                }
                RoundSignature::Cat(Some(_), _) => CollectiveResponse::Complete,
                RoundSignature::Cat(None, dimension) => CollectiveResponse::ConcatenationInputs {
                    dimension,
                    values: values.clone(),
                },
                RoundSignature::FunctionalReduce(Some(root)) if request.context.rank == root => {
                    CollectiveResponse::FunctionalReductionInputs {
                        reducer: reducer.clone().expect("validated reduction callable"),
                        values: values.clone(),
                    }
                }
                RoundSignature::FunctionalReduce(Some(_)) => CollectiveResponse::Complete,
                RoundSignature::FunctionalReduce(None) => {
                    CollectiveResponse::FunctionalReductionInputs {
                        reducer: reducer.clone().expect("validated reduction callable"),
                        values: values.clone(),
                    }
                }
            };
            Ok(completion(request, Ok(response)))
        })
        .collect()
}

fn signature(invocation: &CollectiveInvocation) -> RunnerResult<RoundSignature> {
    match invocation {
        CollectiveInvocation::Barrier => Ok(RoundSignature::Barrier),
        CollectiveInvocation::Broadcast { root, .. } => Ok(RoundSignature::Broadcast(*root)),
        CollectiveInvocation::Gather { root, .. } => Ok(RoundSignature::Gather(*root)),
        CollectiveInvocation::Scatter { root, .. } => Ok(RoundSignature::Scatter(*root)),
        CollectiveInvocation::AllGather { .. } => Ok(RoundSignature::AllGather),
        CollectiveInvocation::Reduce { root, operator, .. } => {
            Ok(RoundSignature::Reduce(*root, *operator))
        }
        CollectiveInvocation::Cat {
            root, dimension, ..
        } => Ok(RoundSignature::Cat(*root, *dimension)),
        CollectiveInvocation::FunctionalReduce { root, .. } => {
            Ok(RoundSignature::FunctionalReduce(*root))
        }
        CollectiveInvocation::Send { .. }
        | CollectiveInvocation::SendReceive { .. }
        | CollectiveInvocation::Receive { .. }
        | CollectiveInvocation::Probe { .. } => Err(RunnerError::Invalid(
            "point-to-point operation cannot enter a synchronized collective round".into(),
        )),
    }
}

fn round_key(request: &CollectiveRequest) -> RoundKey {
    RoundKey {
        gang: request.context.gang.id,
        generation: request.context.gang.generation,
        region: request.context.region,
        id: request.id,
        sequence: request.sequence,
    }
}

fn gang_key(gang: &GangHandle) -> (GangId, u64) {
    (gang.id, gang.generation)
}

fn matches_selection(
    selection: &ReceiveSelection,
    source: LabRank,
    tag: CollectiveMessageTag,
) -> bool {
    selection.source.is_none_or(|expected| expected == source)
        && selection.tag.is_none_or(|expected| expected == tag)
}

fn completion(
    request: CollectiveRequest,
    result: RunnerResult<CollectiveResponse>,
) -> CollectiveCompletion {
    CollectiveCompletion {
        context: request.context,
        id: request.id,
        sequence: request.sequence,
        result,
    }
}

fn invalid(error: impl std::fmt::Display) -> RunnerError {
    RunnerError::Invalid(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{ExecutionScopeId, GangHandle, PoolHandle, PoolId};
    use runmat_types::{LabCount, ProgramFunctionId, RegionId};

    fn context(rank: u32) -> SpmdTaskContext {
        let scope_id = ExecutionScopeId::derive(&[b"collective-runner"]);
        SpmdTaskContext {
            gang: GangHandle {
                id: GangId::derive(&[b"gang"]),
                scope_id,
                generation: 1,
                pool: PoolHandle {
                    id: PoolId::derive(&[b"pool"]),
                    scope_id,
                    generation: 1,
                },
                labs: LabCount(3),
            },
            region: ParallelRegionId(RegionId {
                function: ProgramFunctionId(1),
                ordinal: 2,
            }),
            rank: LabRank(rank),
        }
    }

    fn request(rank: u32, invocation: CollectiveInvocation) -> CollectiveRequest {
        let context = context(rank);
        CollectiveRequest {
            id: CollectiveId {
                region: context.region,
                ordinal: 4,
            },
            context,
            sequence: CollectiveSequence(1),
            invocation,
        }
    }

    fn number(value: f64) -> ValuePayload {
        ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::F64Bits(
            value.to_bits(),
        )))
    }

    fn callable(name: &str) -> ValuePayload {
        let owner = "builtin";
        ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::Callable(
            runmat_execution::value::CallableValue {
                owner_identity: owner.into(),
                qualified_name: name.into(),
                callable_digest: runmat_execution::value::CallableValue::identity_digest(
                    owner, name,
                ),
                captures: Vec::new(),
            },
        )))
    }

    #[test]
    fn allgather_completes_in_rank_order_regardless_of_arrival_order() {
        let mut coordinator = CollectiveCoordinator::default();
        for rank in [3, 1] {
            assert!(coordinator
                .submit(request(
                    rank,
                    CollectiveInvocation::AllGather {
                        value: number(rank as f64),
                    },
                ))
                .unwrap()
                .is_empty());
        }
        let completions = coordinator
            .submit(request(
                2,
                CollectiveInvocation::AllGather { value: number(2.0) },
            ))
            .unwrap();
        assert_eq!(completions.len(), 3);
        let CollectiveResponse::Values { values } = completions[0].result.as_ref().unwrap() else {
            panic!("expected gathered values")
        };
        assert_eq!(values, &vec![number(1.0), number(2.0), number(3.0)]);
    }

    #[test]
    fn receive_wildcards_match_without_reordering_the_message_queue() {
        let mut coordinator = CollectiveCoordinator::default();
        coordinator
            .submit(request(
                1,
                CollectiveInvocation::Send {
                    destination: LabRank(3),
                    tag: CollectiveMessageTag(7),
                    value: number(10.0),
                },
            ))
            .unwrap();
        let mut receive = request(
            3,
            CollectiveInvocation::Receive {
                selection: ReceiveSelection {
                    source: None,
                    tag: Some(CollectiveMessageTag(7)),
                },
            },
        );
        receive.sequence = CollectiveSequence(2);
        let completions = coordinator.submit(receive).unwrap();
        let CollectiveResponse::Received { source, value, .. } =
            completions[0].result.as_ref().unwrap()
        else {
            panic!("expected received value")
        };
        assert_eq!(*source, LabRank(1));
        assert_eq!(value, &number(10.0));
    }

    #[test]
    fn language_owned_aggregates_return_rank_ordered_inputs() {
        let mut coordinator = CollectiveCoordinator::default();
        for rank in [2, 3, 1] {
            let completions = coordinator
                .submit(request(
                    rank,
                    CollectiveInvocation::Cat {
                        root: None,
                        dimension: 2,
                        value: number(rank as f64),
                    },
                ))
                .unwrap();
            if rank == 1 {
                assert_eq!(completions.len(), 3);
                let CollectiveResponse::ConcatenationInputs { dimension, values } =
                    completions[0].result.as_ref().unwrap()
                else {
                    panic!("expected concatenation inputs")
                };
                assert_eq!(*dimension, 2);
                assert_eq!(values, &vec![number(1.0), number(2.0), number(3.0)]);
            } else {
                assert!(completions.is_empty());
            }
        }

        let mut coordinator = CollectiveCoordinator::default();
        for rank in [3, 1, 2] {
            let completions = coordinator
                .submit(request(
                    rank,
                    CollectiveInvocation::FunctionalReduce {
                        root: None,
                        reducer: callable("plus"),
                        value: number(rank as f64),
                    },
                ))
                .unwrap();
            if rank == 2 {
                assert_eq!(completions.len(), 3);
                let CollectiveResponse::FunctionalReductionInputs { reducer, values } =
                    completions[0].result.as_ref().unwrap()
                else {
                    panic!("expected functional reduction inputs")
                };
                assert_eq!(reducer, &callable("plus"));
                assert_eq!(values, &vec![number(1.0), number(2.0), number(3.0)]);
            } else {
                assert!(completions.is_empty());
            }
        }
    }

    #[test]
    fn functional_reduction_rejects_callable_disagreement() {
        let mut coordinator = CollectiveCoordinator::default();
        for rank in [1, 2] {
            assert!(coordinator
                .submit(request(
                    rank,
                    CollectiveInvocation::FunctionalReduce {
                        root: None,
                        reducer: callable("plus"),
                        value: number(rank as f64),
                    },
                ))
                .unwrap()
                .is_empty());
        }
        let error = coordinator
            .submit(request(
                3,
                CollectiveInvocation::FunctionalReduce {
                    root: None,
                    reducer: callable("max"),
                    value: number(3.0),
                },
            ))
            .expect_err("labs must agree on the reducer identity");
        assert!(error.to_string().contains("disagreed on the callable"));
    }

    #[test]
    fn failure_completes_every_blocked_lab_and_clears_diagnostics() {
        let mut coordinator = CollectiveCoordinator::default();
        let first = request(1, CollectiveInvocation::Barrier);
        let gang = first.context.gang.clone();
        coordinator.submit(first).unwrap();
        assert_eq!(coordinator.blocked(&gang).len(), 1);
        let completions = coordinator.fail_gang(&gang, "worker lost");
        assert_eq!(completions.len(), 1);
        assert!(completions[0].result.is_err());
        assert!(coordinator.blocked(&gang).is_empty());
    }

    #[test]
    fn deadlock_requires_every_lab_to_be_blocked_or_completed() {
        let gang = context(1).gang;
        let mut coordinator = CollectiveCoordinator::default();
        assert!(coordinator
            .submit(request(
                1,
                CollectiveInvocation::Receive {
                    selection: ReceiveSelection {
                        source: Some(LabRank(2)),
                        tag: Some(CollectiveMessageTag(7)),
                    },
                },
            ))
            .unwrap()
            .is_empty());
        assert!(coordinator.deadlock(&gang, &[]).is_none());
        let deadlock = coordinator
            .deadlock(&gang, &[LabRank(2), LabRank(3)])
            .expect("completed peers cannot satisfy the blocked receive");
        assert_eq!(deadlock.completed_ranks, vec![LabRank(2), LabRank(3)]);
        assert_eq!(deadlock.blocked[0].waiting_ranks, vec![LabRank(1)]);
        assert!(deadlock.to_string().contains("waits for labs 1"));
    }
}
