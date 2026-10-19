//! Versioned, iteration-owned random streams for logical parallel tasks.

use super::turn::{self, RandomTurn};
use super::{next_u64_state, random_error, RandomState, RngSnapshot, RuntimeRng};
use crate::context::host_allocation::HostRetainedSessionAllocation;
use crate::{build_runtime_error, BuiltinResult};
use runmat_execution::{ParallelRandomStream, ParallelRandomnessContext};
use std::cell::RefCell;
use std::mem::{align_of, size_of};
use std::rc::Rc;
use std::sync::Arc;

/// One task-local RNG owner. RuntimeContext clones within a selected parfor
/// task share it, but the coordinator's stream and gate remain untouched.
pub(crate) struct ParallelRandomScope {
    pub(crate) state: RefCell<RuntimeRng>,
    pub(crate) gate: Arc<turn::RandomTurnGate>,
    _allocation: Option<HostRetainedSessionAllocation>,
}

impl ParallelRandomScope {
    pub(crate) fn new(context: &crate::context::RuntimeContext) -> BuiltinResult<Rc<Self>> {
        let payload = size_of::<Self>()
            .checked_add(size_of::<turn::RandomTurnGate>())
            .and_then(|bytes| bytes.checked_add(4 * size_of::<usize>()))
            .and_then(|bytes| bytes.checked_add(align_of::<Self>()))
            .and_then(|bytes| bytes.checked_add(align_of::<turn::RandomTurnGate>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| random_error("parfor", "parfor: random task metadata size overflow"))?;
        let allocation =
            HostRetainedSessionAllocation::admit(Some(context), payload).map_err(|source| {
                build_runtime_error("parfor: cannot admit random task owner")
                    .with_builtin("parfor")
                    .with_source(source)
                    .build()
            })?;
        Ok(Rc::new(Self {
            state: RefCell::new(RuntimeRng::new()),
            gate: Arc::new(turn::RandomTurnGate::default()),
            _allocation: allocation,
        }))
    }

    pub(crate) fn matches_current_owner(&self, context: &crate::context::RuntimeContext) -> bool {
        let finite = context.service_ports().parallel().is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        });
        if finite {
            self._allocation
                .as_ref()
                .is_some_and(|allocation| allocation.matches_current_owner(context))
        } else {
            !context.is_cancelled()
        }
    }
}

/// Newly issued tasks carry only a checked contiguous logical range. The
/// shipped LCG vector form remains accepted for old task replay.
pub async fn reserve_parallel_randomness_context_async(
    context: &crate::context::RuntimeContext,
    count: usize,
) -> BuiltinResult<ParallelRandomnessContext> {
    let mut turn = turn::acquire(Some(context), "parfor").await?;
    if count == 0 {
        return Ok(ParallelRandomnessContext::Deterministic {
            streams: Vec::new(),
        });
    }
    u64::try_from(count).map_err(|_| {
        random_error(
            "parfor",
            "parfor: logical iteration count exceeds the supported random-stream range",
        )
    })?;
    let legacy = turn.with_state("parfor", |random| {
        matches!(random.state, RandomState::Lcg(_))
    })?;
    let owner = turn.owner().cloned();
    let root = turn.with_state("parfor", |random| {
        random.state.prepare_mutation(owner.as_ref())?;
        Ok::<_, crate::RuntimeError>(next_u64_state(&mut random.state))
    })??;
    if legacy {
        Ok(ParallelRandomnessContext::RunMatLcgStateRangeV1 {
            root,
            start: 0,
            len: count,
        })
    } else {
        Ok(ParallelRandomnessContext::Mt19937InitialSeedRangeV1 {
            root,
            start: 0,
            len: count,
        })
    }
}

/// Reserve one compact, stable stream per logical iteration before any task
/// runs. A failed vector reservation cannot consume the coordinator stream.
pub fn reserve_parallel_random_streams(
    context: &crate::context::RuntimeContext,
    count: usize,
) -> BuiltinResult<Vec<ParallelRandomStream>> {
    let mut turn = turn::try_acquire(Some(context), "parfor")?;
    reserve_parallel_random_streams_in_turn(&mut turn, count)
}

pub async fn reserve_parallel_random_streams_async(
    context: &crate::context::RuntimeContext,
    count: usize,
) -> BuiltinResult<Vec<ParallelRandomStream>> {
    let mut turn = turn::acquire(Some(context), "parfor").await?;
    reserve_parallel_random_streams_in_turn(&mut turn, count)
}

fn reserve_parallel_random_streams_in_turn(
    turn: &mut RandomTurn,
    count: usize,
) -> BuiltinResult<Vec<ParallelRandomStream>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let stream_count = u64::try_from(count).map_err(|_| {
        random_error(
            "parfor",
            "parfor: logical iteration count exceeds the supported random-stream range",
        )
    })?;
    let mut streams = Vec::new();
    streams.try_reserve_exact(count).map_err(|source| {
        build_runtime_error("parfor: cannot reserve logical random streams")
            .with_builtin("parfor")
            .with_source(source)
            .build()
    })?;
    let owner = turn.owner().cloned();
    let (legacy, root) = turn.with_state("parfor", |random| {
        let legacy = matches!(random.state, RandomState::Lcg(_));
        random.state.prepare_mutation(owner.as_ref())?;
        Ok::<_, crate::RuntimeError>((legacy, next_u64_state(&mut random.state)))
    })??;
    for ordinal in 0..stream_count {
        let seed = runmat_execution::task::deterministic_iteration_seed(root, ordinal);
        streams.push(if legacy {
            ParallelRandomStream::RunMatLcgV1 { state: seed }
        } else {
            ParallelRandomStream::Mt19937InitialSeedV1 { seed }
        });
    }
    Ok(streams)
}

/// Restores the coordinator or worker stream on success, error, or cancellation.
pub struct ParallelRandomStreamGuard {
    context: crate::context::RuntimeContext,
    previous: Option<RngSnapshot>,
}

impl Drop for ParallelRandomStreamGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            let mut state = self.context.state().random.borrow_mut();
            *state = RuntimeRng::from(previous);
        }
    }
}

pub fn enter_parallel_random_stream(
    context: &crate::context::RuntimeContext,
    stream: ParallelRandomStream,
) -> BuiltinResult<ParallelRandomStreamGuard> {
    let turn = turn::try_acquire(Some(context), "parfor")?;
    enter_parallel_random_stream_in_turn(context, stream, turn)
}

pub async fn enter_parallel_random_stream_async(
    context: &crate::context::RuntimeContext,
    stream: ParallelRandomStream,
) -> BuiltinResult<ParallelRandomStreamGuard> {
    let turn = turn::acquire(Some(context), "parfor").await?;
    enter_parallel_random_stream_in_turn(context, stream, turn)
}

fn enter_parallel_random_stream_in_turn(
    context: &crate::context::RuntimeContext,
    stream: ParallelRandomStream,
    _turn: RandomTurn,
) -> BuiltinResult<ParallelRandomStreamGuard> {
    let child_state = match stream {
        ParallelRandomStream::RunMatLcgV1 { state } => RandomState::Lcg(state),
        ParallelRandomStream::Mt19937InitialSeedV1 { seed } => RandomState::seeded(seed),
    };
    let scope = context.parallel_random_scope();
    let state_slot = scope.map_or(&context.state().random, |scope| &scope.state);
    let mut random = state_slot.try_borrow_mut().map_err(|_| {
        random_error(
            "parfor",
            "parfor: iteration random stream is already in use",
        )
    })?;
    let previous = scope.is_none().then(|| random.snapshot());
    *random = RuntimeRng::from(RngSnapshot::new(child_state, None));
    drop(random);
    Ok(ParallelRandomStreamGuard {
        context: context.clone(),
        previous,
    })
}
