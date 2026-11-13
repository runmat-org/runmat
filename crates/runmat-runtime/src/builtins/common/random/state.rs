//! Versioned host RNG state and the self-restoring snapshot carried by `rng`.

use super::mt19937::{Mt19937, Mt19937State, StateError, STATE_WORDS};
use super::{advance_state, random_error, DEFAULT_RNG_SEED, DEFAULT_USER_SEED};
use crate::context::host_allocation::{HostAllocationError, HostRetainedSessionAllocation};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum RngAlgorithm {
    Twister,
    RunMatLcgV1,
}

impl RngAlgorithm {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            RngAlgorithm::Twister => "twister",
            RngAlgorithm::RunMatLcgV1 => "runmat-lcg-v1",
        }
    }
}

/// One physically retained MT state. A finite managed claim follows this Arc
/// through snapshots and parallel guards until the last owner releases it.
#[derive(Debug)]
pub(crate) struct MtOwned {
    engine: Mt19937,
    allocation: Option<HostRetainedSessionAllocation>,
}

impl MtOwned {
    fn requested_bytes() -> BuiltinResult<u64> {
        let bytes = std::mem::size_of::<Self>()
            .checked_add(2 * std::mem::size_of::<usize>())
            .and_then(|n| n.checked_add(std::mem::align_of::<Self>()))
            .ok_or_else(|| random_error("rng", "rng: MT state allocation size overflow"))?;
        u64::try_from(bytes)
            .map_err(|_| random_error("rng", "rng: MT state allocation size overflow"))
    }

    fn new(engine: Mt19937, runtime: Option<&RuntimeContext>) -> BuiltinResult<Arc<Self>> {
        let allocation = HostRetainedSessionAllocation::admit(runtime, Self::requested_bytes()?)
            .map_err(allocation_error)?;
        let state = Arc::new(Self { engine, allocation });
        if let Some(runtime) = runtime {
            if runtime.is_cancelled() {
                return Err(allocation_error(HostAllocationError::Cancelled));
            }
            if !state.matches_current_owner(Some(runtime)) {
                return Err(allocation_error(HostAllocationError::StaleAuthority));
            }
        }
        Ok(state)
    }

    fn matches_current_owner(&self, runtime: Option<&RuntimeContext>) -> bool {
        let Some(runtime) = runtime else {
            return true;
        };
        let finite = runtime.service_ports().parallel().is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        });
        if finite {
            self.allocation
                .as_ref()
                .is_some_and(|claim| claim.matches_current_owner(runtime))
        } else {
            self.allocation.is_none()
        }
    }
}

impl Deref for MtOwned {
    type Target = Mt19937;

    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

impl DerefMut for MtOwned {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.engine
    }
}

fn allocation_error(error: HostAllocationError) -> crate::RuntimeError {
    crate::build_runtime_error("rng: retained MT state allocation is unavailable")
        .with_builtin("rng")
        .with_source(error)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

#[derive(Clone, Debug)]
pub(crate) enum RandomState {
    /// No MT heap allocation is made before a finite service can be attached.
    MtSeed(u64),
    Mt(Arc<MtOwned>),
    Lcg(u64),
}

impl PartialEq for RandomState {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Lcg(a), Self::Lcg(b)) => a == b,
            (Self::MtSeed(a), Self::MtSeed(b)) => a == b,
            (Self::Mt(a), Self::Mt(b)) => a.engine == b.engine,
            (Self::MtSeed(seed), Self::Mt(engine)) | (Self::Mt(engine), Self::MtSeed(seed)) => {
                seeded_engine(*seed) == engine.engine
            }
            _ => false,
        }
    }
}

impl Eq for RandomState {}

fn normalized_seed(seed: u64) -> u64 {
    if seed == 0 {
        DEFAULT_RNG_SEED
    } else {
        seed
    }
}

fn seeded_engine(seed: u64) -> Mt19937 {
    let seed = normalized_seed(seed);
    if let Ok(word) = u32::try_from(seed) {
        Mt19937::seed(word)
    } else {
        Mt19937::seed_words(&[seed as u32, (seed >> 32) as u32])
            .expect("a two-word seed is nonempty")
    }
}

impl RandomState {
    pub(crate) fn seeded(seed: u64) -> Self {
        Self::MtSeed(seed)
    }

    #[cfg(test)]
    pub(crate) fn seeded_unbounded(seed: u64) -> Self {
        let mut state = Self::seeded(seed);
        state.prepare_mutation(None).expect("unbounded test state");
        state
    }

    pub(crate) fn algorithm(&self) -> RngAlgorithm {
        match self {
            Self::MtSeed(_) | Self::Mt(_) => RngAlgorithm::Twister,
            Self::Lcg(_) => RngAlgorithm::RunMatLcgV1,
        }
    }

    pub(crate) fn from_mt_words(
        words: [u32; STATE_WORDS],
        index: usize,
    ) -> Result<BuiltinResult<Self>, StateError> {
        let state = Mt19937State::validated(words, index)?;
        let runtime = crate::context::legacy::active();
        Ok(MtOwned::new(Mt19937::from_state(state), runtime.as_ref()).map(Self::Mt))
    }

    pub(crate) fn mt_snapshot(&self) -> Option<Mt19937State> {
        match self {
            Self::MtSeed(seed) => Some(seeded_engine(*seed).snapshot()),
            Self::Mt(engine) => Some(engine.snapshot()),
            Self::Lcg(_) => None,
        }
    }

    /// Claim a distinct state before any copy-on-write mutation or owner
    /// transition. With a unique matching Arc, an ordinary draw allocates none.
    pub(crate) fn prepare_mutation(
        &mut self,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<()> {
        if runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(allocation_error(HostAllocationError::Cancelled));
        }
        match self {
            Self::MtSeed(seed) => {
                let next = MtOwned::new(seeded_engine(*seed), runtime)?;
                *self = Self::Mt(next);
            }
            Self::Mt(current)
                if Arc::strong_count(current) == 1 && current.matches_current_owner(runtime) => {}
            Self::Mt(current) => {
                let next = MtOwned::new(current.engine.clone(), runtime)?;
                *self = Self::Mt(next);
            }
            Self::Lcg(_) => {}
        }
        Ok(())
    }

    pub(crate) fn advance_uniforms(
        &mut self,
        count: u64,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<()> {
        let words = count
            .checked_mul(2)
            .ok_or_else(|| random_error("rng", "rng: uniform draw count exceeds MT word range"))?;
        self.prepare_mutation(runtime)?;
        match self {
            Self::Mt(engine) => Arc::get_mut(engine)
                .expect("MT mutation must have a unique admitted state")
                .advance_words(words),
            Self::Lcg(state) => *state = advance_state(*state, count),
            Self::MtSeed(_) => unreachable!("admitted state was materialized"),
        }
        Ok(())
    }

    pub(crate) fn mt_mut(&mut self) -> Option<&mut Mt19937> {
        match self {
            Self::Mt(engine) => Some(&mut Arc::get_mut(engine)?.engine),
            Self::MtSeed(_) | Self::Lcg(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RngSnapshot {
    pub state: RandomState,
    pub seed: Option<u64>,
}

impl RngSnapshot {
    pub(crate) fn new(state: RandomState, seed: Option<u64>) -> Self {
        Self { state, seed }
    }

    pub(crate) fn algorithm(&self) -> RngAlgorithm {
        self.state.algorithm()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeRng {
    pub(super) state: RandomState,
    pub(super) seed: Option<u64>,
}

impl RuntimeRng {
    pub(crate) fn new() -> Self {
        Self {
            state: RandomState::seeded(DEFAULT_USER_SEED),
            seed: Some(DEFAULT_USER_SEED),
        }
    }

    pub(super) fn snapshot(&self) -> RngSnapshot {
        RngSnapshot {
            state: self.state.clone(),
            seed: self.seed,
        }
    }
}

impl From<RngSnapshot> for RuntimeRng {
    fn from(snapshot: RngSnapshot) -> Self {
        Self {
            state: snapshot.state,
            seed: snapshot.seed,
        }
    }
}
