//! Fixed-capacity, session-owned complete-call profile backing. Portable
//! Strings are reconstructed only for explicit profile export, never while a
//! finite invocation is selected or completed.

use runmat_execution::{
    Digest, ExecutionCandidateKind, PlacementInvalidation, PlacementPermitIdentity,
    PlacementSignature,
};
use runmat_runtime::context::host_effect::{
    HostEffectAllocationError, ManagedHostEffectLedger, ManagedHostEffectStageReservation,
};
use runmat_runtime::context::ManagedObservationPool;
use std::cell::RefCell;
use std::io::{self, Write};
use std::sync::Arc;

use super::{evidence, PlacementFeedbackSeries};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Key {
    pub(super) signature: Digest,
    pub(super) permit: PlacementPermitIdentity,
    pub(super) candidate: Digest,
    pub(super) kind: ExecutionCandidateKind,
}

#[derive(Clone, Copy, Default)]
struct Series {
    key: Option<Key>,
    signature_len: usize,
    counters: evidence::FeedbackNumbers,
}

struct State {
    slots: Vec<Series>,
    signatures: Vec<u8>,
    bytes_per_series: usize,
}

pub(super) struct ManagedFeedbackPool {
    state: RefCell<State>,
    ledger: ManagedHostEffectLedger,
    _backing: Arc<ManagedHostEffectStageReservation>,
}

impl ManagedFeedbackPool {
    pub(super) fn new(
        ledger: &ManagedHostEffectLedger,
        series: usize,
        bytes_per_series: usize,
    ) -> Result<Self, HostEffectAllocationError> {
        let series_bytes = series
            .checked_mul(std::mem::size_of::<Series>())
            .ok_or(HostEffectAllocationError::Overflow)?;
        let signature_bytes = series
            .checked_mul(bytes_per_series)
            .ok_or(HostEffectAllocationError::Overflow)?;
        let reserved = u64::try_from(
            series_bytes
                .checked_add(signature_bytes)
                .ok_or(HostEffectAllocationError::Overflow)?,
        )
        .map_err(|_| HostEffectAllocationError::Overflow)?;
        let backing = ledger.reserve_stage(reserved)?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(series)
            .map_err(|_| HostEffectAllocationError::AllocationFailed)?;
        slots.resize_with(series, Series::default);
        let mut signatures = Vec::new();
        signatures
            .try_reserve_exact(signature_bytes)
            .map_err(|_| HostEffectAllocationError::AllocationFailed)?;
        signatures.resize(signature_bytes, 0);
        Ok(Self {
            state: RefCell::new(State {
                slots,
                signatures,
                bytes_per_series,
            }),
            ledger: ledger.clone(),
            _backing: backing,
        })
    }

    /// Prepare a new series before the selected effect begins. A bounded
    /// writer rejects insufficient configured capacity without a heap mirror.
    pub(super) fn ensure(
        &self,
        signature: &PlacementSignature,
        candidate: Digest,
        kind: ExecutionCandidateKind,
        seed: Option<evidence::FeedbackNumbers>,
        observations: &ManagedObservationPool,
    ) -> Result<Key, HostEffectAllocationError> {
        if !self.ledger.snapshot().current {
            return Err(HostEffectAllocationError::Revoked);
        }
        let key = Key {
            signature: signature.cache_key(),
            permit: signature
                .permit_identity()
                .map_err(|_| HostEffectAllocationError::InvalidInvocation)?,
            candidate,
            kind,
        };
        let mut state = self.state.borrow_mut();
        if state.slots.iter().any(|slot| slot.key == Some(key)) {
            return Ok(key);
        }
        let index = state
            .slots
            .iter()
            .position(|slot| slot.key.is_none())
            .or_else(|| {
                state.slots.iter().position(|slot| {
                    slot.key.is_some_and(|old| {
                        slot.counters.samples == 0
                            && slot.counters.failures == 0
                            && !observations.has_active(old.permit, old.candidate)
                    })
                })
            })
            .ok_or(HostEffectAllocationError::ExceedsCapacity)?;
        let start = index * state.bytes_per_series;
        let end = start + state.bytes_per_series;
        let mut writer = BoundedWriter::new(&mut state.signatures[start..end]);
        serde_json::to_writer(&mut writer, signature)
            .map_err(|_| HostEffectAllocationError::ExceedsCapacity)?;
        let length = writer.written;
        if !self.ledger.snapshot().current {
            return Err(HostEffectAllocationError::Revoked);
        }
        state.slots[index] = Series {
            key: Some(key),
            signature_len: length,
            counters: seed.unwrap_or_default(),
        };
        Ok(key)
    }

    pub(super) fn observed(
        &self,
        signature: Digest,
        candidate: Digest,
        kind: ExecutionCandidateKind,
        tick: u64,
    ) -> Option<runmat_execution::CandidateObservationEvidence> {
        let state = self.state.borrow();
        state
            .slots
            .iter()
            .find(|slot| {
                slot.key.is_some_and(|key| {
                    key.signature == signature && key.candidate == candidate && key.kind == kind
                })
            })
            .and_then(|slot| {
                (slot.counters.samples != 0 || slot.counters.failures != 0)
                    .then(|| evidence::from_numbers(slot.counters, kind, tick))
            })
    }

    pub(super) fn record(
        &self,
        key: Key,
        elapsed_ns: u64,
        succeeded: bool,
        tick: u64,
    ) -> Result<bool, HostEffectAllocationError> {
        if !self.ledger.snapshot().current {
            return Err(HostEffectAllocationError::Revoked);
        }
        let mut state = self.state.borrow_mut();
        let slot = state
            .slots
            .iter_mut()
            .find(|slot| slot.key == Some(key))
            .ok_or(HostEffectAllocationError::InvalidInvocation)?;
        let previous = slot.counters;
        let needs_replan = slot.counters.record(elapsed_ns, succeeded, tick);
        if !self.ledger.snapshot().current {
            slot.counters = previous;
            return Err(HostEffectAllocationError::Revoked);
        }
        Ok(needs_replan)
    }

    pub(super) fn invalidate(&self, invalidation: &PlacementInvalidation) {
        let mut state = self.state.borrow_mut();
        for slot in &mut state.slots {
            if slot
                .key
                .is_some_and(|key| invalidation.matches_permit_identity(&key.permit))
            {
                *slot = Series::default();
            }
        }
    }

    pub(super) fn profile_snapshot(&self) -> Vec<PlacementFeedbackSeries> {
        let state = self.state.borrow();
        let mut result = Vec::new();
        for (index, slot) in state.slots.iter().enumerate() {
            let Some(key) = slot.key else { continue };
            if slot.counters.samples == 0 && slot.counters.failures == 0 {
                continue;
            }
            let start = index * state.bytes_per_series;
            let signature: PlacementSignature =
                serde_json::from_slice(&state.signatures[start..start + slot.signature_len])
                    .expect("selected canonical signature remains valid in managed backing");
            result.push(PlacementFeedbackSeries {
                signature,
                candidate: key.candidate.to_string(),
                kind: key.kind,
                samples: slot.counters.samples,
                failures: slot.counters.failures,
                total_elapsed_ns: slot.counters.total_elapsed_ns,
                total_squared_elapsed_ns: slot.counters.total_squared_elapsed_ns,
                latest_tick: slot.counters.latest_tick,
            });
        }
        result
    }
}

struct BoundedWriter<'a> {
    bytes: &'a mut [u8],
    written: usize,
}

impl<'a> BoundedWriter<'a> {
    fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes, written: 0 }
    }
}

impl Write for BoundedWriter<'_> {
    fn write(&mut self, source: &[u8]) -> io::Result<usize> {
        let end = self
            .written
            .checked_add(source.len())
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::WriteZero, "managed signature capacity")
            })?;
        self.bytes[self.written..end].copy_from_slice(source);
        self.written = end;
        Ok(source.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{PlacementExecutionAuthorityRevision, PlacementRevision};

    fn signature() -> PlacementSignature {
        let digest = Digest::sha256(b"managed profile");
        PlacementSignature {
            region: None,
            operation: "readmatrix".into(),
            runtime_facts: digest,
            revision: PlacementRevision {
                program: None,
                catalog: digest,
                compiler: digest,
                authority: PlacementExecutionAuthorityRevision::HostService(digest),
                policy: digest,
            },
        }
    }

    #[test]
    fn preeffect_signature_capacity_and_exact_profile_roundtrip() {
        let signature = signature();
        let encoded = serde_json::to_vec(&signature).unwrap();
        let candidate = Digest::sha256(b"numeric text route");
        let kind = ExecutionCandidateKind::SharedRuntime;
        let ledger = ManagedHostEffectLedger::new(16 * 1024);
        let pool = ManagedFeedbackPool::new(&ledger, 1, encoded.len()).unwrap();
        let observations = ManagedObservationPool::new(&ledger, 1).unwrap();
        let charged = ledger.snapshot().live_reserved_bytes;
        assert_eq!(
            charged,
            (std::mem::size_of::<Series>() + encoded.len()) as u64
        );
        let key = pool
            .ensure(&signature, candidate, kind, None, &observations)
            .unwrap();
        let state = pool.state.borrow();
        assert_eq!(&state.signatures[..encoded.len()], encoded.as_slice());
        drop(state);
        assert!(pool.observed(key.signature, candidate, kind, 0).is_none());
        pool.record(key, 17, true, 1).unwrap();
        pool.record(key, 19, true, 2).unwrap();
        let snapshot = pool.profile_snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].signature, signature);
        assert_eq!(snapshot[0].candidate, candidate.to_string());
        assert_eq!(snapshot[0].samples, 2);
        pool.invalidate(&PlacementInvalidation::All);
        assert!(pool.profile_snapshot().is_empty());
        drop(pool);
        drop(observations);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn undersized_series_rejects_before_observation_and_reuses_slot() {
        let signature = signature();
        let ledger = ManagedHostEffectLedger::new(16 * 1024);
        let pool = ManagedFeedbackPool::new(&ledger, 1, 1).unwrap();
        let observations = ManagedObservationPool::new(&ledger, 1).unwrap();
        assert!(matches!(
            pool.ensure(
                &signature,
                Digest::sha256(b"route"),
                ExecutionCandidateKind::SharedRuntime,
                None,
                &observations,
            ),
            Err(HostEffectAllocationError::ExceedsCapacity)
        ));
        assert!(pool.state.borrow().slots[0].key.is_none());
    }

    #[test]
    fn abandoned_unobserved_series_reuses_slot_but_active_call_cannot_be_evicted() {
        let ledger = ManagedHostEffectLedger::new(16 * 1024);
        let pool = ManagedFeedbackPool::new(&ledger, 1, 1024).unwrap();
        let observations = ManagedObservationPool::new(&ledger, 1).unwrap();
        let first = signature();
        let candidate = Digest::sha256(b"route");
        let kind = ExecutionCandidateKind::SharedRuntime;
        let first_identity = first.permit_identity().unwrap();
        let active = observations.acquire(first_identity, candidate).unwrap();
        pool.ensure(&first, candidate, kind, None, &observations)
            .unwrap();
        let mut second = first.clone();
        second.operation = "other.readmatrix".into();
        assert!(matches!(
            pool.ensure(&second, candidate, kind, None, &observations),
            Err(HostEffectAllocationError::ExceedsCapacity)
        ));
        drop(active);
        let replacement = pool
            .ensure(&second, candidate, kind, None, &observations)
            .expect("abandoned zero-sample series can be replaced");
        assert_eq!(pool.state.borrow().slots[0].key, Some(replacement));
        assert_eq!(pool.profile_snapshot().len(), 0);
    }
}
