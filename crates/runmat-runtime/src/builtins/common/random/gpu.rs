//! Session-owned explicit GPU stream. Provider pipelines consume a checked
//! request-local reservation; they never own or reseed the language stream.

use crate::{build_runtime_error, BuiltinResult};
pub(crate) use runmat_accelerate_api::random::{
    GpuNormalTransform, GpuRandomAlgorithm, GpuRandomReservation,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GpuRandomSnapshot {
    pub algorithm: GpuRandomAlgorithm,
    pub seed: u64,
    /// Logical item position, independent of the provider's chunk width.
    pub position: u128,
    pub normal_transform: GpuNormalTransform,
}

impl Default for GpuRandomSnapshot {
    fn default() -> Self {
        Self {
            algorithm: GpuRandomAlgorithm::Threefry4x64_20,
            seed: 0,
            position: 0,
            normal_transform: GpuNormalTransform::BoxMuller,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuRandomStream {
    snapshot: GpuRandomSnapshot,
}

impl GpuRandomStream {
    pub fn new() -> Self {
        Self {
            snapshot: GpuRandomSnapshot::default(),
        }
    }

    pub fn snapshot(&self) -> GpuRandomSnapshot {
        self.snapshot
    }

    pub fn replace(&mut self, snapshot: GpuRandomSnapshot) -> GpuRandomSnapshot {
        std::mem::replace(&mut self.snapshot, snapshot)
    }

    /// Advances exactly once per logical output position. Rejection sampling
    /// uses an independent attempt lane within that position, so actual device
    /// chunking and retry count cannot overlap a subsequent call's range.
    pub fn reserve(&mut self, items: usize) -> Option<GpuRandomReservation> {
        let reservation = self.proposed_range(items)?;
        self.commit_range(reservation)?;
        Some(reservation)
    }

    fn proposed_range(&self, items: usize) -> Option<GpuRandomReservation> {
        let count = u64::try_from(items).ok()?;
        self.snapshot.position.checked_add(u128::from(count))?;
        Some(GpuRandomReservation {
            algorithm: self.snapshot.algorithm,
            seed: self.snapshot.seed,
            first: self.snapshot.position,
            count,
            normal_transform: self.snapshot.normal_transform,
        })
    }

    fn commit_range(&mut self, reservation: GpuRandomReservation) -> Option<()> {
        if reservation.first != self.snapshot.position
            || reservation.seed != self.snapshot.seed
            || reservation.algorithm != self.snapshot.algorithm
            || reservation.normal_transform != self.snapshot.normal_transform
        {
            return None;
        }
        self.snapshot.position = self
            .snapshot
            .position
            .checked_add(u128::from(reservation.count))?;
        Some(())
    }
}

fn with_stream<T>(
    label: &str,
    operation: impl FnOnce(&mut GpuRandomStream) -> BuiltinResult<T>,
) -> BuiltinResult<T> {
    if let Some(context) = crate::context::legacy::active() {
        context
            .service_ports()
            .require_selected_local_effect(runmat_types::EffectKind::Randomness)?;
        let mut stream = context
            .state()
            .gpu_random
            .try_borrow_mut()
            .map_err(|source| {
                build_runtime_error(format!("{label}: GPU random stream is already in use"))
                    .with_builtin(label)
                    .with_source(source)
                    .build()
            })?;
        return operation(&mut stream);
    }
    // Direct builtin unit tests may run without a session; normal execution
    // never gives a process-global provider a stream through this branch.
    #[cfg(test)]
    {
        thread_local! {
            static DIRECT_TEST_STREAM: std::cell::RefCell<GpuRandomStream> =
                std::cell::RefCell::new(GpuRandomStream::new());
        }
        DIRECT_TEST_STREAM.with(|stream| {
            let mut stream = stream.try_borrow_mut().map_err(|source| {
                build_runtime_error(format!("{label}: GPU test stream is already in use"))
                    .with_builtin(label)
                    .with_source(source)
                    .build()
            })?;
            operation(&mut stream)
        })
    }
    #[cfg(not(test))]
    Err(build_runtime_error(format!(
        "{label}: GPU random generation requires a live runtime session"
    ))
    .with_builtin(label)
    .build())
}

pub(crate) fn snapshot() -> BuiltinResult<GpuRandomSnapshot> {
    with_stream("gpurng", |stream| Ok(stream.snapshot()))
}

pub(crate) fn replace(snapshot: GpuRandomSnapshot) -> BuiltinResult<GpuRandomSnapshot> {
    with_stream("gpurng", |stream| Ok(stream.replace(snapshot)))
}

pub(crate) fn reserve(items: usize, label: &str) -> BuiltinResult<GpuRandomReservation> {
    with_stream(label, |stream| {
        stream.reserve(items).ok_or_else(|| {
            build_runtime_error(format!("{label}: GPU random counter range overflow"))
                .with_builtin(label)
                .build()
        })
    })
}

/// Commit an asynchronously executed request only after its selected route
/// has been admitted against this exact snapshot. Once committed, all later
/// provider/resource/cancellation failures retain this consumed range and
/// cannot replay the request through a different candidate.
pub(crate) fn reserve_if_snapshot(
    expected: GpuRandomSnapshot,
    items: usize,
    label: &str,
) -> BuiltinResult<GpuRandomReservation> {
    with_stream(label, |stream| {
        if stream.snapshot() != expected {
            return Err(build_runtime_error(format!(
                "{label}: GPU random stream changed after route selection"
            ))
            .with_builtin(label)
            .build());
        }
        stream.reserve(items).ok_or_else(|| {
            build_runtime_error(format!("{label}: GPU random counter range overflow"))
                .with_builtin(label)
                .build()
        })
    })
}

/// A synchronous provider attempt holds the child-stream turn throughout
/// admission and execution. A certified pre-effect refusal leaves the range
/// available; any emitted or unproven attempt consumes it, including failure,
/// so a later request cannot replay a possibly-used counter range.
pub(crate) fn with_reserved_attempt<T>(
    items: usize,
    label: &str,
    attempt: impl FnOnce(GpuRandomReservation) -> (BuiltinResult<T>, bool),
) -> BuiltinResult<T> {
    with_stream(label, |stream| {
        let reservation = stream.proposed_range(items).ok_or_else(|| {
            build_runtime_error(format!("{label}: GPU random counter range overflow"))
                .with_builtin(label)
                .build()
        })?;
        let (result, consumed) = attempt(reservation);
        if consumed {
            stream.commit_range(reservation).ok_or_else(|| {
                build_runtime_error(format!(
                    "{label}: GPU random stream changed during execution"
                ))
                .with_builtin(label)
                .build()
            })?;
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_child_snapshot_restores_exact_next_range_without_host_stream() {
        let mut stream = GpuRandomStream::new();
        let default = stream.snapshot();
        assert_eq!(default.algorithm, GpuRandomAlgorithm::Threefry4x64_20);
        assert_eq!(default.seed, 0);
        assert_eq!(default.position, 0);
        assert_eq!(default.normal_transform, GpuNormalTransform::BoxMuller);

        let first = stream.reserve(3).expect("first range");
        assert_eq!(first.first, 0);
        assert_eq!(first.count, 3);
        let saved = stream.snapshot();
        let second = stream.reserve(2).expect("second range");
        assert_eq!(second.first, 3);
        assert_eq!(second.count, 2);
        stream.replace(saved);
        let repeated = stream.reserve(2).expect("restored range");
        assert_eq!(second, repeated);
        assert_eq!(second.word(0, 0), repeated.word(0, 0));
        assert_eq!(stream.snapshot().position, 5);
    }

    #[test]
    fn attempted_child_range_only_commits_after_effect_or_unproven_execution() {
        let original = GpuRandomSnapshot::default();
        replace(original).expect("reset test child stream");
        let first = with_reserved_attempt(4, "rand", |range| (Ok(range), false))
            .expect("pre-effect attempt");
        assert_eq!(first.first, 0);
        assert_eq!(snapshot().expect("unchanged stream").position, 0);
        let second = with_reserved_attempt::<()>(4, "rand", |range| {
            assert_eq!(range, first);
            (
                Err(build_runtime_error("unproven provider execution").build()),
                true,
            )
        });
        assert!(second.is_err());
        assert_eq!(snapshot().expect("committed stream").position, 4);
        replace(original).expect("restore test child stream");
    }
}
