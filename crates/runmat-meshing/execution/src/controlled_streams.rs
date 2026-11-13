//! Preserve cancellation observed during an outer artifact read, without re-querying its source.
use crate::PreparedMeshingStageObjects;
use runmat_meshing_core::{MeshingCancellationSignal, MeshingChunkStream, MeshingFailureCategory};
use std::sync::atomic::{AtomicBool, Ordering};

struct ObservedCancellation<'a> {
    source: &'a dyn MeshingCancellationSignal,
    observed: AtomicBool,
}
impl MeshingCancellationSignal for ObservedCancellation<'_> {
    #[track_caller]
    fn is_cancelled(&self) -> bool {
        if self.observed.load(Ordering::Relaxed) {
            return true;
        }
        let cancelled = self.source.is_cancelled();
        if cancelled {
            self.observed.store(true, Ordering::Relaxed);
        }
        cancelled
    }
}

#[derive(Debug)]
pub(crate) struct StreamReadFailure {
    pub category: MeshingFailureCategory,
    pub detail: String,
}

pub(crate) fn decode(
    publication: &PreparedMeshingStageObjects,
    source: &dyn MeshingCancellationSignal,
) -> Result<Vec<MeshingChunkStream>, StreamReadFailure> {
    let control = ObservedCancellation {
        source,
        observed: AtomicBool::new(false),
    };
    publication
        .decoded_streams_with_cancellation(&control)
        .map_err(|error| StreamReadFailure {
            category: if control.observed.load(Ordering::Relaxed) {
                MeshingFailureCategory::Cancelled
            } else {
                MeshingFailureCategory::InvalidGeometry
            },
            detail: error.to_string(),
        })
}

/// Move the sole admitted record into a consumer without cloning its payload.
pub(crate) fn take_single_record(
    mut streams: Vec<runmat_meshing_core::MeshingChunkStream>,
    index: usize,
) -> Option<Vec<u8>> {
    if index >= streams.len() || streams[index].records.len() != 1 {
        return None;
    }
    streams.swap_remove(index).records.pop()
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Pulse(AtomicBool);
    impl MeshingCancellationSignal for Pulse {
        fn is_cancelled(&self) -> bool {
            self.0.swap(false, Ordering::Relaxed)
        }
    }
    #[test]
    fn cold_artifact_read_preserves_a_one_shot_cancellation() {
        let artifact = crate::tests::fixture(vec![vec![1; 200], vec![2; 300]], 1024);
        assert_eq!(
            decode(&artifact, &runmat_meshing_core::NeverCancelled).unwrap(),
            artifact.decoded_streams().unwrap()
        );
        let source = Pulse(AtomicBool::new(true));
        let failure = decode(&artifact, &source).unwrap_err();
        assert_eq!(failure.category, MeshingFailureCategory::Cancelled);
        assert!(!source.is_cancelled());
    }
}

/// Charge reconstructed records to the same scope as their consumer. The scope
/// must outlive the returned buffers; original encoded inputs have separate owners.
pub(crate) fn decode_admitted(
    publication: &PreparedMeshingStageObjects,
    source: &dyn MeshingCancellationSignal,
    allocations: &dyn runmat_geometry_core::GeometryEvaluationControl,
) -> Result<Vec<MeshingChunkStream>, StreamReadFailure> {
    let control = ObservedCancellation {
        source,
        observed: AtomicBool::new(false),
    };
    let mut original = None;
    let result = publication.decoded_streams_with_admission(&control, |bytes| {
        allocations
            .checkpoint()
            .and_then(|()| allocations.consume_allocation_bytes(bytes))
            .map_err(|error| {
                let detail = error.to_string();
                original = Some(error);
                crate::MeshingExecutionError::Invalid(detail)
            })
    });
    result.map_err(|error| StreamReadFailure {
        category: match original {
            Some(error) => crate::geometry_failure::category(error.kind),
            None if control.observed.load(Ordering::Relaxed) => MeshingFailureCategory::Cancelled,
            None => MeshingFailureCategory::InvalidGeometry,
        },
        detail: error.to_string(),
    })
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use runmat_geometry_core::GeometryAllocationScope;
    use runmat_meshing_core::NeverCancelled;

    // witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: The finite cold-record fixture rejects zero and one-byte-short stage budgets with
    //   MemoryBudgetExceeded, accepts its exact requested reconstruction budget, preserves original
    //   records and releases the consumer scope completely. This does not establish a complete
    //   projection, original CAD study, native-host ceiling or process RSS bound.
    #[test]
    fn reconstructed_records_share_the_stage_ceiling_and_release_with_consumer_scope() {
        let artifact = crate::tests::fixture(vec![vec![1; 130000], vec![2; 31]], 1024);
        let mut required = 0;
        let expected = artifact
            .decoded_streams_with_admission(&NeverCancelled, |bytes| {
                required += bytes;
                Ok(())
            })
            .unwrap();
        for limit in [0, required - 1, required] {
            let owner = crate::geometry_control::tests::bounded_control(limit);
            {
                let allocations = GeometryAllocationScope::new(&owner);
                let result = decode_admitted(&artifact, &NeverCancelled, &allocations);
                if limit < required {
                    assert_eq!(
                        result.unwrap_err().category,
                        MeshingFailureCategory::MemoryBudgetExceeded
                    );
                } else {
                    let streams = result.unwrap();
                    assert_eq!(streams, expected);
                    assert_eq!(owner.usage().live_allocation_bytes, required);
                    let record = take_single_record(streams, 0);
                    assert!(
                        record.is_none(),
                        "two records cannot be consumed as a singleton"
                    );
                }
            }
            assert_eq!(owner.usage().live_allocation_bytes, 0);
        }
    }
}
