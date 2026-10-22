mod controlled;
mod stage;

use super::*;
use crate::geometry_control::tests::bounded_control;
use std::sync::atomic::{AtomicU64, Ordering};
struct Source {
    calls: AtomicU64,
    missing: bool,
}
impl CacheImport for Source {
    fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        panic!("unbounded input read")
    }
    fn read_verified_bounded(&self, _: Digest, maximum: u64) -> ArtifactResult<Option<Vec<u8>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.missing {
            return Ok(None);
        }
        if maximum < 16 {
            return Err(ArtifactError::Limit("payload exceeds bound".into()));
        }
        Ok(Some(vec![7; 16]))
    }
}
#[test]
fn input_reads_admit_before_provider_and_release_unused_and_scoped_storage() {
    let control = bounded_control(32);
    let scope = GeometryAllocationScope::new(&control);
    let source = Source {
        calls: AtomicU64::new(0),
        missing: false,
    };
    let reader = InputReader::new(&source, &scope);
    let digest = Digest::sha256([7; 16]);
    assert!(reader.read_verified_bounded(digest, 0).is_err());
    assert!(reader.read_verified_bounded(digest, 33).is_err());
    assert_eq!(source.calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        reader.take_failure().unwrap().kind,
        runmat_geometry_core::GeometryEvaluationErrorKind::AllocationBudgetExceeded
    );
    let first = reader.read_verified_bounded(digest, 24).unwrap().unwrap();
    assert_eq!(control.usage().live_allocation_bytes, 16);
    let second = reader.read_verified_bounded(digest, 16).unwrap().unwrap();
    assert_eq!(control.usage().live_allocation_bytes, 32);
    assert!(reader.read_verified_bounded(digest, 1).is_err());
    assert_eq!(source.calls.load(Ordering::Relaxed), 2);
    drop(first);
    drop(second);
    drop(reader);
    drop(scope);
    assert_eq!(control.usage().live_allocation_bytes, 0);
    assert_eq!(control.usage().allocation_bytes, 32);
}
#[test]
fn missing_or_rejected_reads_release_their_entire_reservation() {
    let control = bounded_control(32);
    for missing in [true, false] {
        let scope = GeometryAllocationScope::new(&control);
        let source = Source {
            calls: AtomicU64::new(0),
            missing,
        };
        let reader = InputReader::new(&source, &scope);
        let result = reader.read_verified_bounded(Digest::sha256([]), 8);
        if missing {
            assert_eq!(result.unwrap(), None);
        } else {
            assert!(result.is_err());
        }
        assert_eq!(control.usage().live_allocation_bytes, 0);
        assert!(reader.take_failure().is_none());
    }
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Cold stage identity, manifest and chunks retain original values with actual
//   provider-buffer admission rather than reserving the complete artifact quota. Exact requested
//   peak accepts, one byte less rejects and scope exit releases all storage. The fixture does not
//   qualify original CAD, PDE accuracy or whole-task residency.
#[test]
fn cold_stage_identity_read_admits_actual_storage_instead_of_artifact_quota() {
    use runmat_geometry_core::GeometryEvaluationControl;
    let expected = crate::tests::fixture(vec![vec![1; 200], vec![2; 13000]], 1024);
    let root = expected.root_reference();
    let mut source = crate::tests::MemoryCache::default();
    source.insert_all(&expected.objects);
    let mut observed_peak = 0;
    for ceiling in [8 << 20, 0, 1] {
        let maximum = match ceiling {
            0 => observed_peak,
            1 => observed_peak - 1,
            value => value,
        };
        let control = bounded_control(maximum);
        {
            let scope = GeometryAllocationScope::new(&control);
            let reader = InputReader::new(&source, &scope);
            let codec = crate::codec_control::CodecControl::new(&scope);
            let mut admit = |bytes| {
                codec
                    .consume_allocation_bytes(bytes)
                    .map_err(|error| crate::MeshingExecutionError::Invalid(error.to_string()))
            };
            let result = crate::objects::import_stage_objects_with_admission(
                &reader,
                root.clone(),
                runmat_execution_artifact::object::ObjectInventoryLimits::default(),
                &codec,
                &mut admit,
            );
            if ceiling == 1 {
                assert!(result.is_err(), "one-byte-short admission must reject");
                assert!(codec
                    .take_failure()
                    .or_else(|| reader.take_failure())
                    .is_some());
            } else {
                assert_eq!(result.unwrap().into_raw(), expected);
                if ceiling != 0 {
                    observed_peak = control.usage().allocation_bytes;
                }
                assert!(
                    observed_peak < 8 << 20,
                    "a small identity must not reserve the 512 MiB artifact quota"
                );
            }
        }
        assert_eq!(control.usage().live_allocation_bytes, 0);
    }
}

mod interruptions;
