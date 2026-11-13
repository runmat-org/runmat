use super::*;
use crate::{MeshingSubtaskCacheEntry, MeshingSubtaskCacheError, MeshingSubtaskCacheStore};
use runmat_meshing_core::StableDigest;
use std::collections::BTreeMap;
#[derive(Debug, Default)]
struct Store(Mutex<BTreeMap<StableDigest, (Vec<u8>, u64)>>);
impl MeshingSubtaskCacheStore for Store {
    fn implementation_identity(&self) -> StableDigest {
        StableDigest::from_bytes([9; 32])
    }
    fn maximum_entry_bytes(&self) -> u64 {
        1024 * 1024
    }
    fn read<'a>(
        &self,
        key: StableDigest,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Option<MeshingSubtaskCacheEntry<'a>>, MeshingSubtaskCacheError> {
        let records = self.0.lock().unwrap();
        let Some((bytes, workspace)) = records.get(&key) else {
            return Ok(None);
        };
        let storage = runmat_geometry_core::GeometryAllocationReservation::new(
            control,
            bytes.len() as u64 + workspace,
        )
        .map_err(MeshingSubtaskCacheError::Control)?;
        Ok(Some(MeshingSubtaskCacheEntry {
            bytes: bytes.clone(),
            storage,
            materialization_resident_bytes: *workspace,
        }))
    }
    fn write(
        &self,
        key: StableDigest,
        bytes: &[u8],
        workspace: u64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<bool, MeshingSubtaskCacheError> {
        control
            .checkpoint()
            .map_err(MeshingSubtaskCacheError::Control)?;
        self.0
            .lock()
            .unwrap()
            .insert(key, (bytes.to_vec(), workspace));
        Ok(true)
    }
}
// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Fresh provider instances reuse original sizing without rederivation; changed curve
//   quality forces fresh preparation. The shared in-memory store exercises process-independent
//   interfaces, not filesystem persistence or complete native execution.
#[test]
fn fresh_workers_reuse_original_metric_and_changed_quality_invalidates_it() {
    let geometry = geometry();
    let request = crate::serial_tests::request();
    let count = Arc::new(AtomicUsize::new(0));
    let store = Arc::new(Store::default());
    let control = Control::default();
    let load = |request: &MeshingRequest| {
        let provider = PreparedMetricProvider::new(CountingProvider(Arc::clone(&count)))
            .with_subtask_cache(store.clone());
        let evaluator = provider.evaluator(&geometry, &control).unwrap();
        provider
            .geometry_metric(&geometry, evaluator.as_ref(), &control, request)
            .unwrap()
    };
    let first = load(&request);
    let second = load(&request);
    assert_eq!(*first, *second);
    assert_eq!(count.load(Ordering::Relaxed), 1);
    let mut changed = request.clone();
    changed.quality.curve.maximum_chordal_deviation_m = runmat_types::physical::Quantity::new(
        changed
            .quality
            .curve
            .maximum_chordal_deviation_m
            .canonical_value()
            * (0.5),
    );
    load(&changed);
    assert_eq!(count.load(Ordering::Relaxed), 2);
    load(&request);
    assert_eq!(count.load(Ordering::Relaxed), 2);
}
// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Cold sizing rejects understated decoded inventory and a foreign embedded identity.
//   Numerical accuracy and whole-task storage require their own evidence.
#[test]
fn stored_metric_must_fit_its_admitted_inventory_and_embedded_identity() {
    let geometry = geometry();
    let request = crate::serial_tests::request();
    let store = Arc::new(Store::default());
    let control = Control::default();
    let provider = PreparedMetricProvider::new(CountingProvider(Arc::new(AtomicUsize::new(0))))
        .with_subtask_cache(store.clone());
    let evaluator = provider.evaluator(&geometry, &control).unwrap();
    provider
        .geometry_metric(&geometry, evaluator.as_ref(), &control, &request)
        .unwrap();
    let mut records = store.0.lock().unwrap();
    let (_, entry) = records.iter_mut().next().unwrap();
    entry.1 = 0;
    drop(records);
    let fresh = PreparedMetricProvider::new(CountingProvider(Arc::new(AtomicUsize::new(0))))
        .with_subtask_cache(store.clone());
    assert_eq!(
        fresh
            .geometry_metric(&geometry, evaluator.as_ref(), &control, &request)
            .unwrap_err()
            .kind,
        SharedCurveErrorKind::InvalidContract
    );
    let mut records = store.0.lock().unwrap();
    let (_, entry) = records.iter_mut().next().unwrap();
    entry.1 = 1_000_000;
    entry.0[0] ^= 1;
    drop(records);
    assert_eq!(
        fresh
            .geometry_metric(&geometry, evaluator.as_ref(), &control, &request)
            .unwrap_err()
            .kind,
        SharedCurveErrorKind::InvalidContract
    );
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: The production metric cache encoder releases its temporary workspace across repeated
//   writes of the same original metric. This does not independently establish native cache
//   throughput.
#[test]
fn repeated_metric_encoding_releases_temporary_storage() {
    let ceilings = runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT;
    use crate::MeshingGeometryEvaluationControl;
    use runmat_meshing_core::{MeshingRequestSettings, NeverCancelled};
    use std::{sync::atomic::AtomicU64, time::Instant};
    let geometry = geometry();
    let request = crate::serial_tests::request();
    let settings = MeshingRequestSettings::default();
    let now = Instant::now();
    let control = MeshingGeometryEvaluationControl::new(
        &NeverCancelled,
        now,
        Arc::new(crate::checkpoint_clock::CheckpointClock::new(now)),
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
        &runmat_meshing_core::MeshingResourceBudget::new(settings.limits, ceilings),
        &settings.cancellation,
    );
    let store: Arc<dyn MeshingSubtaskCacheStore> = Arc::new(Store::default());
    for _ in 0..32 {
        super::super::persistent::retain(
            Some(&store),
            &geometry,
            &request,
            &request.metric,
            &control,
        )
        .unwrap();
        assert_eq!(control.usage().live_allocation_bytes, 0);
    }
    assert!(control.usage().allocation_bytes > 0);
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Fresh and cold prepared metrics retain only their reported carrier inventory without
//   retaining construction scratch or hostile-decoder workspace. Their vector capacities can
//   legitimately differ. This finite component witness does not establish whole-study memory or
//   arbitrary CAD throughput.
#[test]
fn fresh_and_cold_metric_retention_excludes_preparation_scratch() {
    let ceilings = runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT;
    use crate::MeshingGeometryEvaluationControl;
    use runmat_meshing_core::{MeshingRequestSettings, NeverCancelled};
    use std::{sync::atomic::AtomicU64, time::Instant};
    let geometry = geometry();
    let request = crate::serial_tests::request();
    let store = Arc::new(Store::default());
    let count = Arc::new(AtomicUsize::new(0));
    let mut original = None;
    for _ in 0..2 {
        let settings = MeshingRequestSettings::default();
        let now = Instant::now();
        let control = MeshingGeometryEvaluationControl::new(
            &NeverCancelled,
            now,
            Arc::new(crate::checkpoint_clock::CheckpointClock::new(now)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            &runmat_meshing_core::MeshingResourceBudget::new(settings.limits, ceilings),
            &settings.cancellation,
        );
        let provider = PreparedMetricProvider::new(CountingProvider(Arc::clone(&count)))
            .with_subtask_cache(store.clone());
        let evaluator = provider.evaluator(&geometry, &control).unwrap();
        let metric = provider
            .geometry_metric(&geometry, evaluator.as_ref(), &control, &request)
            .unwrap();
        if let Some(expected) = &original {
            assert_eq!(metric.as_ref(), expected);
        } else {
            original = Some(metric.as_ref().clone());
        }
        let retained = provider
            .cached
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .resident_bytes;
        assert!(retained > 0);
        assert_eq!(
            control.usage().live_allocation_bytes,
            retained,
            "only the prepared carrier survives; scratch and decoder workspace do not"
        );
    }
    assert_eq!(count.load(Ordering::Relaxed), 1);
}
