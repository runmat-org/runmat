use super::*;
use maintenance::Lifecycle;
struct Cache {
    directory: tempfile::TempDir,
    index: PathBuf,
    objects: PathBuf,
}
impl Cache {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let index = directory.path().join("receipts");
        let objects = directory.path().join("objects");
        std::fs::create_dir_all(&index).unwrap();
        std::fs::create_dir_all(&objects).unwrap();
        Self {
            directory,
            index,
            objects,
        }
    }
    fn receipt(&self, key: Digest, objects: &[(&[u8], u64)]) -> AcceptedMeshingStageReceipt {
        let access = runmat_execution_artifact::ArtifactAccess {
            authorization_scope: "lifecycle-fixture".into(),
            encryption_context: Digest::sha256(b"access"),
        };
        let references = objects
            .iter()
            .map(|(name, length)| {
                let digest = Digest::sha256(name);
                runmat_execution::value::ValueRef {
                    schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
                    id: access.value_id(digest),
                    logical_digest: digest,
                    encoded_length: *length,
                    media_type: "application/octet-stream".into(),
                    value_schema: "fixture".into(),
                    encryption_context: access.encryption_context,
                    kind: runmat_execution::value::ValueRefKind::ResultObject,
                    authorization_scope: access.authorization_scope.clone(),
                    resident_fence: None,
                }
            })
            .collect::<Vec<_>>();
        serde_json::from_value(serde_json::json!({"schema_version":1, "key":key,
            "access":access,"root":references[0],"result_objects":references}))
        .unwrap()
    }
    fn write(&self, key: Digest, receipt: &AcceptedMeshingStageReceipt, objects: &[(&[u8], u64)]) {
        for (name, length) in objects {
            let path = self.objects.join(filename(Digest::sha256(name)));
            if !path.exists() {
                std::fs::File::create(path)
                    .unwrap()
                    .set_len(*length)
                    .unwrap();
            }
        }
        std::fs::write(
            self.index.join(filename(key)),
            serde_json::to_vec(receipt).unwrap(),
        )
        .unwrap();
    }
    fn publish(&self, lifecycle: &mut Lifecycle, id: usize, objects: &[(&[u8], u64)]) {
        let key = Digest::sha256(id.to_le_bytes());
        let receipt = self.receipt(key, objects);
        let requested = objects.iter().map(|(_, bytes)| bytes).sum::<u64>() + RECEIPT_BYTES;
        let reservation = lifecycle
            .reserve(
                &self.index,
                &self.objects,
                requested,
                objects.len(),
                &|| Ok(()),
            )
            .unwrap()
            .unwrap();
        self.write(key, &receipt, objects);
        lifecycle
            .published(
                reservation,
                &self.index.join(filename(key)),
                &self.objects,
                &receipt,
                receipt.object_digests(),
                &|| Ok(()),
            )
            .unwrap();
    }
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A feasible 1,024-receipt working set survives publication with one inventory
//   discovery, and a cold owner preserves the set. This finite metadata fixture does not certify
//   arbitrary model throughput, cache hit rate or whole-study memory.
#[test]
fn cache_lifecycle_retains_feasible_working_set_beyond_256_without_per_stage_discovery() {
    let cache = Cache::new();
    let mut lifecycle = Lifecycle::default();
    for id in 0..1024 {
        cache.publish(&mut lifecycle, id, &[(b"shared", 512)]);
    }
    assert_eq!(std::fs::read_dir(&cache.index).unwrap().count(), 1024);
    assert_eq!(lifecycle.discoveries, 1);
    // A cold owner reads the inventory once and preserves the entire working set.
    let mut cold = Lifecycle::default();
    cache.publish(&mut cold, 1024, &[(b"shared", 512)]);
    assert_eq!(cold.discoveries, 1);
    assert_eq!(std::fs::read_dir(&cache.index).unwrap().count(), 1025);
}
// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Serialized writers invalidate stale inventories, interrupted publication reconciles
//   durable receipts, and unreferenced objects are reclaimed. This fixture does not independently
//   qualify the numerical contents, transport or complete native study.
#[test]
fn cache_lifecycle_reconciles_other_writers_and_interrupted_publications() {
    let cache = Cache::new();
    let mut first = Lifecycle::default();
    cache.publish(&mut first, 0, &[(b"first", 8)]);
    let mut second = Lifecycle::default();
    cache.publish(&mut second, 1, &[(b"second", 8)]);
    cache.publish(&mut first, 2, &[(b"first", 8)]);
    assert_eq!(first.discoveries, 2);
    for completed_receipt in [false, true] {
        let key = Digest::sha256([completed_receipt as u8, 3]);
        let receipt = cache.receipt(key, &[(b"interrupted", 8)]);
        let reservation = first
            .reserve(&cache.index, &cache.objects, RECEIPT_BYTES + 8, 1, &|| {
                Ok(())
            })
            .unwrap()
            .unwrap();
        if completed_receipt {
            cache.write(key, &receipt, &[(b"interrupted", 8)]);
        } else {
            std::fs::write(
                cache.objects.join(filename(Digest::sha256(b"orphan"))),
                b"orphan",
            )
            .unwrap();
        }
        drop(reservation);
        cache.publish(
            &mut first,
            4 + usize::from(completed_receipt),
            &[(b"first", 8)],
        );
        assert!(!cache
            .objects
            .join(filename(Digest::sha256(b"orphan")))
            .exists());
        if completed_receipt {
            assert!(cache.index.join(filename(key)).exists());
        }
    }
    assert_eq!(first.discoveries, 4);
    assert_eq!(std::fs::read_dir(&cache.index).unwrap().count(), 6);
}
#[test]
fn cache_lifecycle_evicts_by_capacity_preserving_shared_live_objects() {
    let cache = Cache::new();
    let mut lifecycle = Lifecycle::default();
    cache.publish(&mut lifecycle, 0, &[(b"shared", 512), (b"old-only", 8192)]);
    cache.publish(&mut lifecycle, 1, &[(b"shared", 512), (b"live-only", 8192)]);
    let reservation = lifecycle
        .reserve(
            &cache.index,
            &cache.objects,
            8 * 1024 * 1024 * 1024 - 12_000,
            0,
            &|| Ok(()),
        )
        .unwrap()
        .unwrap();
    assert!(!cache
        .index
        .join(filename(Digest::sha256(0usize.to_le_bytes())))
        .exists());
    assert!(cache
        .index
        .join(filename(Digest::sha256(1usize.to_le_bytes())))
        .exists());
    assert!(!cache
        .objects
        .join(filename(Digest::sha256(b"old-only")))
        .exists());
    assert!(cache
        .objects
        .join(filename(Digest::sha256(b"shared")))
        .exists());
    drop(reservation);
}
#[test]
fn cache_lifecycle_removes_only_unreferenced_owned_objects_and_bounds_large_entries() {
    let cache = Cache::new();
    let mut lifecycle = Lifecycle::default();
    let orphan = cache.objects.join(filename(Digest::sha256(b"orphan")));
    std::fs::write(&orphan, b"orphan").unwrap();
    assert!(lifecycle
        .reserve(
            &cache.index,
            &cache.objects,
            8 * 1024 * 1024 * 1024 + 1,
            0,
            &|| Ok(())
        )
        .unwrap()
        .is_none());
    assert!(orphan.exists());
    drop(
        lifecycle
            .reserve(&cache.index, &cache.objects, 1, 0, &|| Ok(()))
            .unwrap()
            .unwrap(),
    );
    assert!(!orphan.exists());
    let foreign = cache.objects.join("foreign-file");
    std::fs::write(&foreign, b"preserve").unwrap();
    assert!(lifecycle
        .reserve(&cache.index, &cache.objects, 1, 0, &|| Ok(()))
        .is_err());
    assert!(foreign.exists());
    assert!(cache.directory.path().exists());
}

#[test]
fn cache_lifecycle_duplicate_publication_removes_only_unreferenced_new_objects() {
    let cache = Cache::new();
    let mut lifecycle = Lifecycle::default();
    cache.publish(&mut lifecycle, 0, &[(b"winner", 8)]);
    cache.publish(&mut lifecycle, 1, &[(b"still-live", 8)]);
    let key = Digest::sha256(0usize.to_le_bytes());
    let winner = cache.receipt(key, &[(b"winner", 8)]);
    let reservation = lifecycle
        .reserve(&cache.index, &cache.objects, RECEIPT_BYTES + 8, 2, &|| {
            Ok(())
        })
        .unwrap()
        .unwrap();
    let orphan = Digest::sha256(b"losing-evidence");
    std::fs::write(cache.objects.join(filename(orphan)), b"orphan").unwrap();
    lifecycle
        .published(
            reservation,
            &cache.index.join(filename(key)),
            &cache.objects,
            &winner,
            [orphan, Digest::sha256(b"still-live")].into_iter(),
            &|| Ok(()),
        )
        .unwrap();
    assert!(!cache.objects.join(filename(orphan)).exists());
    assert!(cache
        .objects
        .join(filename(Digest::sha256(b"still-live")))
        .exists());
    assert!(cache
        .objects
        .join(filename(Digest::sha256(b"winner")))
        .exists());
}
#[test]
fn cache_lifecycle_cancellation_and_missing_objects_preserve_committed_receipts() {
    let cache = Cache::new();
    let mut lifecycle = Lifecycle::default();
    cache.publish(&mut lifecycle, 0, &[(b"original", 8)]);
    assert!(lifecycle
        .reserve(&cache.index, &cache.objects, 1, 0, &|| Err(error(
            "cancelled"
        )))
        .is_err());
    let key = Digest::sha256(0usize.to_le_bytes());
    let receipt_path = cache.index.join(filename(key));
    assert!(receipt_path.exists());
    let object = cache.objects.join(filename(Digest::sha256(b"original")));
    std::fs::remove_file(&object).unwrap();
    // Cold reconstruction must never call a missing object an accepted cache hit.
    let mut cold = Lifecycle::default();
    assert!(cold
        .reserve(&cache.index, &cache.objects, 1, 0, &|| Ok(()))
        .is_err());
    assert!(receipt_path.exists());
}
