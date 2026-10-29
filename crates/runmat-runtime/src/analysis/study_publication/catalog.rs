//! Stream authenticated catalog objects into shared immutable binary storage.
//! Numerical admission belongs to the source owner; transport checks remain here.
use crate::analysis::storage::{AnalysisPublicationBuilder, AnalysisPublicationError};
use runmat_execution::{
    value::{PublishedObjectCatalog, PublishedObjectSource},
    Digest,
};

pub(in crate::analysis) fn persist_objects(
    source: &impl PublishedObjectSource,
    catalog: &PublishedObjectCatalog,
    definition: Digest,
    staging: &AnalysisPublicationBuilder,
) -> Result<Vec<String>, AnalysisPublicationError> {
    catalog.validate().map_err(|error| error.to_string())?;
    let _source_phase = staging.begin_source_phase()?;
    staging.require_source_objects(&catalog.objects)?;
    let limit = staging.source_workspace_limit()?;
    let mut paths = Vec::with_capacity(catalog.objects.len());
    for batch in concurrent_batches(&catalog.objects, limit) {
        let resident: u64 = batch.iter().map(|reference| reference.encoded_length).sum();
        // Each object keeps the whole allowance minus the other buffers of its batch, so a
        // batch of one is charged exactly as a sequential write.
        let persist = |reference: &runmat_execution::value::ValueRef| {
            let bytes = source.read_verified(reference)?;
            staging.persist_catalog_object_within(
                definition,
                reference,
                &bytes,
                limit - (resident - reference.encoded_length),
            )
        };
        paths.extend(
            ordered_io_map(batch, persist)
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(paths)
}

/// Consecutive objects whose read and write buffers (twice the object) fit the source
/// workspace together; an object that alone exceeds it forms its own batch and fails there.
fn concurrent_batches(
    objects: &[runmat_execution::value::ValueRef],
    limit: u64,
) -> Vec<&[runmat_execution::value::ValueRef]> {
    let mut batches = Vec::new();
    let mut start = 0;
    let mut charged = 0u64;
    for (index, object) in objects.iter().enumerate() {
        let need = object.encoded_length.saturating_mul(2);
        if index > start && charged.saturating_add(need) > limit {
            batches.push(&objects[start..index]);
            start = index;
            charged = 0;
        }
        charged = charged.saturating_add(need);
    }
    if start < objects.len() {
        batches.push(&objects[start..]);
    }
    batches
}

/// Order-preserving map of independent create-and-fsync I/O over a few host threads. The
/// work is storage latency, not numerical capacity, so it does not take the CPU pool.
fn ordered_io_map<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(8)
            .min(items.len());
        if threads > 1 {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let results = std::sync::Mutex::new(Vec::with_capacity(items.len()));
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(item) = items.get(index) else { break };
                        let value = work(item);
                        results.lock().expect("I/O results").push((index, value));
                    });
                }
            });
            let mut results = results.into_inner().expect("I/O results");
            results.sort_unstable_by_key(|(index, _)| *index);
            return results.into_iter().map(|(_, value)| value).collect();
        }
    }
    items.iter().map(work).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::value::{
        artifact_object_value_id, PublishedObjectCatalog, PublishedObjectSource, ValueRef,
        ValueRefKind,
    };
    struct SubstitutedSource(std::sync::atomic::AtomicUsize);
    impl PublishedObjectSource for SubstitutedSource {
        fn read_verified(&self, _: &ValueRef) -> Result<Vec<u8>, String> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(b"other".to_vec())
        }
    }
    #[test]
    fn shared_catalog_persistence_rejects_substituted_payload_before_writing() {
        let logical_digest = Digest::sha256(b"original");
        let encryption_context = Digest::sha256(b"authority");
        let reference = ValueRef {
            schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
            id: artifact_object_value_id(logical_digest, encryption_context, "authority"),
            logical_digest,
            encoded_length: 8,
            media_type: "application/vnd.runmat.test".into(),
            value_schema: "runmat.test".into(),
            encryption_context,
            kind: ValueRefKind::ResultObject,
            authorization_scope: "authority".into(),
            resident_fence: None,
        };
        let catalog =
            PublishedObjectCatalog::new(vec![reference.clone()], vec![reference]).unwrap();
        let mut staging = crate::analysis::storage::AnalysisPublicationBuilder::new(
            Digest::sha256(b"study"),
            128 << 20,
        )
        .unwrap();
        let source = SubstitutedSource(std::sync::atomic::AtomicUsize::new(0));
        assert!(matches!(
            persist_objects(&source, &catalog, Digest::sha256(b"study"), &staging),
            Err(crate::analysis::storage::AnalysisPublicationError::ResourceLimit { .. })
        ));
        assert_eq!(source.0.load(std::sync::atomic::Ordering::SeqCst), 0);
        staging.admit_source_objects(&catalog.objects).unwrap();
        assert!(
            persist_objects(&source, &catalog, Digest::sha256(b"study"), &staging,)
                .unwrap_err()
                .to_string()
                .contains("digest or length mismatch")
        );
        assert_eq!(source.0.load(std::sync::atomic::Ordering::SeqCst), 1);
        let mut absent = catalog;
        absent.objects.clear();
        assert!(persist_objects(&source, &absent, Digest::sha256(b"study"), &staging,).is_err());
    }
}
