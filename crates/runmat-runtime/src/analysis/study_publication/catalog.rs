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
    staging.persist_catalog_objects(definition, &catalog.objects, |reference| {
        Ok(source.read_verified(reference)?)
    })
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
