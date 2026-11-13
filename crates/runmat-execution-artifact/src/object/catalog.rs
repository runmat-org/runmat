use std::sync::Arc;

use runmat_execution::value::{PublishedObjectSource, ValueRef};

use crate::cache::CacheImport;

/// Artifact-backed implementation of the portable published-object source port.
#[derive(Clone)]
pub struct ArtifactPublishedObjectSource<Source> {
    source: Arc<Source>,
}

impl<Source> ArtifactPublishedObjectSource<Source> {
    pub fn new(source: Source) -> Self {
        Self {
            source: Arc::new(source),
        }
    }
}

impl<Source> PublishedObjectSource for ArtifactPublishedObjectSource<Source>
where
    Source: CacheImport + Send + Sync + 'static,
{
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        let bytes = self
            .source
            .as_ref()
            .read_verified(reference.logical_digest)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "published object source omitted a declared object".to_string())?;
        if bytes.len() as u64 != reference.encoded_length {
            return Err(format!(
                "published object length differs from its declared reference: expected {}, received {}",
                reference.encoded_length,
                bytes.len()
            ));
        }
        if runmat_execution::Digest::sha256(&bytes) != reference.logical_digest {
            return Err("published object digest differs from its declared reference".to_string());
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;
    use runmat_execution::value::{artifact_object_value_id, ValueRefKind};
    use runmat_execution::Digest;

    use super::*;
    use crate::ArtifactResult;

    struct MemoryImport(Option<Vec<u8>>);

    impl CacheImport for MemoryImport {
        fn read_verified(&self, _digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
            Ok(self.0.clone())
        }
    }

    fn reference(bytes: &[u8]) -> ValueRef {
        let logical_digest = Digest::sha256(bytes);
        let encryption_context = Digest::sha256(b"artifact-source-test-context");
        let authorization_scope = "artifact-source-test".to_string();
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
            id: artifact_object_value_id(logical_digest, encryption_context, &authorization_scope),
            logical_digest,
            encoded_length: bytes.len() as u64,
            media_type: "application/vnd.runmat.test".to_string(),
            value_schema: "runmat.test".to_string(),
            encryption_context,
            kind: ValueRefKind::ResultObject,
            authorization_scope,
            resident_fence: None,
        }
    }

    #[test]
    fn source_returns_only_bytes_matching_the_declared_reference() {
        let bytes = b"verified published object".to_vec();
        let source = ArtifactPublishedObjectSource::new(MemoryImport(Some(bytes.clone())));

        assert_eq!(source.read_verified(&reference(&bytes)).unwrap(), bytes);
    }

    #[test]
    fn source_rejects_missing_wrong_length_and_wrong_digest_objects() {
        let bytes = b"verified published object".to_vec();
        let declared = reference(&bytes);

        assert!(ArtifactPublishedObjectSource::new(MemoryImport(None))
            .read_verified(&declared)
            .is_err());
        assert!(
            ArtifactPublishedObjectSource::new(MemoryImport(Some(bytes[..4].to_vec())))
                .read_verified(&declared)
                .is_err()
        );

        let same_length_wrong_bytes = vec![b'x'; bytes.len()];
        assert!(
            ArtifactPublishedObjectSource::new(MemoryImport(Some(same_length_wrong_bytes)))
                .read_verified(&declared)
                .is_err()
        );
    }
}
