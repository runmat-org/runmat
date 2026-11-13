//! Native cache overlays preserve byte admission before copying resident payloads.
use runmat_execution::Digest;
use runmat_execution_artifact::{cache::CacheImport, ArtifactResult};

pub(crate) fn copy_verified(
    digest: Digest,
    bytes: Option<&[u8]>,
    maximum_bytes: u64,
) -> ArtifactResult<Option<Vec<u8>>> {
    runmat_execution_artifact::cache::copy_verified_bounded_with_admission(
        digest,
        bytes,
        maximum_bytes,
        &mut |_| Ok(()),
    )
}

pub(crate) fn read_overlay(
    source: &impl CacheImport,
    digest: Digest,
    resident: Option<&[u8]>,
    maximum_bytes: u64,
) -> ArtifactResult<Option<Vec<u8>>> {
    match copy_verified(digest, resident, maximum_bytes)? {
        Some(bytes) => Ok(Some(bytes)),
        None => source.read_verified_bounded(digest, maximum_bytes),
    }
}

pub(crate) fn read_overlay_admitted(
    source: &impl CacheImport,
    digest: Digest,
    resident: Option<&[u8]>,
    maximum_bytes: u64,
    admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
) -> ArtifactResult<Option<Vec<u8>>> {
    match runmat_execution_artifact::cache::copy_verified_bounded_with_admission(
        digest,
        resident,
        maximum_bytes,
        admit,
    )? {
        Some(bytes) => Ok(Some(bytes)),
        None => source.read_verified_bounded_with_admission(digest, maximum_bytes, admit),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution_artifact::ArtifactError;
    struct BoundedOnly;
    impl CacheImport for BoundedOnly {
        fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
            panic!("bounded overlay must never use unbounded fallback")
        }
        fn read_verified_bounded(
            &self,
            digest: Digest,
            maximum_bytes: u64,
        ) -> ArtifactResult<Option<Vec<u8>>> {
            assert_eq!(maximum_bytes, 4);
            copy_verified(digest, Some(b"data"), maximum_bytes)
        }
    }
    #[test]
    fn resident_bounds_precede_integrity_and_owned_copy() {
        let digest = Digest::sha256(b"data");
        for limit in [0, 3] {
            assert!(matches!(
                copy_verified(digest, Some(b"wrong"), limit),
                Err(ArtifactError::Limit(_))
            ));
        }
        assert!(matches!(
            copy_verified(digest, Some(b"wrong"), 5),
            Err(ArtifactError::Identity(_))
        ));
        assert_eq!(
            copy_verified(digest, Some(b"data"), 4).unwrap(),
            Some(b"data".to_vec())
        );
        assert_eq!(copy_verified(digest, None, 4).unwrap(), None);
        assert!(copy_verified(digest, None, 0).is_err());
    }
    #[test]
    fn missing_overlay_delegates_exact_bound_and_resident_hit_stays_local() {
        let digest = Digest::sha256(b"data");
        assert_eq!(
            read_overlay(&BoundedOnly, digest, None, 4).unwrap(),
            Some(b"data".to_vec())
        );
        // A hit with a different allowance cannot reach BoundedOnly's assertion.
        assert_eq!(
            read_overlay(&BoundedOnly, digest, Some(b"data"), 5).unwrap(),
            Some(b"data".to_vec())
        );
        assert!(read_overlay(&BoundedOnly, digest, None, 0).is_err());
    }
}
