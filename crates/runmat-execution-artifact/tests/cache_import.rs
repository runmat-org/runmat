use runmat_execution::Digest;
use runmat_execution_artifact::cache::{import_verified_object, CacheImport};
use runmat_execution_artifact::{ArtifactResult, LogicalObject, ObjectNamespace};

struct Source(Option<Vec<u8>>);

impl CacheImport for Source {
    fn read_verified(&self, _digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.clone())
    }
}

#[test]
fn cache_import_revalidates_content_without_assuming_layout() {
    let expected = LogicalObject::new(
        ObjectNamespace::PackageRelease,
        "packages/example/release",
        "application/octet-stream",
        b"verified package".to_vec(),
    )
    .unwrap();
    assert_eq!(
        import_verified_object(&Source(Some(expected.bytes.clone())), &expected)
            .unwrap()
            .unwrap(),
        expected
    );
    assert!(import_verified_object(&Source(Some(b"wrong bytes".to_vec())), &expected).is_err());
    assert!(import_verified_object(&Source(None), &expected)
        .unwrap()
        .is_none());
}
