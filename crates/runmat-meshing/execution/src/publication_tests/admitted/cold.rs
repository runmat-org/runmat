use super::*;
use runmat_execution_artifact::{cache::CacheImport, ArtifactResult};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
struct Reads<'a> {
    source: &'a MemoryCache,
    calls: AtomicUsize,
    pointers: Mutex<Vec<(Digest, usize)>>,
}
impl CacheImport for Reads<'_> {
    fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        panic!("cold import must use bounded reads")
    }

    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let bytes = self.source.read_verified_bounded(digest, maximum)?;
        if let Some(bytes) = &bytes {
            self.pointers
                .lock()
                .unwrap()
                .push((digest, bytes.as_ptr() as usize));
        }
        Ok(bytes)
    }
}
#[test]
fn cold_metadata_admits_before_visitors_and_preserves_every_original_read_buffer() {
    let expected = prepare_result_publication(
        fixture(vec![vec![1; 200], vec![2; 13000]], 1024),
        access("run", 1),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut cache = MemoryCache::default();
    cache.insert_all(&expected.stage_objects().objects);
    let runmat_execution::value::ValuePayload::Object(root) = expected.root_output() else {
        panic!("root must be an object")
    };
    let source = Reads {
        source: &cache,
        calls: AtomicUsize::new(0),
        pointers: Mutex::new(vec![]),
    };
    let mut admissions = 0;
    let actual = crate::publication::import_result_publication_with_admission(
        &source,
        root,
        &access("run", 1),
        ObjectInventoryLimits::default(),
        &NeverCancelled,
        &mut |_| {
            admissions += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(actual, expected);
    let pointers = source.pointers.lock().unwrap();
    assert_eq!(
        source.calls.load(Ordering::Relaxed),
        actual.stage_objects().objects.len()
    );
    for object in &actual.stage_objects().objects {
        assert!(
            pointers.contains(&(object.descriptor.digest, object.bytes.as_ptr() as usize)),
            "cold import copied a verified wire buffer"
        );
    }
    drop(pointers);
    for denied in 0..admissions {
        let source = Reads {
            source: &cache,
            calls: AtomicUsize::new(0),
            pointers: Mutex::new(vec![]),
        };
        let mut observed = 0;
        let result = crate::publication::import_result_publication_with_admission(
            &source,
            root,
            &access("run", 1),
            ObjectInventoryLimits::default(),
            &NeverCancelled,
            &mut |_| {
                let ordinal = observed;
                observed += 1;
                if ordinal == denied {
                    Err(crate::MeshingExecutionError::Invalid(
                        "metadata denied".into(),
                    ))
                } else {
                    Ok(())
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(observed, denied + 1);
        if denied == 0 {
            assert_eq!(
                source.calls.load(Ordering::Relaxed),
                1,
                "typed metadata denial must precede component reads"
            );
        }
    }
}
