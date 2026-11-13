//! Verify imported wire ownership against the actual bounded-provider returns.
use runmat_execution::Digest;
use runmat_execution_artifact::{cache::CacheImport, ArtifactResult, LogicalObject};
use std::{collections::HashMap, sync::Mutex};

pub(crate) struct ReadBuffers<'a, S> {
    source: &'a S,
    buffers: Mutex<HashMap<Digest, (usize, usize)>>,
}
impl<'a, S> ReadBuffers<'a, S> {
    pub(crate) fn new(source: &'a S) -> Self {
        Self {
            source,
            buffers: Mutex::new(HashMap::new()),
        }
    }
    pub(crate) fn assert_retained(&self, objects: &[LogicalObject]) {
        let observed = self.buffers.lock().unwrap();
        assert_eq!(observed.len(), objects.len());
        for object in objects {
            assert_eq!(
                observed[&object.descriptor.digest],
                (object.bytes.as_ptr() as usize, object.bytes.capacity()),
                "import replaced the original verified buffer: {}",
                object.descriptor.media_type
            );
        }
    }
}
impl<S: CacheImport> CacheImport for ReadBuffers<'_, S> {
    fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        panic!("cold imports must use the bounded provider route")
    }
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        let bytes =
            self.source
                .read_verified_bounded_with_admission(digest, maximum_bytes, admit)?;
        if let Some(bytes) = &bytes {
            assert!(
                self.buffers
                    .lock()
                    .unwrap()
                    .insert(digest, (bytes.as_ptr() as usize, bytes.capacity()))
                    .is_none(),
                "import repeated a component read"
            );
        }
        Ok(bytes)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        let bytes = self.source.read_verified_bounded(digest, maximum_bytes)?;
        if let Some(bytes) = &bytes {
            assert!(
                self.buffers
                    .lock()
                    .unwrap()
                    .insert(digest, (bytes.as_ptr() as usize, bytes.capacity()))
                    .is_none(),
                "import repeated a component read"
            );
        }
        Ok(bytes)
    }
}
