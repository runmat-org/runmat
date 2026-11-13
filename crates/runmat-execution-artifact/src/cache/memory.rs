//! Bounded content-addressed storage for in-process execution hosts.
use super::{copy_verified_bounded_with_admission, CacheExport, CacheImport};
use crate::{object::ObjectInventoryLimits, ArtifactError, ArtifactResult, LogicalObject};
use runmat_execution::Digest;
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct MemoryObjectStore {
    objects: BTreeMap<Digest, LogicalObject>,
    limits: ObjectInventoryLimits,
    retained_bytes: u64,
}
impl MemoryObjectStore {
    pub fn new(limits: ObjectInventoryLimits) -> ArtifactResult<Self> {
        if limits.max_objects == 0 || limits.max_object_bytes == 0 || limits.max_total_bytes == 0 {
            return Err(ArtifactError::Limit(
                "memory object store requires non-zero inventory limits".into(),
            ));
        }
        Ok(Self {
            objects: BTreeMap::new(),
            limits,
            retained_bytes: 0,
        })
    }
    /// Borrow an already verified object without allocating another payload.
    pub fn logical_object(&self, digest: Digest) -> Option<&LogicalObject> {
        self.objects.get(&digest)
    }
    pub fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }
    pub fn len(&self) -> usize {
        self.objects.len()
    }
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}
impl Default for MemoryObjectStore {
    fn default() -> Self {
        Self::new(ObjectInventoryLimits::default()).expect("default inventory limits are non-zero")
    }
}
impl CacheImport for MemoryObjectStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded(digest, self.limits.max_object_bytes)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded_with_admission(digest, maximum_bytes, &mut |_| Ok(()))
    }
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        copy_verified_bounded_with_admission(
            digest,
            self.objects
                .get(&digest)
                .map(|object| object.bytes.as_slice()),
            maximum_bytes.min(self.limits.max_object_bytes),
            admit,
        )
    }
}
impl CacheExport for MemoryObjectStore {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        object.validate()?;
        if let Some(existing) = self.objects.get(&object.descriptor.digest) {
            if existing == object {
                return Ok(());
            }
            return Err(ArtifactError::Identity(
                "resident object digest reused for different content".into(),
            ));
        }
        let bytes = u64::try_from(object.bytes.len())
            .map_err(|_| ArtifactError::Limit("resident object size overflow".into()))?;
        let retained = self
            .retained_bytes
            .checked_add(bytes)
            .ok_or_else(|| ArtifactError::Limit("resident inventory size overflow".into()))?;
        if self.objects.len() >= self.limits.max_objects
            || bytes > self.limits.max_object_bytes
            || retained > self.limits.max_total_bytes
        {
            return Err(ArtifactError::Limit(
                "resident object exceeds the host's admitted inventory".into(),
            ));
        }
        // No payload allocation happens before inventory admission.
        self.objects
            .insert(object.descriptor.digest, object.clone());
        self.retained_bytes = retained;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn object(name: &str, bytes: &[u8]) -> LogicalObject {
        LogicalObject::new(
            crate::object::ObjectNamespace::InputValue,
            name,
            "application/octet-stream",
            bytes.to_vec(),
        )
        .unwrap()
    }
    #[test]
    fn rejects_excess_before_mutating_inventory_and_deduplicates() {
        let mut store = MemoryObjectStore::new(ObjectInventoryLimits {
            max_objects: 2,
            max_object_bytes: 4,
            max_total_bytes: 5,
        })
        .unwrap();
        let first = object("first", b"abc");
        store.write_verified(&first).unwrap();
        store.write_verified(&first).unwrap();
        assert_eq!(store.retained_bytes(), 3);
        assert_eq!(store.len(), 1);
        assert!(store.write_verified(&object("second", b"def")).is_err());
        assert_eq!(store.retained_bytes(), 3);
        assert_eq!(store.len(), 1);
        store.write_verified(&object("second", b"de")).unwrap();
        assert_eq!(store.retained_bytes(), 5);
        assert!(store.write_verified(&object("third", b"f")).is_err());
    }
    #[test]
    fn bounded_reads_admit_actual_buffer_and_preserve_denial() {
        let mut store = MemoryObjectStore::default();
        let value = object("value", b"abc");
        store.write_verified(&value).unwrap();
        let digest = value.descriptor.digest;
        let mut calls = 0;
        assert!(store
            .read_verified_bounded_with_admission(digest, 2, &mut |_| {
                calls += 1;
                Ok(())
            })
            .is_err());
        assert_eq!(calls, 0);
        assert!(store
            .read_verified_bounded_with_admission(digest, 3, &mut |bytes| {
                assert_eq!(bytes, 3);
                Err(ArtifactError::Limit("no allocation".into()))
            })
            .is_err());
        assert_eq!(
            store.read_verified_bounded(digest, 3).unwrap(),
            Some(b"abc".to_vec())
        );
    }
    #[test]
    fn rejects_corrupt_content_and_zero_capacity() {
        assert!(MemoryObjectStore::new(ObjectInventoryLimits {
            max_objects: 0,
            ..Default::default()
        })
        .is_err());
        let mut store = MemoryObjectStore::default();
        let mut value = object("value", b"abc");
        value.bytes[0] = 0;
        assert!(store.write_verified(&value).is_err());
        assert!(store.is_empty());
    }
}
