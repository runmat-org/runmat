//! Field chunks streamed out of staging as they are produced.
//!
//! The staging budget bounds the bytes in flight (one field's payload and its chunks), not the
//! total field payload of a publication. Each chunk is written to a private spill area as soon as
//! it is chunked and read back one at a time at commit. The spill area lives outside the target
//! store, so an object sweep can never reclaim an uncommitted chunk. Hosts without a filesystem
//! keep the chunks in memory, as before.
use std::collections::HashMap;
use std::sync::Arc;

use runmat_execution::Digest;

#[derive(Debug, Default)]
pub(in crate::analysis::storage) struct SpilledChunks {
    #[cfg(not(target_arch = "wasm32"))]
    area: Option<Arc<tempfile::TempDir>>,
    #[cfg(target_arch = "wasm32")]
    resident: HashMap<Digest, Arc<[u8]>>,
    lengths: HashMap<Digest, u64>,
}

impl SpilledChunks {
    pub(in crate::analysis::storage) fn digests(&self) -> impl Iterator<Item = Digest> + '_ {
        self.lengths.keys().copied()
    }

    /// Content-addressed and idempotent: a repeated digest must carry identical bytes.
    pub(in crate::analysis::storage) fn put(
        &mut self,
        digest: Digest,
        bytes: &[u8],
    ) -> Result<(), String> {
        if Digest::sha256(bytes) != digest {
            return Err("spilled chunk differs from its digest".into());
        }
        if let Some(length) = self.lengths.get(&digest) {
            return if *length == bytes.len() as u64 {
                Ok(())
            } else {
                Err("analysis field chunk digest collision".into())
            };
        }
        self.write(digest, bytes)?;
        self.lengths.insert(digest, bytes.len() as u64);
        Ok(())
    }

    /// Reads one chunk back and re-authenticates it.
    pub(in crate::analysis::storage) fn read(
        &self,
        digest: &Digest,
    ) -> Result<Option<Vec<u8>>, String> {
        let Some(length) = self.lengths.get(digest) else {
            return Ok(None);
        };
        let bytes = self.read_bytes(digest)?;
        if bytes.len() as u64 != *length || Digest::sha256(&bytes) != *digest {
            return Err("spilled chunk changed before commit".into());
        }
        Ok(Some(bytes))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn path(&mut self, digest: Digest) -> Result<std::path::PathBuf, String> {
        if self.area.is_none() {
            let area = tempfile::Builder::new()
                .prefix("runmat-publication-spill-")
                .tempdir()
                .map_err(|error| format!("publication spill area: {error}"))?;
            self.area = Some(Arc::new(area));
        }
        Ok(self.existing_path(digest).expect("spill area exists"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn existing_path(&self, digest: Digest) -> Option<std::path::PathBuf> {
        let hex: String = digest
            .bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        self.area.as_ref().map(|area| area.path().join(hex))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write(&mut self, digest: Digest, bytes: &[u8]) -> Result<(), String> {
        let path = self.path(digest)?;
        std::fs::write(&path, bytes).map_err(|error| format!("publication spill write: {error}"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_bytes(&self, digest: &Digest) -> Result<Vec<u8>, String> {
        let path = self
            .existing_path(*digest)
            .ok_or("publication spill area is absent")?;
        std::fs::read(path).map_err(|error| format!("publication spill read: {error}"))
    }

    #[cfg(target_arch = "wasm32")]
    fn write(&mut self, digest: Digest, bytes: &[u8]) -> Result<(), String> {
        self.resident.insert(digest, Arc::from(bytes));
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    fn read_bytes(&self, digest: &Digest) -> Result<Vec<u8>, String> {
        self.resident
            .get(digest)
            .map(|bytes| bytes.to_vec())
            .ok_or_else(|| "spilled chunk is absent".into())
    }
}
