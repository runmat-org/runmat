use crate::object::CacheObject;
use crate::state::CacheState;
use crate::CacheError;
use runmat_package::ContentDigest;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendSnapshot {
    pub revision: u64,
    pub state: CacheState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectWrite {
    pub object: CacheObject,
    pub bytes: Option<Vec<u8>>,
}

impl ObjectWrite {
    pub fn new(object: CacheObject, bytes: Option<Vec<u8>>) -> Result<Self, CacheError> {
        let write = Self { object, bytes };
        write.validate()?;
        Ok(write)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        self.object.validate()?;
        match (&self.object, self.bytes.as_deref()) {
            (CacheObject::Blob(metadata), Some(bytes)) => metadata.verify(bytes),
            (CacheObject::SourceIndex(metadata), Some(bytes))
                if metadata.byte_len == bytes.len() as u64
                    && metadata.digest == ContentDigest::sha256(bytes) =>
            {
                Ok(())
            }
            (CacheObject::Tree(_), None) => Ok(()),
            (object, _) => Err(CacheError::InvalidObject(format!(
                "{:?} object write has invalid byte payload",
                object.kind()
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheTransaction {
    pub expected_revision: u64,
    pub next_state: CacheState,
    pub writes: BTreeMap<ContentDigest, ObjectWrite>,
    pub deletes: BTreeSet<ContentDigest>,
}

impl CacheTransaction {
    pub fn metadata_only(expected_revision: u64, next_state: CacheState) -> Self {
        Self {
            expected_revision,
            next_state,
            writes: BTreeMap::new(),
            deletes: BTreeSet::new(),
        }
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        self.next_state.validate()?;
        if self
            .writes
            .keys()
            .any(|digest| self.deletes.contains(digest))
        {
            return Err(CacheError::InvalidState(
                "transaction writes and deletes the same object".to_string(),
            ));
        }
        for (digest, write) in &self.writes {
            write.validate()?;
            if digest != write.object.digest()
                || self.next_state.objects.get(digest) != Some(&write.object)
            {
                return Err(CacheError::InvalidState(format!(
                    "object write {digest} does not match next cache state"
                )));
            }
        }
        if self
            .deletes
            .iter()
            .any(|digest| self.next_state.objects.contains_key(digest))
        {
            return Err(CacheError::InvalidState(
                "deleted object remains in next cache state".to_string(),
            ));
        }
        Ok(())
    }

    /// Validates the transaction against the backend's current state.
    ///
    /// Adapters must call this while holding their commit serialization primitive.
    pub fn validate_transition(&self, current: &CacheState) -> Result<(), CacheError> {
        self.validate()?;
        for (digest, next_object) in &self.next_state.objects {
            let changed = current.objects.get(digest) != Some(next_object);
            if changed
                && next_object.stored_payload_bytes() > 0
                && !self.writes.contains_key(digest)
            {
                return Err(CacheError::InvalidState(format!(
                    "new payload-backed object {digest} lacks an atomic write"
                )));
            }
        }
        for digest in current.objects.keys() {
            if !self.next_state.objects.contains_key(digest) && !self.deletes.contains(digest) {
                return Err(CacheError::InvalidState(format!(
                    "removed object {digest} lacks an atomic payload delete"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendCommit {
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "kebab-case")]
pub enum CommitOutcome {
    Committed(BackendCommit),
    Conflict { actual_revision: u64 },
}
