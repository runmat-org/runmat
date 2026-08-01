use super::{AccessRecord, CorruptionRecord, QuotaRecord};
use crate::lease::{Lease, LeaseId};
use crate::materialize::MaterializationRecord;
use crate::object::{CacheObject, Pin, PinId};
use crate::CacheError;
use runmat_package::ContentDigest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CACHE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheState {
    pub schema_version: u32,
    pub objects: BTreeMap<ContentDigest, CacheObject>,
    pub access: BTreeMap<ContentDigest, AccessRecord>,
    pub leases: BTreeMap<LeaseId, Lease>,
    pub pins: BTreeMap<PinId, Pin>,
    pub corruptions: BTreeMap<ContentDigest, CorruptionRecord>,
    pub materializations: BTreeMap<ContentDigest, MaterializationRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quota: Option<QuotaRecord>,
}

impl Default for CacheState {
    fn default() -> Self {
        Self {
            schema_version: CACHE_SCHEMA_VERSION,
            objects: BTreeMap::new(),
            access: BTreeMap::new(),
            leases: BTreeMap::new(),
            pins: BTreeMap::new(),
            corruptions: BTreeMap::new(),
            materializations: BTreeMap::new(),
            quota: None,
        }
    }
}

impl CacheState {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != CACHE_SCHEMA_VERSION {
            return Err(CacheError::InvalidState(format!(
                "unsupported cache schema {}",
                self.schema_version
            )));
        }
        for (digest, object) in &self.objects {
            if digest != object.digest() {
                return Err(CacheError::InvalidState(format!(
                    "object key {digest} does not match object digest {}",
                    object.digest()
                )));
            }
            object.validate()?;
            for reference in object.references() {
                if !self.objects.contains_key(&reference) {
                    return Err(CacheError::InvalidState(format!(
                        "object {digest} references missing object {reference}"
                    )));
                }
            }
        }
        if self
            .access
            .keys()
            .any(|digest| !self.objects.contains_key(digest))
        {
            return Err(CacheError::InvalidState(
                "access record references a missing object".to_string(),
            ));
        }
        for (id, lease) in &self.leases {
            if LeaseId::new(id.as_str()).is_err()
                || crate::LeaseOwner::new(lease.owner.as_str()).is_err()
                || id != &lease.id
                || lease.acquired_at_ms >= lease.expires_at_ms
                || lease
                    .objects
                    .iter()
                    .any(|digest| !self.objects.contains_key(digest))
            {
                return Err(CacheError::InvalidState(format!(
                    "lease {} is internally inconsistent",
                    lease.id
                )));
            }
        }
        for (id, pin) in &self.pins {
            if PinId::new(id.as_str()).is_err()
                || id != &pin.id
                || pin
                    .objects
                    .iter()
                    .any(|digest| !self.objects.contains_key(digest))
            {
                return Err(CacheError::InvalidState(format!(
                    "pin {id} is internally inconsistent"
                )));
            }
        }
        for (digest, materialization) in &self.materializations {
            if !self.objects.contains_key(digest)
                || !self.leases.contains_key(&materialization.lease)
            {
                return Err(CacheError::InvalidState(format!(
                    "materialization {digest} lacks its object or lease"
                )));
            }
        }
        Ok(())
    }

    pub fn total_logical_bytes(&self) -> u64 {
        self.objects.values().fold(0u64, |total, object| {
            total.saturating_add(object.logical_byte_len())
        })
    }

    pub fn total_stored_payload_bytes(&self) -> u64 {
        self.objects.values().fold(0u64, |total, object| {
            total.saturating_add(object.stored_payload_bytes())
        })
    }
}
