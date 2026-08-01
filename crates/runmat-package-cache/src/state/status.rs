use super::CacheState;
use crate::CacheObjectKind;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheStatus {
    pub schema_version: u32,
    pub object_count: u64,
    pub objects_by_kind: BTreeMap<CacheObjectKind, u64>,
    pub logical_bytes: u64,
    pub stored_payload_bytes: u64,
    pub pin_count: u64,
    pub lease_count: u64,
    pub corruption_count: u64,
    pub materialization_count: u64,
}

impl CacheStatus {
    pub fn from_state(state: &CacheState) -> Self {
        let mut objects_by_kind = BTreeMap::new();
        for object in state.objects.values() {
            *objects_by_kind.entry(object.kind()).or_insert(0) += 1;
        }
        Self {
            schema_version: state.schema_version,
            object_count: state.objects.len() as u64,
            objects_by_kind,
            logical_bytes: state.total_logical_bytes(),
            stored_payload_bytes: state.total_stored_payload_bytes(),
            pin_count: state.pins.len() as u64,
            lease_count: state.leases.len() as u64,
            corruption_count: state.corruptions.len() as u64,
            materialization_count: state.materializations.len() as u64,
        }
    }
}
