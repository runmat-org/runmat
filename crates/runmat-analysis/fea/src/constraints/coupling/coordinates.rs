//! Borrow coordinate authority without rebuilding an explicit-surface map.
use super::super::node_index::NodeValues;
use super::*;
use std::collections::BTreeMap;
pub(in crate::constraints) trait CoordinateLookup {
    fn position(&self, identity: &StableDigest) -> Option<[f64; 3]>;
}
impl CoordinateLookup for NodeValues<[f64; 3]> {
    fn position(&self, id: &StableDigest) -> Option<[f64; 3]> {
        self.get(id).copied()
    }
}
impl CoordinateLookup for BTreeMap<StableDigest, [f64; 3]> {
    fn position(&self, id: &StableDigest) -> Option<[f64; 3]> {
        self.get(id).copied()
    }
}
