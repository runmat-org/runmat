//! Borrow coordinate authority without rebuilding an explicit-surface map.
use super::*;
use crate::physics::structural::topology::explicit_surface::ExplicitShellSurfaceNode;
use crate::topology::storage::CanonicalNodeValues as NodeValues;
pub(in crate::constraints) trait CoordinateLookup {
    fn position(&self, identity: &StableDigest) -> Option<[f64; 3]>;
}
impl CoordinateLookup for NodeValues<[f64; 3]> {
    fn position(&self, id: &StableDigest) -> Option<[f64; 3]> {
        self.get(id).copied()
    }
}
impl CoordinateLookup for NodeValues<ExplicitShellSurfaceNode> {
    fn position(&self, id: &StableDigest) -> Option<[f64; 3]> {
        self.get(id)
            .map(|node| node.coordinates_m.canonical_value())
    }
}
