//! One original selection identity contract and its borrowed codec workspace inventory.
use super::super::workspace::{check_cancelled, slots};
use super::*;
use runmat_meshing_core::PersistentEntityId;
use serde::Serialize;

/// Count-only maximal digest widths. These values never enter a numerical identity.
struct NodeIdentityInventory(usize);
impl Serialize for NodeIdentityInventory {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut sequence = serializer.serialize_seq(Some(self.0))?;
        let width = StableDigest::from_bytes([255; 32]);
        for ordinal in 0..self.0 {
            if ordinal % 1024 == 0 && crate::is_fea_cancelled() {
                return Err(serde::ser::Error::custom(
                    "constraint selection profiling cancelled",
                ));
            }
            sequence.serialize_element(&width)?;
        }
        sequence.end()
    }
}

pub(in crate::constraints) fn workspace_from_count<E: Serialize + ?Sized>(
    target: &RegionTarget,
    entities: &E,
    node_count: usize,
) -> Result<u64, ConstraintPlanError> {
    check_cancelled()?;
    slots::<StableDigest>(node_count)?;
    super::super::plan::canonical_value_workspace_bytes(&(
        target,
        entities,
        NodeIdentityInventory(node_count),
    ))
}

pub(in crate::constraints) fn digest<E: Serialize + ?Sized>(
    target: &RegionTarget,
    entities: &E,
    nodes: &[StableDigest],
) -> Result<runmat_execution::Digest, ConstraintPlanError> {
    // Validate the actual borrowed materialization inventory before encoding. Count-only
    // preparation bounds this phase before the incoming selections are allocated.
    let identity = (target, entities, nodes);
    let requested = super::super::plan::canonical_value_workspace_bytes(&identity)?;
    usize::try_from(requested).map_err(|_| super::super::workspace::overflow())?;
    super::super::plan::canonical_digest("resolved-constraint-target", &identity)
}

pub(in crate::constraints) fn explicit_workspace(
    target: &RegionTarget,
    node_count: usize,
) -> Result<u64, ConstraintPlanError> {
    workspace_from_count(target, &[] as &[PersistentEntityId], node_count)
}

#[cfg(test)]
mod tests;
