use std::collections::BTreeSet;

use runmat_analysis_core::{StructuralElementKind, StructuralModel};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

pub(crate) mod explicit_surface;
mod identity;
pub(crate) mod node_space;
pub(crate) mod target;

pub(crate) use identity::{
    explicit_element_identity, explicit_identity_resource_bytes, explicit_node_identity,
    explicit_topology_digest,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CanonicalStructuralTopologyIdentity {
    Solid {
        mesh_digest: StableDigest,
    },
    Explicit {
        structural_topology_digest: Digest,
    },
    ConformalMixed {
        mesh_digest: StableDigest,
        structural_topology_digest: Digest,
    },
}

impl CanonicalStructuralTopologyIdentity {
    pub const fn mesh_digest(self) -> Option<StableDigest> {
        match self {
            Self::Solid { mesh_digest } | Self::ConformalMixed { mesh_digest, .. } => {
                Some(mesh_digest)
            }
            Self::Explicit { .. } => None,
        }
    }

    pub fn discretization_digest(self) -> Digest {
        match self {
            Self::Solid { mesh_digest } => Digest::from_bytes(*mesh_digest.bytes()),
            Self::Explicit {
                structural_topology_digest,
            } => structural_topology_digest,
            Self::ConformalMixed {
                mesh_digest,
                structural_topology_digest,
            } => {
                let mut identity = Vec::with_capacity(96);
                identity.extend_from_slice(b"runmat-fea-conformal-mixed-discretization/v1\0");
                identity.extend_from_slice(mesh_digest.bytes());
                identity.extend_from_slice(structural_topology_digest.bytes());
                Digest::sha256(identity)
            }
        }
    }

    pub fn has_valid_identity(self) -> bool {
        self.mesh_digest()
            .is_none_or(|mesh_digest| mesh_digest != StableDigest::ZERO)
            && self.discretization_digest().bytes() != &[0; 32]
            && self.discretization_digest() != Digest::sha256([])
    }
}

pub(crate) fn explicit_target_node_indices(
    structural: &StructuralModel,
    selector: &str,
) -> Vec<usize> {
    let selected_node = explicit_node_selector(selector);
    structural
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| selected_node == Some(node.node_id))
        .map(|(index, _)| index)
        .chain(
            structural
                .elements
                .iter()
                .filter(|element| element.region_id == selector)
                .flat_map(|element| match &element.kind {
                    StructuralElementKind::Beam(beam) => beam.node_ids.as_slice(),
                    StructuralElementKind::Shell(shell) => shell.node_ids.as_slice(),
                })
                .filter_map(|node_id| {
                    structural
                        .nodes
                        .iter()
                        .position(|node| node.node_id == *node_id)
                }),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn explicit_node_selector(selector: &str) -> Option<u32> {
    selector.strip_prefix("node:")?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};

    #[test]
    fn topology_identity_is_stable_and_geometry_sensitive() {
        let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let structural = model.structural.as_ref().expect("explicit topology");
        let first = explicit_topology_digest(structural).expect("identity");
        assert_eq!(
            first,
            explicit_topology_digest(structural).expect("repeat identity")
        );

        let mut changed = structural.clone();
        changed.nodes[1].coordinates_m[0] += 0.25;
        assert_ne!(
            first,
            explicit_topology_digest(&changed).expect("changed identity")
        );

        let mut section_only = structural.clone();
        section_only.beam_sections[0].area_m2 *= 2.0;
        assert_eq!(
            first,
            explicit_topology_digest(&section_only).expect("section-independent identity")
        );
    }

    #[test]
    fn explicit_target_resolution_has_one_deterministic_owner() {
        let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let structural = model.structural.as_ref().expect("explicit topology");
        assert_eq!(explicit_target_node_indices(structural, "node:1"), vec![0]);
        assert_eq!(
            explicit_target_node_indices(structural, "beam_span"),
            vec![0, 1]
        );
        assert!(explicit_target_node_indices(structural, "absent").is_empty());
    }

    #[test]
    fn mixed_discretization_identity_binds_both_topology_domains() {
        let identity = CanonicalStructuralTopologyIdentity::ConformalMixed {
            mesh_digest: StableDigest::from_bytes([1; 32]),
            structural_topology_digest: Digest::from_bytes([2; 32]),
        };
        let changed_mesh = CanonicalStructuralTopologyIdentity::ConformalMixed {
            mesh_digest: StableDigest::from_bytes([3; 32]),
            structural_topology_digest: Digest::from_bytes([2; 32]),
        };
        let changed_explicit = CanonicalStructuralTopologyIdentity::ConformalMixed {
            mesh_digest: StableDigest::from_bytes([1; 32]),
            structural_topology_digest: Digest::from_bytes([4; 32]),
        };

        assert_ne!(
            identity.discretization_digest(),
            changed_mesh.discretization_digest()
        );
        assert_ne!(
            identity.discretization_digest(),
            changed_explicit.discretization_digest()
        );
    }
}
