use std::collections::BTreeSet;

use runmat_analysis_core::StructuralModel;
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{
    CanonicalMeshingContract, PersistentEntityKind, SolverMeshArtifact, StableDigest,
};
use serde::Serialize;
use thiserror::Error;

use super::{
    explicit_node_identity, explicit_topology_digest, CanonicalStructuralTopologyIdentity,
};

const IDENTITY_PREFIX: &[u8] = b"runmat-fea-mixed-structural-topology/v1\0";
const IDENTITY_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 4_000_000, 4_000_000, 64);

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct CanonicalStructuralNode {
    pub identity: StableDigest,
    pub coordinates_m: [f64; 3],
    pub owns_solid_translation: bool,
    pub owns_explicit_rotation: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct CanonicalStructuralNodeSpace {
    pub nodes: Vec<CanonicalStructuralNode>,
    pub solid_node_count: usize,
    pub explicit_node_indices: Vec<usize>,
    pub identity: CanonicalStructuralTopologyIdentity,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub(crate) enum StructuralNodeSpaceError {
    #[error("mixed structural topology requires a valid canonical solver mesh: {0}")]
    InvalidMesh(String),
    #[error(
        "explicit node {node_id} attachment `{attachment_id}` did not resolve to one solid-mesh node"
    )]
    MissingAttachment { node_id: u32, attachment_id: String },
    #[error(
        "explicit node {node_id} attachment `{attachment_id}` resolved to {matched_node_count} solid-mesh nodes"
    )]
    AmbiguousAttachment {
        node_id: u32,
        attachment_id: String,
        matched_node_count: usize,
    },
    #[error(
        "explicit node {node_id} is {distance_m} m from attachment `{attachment_id}`, exceeding topology tolerance {tolerance_m} m"
    )]
    AttachmentCoordinateMismatch {
        node_id: u32,
        attachment_id: String,
        distance_m: f64,
        tolerance_m: f64,
    },
    #[error(
        "explicit nodes {first_node_id} and {second_node_id} both attach to the same solid-mesh node"
    )]
    DuplicateAttachment {
        first_node_id: u32,
        second_node_id: u32,
    },
    #[error("mixed structural topology identity encoding failed: {0}")]
    Identity(String),
}

impl CanonicalStructuralNodeSpace {
    pub fn derive_explicit(structural: &StructuralModel) -> Result<Self, StructuralNodeSpaceError> {
        let structural_topology_digest = explicit_topology_digest(structural)
            .map_err(|error| StructuralNodeSpaceError::Identity(error.to_string()))?;
        let nodes = structural
            .nodes
            .iter()
            .map(|node| {
                explicit_node_identity(structural_topology_digest, node.node_id, node.coordinates_m)
                    .map(|identity| CanonicalStructuralNode {
                        identity,
                        coordinates_m: node.coordinates_m,
                        owns_solid_translation: false,
                        owns_explicit_rotation: true,
                    })
                    .map_err(|error| StructuralNodeSpaceError::Identity(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let unique_identities = nodes
            .iter()
            .map(|node| node.identity)
            .collect::<BTreeSet<_>>();
        if unique_identities.len() != nodes.len() {
            return Err(StructuralNodeSpaceError::Identity(
                "canonical explicit-node identities are not unique".into(),
            ));
        }
        Ok(Self {
            explicit_node_indices: (0..nodes.len()).collect(),
            nodes,
            solid_node_count: 0,
            identity: CanonicalStructuralTopologyIdentity::Explicit {
                structural_topology_digest,
            },
        })
    }

    pub fn derive(
        mesh: &SolverMeshArtifact,
        structural: &StructuralModel,
    ) -> Result<Self, StructuralNodeSpaceError> {
        mesh.validate_canonical()
            .map_err(|error| StructuralNodeSpaceError::InvalidMesh(error.to_string()))?;
        let tolerance_m = mesh
            .resolved_request
            .tolerance
            .equivalence_tolerance_m(model_scale(mesh))
            .map_err(|error| StructuralNodeSpaceError::InvalidMesh(error.to_string()))?;
        let structural_topology_digest = explicit_topology_digest(structural)
            .map_err(|error| StructuralNodeSpaceError::Identity(error.to_string()))?;
        let mut nodes = mesh
            .topology
            .nodes
            .iter()
            .map(|node| CanonicalStructuralNode {
                identity: node.stable_identity,
                coordinates_m: node.coordinates_m,
                owns_solid_translation: true,
                owns_explicit_rotation: false,
            })
            .collect::<Vec<_>>();
        let solid_node_count = nodes.len();
        let mut explicit_node_indices = Vec::with_capacity(structural.nodes.len());
        let mut attached = std::collections::BTreeMap::<usize, u32>::new();
        for node in &structural.nodes {
            let canonical_index = if let Some(attachment) = &node.solid_attachment {
                let attachment_id = attachment_label(attachment);
                let matches = mesh
                    .topology
                    .nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, candidate)| attachment_matches(attachment, candidate))
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>();
                let canonical_index = match matches.as_slice() {
                    [] => {
                        return Err(StructuralNodeSpaceError::MissingAttachment {
                            node_id: node.node_id,
                            attachment_id,
                        })
                    }
                    [index] => *index,
                    _ => {
                        return Err(StructuralNodeSpaceError::AmbiguousAttachment {
                            node_id: node.node_id,
                            attachment_id,
                            matched_node_count: matches.len(),
                        })
                    }
                };
                if let Some(first_node_id) = attached.insert(canonical_index, node.node_id) {
                    return Err(StructuralNodeSpaceError::DuplicateAttachment {
                        first_node_id,
                        second_node_id: node.node_id,
                    });
                }
                let distance_m = distance(
                    node.coordinates_m,
                    mesh.topology.nodes[canonical_index].coordinates_m,
                );
                if !distance_m.is_finite() || distance_m > tolerance_m {
                    return Err(StructuralNodeSpaceError::AttachmentCoordinateMismatch {
                        node_id: node.node_id,
                        attachment_id,
                        distance_m,
                        tolerance_m,
                    });
                }
                nodes[canonical_index].owns_explicit_rotation = true;
                canonical_index
            } else {
                let identity = explicit_node_identity(
                    structural_topology_digest,
                    node.node_id,
                    node.coordinates_m,
                )
                .map_err(|error| StructuralNodeSpaceError::Identity(error.to_string()))?;
                nodes.push(CanonicalStructuralNode {
                    identity,
                    coordinates_m: node.coordinates_m,
                    owns_solid_translation: false,
                    owns_explicit_rotation: true,
                });
                nodes.len() - 1
            };
            explicit_node_indices.push(canonical_index);
        }
        let unique_identities = nodes
            .iter()
            .map(|node| node.identity)
            .collect::<BTreeSet<_>>();
        if unique_identities.len() != nodes.len() {
            return Err(StructuralNodeSpaceError::Identity(
                "canonical mixed-node identities are not unique".into(),
            ));
        }
        let structural_topology_digest = identity(
            mesh.canonical_digest,
            structural_topology_digest,
            &nodes,
            &explicit_node_indices,
        )?;
        Ok(Self {
            nodes,
            solid_node_count,
            explicit_node_indices,
            identity: CanonicalStructuralTopologyIdentity::ConformalMixed {
                mesh_digest: mesh.canonical_digest,
                structural_topology_digest,
            },
        })
    }
}

fn attachment_matches(
    attachment: &runmat_analysis_core::ConformalSolidAttachment,
    candidate: &runmat_meshing_core::SolverMeshNode,
) -> bool {
    match attachment {
        runmat_analysis_core::ConformalSolidAttachment::GeometryVertex { entity } => {
            entity.kind == PersistentEntityKind::Vertex
                && candidate.provenance.iter().any(|value| value == entity)
        }
        runmat_analysis_core::ConformalSolidAttachment::SolverNode { stable_identity } => {
            candidate.stable_identity.0 == *stable_identity
        }
    }
}

fn attachment_label(attachment: &runmat_analysis_core::ConformalSolidAttachment) -> String {
    match attachment {
        runmat_analysis_core::ConformalSolidAttachment::GeometryVertex { entity } => entity
            .analysis_selection_id()
            .unwrap_or_else(|_| entity.source_topology_id.clone()),
        runmat_analysis_core::ConformalSolidAttachment::SolverNode { stable_identity } => {
            stable_identity
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        }
    }
}

fn identity(
    mesh_digest: StableDigest,
    structural_topology_digest: Digest,
    nodes: &[CanonicalStructuralNode],
    explicit_node_indices: &[usize],
) -> Result<Digest, StructuralNodeSpaceError> {
    encode_contract(
        IDENTITY_PREFIX,
        "mixed-structural-topology",
        &(
            mesh_digest,
            structural_topology_digest,
            nodes,
            explicit_node_indices,
        ),
        IDENTITY_LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|error| StructuralNodeSpaceError::Identity(error.to_string()))
}

fn model_scale(mesh: &SolverMeshArtifact) -> f64 {
    let bounds = std::array::from_fn::<_, 3, _>(|axis| {
        mesh.topology.nodes.iter().fold(
            [f64::INFINITY, f64::NEG_INFINITY],
            |[minimum, maximum], node| {
                [
                    minimum.min(node.coordinates_m[axis]),
                    maximum.max(node.coordinates_m[axis]),
                ]
            },
        )
    });
    bounds
        .into_iter()
        .map(|[minimum, maximum]| (maximum - minimum).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::ConformalSolidAttachment;
    use runmat_meshing_core::{ElementOrder, PersistentEntityId, StableDigest};

    use super::*;

    #[test]
    fn explicit_node_space_uses_the_authored_topology_without_a_mesh() {
        let structural = crate::fixtures::fixture_model(
            crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
        )
        .structural
        .expect("beam topology");
        let topology = CanonicalStructuralNodeSpace::derive_explicit(&structural).unwrap();
        assert_eq!(topology.solid_node_count, 0);
        assert_eq!(topology.explicit_node_indices, vec![0, 1]);
        assert!(topology
            .nodes
            .iter()
            .all(|node| { !node.owns_solid_translation && node.owns_explicit_rotation }));
        assert!(matches!(
            topology.identity,
            super::super::CanonicalStructuralTopologyIdentity::Explicit { .. }
        ));
    }

    #[test]
    fn attached_explicit_node_upgrades_one_solid_node_without_duplicate_translation() {
        let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
        let vertex = PersistentEntityId {
            kind: PersistentEntityKind::Vertex,
            source_topology_id: "beam-root".into(),
            assembly_path: vec!["root".into()],
        };
        mesh.topology.nodes[0].provenance.push(vertex.clone());
        reseal(&mut mesh);
        let mut structural = crate::fixtures::fixture_model(
            crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
        )
        .structural
        .expect("beam topology");
        structural.nodes[0].solid_attachment =
            Some(ConformalSolidAttachment::GeometryVertex { entity: vertex });

        let projected = CanonicalStructuralNodeSpace::derive(&mesh, &structural).unwrap();
        assert_eq!(projected.solid_node_count, 4);
        assert_eq!(projected.nodes.len(), 5);
        assert_eq!(projected.explicit_node_indices, vec![0, 4]);
        assert!(projected.nodes[0].owns_solid_translation);
        assert!(projected.nodes[0].owns_explicit_rotation);
        assert!(!projected.nodes[4].owns_solid_translation);
        assert!(projected.nodes[4].owns_explicit_rotation);
    }

    #[test]
    fn attachments_fail_closed_on_missing_ambiguous_and_displaced_vertices() {
        let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
        let mut structural = crate::fixtures::fixture_model(
            crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
        )
        .structural
        .expect("beam topology");
        structural.nodes[0].solid_attachment = Some(ConformalSolidAttachment::GeometryVertex {
            entity: runmat_geometry_core::PersistentEntityId {
                kind: runmat_geometry_core::PersistentEntityKind::Vertex,
                source_topology_id: "beam-root".into(),
                assembly_path: Vec::new(),
            },
        });
        assert!(matches!(
            CanonicalStructuralNodeSpace::derive(&mesh, &structural),
            Err(StructuralNodeSpaceError::MissingAttachment { .. })
        ));

        let vertex = PersistentEntityId {
            kind: PersistentEntityKind::Vertex,
            source_topology_id: "beam-root".into(),
            assembly_path: vec!["root".into()],
        };
        mesh.topology.nodes[0].provenance.push(vertex.clone());
        mesh.topology.nodes[1].provenance.push(vertex.clone());
        structural.nodes[0].solid_attachment =
            Some(ConformalSolidAttachment::GeometryVertex { entity: vertex });
        reseal(&mut mesh);
        assert!(matches!(
            CanonicalStructuralNodeSpace::derive(&mesh, &structural),
            Err(StructuralNodeSpaceError::AmbiguousAttachment { .. })
        ));

        mesh.topology.nodes[1]
            .provenance
            .retain(|entity| entity.kind != PersistentEntityKind::Vertex);
        reseal(&mut mesh);
        structural.nodes[0].coordinates_m[0] += 0.25;
        assert!(matches!(
            CanonicalStructuralNodeSpace::derive(&mesh, &structural),
            Err(StructuralNodeSpaceError::AttachmentCoordinateMismatch { .. })
        ));
    }

    #[test]
    fn topology_identity_changes_with_attachment_ownership() {
        let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
        let vertex = PersistentEntityId {
            kind: PersistentEntityKind::Vertex,
            source_topology_id: "beam-root".into(),
            assembly_path: vec!["root".into()],
        };
        mesh.topology.nodes[0].provenance.push(vertex.clone());
        reseal(&mut mesh);
        let mut structural = crate::fixtures::fixture_model(
            crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
        )
        .structural
        .expect("beam topology");
        let disconnected = CanonicalStructuralNodeSpace::derive(&mesh, &structural).unwrap();
        structural.nodes[0].solid_attachment =
            Some(ConformalSolidAttachment::GeometryVertex { entity: vertex });
        let attached = CanonicalStructuralNodeSpace::derive(&mesh, &structural).unwrap();
        assert_ne!(disconnected.identity, attached.identity);
    }

    fn reseal(mesh: &mut SolverMeshArtifact) {
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest().expect("canonical test mesh");
    }
}
