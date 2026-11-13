//! Counted node-space construction; authored order and attachment ownership are preserved.
use super::*;
impl CanonicalStructuralNodeSpace {
    pub fn derive_solid(mesh: &SolverMeshArtifact) -> Result<Self, StructuralNodeSpaceError> {
        node_space_workspace_profile(Some(mesh), None)?;
        mesh.validate_canonical()
            .map_err(|error| StructuralNodeSpaceError::InvalidMesh(error.to_string()))?;
        let mut nodes = storage::reserve(mesh.topology.nodes.len())?;
        nodes.extend(
            mesh.topology
                .nodes
                .iter()
                .map(|node| CanonicalStructuralNode {
                    identity: node.stable_identity,
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(node.coordinates_m.canonical_value()),
                    owns_solid_translation: true,
                    owns_explicit_rotation: false,
                }),
        );
        if !unique_nodes(&nodes)? {
            return Err(StructuralNodeSpaceError::InvalidMesh(
                "canonical solid-node identities are not unique".into(),
            ));
        }
        Ok(Self {
            solid_node_count: nodes.len(),
            nodes,
            explicit_node_indices: Vec::new(),
            identity: CanonicalStructuralTopologyIdentity::Solid {
                mesh_digest: mesh.canonical_digest,
            },
        })
    }

    pub fn derive_explicit(structural: &StructuralModel) -> Result<Self, StructuralNodeSpaceError> {
        node_space_workspace_profile(None, Some(structural))?;
        let structural_topology_digest =
            explicit_topology_digest(structural).map_err(identity_error)?;
        let mut nodes = storage::reserve(structural.nodes.len())?;
        for node in &structural.nodes {
            storage::check_cancelled()?;
            nodes.push(CanonicalStructuralNode {
                identity: explicit_node_identity(
                    structural_topology_digest,
                    node.node_id,
                    node.coordinates_m.canonical_value(),
                )
                .map_err(identity_error)?,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(node.coordinates_m.canonical_value()),
                owns_solid_translation: false,
                owns_explicit_rotation: true,
            });
        }
        if !unique_nodes(&nodes)? {
            return Err(StructuralNodeSpaceError::Identity(
                "canonical explicit-node identities are not unique".into(),
            ));
        }
        let mut explicit_node_indices = storage::reserve(nodes.len())?;
        explicit_node_indices.extend(0..nodes.len());
        Ok(Self {
            explicit_node_indices,
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
        let profile = node_space_workspace_profile(Some(mesh), Some(structural))?;
        mesh.validate_canonical()
            .map_err(|error| StructuralNodeSpaceError::InvalidMesh(error.to_string()))?;
        let tolerance_m = mesh
            .resolved_request
            .tolerance
            .equivalence_tolerance_m(model_scale(mesh))
            .map_err(|error| StructuralNodeSpaceError::InvalidMesh(error.to_string()))?;
        let structural_topology_digest =
            explicit_topology_digest(structural).map_err(identity_error)?;
        let mut nodes = storage::reserve(profile.node_slots)?;
        nodes.extend(
            mesh.topology
                .nodes
                .iter()
                .map(|node| CanonicalStructuralNode {
                    identity: node.stable_identity,
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(node.coordinates_m.canonical_value()),
                    owns_solid_translation: true,
                    owns_explicit_rotation: false,
                }),
        );
        let solid_node_count = nodes.len();
        let mut explicit_node_indices = storage::reserve(structural.nodes.len())?;
        let mut attached = storage::reserve(solid_node_count)?;
        attached.resize(solid_node_count, None);
        for node in &structural.nodes {
            storage::check_cancelled()?;
            let canonical_index = if let Some(attachment) = &node.solid_attachment {
                let mut first = None;
                let mut matched_node_count = 0;
                for (index, candidate) in mesh.topology.nodes.iter().enumerate() {
                    if index % 1024 == 0 {
                        storage::check_cancelled()?;
                    }
                    if attachment_matches(attachment, candidate) {
                        first.get_or_insert(index);
                        matched_node_count += 1;
                    }
                }
                let canonical_index = match (first, matched_node_count) {
                    (None, _) => {
                        return Err(StructuralNodeSpaceError::MissingAttachment {
                            node_id: node.node_id,
                            attachment_id: attachment_label(attachment),
                        })
                    }
                    (Some(index), 1) => index,
                    _ => {
                        return Err(StructuralNodeSpaceError::AmbiguousAttachment {
                            node_id: node.node_id,
                            attachment_id: attachment_label(attachment),
                            matched_node_count,
                        })
                    }
                };
                if let Some(first_node_id) = attached[canonical_index].replace(node.node_id) {
                    return Err(StructuralNodeSpaceError::DuplicateAttachment {
                        first_node_id,
                        second_node_id: node.node_id,
                    });
                }
                let distance_m = distance(
                    node.coordinates_m.canonical_value(),
                    mesh.topology.nodes[canonical_index]
                        .coordinates_m
                        .canonical_value(),
                );
                if !distance_m.is_finite() || distance_m > tolerance_m {
                    return Err(StructuralNodeSpaceError::AttachmentCoordinateMismatch {
                        node_id: node.node_id,
                        attachment_id: attachment_label(attachment),
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
                    node.coordinates_m.canonical_value(),
                )
                .map_err(identity_error)?;
                nodes.push(CanonicalStructuralNode {
                    identity,
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(node.coordinates_m.canonical_value()),
                    owns_solid_translation: false,
                    owns_explicit_rotation: true,
                });
                nodes.len() - 1
            };
            explicit_node_indices.push(canonical_index);
        }
        if !unique_nodes(&nodes)? {
            return Err(StructuralNodeSpaceError::Identity(
                "canonical mixed-node identities are not unique".into(),
            ));
        }
        drop(attached);
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

fn unique_nodes(nodes: &[CanonicalStructuralNode]) -> Result<bool, TopologyStorageError> {
    let mut identities = storage::reserve(nodes.len())?;
    identities.extend(nodes.iter().map(|node| node.identity));
    identities.sort_unstable();
    Ok(!identities.windows(2).any(|p| p[0] == p[1]))
}
fn identity_error(
    error: super::super::identity::ExplicitStructuralTopologyError,
) -> StructuralNodeSpaceError {
    match error {
        super::super::identity::ExplicitStructuralTopologyError::Storage(error) => error.into(),
        other => StructuralNodeSpaceError::Identity(other.to_string()),
    }
}
