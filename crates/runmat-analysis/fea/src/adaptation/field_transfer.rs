use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisField, AnalysisFieldValues};
use runmat_meshing_core::{
    ElementOrder, FieldTopologyLocation, SolverEntityTransfer, SolverMeshArtifact,
    SolverMeshTransferMap, SolverTransferMethod, StableDigest, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverFieldTransferMethod {
    StableIdentity,
    QuadraticEdgeInterpolation,
    BarycentricInterpolation,
    CentroidProjection,
    ConservativeProjection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverFieldTransferEvidence {
    pub source_artifact_digest: StableDigest,
    pub target_artifact_digest: StableDigest,
    pub topology_id: String,
    pub location: FieldTopologyLocation,
    pub component_count: usize,
    pub copied_entity_count: usize,
    pub projected_entity_count: usize,
    pub methods: Vec<SolverFieldTransferMethod>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolverFieldTransferResult {
    pub field: AnalysisField,
    pub evidence: SolverFieldTransferEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverFieldTransferError {
    InvalidSourceArtifact(String),
    InvalidTargetArtifact(String),
    InvalidTransferMap(String),
    GeometryRevisionMismatch,
    MissingSourceTopology,
    MissingTargetTopology,
    TopologyLocationMismatch,
    InvalidFieldShape,
    DeviceFieldRequiresHostTransfer,
    UnsupportedTopologyChange,
    InconsistentQuadraticConnectivity,
}

pub fn transfer_solver_field(
    source: &SolverMeshArtifact,
    target: &SolverMeshArtifact,
    transfer_map: Option<&SolverMeshTransferMap>,
    topology_id: &str,
    field: &AnalysisField,
) -> Result<SolverFieldTransferResult, SolverFieldTransferError> {
    source
        .validate()
        .map_err(|failure| SolverFieldTransferError::InvalidSourceArtifact(failure.to_string()))?;
    target
        .validate()
        .map_err(|failure| SolverFieldTransferError::InvalidTargetArtifact(failure.to_string()))?;
    if source.geometry != target.geometry {
        return Err(SolverFieldTransferError::GeometryRevisionMismatch);
    }
    if let Some(transfer_map) = transfer_map {
        transfer_map
            .validate_against(source, target)
            .map_err(|failure| SolverFieldTransferError::InvalidTransferMap(failure.to_string()))?;
    }
    let source_map = source
        .topology
        .field_topologies
        .iter()
        .find(|map| map.topology_id == topology_id)
        .ok_or(SolverFieldTransferError::MissingSourceTopology)?;
    let target_map = target
        .topology
        .field_topologies
        .iter()
        .find(|map| map.topology_id == topology_id)
        .ok_or(SolverFieldTransferError::MissingTargetTopology)?;
    if source_map.location != target_map.location {
        return Err(SolverFieldTransferError::TopologyLocationMismatch);
    }
    let values = match &field.values {
        AnalysisFieldValues::HostF64(values) => values,
        AnalysisFieldValues::DeviceRef(_) => {
            return Err(SolverFieldTransferError::DeviceFieldRequiresHostTransfer)
        }
    };
    let component_count =
        component_count(field, source_map.ordered_entity_ids.len(), values.len())?;
    let source_identities = stable_identities(source, source_map.location);
    let target_identities = stable_identities(target, target_map.location);
    let source_values = source_map
        .ordered_entity_ids
        .iter()
        .enumerate()
        .map(|(index, entity_id)| {
            let start = index * component_count;
            (
                source_identities[entity_id],
                &values[start..start + component_count],
            )
        })
        .collect::<BTreeMap<_, _>>();
    let midpoint_edges = if source_map.location == FieldTopologyLocation::Node {
        target_midpoint_edges(target)?
    } else {
        BTreeMap::new()
    };
    let projections = transfer_map
        .map(|transfer_map| transfer_map_for_location(transfer_map, source_map.location))
        .unwrap_or_default();
    let mut output = Vec::with_capacity(target_map.ordered_entity_ids.len() * component_count);
    let mut copied_entity_count = 0;
    let mut projected_entity_count = 0;
    let mut applied_projection_methods = Vec::new();
    let mut resolved_values = source_values
        .iter()
        .map(|(identity, values)| (*identity, values.to_vec()))
        .collect::<BTreeMap<_, _>>();
    for entity_id in &target_map.ordered_entity_ids {
        let stable_identity = target_identities[entity_id];
        if let Some(values) = source_values.get(&stable_identity) {
            output.extend_from_slice(values);
            copied_entity_count += 1;
            continue;
        }
        let (values, method) = if let Some(projection) = projections.get(&stable_identity) {
            (
                apply_projection(projection, &source_values, component_count)?,
                field_transfer_method(projection.method),
            )
        } else if let Some([left, right]) = midpoint_edges.get(&stable_identity).copied() {
            let left = resolved_values
                .get(&left)
                .ok_or(SolverFieldTransferError::UnsupportedTopologyChange)?;
            let right = resolved_values
                .get(&right)
                .ok_or(SolverFieldTransferError::UnsupportedTopologyChange)?;
            (
                left.iter()
                    .zip(right)
                    .map(|(left, right)| 0.5 * (left + right))
                    .collect(),
                SolverFieldTransferMethod::QuadraticEdgeInterpolation,
            )
        } else {
            return Err(SolverFieldTransferError::UnsupportedTopologyChange);
        };
        output.extend_from_slice(&values);
        resolved_values.insert(stable_identity, values);
        projected_entity_count += 1;
        if !applied_projection_methods.contains(&method) {
            applied_projection_methods.push(method);
        }
    }
    let mut shape = field.shape.clone();
    shape[0] = target_map.ordered_entity_ids.len();
    let mut methods = vec![SolverFieldTransferMethod::StableIdentity];
    methods.extend(applied_projection_methods);
    Ok(SolverFieldTransferResult {
        field: AnalysisField::host_f64(field.field_id.clone(), shape, output),
        evidence: SolverFieldTransferEvidence {
            source_artifact_digest: source.canonical_digest,
            target_artifact_digest: target.canonical_digest,
            topology_id: topology_id.to_owned(),
            location: source_map.location,
            component_count,
            copied_entity_count,
            projected_entity_count,
            methods,
        },
    })
}

fn transfer_map_for_location(
    transfer_map: &SolverMeshTransferMap,
    location: FieldTopologyLocation,
) -> BTreeMap<StableDigest, &SolverEntityTransfer> {
    let transfers = match location {
        FieldTopologyLocation::Node => &transfer_map.node_transfers,
        FieldTopologyLocation::VolumeElement => &transfer_map.volume_element_transfers,
        FieldTopologyLocation::BoundaryFace => &transfer_map.boundary_face_transfers,
        FieldTopologyLocation::BoundaryEdge => &transfer_map.boundary_edge_transfers,
    };
    transfers
        .iter()
        .map(|transfer| (transfer.target_stable_identity, transfer))
        .collect()
}

fn apply_projection(
    projection: &SolverEntityTransfer,
    source_values: &BTreeMap<StableDigest, &[f64]>,
    component_count: usize,
) -> Result<Vec<f64>, SolverFieldTransferError> {
    let mut result = vec![0.0; component_count];
    for source in &projection.sources {
        let values = source_values
            .get(&source.stable_identity)
            .ok_or(SolverFieldTransferError::UnsupportedTopologyChange)?;
        for (result, value) in result.iter_mut().zip(*values) {
            *result += source.weight * value;
        }
    }
    Ok(result)
}

fn field_transfer_method(method: SolverTransferMethod) -> SolverFieldTransferMethod {
    match method {
        SolverTransferMethod::BarycentricInterpolation => {
            SolverFieldTransferMethod::BarycentricInterpolation
        }
        SolverTransferMethod::CentroidProjection => SolverFieldTransferMethod::CentroidProjection,
        SolverTransferMethod::ConservativeProjection => {
            SolverFieldTransferMethod::ConservativeProjection
        }
    }
}

fn stable_identities(
    artifact: &SolverMeshArtifact,
    location: FieldTopologyLocation,
) -> BTreeMap<u64, StableDigest> {
    match location {
        FieldTopologyLocation::Node => artifact
            .topology
            .nodes
            .iter()
            .map(|entity| (entity.node_id, entity.stable_identity))
            .collect(),
        FieldTopologyLocation::VolumeElement => artifact
            .topology
            .volume_elements
            .iter()
            .map(|entity| (entity.element_id, entity.stable_identity))
            .collect(),
        FieldTopologyLocation::BoundaryFace => artifact
            .topology
            .boundary_faces
            .iter()
            .map(|entity| (entity.face_id, entity.stable_identity))
            .collect(),
        FieldTopologyLocation::BoundaryEdge => artifact
            .topology
            .boundary_edges
            .iter()
            .map(|entity| (entity.edge_id, entity.stable_identity))
            .collect(),
    }
}

fn component_count(
    field: &AnalysisField,
    entity_count: usize,
    value_count: usize,
) -> Result<usize, SolverFieldTransferError> {
    if field.shape.is_empty() || field.shape[0] != entity_count {
        return Err(SolverFieldTransferError::InvalidFieldShape);
    }
    let components = field.shape[1..]
        .iter()
        .try_fold(1_usize, |product, count| product.checked_mul(*count))
        .ok_or(SolverFieldTransferError::InvalidFieldShape)?;
    if components == 0
        || entity_count
            .checked_mul(components)
            .is_none_or(|expected| expected != value_count)
    {
        return Err(SolverFieldTransferError::InvalidFieldShape);
    }
    Ok(components)
}

fn target_midpoint_edges(
    target: &SolverMeshArtifact,
) -> Result<BTreeMap<StableDigest, [StableDigest; 2]>, SolverFieldTransferError> {
    if target.resolved_request.element_order == ElementOrder::Tet4 {
        return Ok(BTreeMap::new());
    }
    let mut result = BTreeMap::new();
    let identities = stable_identities(target, FieldTopologyLocation::Node);
    for element in &target.topology.volume_elements {
        for (local_edge, corners) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.iter().enumerate() {
            let mut edge = [
                identities[&element.node_ids[corners[0]]],
                identities[&element.node_ids[corners[1]]],
            ];
            edge.sort_unstable();
            let midpoint = identities[&element.node_ids[4 + local_edge]];
            if result
                .insert(midpoint, edge)
                .is_some_and(|existing| existing != edge)
            {
                return Err(SolverFieldTransferError::InconsistentQuadraticConnectivity);
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::solver_solid::tests::artifact;
    use runmat_meshing_core::{
        solver_boundary_edge_identity, solver_boundary_face_identity,
        solver_volume_element_identity, CanonicalMeshingContract, SolverTransferSource,
        SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
    };

    #[test]
    fn p_elevation_interpolates_nodes_and_preserves_element_history() {
        let linear = artifact(ElementOrder::Tet4);
        let quadratic = artifact(ElementOrder::Tet10);
        let nodal = AnalysisField::host_f64(
            "displacement",
            vec![4, 3],
            vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0],
        );
        let elevated = transfer_solver_field(&linear, &quadratic, None, "nodes", &nodal).unwrap();
        assert_eq!(elevated.field.shape, vec![10, 3]);
        assert_eq!(elevated.evidence.copied_entity_count, 4);
        assert_eq!(elevated.evidence.projected_entity_count, 6);
        let values = elevated.field.as_host_f64().unwrap();
        assert_eq!(&values[12..15], &[1.0, 0.0, 0.0]);
        assert_eq!(&values[15..18], &[1.0, 2.0, 0.0]);
        assert_eq!(&values[27..30], &[0.0, 2.0, 3.0]);

        let history = AnalysisField::host_f64("plastic_strain", vec![1, 2], vec![0.2, 0.3]);
        let transferred =
            transfer_solver_field(&linear, &quadratic, None, "elements", &history).unwrap();
        assert_eq!(transferred.field, history);
        assert_eq!(transferred.evidence.projected_entity_count, 0);

        let contact_state =
            AnalysisField::host_f64("contact_state", vec![4], vec![0.0, 1.0, 1.0, 0.0]);
        let transferred =
            transfer_solver_field(&linear, &quadratic, None, "faces", &contact_state).unwrap();
        assert_eq!(transferred.field, contact_state);
        assert_eq!(
            transferred.evidence.location,
            FieldTopologyLocation::BoundaryFace
        );
    }

    #[test]
    fn p_restriction_preserves_corners_and_rejects_unrelated_or_device_fields() {
        let linear = artifact(ElementOrder::Tet4);
        let quadratic = artifact(ElementOrder::Tet10);
        let values = (0..30).map(f64::from).collect::<Vec<_>>();
        let quadratic_field = AnalysisField::host_f64("state", vec![10, 3], values);
        let restricted =
            transfer_solver_field(&quadratic, &linear, None, "nodes", &quadratic_field).unwrap();
        assert_eq!(restricted.field.shape, vec![4, 3]);
        assert_eq!(
            restricted.field.as_host_f64().unwrap(),
            &(0..12).map(f64::from).collect::<Vec<_>>()
        );

        let mut unrelated = artifact(ElementOrder::Tet10);
        unrelated.geometry.geometry_revision += 1;
        unrelated.seal_canonical_digest().unwrap();
        assert_eq!(
            transfer_solver_field(
                &linear,
                &unrelated,
                None,
                "nodes",
                &AnalysisField::host_f64("x", vec![4], vec![0.0; 4])
            )
            .unwrap_err(),
            SolverFieldTransferError::GeometryRevisionMismatch
        );

        let device = AnalysisField {
            field_id: "x".into(),
            shape: vec![4],
            values: AnalysisFieldValues::DeviceRef(runmat_analysis_core::DeviceFieldRef {
                backend: "gpu".into(),
                token: "token".into(),
                element_count: 4,
            }),
        };
        assert_eq!(
            transfer_solver_field(&linear, &quadratic, None, "nodes", &device).unwrap_err(),
            SolverFieldTransferError::DeviceFieldRequiresHostTransfer
        );
    }

    #[test]
    fn transfer_rejects_bad_shape_and_unsealed_artifacts() {
        let linear = artifact(ElementOrder::Tet4);
        let quadratic = artifact(ElementOrder::Tet10);
        assert_eq!(
            transfer_solver_field(
                &linear,
                &quadratic,
                None,
                "nodes",
                &AnalysisField::host_f64("bad", vec![3], vec![0.0; 3]),
            )
            .unwrap_err(),
            SolverFieldTransferError::InvalidFieldShape
        );
        let mut tampered = quadratic;
        tampered.topology.nodes[0].coordinates_m[0] = 0.25;
        assert!(matches!(
            transfer_solver_field(
                &linear,
                &tampered,
                None,
                "nodes",
                &AnalysisField::host_f64("x", vec![4], vec![0.0; 4]),
            ),
            Err(SolverFieldTransferError::InvalidTargetArtifact(_))
        ));
    }

    #[test]
    fn transfer_uses_stable_identity_when_numeric_node_ids_shift() {
        let source = artifact(ElementOrder::Tet4);
        let mut target = source.clone();
        let first_identity = target.topology.nodes[0].stable_identity;
        let second_identity = target.topology.nodes[1].stable_identity;
        target.topology.nodes[0].stable_identity = second_identity;
        target.topology.nodes[1].stable_identity = first_identity;
        let identities = target
            .topology
            .nodes
            .iter()
            .map(|node| (node.node_id, node.stable_identity))
            .collect::<BTreeMap<_, _>>();
        for face in &mut target.topology.boundary_faces {
            face.stable_identity =
                runmat_meshing_core::solver_boundary_face_identity(std::array::from_fn(|index| {
                    identities[&face.node_ids[index]]
                }));
        }
        for edge in &mut target.topology.boundary_edges {
            edge.stable_identity =
                runmat_meshing_core::solver_boundary_edge_identity(std::array::from_fn(|index| {
                    identities[&edge.node_ids[index]]
                }));
        }
        target.seal_canonical_digest().unwrap();

        let field = AnalysisField::host_f64("temperature", vec![4], vec![10.0, 20.0, 30.0, 40.0]);
        let transferred = transfer_solver_field(&source, &target, None, "nodes", &field).unwrap();
        assert_eq!(
            transferred.field.as_host_f64().unwrap(),
            &[20.0, 10.0, 30.0, 40.0]
        );
    }

    #[test]
    fn weighted_h_transfer_projects_nodes_elements_and_boundary_state() {
        let source = artifact(ElementOrder::Tet4);
        let mut target = source.clone();
        target.topology.nodes[0].stable_identity = StableDigest::from_bytes([99; 32]);
        let identities = stable_identities(&target, FieldTopologyLocation::Node);
        target.topology.volume_elements[0].stable_identity =
            solver_volume_element_identity(std::array::from_fn(|index| {
                identities[&target.topology.volume_elements[0].node_ids[index]]
            }));
        for face in &mut target.topology.boundary_faces {
            face.stable_identity = solver_boundary_face_identity(std::array::from_fn(|index| {
                identities[&face.node_ids[index]]
            }));
        }
        for edge in &mut target.topology.boundary_edges {
            edge.stable_identity = solver_boundary_edge_identity(std::array::from_fn(|index| {
                identities[&edge.node_ids[index]]
            }));
        }
        target.seal_canonical_digest().unwrap();

        let mut transfer_map = SolverMeshTransferMap {
            schema_version: SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
            source_artifact_digest: source.canonical_digest,
            target_artifact_digest: target.canonical_digest,
            geometry: source.geometry.clone(),
            node_transfers: vec![SolverEntityTransfer {
                target_stable_identity: target.topology.nodes[0].stable_identity,
                method: SolverTransferMethod::BarycentricInterpolation,
                sources: source
                    .topology
                    .nodes
                    .iter()
                    .map(|node| SolverTransferSource {
                        stable_identity: node.stable_identity,
                        weight: 0.25,
                    })
                    .collect(),
            }],
            volume_element_transfers: changed_entity_transfers(
                source
                    .topology
                    .volume_elements
                    .iter()
                    .map(|entity| entity.stable_identity),
                target
                    .topology
                    .volume_elements
                    .iter()
                    .map(|entity| entity.stable_identity),
                SolverTransferMethod::CentroidProjection,
            ),
            boundary_face_transfers: changed_entity_transfers(
                source
                    .topology
                    .boundary_faces
                    .iter()
                    .map(|entity| entity.stable_identity),
                target
                    .topology
                    .boundary_faces
                    .iter()
                    .map(|entity| entity.stable_identity),
                SolverTransferMethod::CentroidProjection,
            ),
            boundary_edge_transfers: changed_entity_transfers(
                source
                    .topology
                    .boundary_edges
                    .iter()
                    .map(|entity| entity.stable_identity),
                target
                    .topology
                    .boundary_edges
                    .iter()
                    .map(|entity| entity.stable_identity),
                SolverTransferMethod::CentroidProjection,
            ),
        };
        transfer_map.validate_against(&source, &target).unwrap();
        let encoded = transfer_map.canonical_encode().unwrap();
        assert_eq!(
            SolverMeshTransferMap::canonical_decode(&encoded).unwrap(),
            transfer_map
        );

        let nodal = AnalysisField::host_f64("temperature", vec![4], vec![10.0, 20.0, 30.0, 40.0]);
        let transferred =
            transfer_solver_field(&source, &target, Some(&transfer_map), "nodes", &nodal).unwrap();
        assert_eq!(
            transferred.field.as_host_f64().unwrap(),
            &[25.0, 20.0, 30.0, 40.0]
        );
        assert!(transferred
            .evidence
            .methods
            .contains(&SolverFieldTransferMethod::BarycentricInterpolation));

        let history = AnalysisField::host_f64("history", vec![1, 2], vec![0.2, 0.3]);
        let transferred =
            transfer_solver_field(&source, &target, Some(&transfer_map), "elements", &history)
                .unwrap();
        assert_eq!(transferred.field, history);
        assert!(transferred
            .evidence
            .methods
            .contains(&SolverFieldTransferMethod::CentroidProjection));

        transfer_map.node_transfers[0].sources[0].weight = 0.5;
        assert!(matches!(
            transfer_solver_field(&source, &target, Some(&transfer_map), "nodes", &nodal),
            Err(SolverFieldTransferError::InvalidTransferMap(_))
        ));
    }

    fn changed_entity_transfers(
        source: impl Iterator<Item = StableDigest>,
        target: impl Iterator<Item = StableDigest>,
        method: SolverTransferMethod,
    ) -> Vec<SolverEntityTransfer> {
        let mut transfers = source
            .zip(target)
            .filter(|(source, target)| source != target)
            .map(|(source, target)| SolverEntityTransfer {
                target_stable_identity: target,
                method,
                sources: vec![SolverTransferSource {
                    stable_identity: source,
                    weight: 1.0,
                }],
            })
            .collect::<Vec<_>>();
        transfers.sort_by_key(|transfer| transfer.target_stable_identity);
        transfers
    }
}
