use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisField, AnalysisFieldValues};
use runmat_meshing_core::{
    ElementOrder, FieldTopologyLocation, SolverMeshArtifact, StableDigest,
    TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverFieldTransferMethod {
    StableIdentity,
    QuadraticEdgeInterpolation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverFieldTransferEvidence {
    pub source_artifact_digest: StableDigest,
    pub target_artifact_digest: StableDigest,
    pub topology_id: String,
    pub location: FieldTopologyLocation,
    pub component_count: usize,
    pub copied_entity_count: usize,
    pub interpolated_entity_count: usize,
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
    let mut output = Vec::with_capacity(target_map.ordered_entity_ids.len() * component_count);
    let mut copied_entity_count = 0;
    let mut interpolated_entity_count = 0;
    for entity_id in &target_map.ordered_entity_ids {
        let stable_identity = target_identities[entity_id];
        if let Some(values) = source_values.get(&stable_identity) {
            output.extend_from_slice(values);
            copied_entity_count += 1;
            continue;
        }
        let [left, right] = midpoint_edges
            .get(&stable_identity)
            .copied()
            .ok_or(SolverFieldTransferError::UnsupportedTopologyChange)?;
        let left = source_values
            .get(&left)
            .ok_or(SolverFieldTransferError::UnsupportedTopologyChange)?;
        let right = source_values
            .get(&right)
            .ok_or(SolverFieldTransferError::UnsupportedTopologyChange)?;
        output.extend(
            left.iter()
                .zip(*right)
                .map(|(left, right)| 0.5 * (left + right)),
        );
        interpolated_entity_count += 1;
    }
    let mut shape = field.shape.clone();
    shape[0] = target_map.ordered_entity_ids.len();
    let mut methods = vec![SolverFieldTransferMethod::StableIdentity];
    if interpolated_entity_count > 0 {
        methods.push(SolverFieldTransferMethod::QuadraticEdgeInterpolation);
    }
    Ok(SolverFieldTransferResult {
        field: AnalysisField::host_f64(field.field_id.clone(), shape, output),
        evidence: SolverFieldTransferEvidence {
            source_artifact_digest: source.canonical_digest,
            target_artifact_digest: target.canonical_digest,
            topology_id: topology_id.to_owned(),
            location: source_map.location,
            component_count,
            copied_entity_count,
            interpolated_entity_count,
            methods,
        },
    })
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

    #[test]
    fn p_elevation_interpolates_nodes_and_preserves_element_history() {
        let linear = artifact(ElementOrder::Tet4);
        let quadratic = artifact(ElementOrder::Tet10);
        let nodal = AnalysisField::host_f64(
            "displacement",
            vec![4, 3],
            vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0],
        );
        let elevated = transfer_solver_field(&linear, &quadratic, "nodes", &nodal).unwrap();
        assert_eq!(elevated.field.shape, vec![10, 3]);
        assert_eq!(elevated.evidence.copied_entity_count, 4);
        assert_eq!(elevated.evidence.interpolated_entity_count, 6);
        let values = elevated.field.as_host_f64().unwrap();
        assert_eq!(&values[12..15], &[1.0, 0.0, 0.0]);
        assert_eq!(&values[15..18], &[1.0, 2.0, 0.0]);
        assert_eq!(&values[27..30], &[0.0, 2.0, 3.0]);

        let history = AnalysisField::host_f64("plastic_strain", vec![1, 2], vec![0.2, 0.3]);
        let transferred = transfer_solver_field(&linear, &quadratic, "elements", &history).unwrap();
        assert_eq!(transferred.field, history);
        assert_eq!(transferred.evidence.interpolated_entity_count, 0);

        let contact_state =
            AnalysisField::host_f64("contact_state", vec![4], vec![0.0, 1.0, 1.0, 0.0]);
        let transferred =
            transfer_solver_field(&linear, &quadratic, "faces", &contact_state).unwrap();
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
            transfer_solver_field(&quadratic, &linear, "nodes", &quadratic_field).unwrap();
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
            transfer_solver_field(&linear, &quadratic, "nodes", &device).unwrap_err(),
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
        let transferred = transfer_solver_field(&source, &target, "nodes", &field).unwrap();
        assert_eq!(
            transferred.field.as_host_f64().unwrap(),
            &[20.0, 10.0, 30.0, 40.0]
        );
    }
}
