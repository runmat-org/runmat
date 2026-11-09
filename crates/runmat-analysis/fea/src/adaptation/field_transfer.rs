use std::collections::BTreeMap;

mod error_measurement;
pub use error_measurement::measure_solver_field_transfer_error;
mod projection;
#[cfg(test)]
mod quadratic_tests;
mod topology;

use runmat_analysis_core::{AnalysisField, AnalysisFieldValues};
use runmat_meshing_core::{
    ElementOrder, FieldTopologyLocation, SolverMeshArtifact, SolverMeshTransferMap, StableDigest,
};
use serde::{Deserialize, Serialize};

use projection::{apply_projection, field_transfer_method};
use topology::{
    component_count, stable_identities, target_midpoint_edges, transfer_map_for_location,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverFieldTransferMethod {
    StableIdentity,
    LinearEdgeInterpolation,
    QuadraticLagrangeInterpolation,
    BarycentricInterpolation,
    CentroidProjection,
    ConservativeProjection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverFieldTransferErrorEvidence {
    pub transfer: SolverFieldTransferEvidence,
    pub transferred_field_digest: StableDigest,
    pub reference_field_digest: StableDigest,
    pub absolute_l2_error: f64,
    pub relative_l2_error: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolverFieldTransferResult {
    pub field: AnalysisField,
    pub evidence: SolverFieldTransferEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverFieldTransferError {
    Cancelled,
    InvalidSourceArtifact(String),
    InvalidTargetArtifact(String),
    InvalidTransferMap(String),
    GeometryRevisionMismatch,
    MissingSourceTopology,
    MissingTargetTopology,
    TopologyLocationMismatch,
    InvalidFieldShape,
    InvalidFieldValues,
    DeviceFieldRequiresHostTransfer,
    UnsupportedTopologyChange,
    InconsistentQuadraticConnectivity,
    InvalidReferenceField,
}

pub fn transfer_solver_field(
    source: &SolverMeshArtifact,
    target: &SolverMeshArtifact,
    transfer_map: Option<&SolverMeshTransferMap>,
    topology_id: &str,
    field: &AnalysisField,
) -> Result<SolverFieldTransferResult, SolverFieldTransferError> {
    if crate::is_fea_cancelled() {
        return Err(SolverFieldTransferError::Cancelled);
    }
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
    if values.iter().any(|v| !v.is_finite()) {
        return Err(SolverFieldTransferError::InvalidFieldValues);
    }
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
    let quadratic_nodal_source = source_map.location == FieldTopologyLocation::Node
        && source.resolved_request.element_order == ElementOrder::Tet10;
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
    // Original values are already borrowed above. Retain only newly projected values
    // needed by later linear midsides; do not duplicate every original component array.
    let mut resolved_values: BTreeMap<_, Vec<f64>> = BTreeMap::new();
    for (ordinal, entity_id) in target_map.ordered_entity_ids.iter().enumerate() {
        if ordinal % 1024 == 0 && crate::is_fea_cancelled() {
            return Err(SolverFieldTransferError::Cancelled);
        }
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
            // Endpoint averaging is exact only when elevating a linear source.
            if quadratic_nodal_source {
                return Err(SolverFieldTransferError::UnsupportedTopologyChange);
            }
            let left = source_values
                .get(&left)
                .copied()
                .or_else(|| resolved_values.get(&left).map(Vec::as_slice))
                .ok_or(SolverFieldTransferError::UnsupportedTopologyChange)?;
            let right = source_values
                .get(&right)
                .copied()
                .or_else(|| resolved_values.get(&right).map(Vec::as_slice))
                .ok_or(SolverFieldTransferError::UnsupportedTopologyChange)?;
            (
                left.iter()
                    .zip(right)
                    .map(|(left, right)| projection::linear_midpoint(*left, *right))
                    .collect(),
                SolverFieldTransferMethod::LinearEdgeInterpolation,
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
        field: AnalysisField::host_f64(field.field_id.clone(), field.layout.clone(), shape, output),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::solver_solid::tests::artifact;
    use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ObservationQuantity};
    use runmat_meshing_core::{
        solver_boundary_edge_identity, solver_boundary_face_identity,
        solver_volume_element_identity, CanonicalMeshingContract, ElementOrder,
        SolverEntityTransfer, SolverTransferMethod, SolverTransferSource,
        SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
    };

    #[test]
    fn p_elevation_interpolates_nodes_and_preserves_element_history() {
        let linear = artifact(ElementOrder::Tet4);
        let quadratic = artifact(ElementOrder::Tet10);
        let nodal = AnalysisField::host_f64(
            "displacement",
            AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node),
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

        let history = AnalysisField::host_f64(
            "plastic_strain",
            AnalysisFieldLayout::components(
                ObservationQuantity::PlasticStrain,
                FieldLocation::IntegrationPoint,
                2,
            ),
            vec![1, 2],
            vec![0.2, 0.3],
        );
        let transferred =
            transfer_solver_field(&linear, &quadratic, None, "elements", &history).unwrap();
        assert_eq!(transferred.field, history);
        assert_eq!(transferred.evidence.projected_entity_count, 0);

        let contact_state = AnalysisField::host_f64(
            "contact_state",
            AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global),
            vec![4],
            vec![0.0, 1.0, 1.0, 0.0],
        );
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
        let quadratic_field = AnalysisField::host_f64(
            "state",
            AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node),
            vec![10, 3],
            values,
        );
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
                &AnalysisField::host_f64(
                    "x",
                    AnalysisFieldLayout::scalar(
                        ObservationQuantity::Residual,
                        FieldLocation::Global
                    ),
                    vec![4],
                    vec![0.0; 4]
                )
            )
            .unwrap_err(),
            SolverFieldTransferError::GeometryRevisionMismatch
        );

        let device = AnalysisField {
            field_id: "x".into(),
            layout: runmat_analysis_core::AnalysisFieldLayout::scalar(
                runmat_analysis_core::ObservationQuantity::Residual,
                runmat_analysis_core::FieldLocation::Global,
            ),
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
                &AnalysisField::host_f64(
                    "bad",
                    AnalysisFieldLayout::scalar(
                        ObservationQuantity::Residual,
                        FieldLocation::Global
                    ),
                    vec![3],
                    vec![0.0; 3]
                ),
            )
            .unwrap_err(),
            SolverFieldTransferError::InvalidFieldShape
        );
        let mut tampered = quadratic;
        {
            let quantity_change = 0.25;
            let quantity_index = 0;
            let quantity_slot = &mut tampered.topology.nodes[0].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] = quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        assert!(matches!(
            transfer_solver_field(
                &linear,
                &tampered,
                None,
                "nodes",
                &AnalysisField::host_f64(
                    "x",
                    AnalysisFieldLayout::scalar(
                        ObservationQuantity::Residual,
                        FieldLocation::Global
                    ),
                    vec![4],
                    vec![0.0; 4]
                ),
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

        let field = AnalysisField::host_f64(
            "temperature",
            AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node),
            vec![4],
            vec![10.0, 20.0, 30.0, 40.0],
        );
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

        // The transferred first component is 2, one unit from the reference 3.
        // Successive Pythagorean triples (3,4,5), (5,12,13), (13,84,85)
        // make its reference norm exact even with rescaled hypot reduction.
        // This checks exact transfer/error values, not platform libm rounding.
        let nodal = AnalysisField::host_f64(
            "temperature",
            AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node),
            vec![4],
            vec![-92.0, 4.0, 12.0, 84.0],
        );
        let transferred =
            transfer_solver_field(&source, &target, Some(&transfer_map), "nodes", &nodal).unwrap();
        assert_eq!(
            transferred.field.as_host_f64().unwrap(),
            &[2.0, 4.0, 12.0, 84.0]
        );
        assert!(transferred
            .evidence
            .methods
            .contains(&SolverFieldTransferMethod::BarycentricInterpolation));
        let reference = AnalysisField::host_f64(
            "temperature",
            AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node),
            vec![4],
            vec![3.0, 4.0, 12.0, 84.0],
        );
        let transfer_error = measure_solver_field_transfer_error(&transferred, &reference).unwrap();
        assert_eq!(transfer_error.absolute_l2_error, 1.0);
        assert_eq!(transfer_error.relative_l2_error, Some(1.0 / 85.0));
        let wrong_reference = AnalysisField::host_f64(
            "other",
            AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global),
            vec![4],
            vec![0.0; 4],
        );
        assert_eq!(
            measure_solver_field_transfer_error(&transferred, &wrong_reference),
            Err(SolverFieldTransferError::InvalidReferenceField)
        );

        let history = AnalysisField::host_f64(
            "history",
            AnalysisFieldLayout::components(
                ObservationQuantity::Residual,
                FieldLocation::Global,
                2,
            ),
            vec![1, 2],
            vec![0.2, 0.3],
        );
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
