use std::collections::HashMap;

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{
    quality::predicate::{
        tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_volume,
    },
    BoundaryFaceRole, ElementOrder, SolverMeshArtifact,
};
use runmat_value::{IntegerStorage, Tensor, Value};

use super::object::payload_from_value;
use super::{error, BOUNDARY_FACES_NAME, ELEMENTS_NAME, NODES_NAME, QUALITY_NAME, REGIONS_NAME};
use crate::BuiltinResult;

pub(super) fn nodes(mesh: Value) -> BuiltinResult<Value> {
    let payload = payload_from_value(NODES_NAME, &mesh)?;
    let nodes = &payload.artifact.topology.nodes;
    let mut coordinates = Vec::with_capacity(nodes.len() * 3);
    for axis in 0..3 {
        coordinates.extend(
            nodes
                .iter()
                .map(|node| node.coordinates_m.canonical_value()[axis]),
        );
    }
    let mut result = runmat_value::UnboundedStructValueBuilder::new();
    result.insert(
        "ids",
        u64_tensor(
            nodes.iter().map(|node| node.node_id).collect(),
            nodes.len(),
            1,
        )?,
    );
    result.insert(
        "coordinates",
        Value::Tensor(
            Tensor::new_2d(coordinates, nodes.len(), 3)
                .map_err(|cause| error(NODES_NAME, cause.to_string()))?,
        ),
    );
    Ok(Value::Struct(result.finish()))
}

pub(super) fn elements(mesh: Value) -> BuiltinResult<Value> {
    let payload = payload_from_value(ELEMENTS_NAME, &mesh)?;
    let elements = &payload.artifact.topology.volume_elements;
    let width = elements.first().map_or(0, |element| element.node_ids.len());
    if !elements
        .iter()
        .all(|element| element.node_ids.len() == width)
    {
        return Err(error(
            ELEMENTS_NAME,
            "mixed element connectivity widths are unsupported",
        ));
    }
    let connectivity = column_major_u64(elements.len(), width, |row, column| {
        elements[row].node_ids[column]
    });
    let mut result = runmat_value::UnboundedStructValueBuilder::new();
    result.insert(
        "ids",
        u64_tensor(
            elements.iter().map(|element| element.element_id).collect(),
            elements.len(),
            1,
        )?,
    );
    result.insert(
        "connectivity",
        u64_tensor(connectivity, elements.len(), width)?,
    );
    result.insert(
        "order",
        Value::String(
            match payload.artifact.resolved_request.element_order {
                ElementOrder::Tet4 => "tet4",
                ElementOrder::Tet10 => "tet10",
            }
            .to_string()
            .into(),
        ),
    );
    result.insert(
        "region_ids",
        string_cell(
            elements
                .iter()
                .map(|element| element.region_id.source_topology_id.clone())
                .collect(),
        )?,
    );
    Ok(Value::Struct(result.finish()))
}

pub(super) fn boundary_faces(mesh: Value) -> BuiltinResult<Value> {
    let payload = payload_from_value(BOUNDARY_FACES_NAME, &mesh)?;
    let faces = &payload.artifact.topology.boundary_faces;
    let width = faces.first().map_or(0, |face| face.node_ids.len());
    if !faces.iter().all(|face| face.node_ids.len() == width) {
        return Err(error(
            BOUNDARY_FACES_NAME,
            "mixed boundary connectivity widths are unsupported",
        ));
    }
    let connectivity = column_major_u64(faces.len(), width, |row, column| {
        faces[row].node_ids[column]
    });
    let mut result = runmat_value::UnboundedStructValueBuilder::new();
    result.insert(
        "ids",
        u64_tensor(
            faces.iter().map(|face| face.face_id).collect(),
            faces.len(),
            1,
        )?,
    );
    result.insert(
        "connectivity",
        u64_tensor(connectivity, faces.len(), width)?,
    );
    result.insert(
        "roles",
        string_cell(
            faces
                .iter()
                .map(|face| boundary_face_role(face.role).to_string())
                .collect(),
        )?,
    );
    result.insert(
        "adjacent_element_ids",
        crate::make_cell_with_shape(
            faces
                .iter()
                .map(|face| {
                    u64_tensor(
                        face.adjacent_volume_element_ids.clone(),
                        1,
                        face.adjacent_volume_element_ids.len(),
                    )
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            vec![faces.len(), 1],
        )
        .map_err(|message| error(BOUNDARY_FACES_NAME, message))?,
    );
    result.insert(
        "provenance",
        crate::make_cell_with_shape(
            faces
                .iter()
                .map(|face| {
                    crate::make_cell_with_shape(
                        face.provenance
                            .iter()
                            .map(persistent_entity_value)
                            .collect::<BuiltinResult<Vec<_>>>()?,
                        vec![1, face.provenance.len()],
                    )
                    .map_err(|message| error(BOUNDARY_FACES_NAME, message))
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            vec![faces.len(), 1],
        )
        .map_err(|message| error(BOUNDARY_FACES_NAME, message))?,
    );
    Ok(Value::Struct(result.finish()))
}

pub(super) fn regions(mesh: Value) -> BuiltinResult<Value> {
    let payload = payload_from_value(REGIONS_NAME, &mesh)?;
    let values = payload
        .artifact
        .topology
        .regions
        .iter()
        .map(|region| {
            let mut value = runmat_value::UnboundedStructValueBuilder::new();
            value.insert(
                "id",
                Value::String(region.region_id.source_topology_id.clone().into()),
            );
            value.insert(
                "kind",
                Value::String(
                    persistent_entity_kind(region.region_id.kind)
                        .to_string()
                        .into(),
                ),
            );
            value.insert(
                "assembly_path",
                string_cell(region.region_id.assembly_path.clone())?,
            );
            value.insert(
                "element_count",
                Value::Int(runmat_value::IntValue::U64(region.element_ids.len() as u64)),
            );
            Ok(Value::Struct(value.finish()))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    crate::make_cell_with_shape(values, vec![1, payload.artifact.topology.regions.len()])
        .map_err(|message| error(REGIONS_NAME, message))
}

pub(super) fn quality(mesh: Value) -> BuiltinResult<Value> {
    let payload = payload_from_value(QUALITY_NAME, &mesh)?;
    quality_value(&payload.artifact)
}

pub(super) fn quality_value(mesh: &SolverMeshArtifact) -> BuiltinResult<Value> {
    if mesh.topology.volume_elements.is_empty() {
        return Err(error(
            QUALITY_NAME,
            "solver mesh contains no volume elements",
        ));
    }
    let nodes = mesh
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m.canonical_value()))
        .collect::<HashMap<_, _>>();
    let mut ids = Vec::with_capacity(mesh.topology.volume_elements.len());
    let mut jacobians = Vec::with_capacity(ids.capacity());
    let mut aspects = Vec::with_capacity(ids.capacity());
    let mut volumes = Vec::with_capacity(ids.capacity());
    for element in &mesh.topology.volume_elements {
        let corners = element
            .node_ids
            .get(..4)
            .ok_or_else(|| error(QUALITY_NAME, "tetrahedron has fewer than four nodes"))?;
        let points = [
            *nodes
                .get(&corners[0])
                .ok_or_else(|| error(QUALITY_NAME, "missing node"))?,
            *nodes
                .get(&corners[1])
                .ok_or_else(|| error(QUALITY_NAME, "missing node"))?,
            *nodes
                .get(&corners[2])
                .ok_or_else(|| error(QUALITY_NAME, "missing node"))?,
            *nodes
                .get(&corners[3])
                .ok_or_else(|| error(QUALITY_NAME, "missing node"))?,
        ];
        let jacobian = tetrahedron_scaled_jacobian(points).map_err(|cause| {
            let identifier =
                crate::analysis::fea_run_admission::FeaRunDenied::Accounting(cause).error_code();
            crate::build_runtime_error(format!("{QUALITY_NAME}: {cause}"))
                .with_builtin(QUALITY_NAME)
                .with_identifier(identifier)
                .with_source(cause)
                .build()
        })?;
        ids.push(element.element_id);
        jacobians.push(jacobian);
        aspects.push(tetrahedron_edge_aspect_ratio(points));
        volumes.push(tetrahedron_volume(points));
    }
    let mut value = runmat_value::UnboundedStructValueBuilder::new();
    value.insert(
        "element_order",
        Value::String(
            match mesh.resolved_request.element_order {
                ElementOrder::Tet4 => "tet4",
                ElementOrder::Tet10 => "tet10",
            }
            .to_string()
            .into(),
        ),
    );
    value.insert("element_ids", u64_tensor(ids, jacobians.len(), 1)?);
    value.insert(
        "corner_scaled_jacobian",
        Value::Tensor(
            Tensor::new(jacobians.clone(), vec![jacobians.len(), 1])
                .map_err(|cause| error(QUALITY_NAME, cause.to_string()))?,
        ),
    );
    value.insert(
        "corner_aspect_ratio",
        Value::Tensor(
            Tensor::new(aspects.clone(), vec![aspects.len(), 1])
                .map_err(|cause| error(QUALITY_NAME, cause.to_string()))?,
        ),
    );
    value.insert(
        "corner_volume_m3",
        Value::Tensor(
            Tensor::new(volumes.clone(), vec![volumes.len(), 1])
                .map_err(|cause| error(QUALITY_NAME, cause.to_string()))?,
        ),
    );
    value.insert(
        "minimum_corner_scaled_jacobian",
        Value::Num(jacobians.iter().copied().fold(f64::INFINITY, f64::min)),
    );
    value.insert("mean_corner_scaled_jacobian", Value::Num(mean(&jacobians)));
    value.insert(
        "p05_corner_scaled_jacobian",
        Value::Num(percentile(&jacobians, 0.05)),
    );
    value.insert(
        "maximum_corner_aspect_ratio",
        Value::Num(aspects.iter().copied().fold(0.0, f64::max)),
    );
    value.insert("mean_corner_aspect_ratio", Value::Num(mean(&aspects)));
    value.insert(
        "p95_corner_aspect_ratio",
        Value::Num(percentile(&aspects, 0.95)),
    );
    value.insert(
        "minimum_corner_volume_m3",
        Value::Num(volumes.iter().copied().fold(f64::INFINITY, f64::min)),
    );
    value.insert("total_corner_volume_m3", Value::Num(volumes.iter().sum()));
    Ok(Value::Struct(value.finish()))
}

fn persistent_entity_value(id: &PersistentEntityId) -> BuiltinResult<Value> {
    let mut value = runmat_value::UnboundedStructValueBuilder::new();
    value.insert(
        "kind",
        Value::String(persistent_entity_kind(id.kind).to_string().into()),
    );
    value.insert(
        "source_topology_id",
        Value::String(id.source_topology_id.clone().into()),
    );
    value.insert("assembly_path", string_cell(id.assembly_path.clone())?);
    Ok(Value::Struct(value.finish()))
}

fn persistent_entity_kind(kind: PersistentEntityKind) -> &'static str {
    match kind {
        PersistentEntityKind::Assembly => "assembly",
        PersistentEntityKind::Instance => "instance",
        PersistentEntityKind::Body => "body",
        PersistentEntityKind::Lump => "lump",
        PersistentEntityKind::Solid => "solid",
        PersistentEntityKind::Shell => "shell",
        PersistentEntityKind::Face => "face",
        PersistentEntityKind::Wire => "wire",
        PersistentEntityKind::Coedge => "coedge",
        PersistentEntityKind::Edge => "edge",
        PersistentEntityKind::Vertex => "vertex",
        PersistentEntityKind::Region => "region",
        PersistentEntityKind::Contact => "contact",
    }
}

fn boundary_face_role(role: BoundaryFaceRole) -> &'static str {
    match role {
        BoundaryFaceRole::Exterior => "exterior",
        BoundaryFaceRole::ConformalInterface => "conformal_interface",
        BoundaryFaceRole::ContactPrimary => "contact_primary",
        BoundaryFaceRole::ContactSecondary => "contact_secondary",
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

fn u64_tensor(data: Vec<u64>, rows: usize, columns: usize) -> BuiltinResult<Value> {
    Tensor::new_integer(IntegerStorage::U64(data), vec![rows, columns])
        .map(Value::Tensor)
        .map_err(|cause| error("meshing", cause.to_string()))
}

fn column_major_u64(rows: usize, columns: usize, value: impl Fn(usize, usize) -> u64) -> Vec<u64> {
    let mut data = Vec::with_capacity(rows * columns);
    for column in 0..columns {
        for row in 0..rows {
            data.push(value(row, column));
        }
    }
    data
}

fn string_cell(values: Vec<String>) -> BuiltinResult<Value> {
    let columns = values.len();
    crate::make_cell_with_shape(
        values.into_iter().map(Value::string).collect(),
        vec![columns, 1],
    )
    .map_err(|message| error("meshing", message))
}
