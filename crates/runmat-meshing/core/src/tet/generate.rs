use std::collections::{BTreeMap, BTreeSet};

pub use crate::contracts::{Tet4Element, TetBoundaryFace, TetMesh, TetMeshNode};
use crate::{
    contracts::{MeshingStage, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId},
    predicate::{tet_scaled_jacobian, tet_signed_volume},
};

pub const MODULE_PURPOSE: &str = "deterministic Tet4 generation from a validated PLC";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetGenerationError {
    InvalidProtectedBoundaryComplex,
    EmptyProtectedBoundaryComplex,
    MissingPlcNode { node_id: String },
    NonFinitePlcNode { node_id: String },
    NonFiniteInteriorPoint,
    DegeneratePlcBounds,
    UnsupportedStructuredBoxPlc,
    DegenerateBoundaryFacet { facet_id: String },
}

impl std::fmt::Display for TetGenerationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProtectedBoundaryComplex => {
                write!(formatter, "Tet generation requires a validated PLC")
            }
            Self::EmptyProtectedBoundaryComplex => {
                write!(formatter, "validated PLC has no nodes or facets")
            }
            Self::MissingPlcNode { node_id } => {
                write!(formatter, "PLC facet references missing node {node_id}")
            }
            Self::NonFinitePlcNode { node_id } => {
                write!(formatter, "PLC node {node_id} has non-finite coordinates")
            }
            Self::NonFiniteInteriorPoint => {
                write!(formatter, "PLC interior insertion point is non-finite")
            }
            Self::DegeneratePlcBounds => {
                write!(formatter, "validated PLC bounds are degenerate")
            }
            Self::UnsupportedStructuredBoxPlc => {
                write!(
                    formatter,
                    "validated PLC is not an axis-aligned structured box"
                )
            }
            Self::DegenerateBoundaryFacet { facet_id } => {
                write!(formatter, "PLC facet {facet_id} creates a degenerate Tet4")
            }
        }
    }
}

impl std::error::Error for TetGenerationError {}

pub fn generate_initial_tet_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetMesh, TetGenerationError> {
    if !plc.validation.valid_for_volume_meshing() {
        return Err(TetGenerationError::InvalidProtectedBoundaryComplex);
    }
    if plc.nodes.is_empty() || plc.facets.is_empty() {
        return Err(TetGenerationError::EmptyProtectedBoundaryComplex);
    }

    let mut coordinates_by_id = BTreeMap::<TopologyEntityId, [f64; 3]>::new();
    for node in &plc.nodes {
        if !node
            .coordinates_m
            .iter()
            .all(|coordinate| coordinate.is_finite())
        {
            return Err(TetGenerationError::NonFinitePlcNode {
                node_id: node.node_id.id.clone(),
            });
        }
        coordinates_by_id.insert(node.node_id.clone(), node.coordinates_m);
    }

    let interior = plc_node_average(plc)?;
    let interior_id = TopologyEntityId {
        stage: MeshingStage::TetMesh,
        id: "tet_interior_seed_0".to_string(),
    };
    let mut nodes = plc
        .nodes
        .iter()
        .map(|node| TetMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    nodes.push(TetMeshNode {
        node_id: interior_id.clone(),
        coordinates_m: interior,
    });

    let mut elements = Vec::<Tet4Element>::with_capacity(plc.facets.len());
    let mut boundary_faces = Vec::<TetBoundaryFace>::with_capacity(plc.facets.len());
    let mut min_signed_volume = f64::INFINITY;
    for (element_index, facet) in plc.facets.iter().enumerate() {
        let mut node_ids = [
            facet.node_ids[0].clone(),
            facet.node_ids[1].clone(),
            facet.node_ids[2].clone(),
            interior_id.clone(),
        ];
        let points = [
            *coordinates_by_id.get(&facet.node_ids[0]).ok_or_else(|| {
                TetGenerationError::MissingPlcNode {
                    node_id: facet.node_ids[0].id.clone(),
                }
            })?,
            *coordinates_by_id.get(&facet.node_ids[1]).ok_or_else(|| {
                TetGenerationError::MissingPlcNode {
                    node_id: facet.node_ids[1].id.clone(),
                }
            })?,
            *coordinates_by_id.get(&facet.node_ids[2]).ok_or_else(|| {
                TetGenerationError::MissingPlcNode {
                    node_id: facet.node_ids[2].id.clone(),
                }
            })?,
            interior,
        ];
        let signed_volume = tet_signed_volume(points);
        if signed_volume.abs() <= f64::EPSILON {
            return Err(TetGenerationError::DegenerateBoundaryFacet {
                facet_id: facet.facet_id.id.clone(),
            });
        }
        if signed_volume < 0.0 {
            node_ids.swap(1, 2);
        }
        min_signed_volume = min_signed_volume.min(signed_volume.abs());

        elements.push(Tet4Element {
            element_id: TopologyEntityId {
                stage: MeshingStage::TetMesh,
                id: format!("tet_{element_index}"),
            },
            node_ids,
            material_region_id: facet
                .material_interface_ids
                .first()
                .cloned()
                .unwrap_or_else(|| "body".to_string()),
        });
        boundary_faces.push(TetBoundaryFace {
            face_id: facet.facet_id.clone(),
            node_ids: facet.node_ids.clone(),
            source_face_id: facet.source_face_id.clone(),
        });
    }

    let mut evidence = StageEvidence::complete(MeshingStage::TetMesh);
    evidence
        .entity_counts
        .insert("nodes".to_string(), nodes.len());
    evidence
        .entity_counts
        .insert("tet4_elements".to_string(), elements.len());
    evidence
        .entity_counts
        .insert("boundary_faces".to_string(), boundary_faces.len());
    evidence.min_scaled_jacobian = Some(min_signed_volume);

    Ok(TetMesh {
        mesh_id: "initial_plc_tet_mesh".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

pub fn generate_structured_box_tet_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetMesh, TetGenerationError> {
    if !plc.validation.valid_for_volume_meshing() {
        return Err(TetGenerationError::InvalidProtectedBoundaryComplex);
    }
    if plc.nodes.is_empty() || plc.facets.is_empty() {
        return Err(TetGenerationError::EmptyProtectedBoundaryComplex);
    }

    let bounds = plc_bounds(plc)?;
    validate_structured_box_plc(plc, bounds)?;
    let material_region_ids = plc_material_region_ids(plc);

    let mut nodes = plc
        .nodes
        .iter()
        .map(|node| TetMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    let corner_ids = structured_box_corner_nodes(bounds, &mut nodes);
    let tet_corners = [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ];
    let coordinates_by_id = nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut elements = Vec::<Tet4Element>::with_capacity(tet_corners.len());
    let mut min_scaled_jacobian = f64::INFINITY;
    for (tet_index, corners) in tet_corners.iter().enumerate() {
        let mut node_ids = corners.map(|corner| corner_ids[corner].clone());
        let points = node_ids.clone().map(|node_id| coordinates_by_id[&node_id]);
        if tet_signed_volume(points) < 0.0 {
            node_ids.swap(1, 2);
        }
        let points = node_ids.clone().map(|node_id| coordinates_by_id[&node_id]);
        min_scaled_jacobian = min_scaled_jacobian.min(tet_scaled_jacobian(points));
        elements.push(Tet4Element {
            element_id: TopologyEntityId {
                stage: MeshingStage::TetMesh,
                id: format!("structured_box_tet_{tet_index}"),
            },
            node_ids,
            material_region_id: material_region_ids[tet_index % material_region_ids.len()].clone(),
        });
    }

    let boundary_faces = plc
        .facets
        .iter()
        .map(|facet| TetBoundaryFace {
            face_id: facet.facet_id.clone(),
            node_ids: facet.node_ids.clone(),
            source_face_id: facet.source_face_id.clone(),
        })
        .collect::<Vec<_>>();

    let mut evidence = StageEvidence::complete(MeshingStage::TetMesh);
    evidence
        .entity_counts
        .insert("nodes".to_string(), nodes.len());
    evidence
        .entity_counts
        .insert("tet4_elements".to_string(), elements.len());
    evidence
        .entity_counts
        .insert("boundary_faces".to_string(), boundary_faces.len());
    evidence
        .entity_counts
        .insert("plc_boundary_nodes".to_string(), plc.nodes.len());
    evidence.min_scaled_jacobian = Some(min_scaled_jacobian);

    Ok(TetMesh {
        mesh_id: "structured_box_tet_mesh".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

fn plc_node_average(plc: &ProtectedBoundaryComplex) -> Result<[f64; 3], TetGenerationError> {
    let mut sum = [0.0; 3];
    for node in &plc.nodes {
        for (axis, coordinate) in node.coordinates_m.iter().enumerate() {
            sum[axis] += coordinate;
        }
    }
    let count = plc.nodes.len() as f64;
    let interior = [sum[0] / count, sum[1] / count, sum[2] / count];
    if interior.iter().all(|coordinate| coordinate.is_finite()) {
        Ok(interior)
    } else {
        Err(TetGenerationError::NonFiniteInteriorPoint)
    }
}

fn plc_bounds(plc: &ProtectedBoundaryComplex) -> Result<[[f64; 3]; 2], TetGenerationError> {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for node in &plc.nodes {
        if !node
            .coordinates_m
            .iter()
            .all(|coordinate| coordinate.is_finite())
        {
            return Err(TetGenerationError::NonFinitePlcNode {
                node_id: node.node_id.id.clone(),
            });
        }
        for axis in 0..3 {
            min[axis] = min[axis].min(node.coordinates_m[axis]);
            max[axis] = max[axis].max(node.coordinates_m[axis]);
        }
    }
    if (0..3).all(|axis| {
        min[axis].is_finite() && max[axis].is_finite() && max[axis] - min[axis] > f64::EPSILON
    }) {
        Ok([min, max])
    } else {
        Err(TetGenerationError::DegeneratePlcBounds)
    }
}

fn plc_material_region_ids(plc: &ProtectedBoundaryComplex) -> Vec<String> {
    let material_region_ids = plc
        .facets
        .iter()
        .flat_map(|facet| facet.material_interface_ids.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if material_region_ids.is_empty() {
        vec!["solid_body".to_string()]
    } else {
        material_region_ids
    }
}

fn validate_structured_box_plc(
    plc: &ProtectedBoundaryComplex,
    bounds: [[f64; 3]; 2],
) -> Result<(), TetGenerationError> {
    let [min, max] = bounds;
    let tolerance = ((max[0] - min[0])
        .abs()
        .max((max[1] - min[1]).abs())
        .max((max[2] - min[2]).abs())
        * 1.0e-9)
        .max(1.0e-12);
    let coordinates_by_id = plc
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut covered_sides = [false; 6];
    for facet in &plc.facets {
        let coordinates = facet
            .node_ids
            .iter()
            .map(|node_id| {
                coordinates_by_id.get(node_id).copied().ok_or_else(|| {
                    TetGenerationError::MissingPlcNode {
                        node_id: node_id.id.clone(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let side_index = structured_box_side_index(&coordinates, min, max, tolerance)
            .ok_or(TetGenerationError::UnsupportedStructuredBoxPlc)?;
        covered_sides[side_index] = true;
    }
    if covered_sides.iter().all(|covered| *covered) {
        Ok(())
    } else {
        Err(TetGenerationError::UnsupportedStructuredBoxPlc)
    }
}

fn structured_box_side_index(
    coordinates: &[[f64; 3]],
    min: [f64; 3],
    max: [f64; 3],
    tolerance: f64,
) -> Option<usize> {
    for axis in 0..3 {
        if coordinates
            .iter()
            .all(|point| (point[axis] - min[axis]).abs() <= tolerance)
        {
            return Some(axis * 2);
        }
        if coordinates
            .iter()
            .all(|point| (point[axis] - max[axis]).abs() <= tolerance)
        {
            return Some(axis * 2 + 1);
        }
    }
    None
}

fn structured_box_corner_nodes(
    bounds: [[f64; 3]; 2],
    nodes: &mut Vec<TetMeshNode>,
) -> [TopologyEntityId; 8] {
    let [min, max] = bounds;
    let corners = [
        [min[0], min[1], min[2]],
        [max[0], min[1], min[2]],
        [min[0], max[1], min[2]],
        [max[0], max[1], min[2]],
        [min[0], min[1], max[2]],
        [max[0], min[1], max[2]],
        [min[0], max[1], max[2]],
        [max[0], max[1], max[2]],
    ];
    corners.each_ref().map(|coordinates| {
        let node_id = TopologyEntityId {
            stage: MeshingStage::TetMesh,
            id: format!("structured_box_node_{}", nodes.len()),
        };
        nodes.push(TetMeshNode {
            node_id: node_id.clone(),
            coordinates_m: *coordinates,
        });
        node_id
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{
        PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary, ProtectedBoundaryComplex,
    };

    #[test]
    fn generates_positive_tets_from_validated_tetra_plc() {
        let mesh = generate_initial_tet_mesh_from_plc(&tetra_plc())
            .expect("validated tetra PLC should generate an initial Tet mesh");

        assert_eq!(mesh.nodes.len(), 5);
        assert_eq!(mesh.elements.len(), 4);
        assert_eq!(mesh.boundary_faces.len(), 4);
        assert!(!mesh.recovery_complete);
        assert!(!mesh.quality_optimized);
        assert_eq!(mesh.evidence.entity_counts["tet4_elements"], 4);
        assert!(mesh.evidence.min_scaled_jacobian.expect("volume evidence") > 0.0);
    }

    #[test]
    fn rejects_unvalidated_plc_before_tet_generation() {
        let mut plc = tetra_plc();
        plc.validation.watertight = false;

        assert_eq!(
            generate_initial_tet_mesh_from_plc(&plc),
            Err(TetGenerationError::InvalidProtectedBoundaryComplex)
        );
    }

    #[test]
    fn rejects_degenerate_plc_facet() {
        let mut plc = tetra_plc();
        plc.facets[0].node_ids = [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        ];

        assert!(matches!(
            generate_initial_tet_mesh_from_plc(&plc),
            Err(TetGenerationError::DegenerateBoundaryFacet { .. })
        ));
    }

    #[test]
    fn generates_structured_box_tets_from_validated_plc_bounds() {
        let mesh = generate_structured_box_tet_mesh_from_plc(&box_plc())
            .expect("validated box PLC should generate structured Tet mesh");

        assert_eq!(mesh.elements.len(), 6);
        assert_eq!(mesh.boundary_faces.len(), 12);
        assert_eq!(mesh.evidence.entity_counts["plc_boundary_nodes"], 8);
        assert!(mesh.evidence.min_scaled_jacobian.expect("quality") >= 0.15);
        for element in &mesh.elements {
            let points = element.node_ids.clone().map(|node_id| {
                mesh.nodes
                    .iter()
                    .find(|node| node.node_id == node_id)
                    .expect("node exists")
                    .coordinates_m
            });
            assert!(tet_signed_volume(points) > 0.0);
        }
    }

    #[test]
    fn structured_box_generation_rejects_degenerate_bounds() {
        let mut plc = tetra_plc();
        for node in &mut plc.nodes {
            node.coordinates_m[2] = 0.0;
        }

        assert_eq!(
            generate_structured_box_tet_mesh_from_plc(&plc),
            Err(TetGenerationError::DegeneratePlcBounds)
        );
    }

    #[test]
    fn structured_box_generation_rejects_non_box_plc() {
        assert_eq!(
            generate_structured_box_tet_mesh_from_plc(&tetra_plc()),
            Err(TetGenerationError::UnsupportedStructuredBoxPlc)
        );
    }

    fn tetra_plc() -> ProtectedBoundaryComplex {
        ProtectedBoundaryComplex {
            complex_id: "tetra".to_string(),
            nodes: vec![
                node("0", [0.0, 0.0, 0.0]),
                node("1", [1.0, 0.0, 0.0]),
                node("2", [0.0, 1.0, 0.0]),
                node("3", [0.0, 0.0, 1.0]),
            ],
            facets: vec![
                facet("0", ["0", "2", "1"]),
                facet("1", ["0", "1", "3"]),
                facet("2", ["1", "2", "3"]),
                facet("3", ["2", "0", "3"]),
            ],
            protected_edges: Vec::<PlcProtectedEdge>::new(),
            validation: PlcValidationSummary {
                watertight: true,
                manifold: true,
                shell_nesting_classified: true,
                material_interfaces_classified: true,
            },
            evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
        }
    }

    fn box_plc() -> ProtectedBoundaryComplex {
        ProtectedBoundaryComplex {
            complex_id: "box".to_string(),
            nodes: vec![
                node("0", [0.0, 0.0, 0.0]),
                node("1", [1.0, 0.0, 0.0]),
                node("2", [1.0, 1.0, 0.0]),
                node("3", [0.0, 1.0, 0.0]),
                node("4", [0.0, 0.0, 1.0]),
                node("5", [1.0, 0.0, 1.0]),
                node("6", [1.0, 1.0, 1.0]),
                node("7", [0.0, 1.0, 1.0]),
            ],
            facets: vec![
                facet("0", ["0", "1", "2"]),
                facet("1", ["0", "2", "3"]),
                facet("2", ["4", "6", "5"]),
                facet("3", ["4", "7", "6"]),
                facet("4", ["0", "4", "5"]),
                facet("5", ["0", "5", "1"]),
                facet("6", ["1", "5", "6"]),
                facet("7", ["1", "6", "2"]),
                facet("8", ["2", "6", "7"]),
                facet("9", ["2", "7", "3"]),
                facet("10", ["3", "7", "4"]),
                facet("11", ["3", "4", "0"]),
            ],
            protected_edges: Vec::<PlcProtectedEdge>::new(),
            validation: PlcValidationSummary {
                watertight: true,
                manifold: true,
                shell_nesting_classified: true,
                material_interfaces_classified: true,
            },
            evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
        }
    }

    fn node(id: &str, coordinates_m: [f64; 3]) -> PlcNode {
        PlcNode {
            node_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
            coordinates_m,
        }
    }

    fn facet(id: &str, node_ids: [&str; 3]) -> PlcFacet {
        PlcFacet {
            facet_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, node_ids[0]),
                entity(MeshingStage::ProtectedBoundaryComplex, node_ids[1]),
                entity(MeshingStage::ProtectedBoundaryComplex, node_ids[2]),
            ],
            source_face_id: entity(MeshingStage::SurfaceMesh, id),
            material_interface_ids: vec!["body".to_string()],
        }
    }

    fn entity(stage: MeshingStage, id: &str) -> TopologyEntityId {
        TopologyEntityId {
            stage,
            id: id.to_string(),
        }
    }
}
