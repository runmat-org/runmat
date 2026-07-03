use std::collections::BTreeMap;

pub use runmat_meshing_core::contracts::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronMesh, TetrahedronMeshNode,
};
use runmat_meshing_core::{
    contracts::{MeshingStage, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId},
    predicate::tetrahedron_signed_volume,
};

pub const MODULE_PURPOSE: &str = "deterministic Tetrahedron4 generation from a validated PLC";

#[path = "generate/structured_box.rs"]
mod structured_box;
pub use structured_box::generate_structured_box_tetrahedron_mesh_from_plc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetrahedronGenerationError {
    InvalidProtectedBoundaryComplex,
    EmptyProtectedBoundaryComplex,
    MissingPlcNode { node_id: String },
    NonFinitePlcNode { node_id: String },
    NonFiniteInteriorPoint,
    DegeneratePlcBounds,
    UnsupportedStructuredBoxPlc,
    DegenerateBoundaryFacet { facet_id: String },
}

impl std::fmt::Display for TetrahedronGenerationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProtectedBoundaryComplex => {
                write!(formatter, "Tetrahedron generation requires a validated PLC")
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
                write!(
                    formatter,
                    "PLC facet {facet_id} creates a degenerate Tetrahedron4"
                )
            }
        }
    }
}

impl std::error::Error for TetrahedronGenerationError {}

pub fn generate_initial_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    if !plc.validation.valid_for_volume_meshing() {
        return Err(TetrahedronGenerationError::InvalidProtectedBoundaryComplex);
    }
    if plc.nodes.is_empty() || plc.facets.is_empty() {
        return Err(TetrahedronGenerationError::EmptyProtectedBoundaryComplex);
    }

    let mut coordinates_by_id = BTreeMap::<TopologyEntityId, [f64; 3]>::new();
    for node in &plc.nodes {
        if !node
            .coordinates_m
            .iter()
            .all(|coordinate| coordinate.is_finite())
        {
            return Err(TetrahedronGenerationError::NonFinitePlcNode {
                node_id: node.node_id.id.clone(),
            });
        }
        coordinates_by_id.insert(node.node_id.clone(), node.coordinates_m);
    }

    let interior = plc_node_average(plc)?;
    let interior_id = TopologyEntityId {
        stage: MeshingStage::TetrahedronMesh,
        id: "tetrahedron_interior_seed_0".to_string(),
    };
    let mut nodes = plc
        .nodes
        .iter()
        .map(|node| TetrahedronMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    nodes.push(TetrahedronMeshNode {
        node_id: interior_id.clone(),
        coordinates_m: interior,
    });

    let mut elements = Vec::<Tetrahedron4Element>::with_capacity(plc.facets.len());
    let mut boundary_faces = Vec::<TetrahedronBoundaryFace>::with_capacity(plc.facets.len());
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
                TetrahedronGenerationError::MissingPlcNode {
                    node_id: facet.node_ids[0].id.clone(),
                }
            })?,
            *coordinates_by_id.get(&facet.node_ids[1]).ok_or_else(|| {
                TetrahedronGenerationError::MissingPlcNode {
                    node_id: facet.node_ids[1].id.clone(),
                }
            })?,
            *coordinates_by_id.get(&facet.node_ids[2]).ok_or_else(|| {
                TetrahedronGenerationError::MissingPlcNode {
                    node_id: facet.node_ids[2].id.clone(),
                }
            })?,
            interior,
        ];
        let signed_volume = tetrahedron_signed_volume(points);
        if signed_volume.abs() <= f64::EPSILON {
            return Err(TetrahedronGenerationError::DegenerateBoundaryFacet {
                facet_id: facet.facet_id.id.clone(),
            });
        }
        if signed_volume < 0.0 {
            node_ids.swap(1, 2);
        }
        min_signed_volume = min_signed_volume.min(signed_volume.abs());

        elements.push(Tetrahedron4Element {
            element_id: TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("tetrahedron_{element_index}"),
            },
            node_ids,
            material_region_id: facet
                .material_interface_ids
                .first()
                .cloned()
                .unwrap_or_else(|| "body".to_string()),
        });
        boundary_faces.push(TetrahedronBoundaryFace {
            face_id: facet.facet_id.clone(),
            node_ids: facet.node_ids.clone(),
            source_face_id: facet.source_face_id.clone(),
        });
    }

    let mut evidence = StageEvidence::complete(MeshingStage::TetrahedronMesh);
    evidence
        .entity_counts
        .insert("nodes".to_string(), nodes.len());
    evidence
        .entity_counts
        .insert("tetrahedron4_elements".to_string(), elements.len());
    evidence
        .entity_counts
        .insert("boundary_faces".to_string(), boundary_faces.len());
    evidence.min_scaled_jacobian = Some(min_signed_volume);

    Ok(TetrahedronMesh {
        mesh_id: "initial_plc_tetrahedron_mesh".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

fn plc_node_average(
    plc: &ProtectedBoundaryComplex,
) -> Result<[f64; 3], TetrahedronGenerationError> {
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
        Err(TetrahedronGenerationError::NonFiniteInteriorPoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::contracts::{
        PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary, ProtectedBoundaryComplex,
    };

    #[test]
    fn generates_positive_tetrahedra_from_validated_tetra_plc() {
        let mesh = generate_initial_tetrahedron_mesh_from_plc(&tetra_plc())
            .expect("validated tetra PLC should generate an initial Tetrahedron mesh");

        assert_eq!(mesh.nodes.len(), 5);
        assert_eq!(mesh.elements.len(), 4);
        assert_eq!(mesh.boundary_faces.len(), 4);
        assert!(!mesh.recovery_complete);
        assert!(!mesh.quality_optimized);
        assert_eq!(mesh.evidence.entity_counts["tetrahedron4_elements"], 4);
        assert!(mesh.evidence.min_scaled_jacobian.expect("volume evidence") > 0.0);
    }

    #[test]
    fn rejects_unvalidated_plc_before_tetrahedron_generation() {
        let mut plc = tetra_plc();
        plc.validation.watertight = false;

        assert_eq!(
            generate_initial_tetrahedron_mesh_from_plc(&plc),
            Err(TetrahedronGenerationError::InvalidProtectedBoundaryComplex)
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
            generate_initial_tetrahedron_mesh_from_plc(&plc),
            Err(TetrahedronGenerationError::DegenerateBoundaryFacet { .. })
        ));
    }

    #[test]
    fn generates_structured_box_tetrahedra_from_validated_plc_bounds() {
        let mesh = generate_structured_box_tetrahedron_mesh_from_plc(&box_plc())
            .expect("validated box PLC should generate structured Tetrahedron mesh");

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
            assert!(tetrahedron_signed_volume(points) > 0.0);
        }
    }

    #[test]
    fn structured_box_generation_rejects_degenerate_bounds() {
        let mut plc = tetra_plc();
        for node in &mut plc.nodes {
            node.coordinates_m[2] = 0.0;
        }

        assert_eq!(
            generate_structured_box_tetrahedron_mesh_from_plc(&plc),
            Err(TetrahedronGenerationError::DegeneratePlcBounds)
        );
    }

    #[test]
    fn structured_box_generation_rejects_non_box_plc() {
        assert_eq!(
            generate_structured_box_tetrahedron_mesh_from_plc(&tetra_plc()),
            Err(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)
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
