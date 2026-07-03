use runmat_meshing_core::contracts::ProtectedBoundaryComplex;
use runmat_meshing_plc::validate::validate_protected_boundary_complex;
use runmat_meshing_tetrahedron::generate::{
    generate_solver_tetrahedron_mesh_from_plc, TetrahedronMesh,
};

use super::SolidMeshingError;

pub(super) fn generate_solid_tetrahedron_mesh(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, SolidMeshingError> {
    validate_protected_boundary_complex(plc)
        .map_err(SolidMeshingError::ProtectedBoundaryValidation)?;
    generate_solver_tetrahedron_mesh_from_plc(plc).map_err(SolidMeshingError::Tetrahedron)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::contracts::{
        MeshingStage, PlcFacet, PlcNode, PlcValidationSummary, StageEvidence, TopologyEntityId,
    };

    #[test]
    fn rejects_invalid_plc_before_tetrahedron_generation() {
        let mut plc = tetrahedron_plc();
        plc.validation.watertight = false;

        assert!(matches!(
            generate_solid_tetrahedron_mesh(&plc),
            Err(SolidMeshingError::ProtectedBoundaryValidation(_))
        ));
    }

    fn tetrahedron_plc() -> ProtectedBoundaryComplex {
        ProtectedBoundaryComplex {
            complex_id: "tetrahedron_plc".to_string(),
            nodes: vec![
                node("0", [0.0, 0.0, 0.0]),
                node("1", [1.0, 0.0, 0.0]),
                node("2", [0.0, 1.0, 0.0]),
                node("3", [0.0, 0.0, 1.0]),
            ],
            facets: vec![
                facet("f0", ["0", "2", "1"]),
                facet("f1", ["0", "1", "3"]),
                facet("f2", ["1", "2", "3"]),
                facet("f3", ["2", "0", "3"]),
            ],
            protected_edges: Vec::new(),
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
            node_id: entity(id),
            coordinates_m,
        }
    }

    fn facet(id: &str, node_ids: [&str; 3]) -> PlcFacet {
        PlcFacet {
            facet_id: entity(id),
            node_ids: [
                entity(node_ids[0]),
                entity(node_ids[1]),
                entity(node_ids[2]),
            ],
            source_face_id: entity(id),
            material_interface_ids: Vec::new(),
        }
    }

    fn entity(id: &str) -> TopologyEntityId {
        TopologyEntityId {
            stage: MeshingStage::ProtectedBoundaryComplex,
            id: id.to_string(),
        }
    }
}
