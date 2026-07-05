use crate::{
    contracts::{
        artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisBoundaryEdge, AnalysisBoundaryFace,
        AnalysisMeshArtifact, AnalysisMeshNode, AnalysisMeshProvenance, AnalysisVolumeElement,
        BoundaryElementKind, MeshEntityProvenance, SourceEntityKind, VolumeElementKind,
    },
    quality::AnalysisMeshQualityReport,
    size::field::MeshSizingField,
};

pub(super) fn valid_tetrahedron_mesh() -> AnalysisMeshArtifact {
    AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: "mesh_valid".to_string(),
        nodes: vec![
            AnalysisMeshNode {
                node_id: 1,
                coordinates_m: [0.0, 0.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 2,
                coordinates_m: [1.0, 0.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 3,
                coordinates_m: [0.0, 1.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 4,
                coordinates_m: [0.0, 0.0, 1.0],
                provenance: Vec::new(),
            },
        ],
        volume_elements: vec![AnalysisVolumeElement {
            element_id: "e1".to_string(),
            kind: VolumeElementKind::Tetrahedron4,
            node_ids: vec![1, 2, 3, 4],
            material_region_id: "mat_region".to_string(),
            provenance: Vec::new(),
        }],
        boundary_faces: vec![AnalysisBoundaryFace {
            face_id: "f1".to_string(),
            kind: BoundaryElementKind::Tri3,
            node_ids: vec![1, 2, 3],
            adjacent_volume_element_ids: vec!["e1".to_string()],
            region_ids: vec!["fixed".to_string()],
            provenance: Vec::new(),
        }],
        boundary_edges: Vec::new(),
        quality: AnalysisMeshQualityReport::default(),
        sizing: MeshSizingField::default(),
        field_topology: Vec::new(),
        backend: Default::default(),
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "test".to_string(),
            source_geometry_id: "geo".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
        },
    }
}

pub(super) fn solid_tetrahedron_mesh_with_plc_input_evidence() -> AnalysisMeshArtifact {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.backend = "solid".to_string();
    mesh.backend.algorithm = "plc_tetrahedron/v1".to_string();
    mesh.backend.plc_input_node_count = 4;
    mesh.backend.plc_input_facet_count = 4;
    mesh.backend.plc_input_protected_edge_count = 1;
    mesh.backend.plc_input_boundary_component_count = 1;
    mesh.backend.plc_input_boundary_component_node_count = 4;
    mesh.backend.plc_input_max_boundary_component_node_count = 4;
    mesh.backend.plc_input_shell_nesting_classified = true;
    mesh.backend.plc_input_outer_shell_count = 1;
    mesh.backend.plc_input_material_region_count = 1;
    mesh.backend.plc_input_material_region_facet_count = 4;
    mesh.backend.plc_input_surface_boundary_node_count = 4;
    mesh.backend.tetrahedron_generation_family = "single_tetrahedron".to_string();
    mesh.backend.tetrahedron_generation_attempted_family_count = 1;
    mesh.backend.tetrahedron_generation_rejected_family_count = 0;
    mesh.backend.tetrahedron_generation_selected_family_index = 1;
    mesh.backend.tetrahedron_material_region_count = 1;
    mesh.backend.tetrahedron_unclassified_material_element_count = 0;
    mesh.boundary_faces[0]
        .provenance
        .push(source_provenance(SourceEntityKind::Face, "source_face_1"));
    mesh.boundary_edges = vec![AnalysisBoundaryEdge {
        edge_id: "edge_1".to_string(),
        node_ids: [1, 2],
        adjacent_boundary_face_ids: vec!["f1".to_string()],
        region_ids: vec!["fixed".to_string()],
        provenance: vec![source_provenance(SourceEntityKind::Edge, "source_edge_1")],
    }];
    mesh
}

fn source_provenance(kind: SourceEntityKind, entity_id: &str) -> MeshEntityProvenance {
    MeshEntityProvenance {
        source_geometry_id: "geo".to_string(),
        source_geometry_revision: 1,
        source_entity_kind: kind,
        source_entity_id: entity_id.to_string(),
        region_ids: Vec::new(),
    }
}
