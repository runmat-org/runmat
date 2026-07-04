use crate::{
    contracts::{
        artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisBoundaryFace, AnalysisMeshArtifact,
        AnalysisMeshNode, AnalysisMeshProvenance, AnalysisVolumeElement, BoundaryElementKind,
        VolumeElementKind,
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
