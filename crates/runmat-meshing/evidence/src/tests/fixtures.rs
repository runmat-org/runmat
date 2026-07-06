use runmat_meshing_core::{
    contracts::{
        artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisBoundaryEdge, AnalysisBoundaryFace,
        AnalysisMeshArtifact, AnalysisMeshNode, AnalysisMeshProvenance, AnalysisVolumeElement,
        BoundaryElementKind, MeshBackendSummary, VolumeElementKind,
    },
    quality::{AnalysisMeshQualityReport, ElementQuality},
};
use runmat_meshing_size::field::MeshSizingField;

pub(super) fn node(node_id: u32, coordinates_m: [f64; 3]) -> AnalysisMeshNode {
    AnalysisMeshNode {
        node_id,
        coordinates_m,
        provenance: Vec::new(),
    }
}

pub(super) fn boundary_edge(edge_id: &str, node_ids: [u32; 2]) -> AnalysisBoundaryEdge {
    AnalysisBoundaryEdge {
        edge_id: edge_id.to_string(),
        node_ids,
        adjacent_boundary_face_ids: vec!["face_1".to_string()],
        region_ids: vec!["fixed".to_string()],
        provenance: Vec::new(),
    }
}

pub(super) fn minimal_evidence_mesh() -> AnalysisMeshArtifact {
    AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: "adaptive_mesh".to_string(),
        nodes: vec![
            node(1, [0.0, 0.0, 0.0]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
        ],
        volume_elements: vec![AnalysisVolumeElement {
            element_id: "tetrahedron_1".to_string(),
            kind: VolumeElementKind::Tetrahedron4,
            node_ids: vec![1, 2, 3, 4],
            material_region_id: "solid".to_string(),
            provenance: Vec::new(),
        }],
        boundary_faces: vec![AnalysisBoundaryFace {
            face_id: "face_1".to_string(),
            kind: BoundaryElementKind::Tri3,
            node_ids: vec![1, 2, 3],
            adjacent_volume_element_ids: vec!["tetrahedron_1".to_string()],
            region_ids: vec!["fixed".to_string()],
            provenance: Vec::new(),
        }],
        boundary_edges: vec![
            boundary_edge("edge_1", [1, 2]),
            boundary_edge("edge_2", [2, 3]),
            boundary_edge("edge_3", [1, 3]),
        ],
        sizing: MeshSizingField::default(),
        quality: AnalysisMeshQualityReport {
            min_scaled_jacobian: 0.5,
            min_exact_scaled_jacobian: 0.45,
            mean_aspect_ratio: 2.0,
            max_aspect_ratio: 2.0,
            inverted_element_count: 0,
            mean_boundary_projection_error_m: 0.0,
            max_boundary_projection_error_m: 0.0,
            elements: vec![ElementQuality {
                element_id: "tetrahedron_1".to_string(),
                scaled_jacobian: 0.5,
                exact_scaled_jacobian: 0.45,
                aspect_ratio: 2.0,
                volume_m3: 1.0 / 6.0,
            }],
        },
        field_topology: Vec::new(),
        backend: MeshBackendSummary::default(),
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "test".to_string(),
            source_geometry_id: "geo".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
        },
    }
}

#[cfg(feature = "dev-evidence")]
pub(super) fn boundary_face(face_id: &str, node_ids: [u32; 3]) -> AnalysisBoundaryFace {
    AnalysisBoundaryFace {
        face_id: face_id.to_string(),
        kind: BoundaryElementKind::Tri3,
        node_ids: node_ids.into(),
        adjacent_volume_element_ids: vec!["tetrahedron_1".to_string()],
        region_ids: vec!["fixed".to_string()],
        provenance: Vec::new(),
    }
}

#[cfg(feature = "dev-evidence")]
pub(super) fn volume_element(element_id: &str, node_ids: [u32; 4]) -> AnalysisVolumeElement {
    AnalysisVolumeElement {
        element_id: element_id.to_string(),
        kind: VolumeElementKind::Tetrahedron4,
        node_ids: node_ids.into(),
        material_region_id: "solid".to_string(),
        provenance: Vec::new(),
    }
}

#[cfg(feature = "dev-evidence")]
pub(super) fn quality_report() -> AnalysisMeshQualityReport {
    AnalysisMeshQualityReport {
        min_scaled_jacobian: 0.5,
        min_exact_scaled_jacobian: 0.45,
        mean_aspect_ratio: 2.0,
        max_aspect_ratio: 2.0,
        inverted_element_count: 0,
        mean_boundary_projection_error_m: 0.0,
        max_boundary_projection_error_m: 0.0,
        elements: vec![ElementQuality {
            element_id: "tetrahedron_1".to_string(),
            scaled_jacobian: 0.5,
            exact_scaled_jacobian: 0.45,
            aspect_ratio: 2.0,
            volume_m3: 1.0 / 6.0,
        }],
    }
}
