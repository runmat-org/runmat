//! Deterministic, typed FEA example inputs. The exporter/checker serializes
//! these through the same evidence builders used by the analysis pipeline.

use runmat_geometry_core::GeometryAsset;
use runmat_meshing_core::{
    contracts::{
        artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisBoundaryEdge, AnalysisBoundaryFace,
        AnalysisMeshArtifact, AnalysisMeshNode, AnalysisMeshProvenance, AnalysisVolumeElement,
        BoundaryElementKind, MeshBackendSummary, VolumeElementKind,
    },
    quality::{AnalysisMeshQualityReport, ElementQuality},
    validate_analysis_mesh_with_options, AnalysisMeshValidationOptions, MeshSizingField,
};
use serde::Serialize;

use crate::{build_mesh_authoring_summary, build_mesh_evidence_artifact};

pub const TETRA_STL: &str = concat!(
    "solid tetra\n",
    "facet normal 0 0 -1\nouter loop\nvertex 0 0 0\nvertex 0 1 0\nvertex 1 0 0\nendloop\nendfacet\n",
    "facet normal 0 -1 0\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 0 1\nendloop\nendfacet\n",
    "facet normal -1 0 0\nouter loop\nvertex 0 0 0\nvertex 0 0 1\nvertex 0 1 0\nendloop\nendfacet\n",
    "facet normal 1 1 1\nouter loop\nvertex 1 0 0\nvertex 0 1 0\nvertex 0 0 1\nendloop\nendfacet\n",
    "endsolid tetra\n"
);

pub const TETRA_PATH: &str = "tetra.stl";
pub const MESH_PATH: &str = "analysis_mesh.json";
pub const EVIDENCE_PATH: &str = "mesh_evidence.json";
pub const SUMMARY_PATH: &str = "mesh_summary.json";
const REGION: &str = "region_default";

pub struct AuthorStudyExampleFiles {
    pub mesh: Vec<u8>,
    pub evidence: Vec<u8>,
    pub summary: Vec<u8>,
}

impl AuthorStudyExampleFiles {
    pub fn entries(&self) -> [(&'static str, &[u8]); 4] {
        [
            (MESH_PATH, &self.mesh),
            (EVIDENCE_PATH, &self.evidence),
            (SUMMARY_PATH, &self.summary),
            (TETRA_PATH, TETRA_STL.as_bytes()),
        ]
    }
}

/// `geometry` must be the result of importing `TETRA_STL` as `TETRA_PATH`.
/// No geometry ID, revision, or source hash is guessed by the fixture builder.
pub fn author_study_example_files(
    geometry: &GeometryAsset,
) -> Result<AuthorStudyExampleFiles, Box<dyn std::error::Error>> {
    if geometry.geometry_id != format!("geo:{TETRA_PATH}")
        || geometry.revision != 1
        || !geometry
            .regions
            .iter()
            .any(|region| region.region_id == REGION)
    {
        return Err("fixture requires the imported tetra STL and its default region".into());
    }
    let mesh = tetrahedron_mesh(geometry);
    let validation = AnalysisMeshValidationOptions {
        required_boundary_region_ids: vec![REGION.to_string()],
        required_material_region_ids: vec![REGION.to_string()],
        ..AnalysisMeshValidationOptions::default()
    };
    validate_analysis_mesh_with_options(&mesh, validation.clone())
        .map_err(|error| format!("typed author-study mesh failed validation: {error:?}"))?;
    let evidence = build_mesh_evidence_artifact(&mesh, &validation);
    let summary = build_mesh_authoring_summary(&evidence);
    if !summary.solve_ready {
        return Err("typed author-study fixture is not solve-ready".into());
    }
    #[derive(Serialize)]
    struct MeshEnvelope<'a> {
        schema_version: &'static str,
        mesh_evidence_artifact_path: &'static str,
        mesh_validation_options: &'a AnalysisMeshValidationOptions,
        mesh: &'a AnalysisMeshArtifact,
    }
    #[derive(Serialize)]
    struct EvidenceEnvelope<'a> {
        schema_version: &'static str,
        mesh_validation_options: &'a AnalysisMeshValidationOptions,
        mesh_authoring_summary: &'a crate::MeshAuthoringSummary,
        mesh_evidence: &'a crate::MeshEvidenceArtifact,
    }
    #[derive(Serialize)]
    struct SummaryEnvelope<'a> {
        mesh_authoring_summary: &'a crate::MeshAuthoringSummary,
    }
    Ok(AuthorStudyExampleFiles {
        mesh: serde_json::to_vec_pretty(&MeshEnvelope {
            schema_version: "fea_study_analysis_mesh_artifact/v1",
            mesh_evidence_artifact_path: EVIDENCE_PATH,
            mesh_validation_options: &validation,
            mesh: &mesh,
        })?,
        evidence: serde_json::to_vec_pretty(&EvidenceEnvelope {
            schema_version: "fea_study_mesh_evidence_artifact/v1",
            mesh_validation_options: &validation,
            mesh_authoring_summary: &summary,
            mesh_evidence: &evidence,
        })?,
        summary: serde_json::to_vec_pretty(&SummaryEnvelope {
            mesh_authoring_summary: &summary,
        })?,
    })
}

fn tetrahedron_mesh(geometry: &GeometryAsset) -> AnalysisMeshArtifact {
    let points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let faces = [
        ("face_1", [1, 3, 2]),
        ("face_2", [1, 2, 4]),
        ("face_3", [1, 4, 3]),
        ("face_4", [2, 3, 4]),
    ];
    let edges = [
        ("edge_1_2", [1, 2], ["face_1", "face_2"]),
        ("edge_1_3", [1, 3], ["face_1", "face_3"]),
        ("edge_1_4", [1, 4], ["face_2", "face_3"]),
        ("edge_2_3", [2, 3], ["face_1", "face_4"]),
        ("edge_2_4", [2, 4], ["face_2", "face_4"]),
        ("edge_3_4", [3, 4], ["face_3", "face_4"]),
    ];
    let mut mesh = AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: "mesh_author_study_example".to_string(),
        nodes: points
            .into_iter()
            .enumerate()
            .map(|(index, coordinates_m)| AnalysisMeshNode {
                node_id: (index + 1) as u32,
                coordinates_m,
                provenance: Vec::new(),
            })
            .collect(),
        volume_elements: vec![AnalysisVolumeElement {
            element_id: "tetrahedron_1".to_string(),
            kind: VolumeElementKind::Tetrahedron4,
            node_ids: vec![1, 2, 3, 4],
            material_region_id: REGION.to_string(),
            provenance: Vec::new(),
        }],
        boundary_faces: faces
            .into_iter()
            .map(|(face_id, node_ids)| AnalysisBoundaryFace {
                face_id: face_id.to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: node_ids.to_vec(),
                adjacent_volume_element_ids: vec!["tetrahedron_1".to_string()],
                region_ids: vec![REGION.to_string()],
                provenance: Vec::new(),
            })
            .collect(),
        boundary_edges: edges
            .into_iter()
            .map(|(edge_id, node_ids, adjacent)| AnalysisBoundaryEdge {
                edge_id: edge_id.to_string(),
                node_ids,
                adjacent_boundary_face_ids: adjacent.into_iter().map(str::to_string).collect(),
                region_ids: vec![REGION.to_string()],
                provenance: Vec::new(),
            })
            .collect(),
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
        sizing: MeshSizingField::default(),
        field_topology: Vec::new(),
        backend: MeshBackendSummary {
            backend: "artifact_fixture".to_string(),
            algorithm: "artifact_fixture".to_string(),
            tetrahedron_generation_family: "artifact_fixture".to_string(),
            tetrahedron_element_count: 1,
            tetrahedron_material_region_count: 1,
            tetrahedron_recovered_component_ratio: 1.0,
            tetrahedron_recovered_boundary_face_count: 4,
            ..MeshBackendSummary::default()
        },
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "artifact_fixture".to_string(),
            source_geometry_id: geometry.geometry_id.clone(),
            source_geometry_revision: geometry.revision,
            source_geometry_sha256: (!geometry.source.sha256.is_empty())
                .then(|| geometry.source.sha256.clone()),
        },
    };
    mesh.refresh_field_topology();
    mesh
}
