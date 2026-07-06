use super::*;
use runmat_meshing_core::{
    contracts::RefinementIndicatorMode,
    contracts::{
        artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisBoundaryFace, AnalysisMeshProvenance,
        AnalysisVolumeElement, BoundaryElementKind, MeshEntityProvenance, SourceEntityKind,
        VolumeElementKind,
    },
    contracts::{AnalysisMeshArtifact, MeshBackendSummary},
    quality::{AnalysisMeshQualityReport, ElementQuality},
    validation::AnalysisMeshValidationOptions,
};
use runmat_meshing_size::adaptive::{
    AdaptiveConvergenceStatus, AdaptiveIterationSummary, RefinementIndicatorStatus,
    RefinementIndicatorSummary, RefinementMarker, SizingFieldUpdate,
};
use runmat_meshing_size::field::{
    AnisotropicSizingSample, MeshSizingField, SizingSample, SizingSampleApplication,
    SizingSampleRejection,
};
use std::collections::BTreeMap;

mod adaptive;
mod artifact_summary;
mod authoring;
mod fixtures;

use fixtures::*;

#[cfg(feature = "dev-evidence")]
#[test]
fn dev_mesh_evidence_caps_debug_events() {
    let mesh = AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: "debug_mesh".to_string(),
        nodes: vec![
            node(1, [0.0, 0.0, 0.0]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
        ],
        volume_elements: vec![volume_element("tetrahedron_1", [1, 2, 3, 4])],
        boundary_faces: vec![boundary_face("face_1", [1, 2, 3])],
        boundary_edges: vec![
            boundary_edge("edge_1", [1, 2]),
            boundary_edge("edge_2", [2, 3]),
            boundary_edge("edge_3", [1, 3]),
        ],
        sizing: MeshSizingField::default(),
        quality: quality_report(),
        field_topology: Vec::new(),
        backend: MeshBackendSummary::default(),
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "test".to_string(),
            source_geometry_id: "geo".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
        },
    };

    let evidence = build_mesh_evidence_artifact_with_debug(
        &mesh,
        &AnalysisMeshValidationOptions::default(),
        vec![
            MeshDebugEvent::new("surface", "info", "surface recovery accepted"),
            MeshDebugEvent::new("volume", "warning", "Tetrahedron quality improved"),
            MeshDebugEvent::new("validation", "info", "solve readiness checked"),
        ],
        2,
    );

    let debug = evidence.debug.expect("dev evidence should include debug");
    assert_eq!(debug.event_cap, 2);
    assert_eq!(debug.event_count, 3);
    assert_eq!(debug.emitted_event_count, 2);
    assert_eq!(debug.truncated_event_count, 1);
    assert_eq!(debug.events[0].stage, "surface");
    assert_eq!(debug.events[1].stage, "volume");

    let encoded = serde_json::to_value(&debug).expect("serialize debug evidence");
    assert_eq!(encoded["events"].as_array().map(Vec::len), Some(2));
}
