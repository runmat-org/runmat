use super::common::*;
use super::*;

#[test]
fn invalid_open_shell_returns_meshing_error() {
    let mut geometry = cube_geometry();
    geometry.surface_meshes[0].triangles.pop();
    geometry.meshes[0].element_count -= 1;

    let err = generate_analysis_mesh(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredGridTetrahedron,
            ..VolumeMeshingOptions::default()
        },
    )
    .expect_err("open shell should fail");

    assert!(matches!(err, MeshingError::BoundaryInput(_)));
    assert!(err.to_string().contains("incidence"));
}
#[test]
fn unsupported_element_kind_is_rejected() {
    let geometry = cube_geometry();
    let err = generate_analysis_mesh(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredGridTetrahedron,
            element: VolumeElementKind::Hex8,
            ..VolumeMeshingOptions::default()
        },
    )
    .expect_err("hex backend is not implemented");

    assert_eq!(
        err,
        MeshingError::UnsupportedElementKind(VolumeElementKind::Hex8)
    );
}
#[test]
fn unsupported_mesh_kind_is_rejected() {
    let geometry = cube_geometry();
    let err = generate_analysis_mesh(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredGridTetrahedron,
            kind: MeshKindRequest::Surrogate,
            ..VolumeMeshingOptions::default()
        },
    )
    .expect_err("surrogate mesh kind is not an analysis mesh backend");

    assert_eq!(
        err,
        MeshingError::UnsupportedMeshKind(MeshKindRequest::Surrogate)
    );
    assert!(err.to_string().contains("unsupported analysis mesh kind"));
}
#[test]
fn invalid_sizing_envelope_options_are_rejected() {
    let geometry = cube_geometry();
    let err = generate_analysis_mesh(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredGridTetrahedron,
            min_size_m: Some(0.2),
            max_size_m: Some(0.1),
            ..VolumeMeshingOptions::default()
        },
    )
    .expect_err("invalid min/max envelope should fail");

    assert_eq!(err, MeshingError::InvalidTargetSize);

    let err = generate_analysis_mesh(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredGridTetrahedron,
            growth_rate: Some(0.99),
            ..VolumeMeshingOptions::default()
        },
    )
    .expect_err("invalid growth rate should fail");

    assert_eq!(err, MeshingError::InvalidTargetSize);
}
#[test]
fn volume_meshing_options_require_backend_when_deserializing() {
    let err = serde_json::from_value::<VolumeMeshingOptions>(serde_json::json!({
        "kind": "solid",
        "element": "tetrahedron4",
        "element_order": "linear",
        "profile": "analysis_ready",
        "max_elements": 250000,
        "target_size": "auto",
        "refinement": {
            "strategy": "auto",
            "max_iterations": 4,
            "convergence": {
                "field_change_tolerance": 0.05,
                "energy_change_tolerance": 0.02
            },
            "focus": {
                "loads": "fine",
                "constraints": "fine",
                "interfaces": "normal",
                "curvature": true,
                "small_features": true
            }
        }
    }))
    .expect_err("serialized mesh options must name the backend explicitly");

    assert!(err.to_string().contains("missing field `backend`"));
}
