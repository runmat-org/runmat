use super::common::*;
use super::*;

#[test]
fn auto_backend_is_not_owned_by_structured_grid_stage() {
    let geometry = cube_geometry();
    let err = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
        .expect_err("auto backend selection belongs to the orchestration layer");

    assert!(matches!(
        err,
        MeshingError::UnsupportedBackend(MeshBackendKind::Auto)
    ));
}
#[test]
fn explicit_solid_backend_is_not_owned_by_structured_grid_stage() {
    let geometry = cube_geometry();
    let err = generate_analysis_mesh(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::Solid,
            ..VolumeMeshingOptions::default()
        },
    )
    .expect_err("solid backend belongs to the solid meshing orchestration layer");

    assert!(matches!(
        err,
        MeshingError::UnsupportedBackend(MeshBackendKind::Solid)
    ));
}
#[test]
fn explicit_solid_backend_with_sizing_is_not_owned_by_structured_grid_stage() {
    let geometry = cube_geometry();
    let mut options = VolumeMeshingOptions {
        backend: MeshBackendKind::Solid,
        target_size: MeshTargetSize::LengthM(1.0),
        min_size_m: Some(0.4),
        max_size_m: Some(0.75),
        growth_rate: Some(1.25),
        max_elements: 10_000,
        ..VolumeMeshingOptions::default()
    };
    options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    let sizing = MeshSizingField {
        samples: vec![SizingSample {
            position_m: [0.33, 0.47, 0.61],
            target_size_m: 0.5,
            reason: Some("structural.stress_gradient".to_string()),
        }],
        ..MeshSizingField::default()
    };

    let err = generate_analysis_mesh_with_sizing(&geometry, options, &sizing)
        .expect_err("solid backend belongs to the solid meshing orchestration layer");

    assert!(matches!(
        err,
        MeshingError::UnsupportedBackend(MeshBackendKind::Solid)
    ));
}
