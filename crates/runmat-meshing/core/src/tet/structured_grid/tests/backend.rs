use super::common::*;
use super::*;

#[test]
fn auto_backend_uses_solid_backend_by_default() {
    let geometry = cube_geometry();
    let mesh = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
        .expect("auto backend should generate with solid backend");

    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("auto solid mesh should validate");
    assert_eq!(mesh.provenance.algorithm, "plc_tet/v1");
    assert_eq!(mesh.backend.backend, "solid");
}
#[test]
fn explicit_solid_backend_generates_analysis_mesh() {
    let geometry = cube_geometry();
    let mesh = generate_analysis_mesh(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::Solid,
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("solid backend should generate an analysis mesh");

    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("solid analysis mesh should validate");
    assert_eq!(mesh.provenance.algorithm, "plc_tet/v1");
}
#[test]
fn solid_backend_carries_external_sizing_field() {
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

    let refined = generate_analysis_mesh_with_sizing(&geometry, options, &sizing)
        .expect("solid sizing-driven mesh should generate");

    assert_eq!(refined.backend.backend, "solid");
    assert_eq!(
        refined.backend.tet_element_count,
        refined.volume_elements.len()
    );
    assert_eq!(
        refined.sizing.global_target_size_m,
        Some(0.6),
        "external sizing should lower the solid target size within the growth envelope"
    );
    assert_eq!(refined.sizing.min_size_m, Some(0.4));
    assert_eq!(refined.sizing.max_size_m, Some(0.75));
    assert_eq!(refined.sizing.growth_rate, Some(1.25));
    assert!(refined.sizing.applied_samples.is_empty());
    assert_eq!(refined.sizing.rejected_samples.len(), 1);
    assert_eq!(refined.sizing.rejected_samples[0].target_size_m, 0.5);
    assert_eq!(
        refined.sizing.rejected_samples[0].status.as_str(),
        "not_inserted_by_tet_generation"
    );
    assert_eq!(
        refined.sizing.rejected_samples[0].reason.as_deref(),
        Some("structural.stress_gradient")
    );
    assert_eq!(refined.backend.tet_requested_refinement_point_count, 1);
    assert_eq!(
        refined
            .backend
            .tet_accepted_requested_refinement_point_count,
        0
    );
    assert_eq!(
        refined
            .backend
            .tet_rejected_requested_refinement_point_count,
        1
    );
}
