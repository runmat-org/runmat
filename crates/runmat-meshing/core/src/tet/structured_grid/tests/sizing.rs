use super::common::*;
use super::*;

#[test]
fn sizing_field_controls_structured_tet_density() {
    let geometry = cube_geometry();
    let base_options = VolumeMeshingOptions {
        backend: MeshBackendKind::StructuredTetFallback,
        target_size: MeshTargetSize::LengthM(1.0),
        max_elements: 10_000,
        ..VolumeMeshingOptions::default()
    };
    let mut base_options = base_options;
    base_options.refinement.focus.curvature = false;
    base_options.refinement.focus.small_features = false;
    let coarse = generate_analysis_mesh(&geometry, base_options.clone())
        .expect("coarse mesh should generate");
    let sizing = MeshSizingField {
        samples: vec![SizingSample {
            position_m: [0.33, 0.47, 0.61],
            target_size_m: 0.5,
            reason: Some("structural.stress_gradient".to_string()),
        }],
        ..MeshSizingField::default()
    };

    let refined = generate_analysis_mesh_with_sizing(&geometry, base_options, &sizing)
        .expect("sizing-driven mesh should generate");

    assert!(refined.volume_elements.len() > coarse.volume_elements.len());
    assert_eq!(refined.sizing.samples.len(), 1);
    assert_eq!(
        refined.sizing.samples[0].reason.as_deref(),
        Some("structural.stress_gradient")
    );
}
#[test]
fn sizing_field_creates_local_structured_breakpoints() {
    let geometry = cube_geometry();
    let mut options = VolumeMeshingOptions {
        backend: MeshBackendKind::StructuredTetFallback,
        target_size: MeshTargetSize::LengthM(1.0),
        max_elements: 10_000,
        ..VolumeMeshingOptions::default()
    };
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = false;
    let sizing = MeshSizingField {
        global_target_size_m: Some(1.0),
        growth_rate: Some(2.0),
        samples: vec![SizingSample {
            position_m: [0.4, 0.4, 0.4],
            target_size_m: 0.25,
            reason: Some("structural.load_regions".to_string()),
        }],
        ..MeshSizingField::default()
    };

    let mesh = generate_analysis_mesh_with_sizing(&geometry, options, &sizing)
        .expect("local sizing-driven mesh should generate");

    validate_analysis_mesh(&mesh, Default::default()).expect("local mesh should validate");
    assert_eq!(mesh.sizing.applied_samples.len(), 1);
    assert_eq!(
        mesh.sizing.applied_samples[0].reason.as_deref(),
        Some("structural.load_regions")
    );
    assert_eq!(mesh.sizing.growth_rate, Some(2.0));
    assert_eq!(mesh.sizing.applied_samples[0].target_size_m, 0.5);
    assert!(mesh.sizing.applied_samples[0].inserted_breakpoint_count > 0);
    let x = unique_axis_coordinates(&mesh, 0);
    assert!(x.iter().any(|value| (*value - 0.4).abs() <= 1.0e-12));
    let spacings = x
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let min_spacing = spacings.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(min_spacing <= 0.5 + 1.0e-12);
}
#[test]
fn structured_fallback_consumes_valid_anisotropic_sizing_samples() {
    let geometry = cube_geometry();
    let mut options = VolumeMeshingOptions {
        backend: MeshBackendKind::StructuredTetFallback,
        target_size: MeshTargetSize::LengthM(1.0),
        max_elements: 10_000,
        ..VolumeMeshingOptions::default()
    };
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = false;
    let sizing = MeshSizingField {
        anisotropic_samples: vec![
            AnisotropicSizingSample {
                position_m: [0.4, 0.4, 0.4],
                target_sizes_m: [0.25, 0.5, 0.75],
                directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                reason: Some("boundary_layer".to_string()),
            },
            AnisotropicSizingSample {
                position_m: [0.6, 0.6, 0.6],
                target_sizes_m: [0.25, -0.5, 0.75],
                directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                reason: Some("cad.proximity".to_string()),
            },
        ],
        ..MeshSizingField::default()
    };

    let mesh = generate_analysis_mesh_with_sizing(&geometry, options, &sizing)
        .expect("anisotropic sizing-driven fallback mesh should generate");

    assert_eq!(mesh.sizing.applied_samples.len(), 1);
    assert_eq!(
        mesh.sizing.applied_samples[0].reason.as_deref(),
        Some("boundary_layer")
    );
    assert_eq!(mesh.sizing.applied_samples[0].target_size_m, 0.25);
    assert!(mesh.sizing.applied_samples[0].inserted_breakpoint_count > 0);
    assert!(mesh.sizing.rejected_samples.iter().any(|sample| {
        sample.reason.as_deref() == Some("cad.proximity") && sample.status == "skipped_invalid"
    }));
    let x = unique_axis_coordinates(&mesh, 0);
    assert!(x.iter().any(|value| (*value - 0.4).abs() <= 1.0e-12));
}
#[test]
fn sizing_field_reports_duplicate_and_invalid_samples() {
    let geometry = cube_geometry();
    let mut options = VolumeMeshingOptions {
        backend: MeshBackendKind::StructuredTetFallback,
        target_size: MeshTargetSize::LengthM(1.0),
        max_elements: 10_000,
        ..VolumeMeshingOptions::default()
    };
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = false;
    let sizing = MeshSizingField {
        samples: vec![
            SizingSample {
                position_m: [0.5, 0.5, 0.5],
                target_size_m: 0.5,
                reason: Some("structural.stress_gradient".to_string()),
            },
            SizingSample {
                position_m: [0.5, 0.5, 0.5],
                target_size_m: 0.5,
                reason: Some("structural.stress_gradient".to_string()),
            },
            SizingSample {
                position_m: [f64::NAN, 0.5, 0.5],
                target_size_m: 0.25,
                reason: Some("structural.invalid_position".to_string()),
            },
            SizingSample {
                position_m: [0.25, 0.25, 0.25],
                target_size_m: f64::NAN,
                reason: Some("structural.invalid_size".to_string()),
            },
        ],
        ..MeshSizingField::default()
    };

    let mesh = generate_analysis_mesh_with_sizing(&geometry, options, &sizing)
        .expect("audited sizing mesh should generate");

    assert_eq!(mesh.sizing.applied_samples.len(), 1);
    assert_eq!(
        mesh.sizing.applied_samples[0].reason.as_deref(),
        Some("structural.stress_gradient")
    );
    assert!(mesh.sizing.rejected_samples.iter().any(|rejection| {
        rejection.status == "skipped_duplicate"
            && rejection.reason.as_deref() == Some("structural.stress_gradient")
    }));
    assert!(mesh.sizing.rejected_samples.iter().any(|rejection| {
        rejection.status == "skipped_invalid"
            && rejection.reason.as_deref() == Some("structural.invalid_position")
    }));
    assert!(mesh.sizing.rejected_samples.iter().any(|rejection| {
        rejection.status == "skipped_invalid"
            && rejection.reason.as_deref() == Some("structural.invalid_size")
    }));
}
#[test]
fn sizing_field_skips_breakpoints_that_would_violate_quality() {
    let geometry = cube_geometry();
    let mut options = VolumeMeshingOptions {
        backend: MeshBackendKind::StructuredTetFallback,
        target_size: MeshTargetSize::LengthM(1.0),
        max_elements: 10_000,
        ..VolumeMeshingOptions::default()
    };
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = false;
    let sizing = MeshSizingField {
        samples: vec![SizingSample {
            position_m: [0.2, 0.2, 0.2],
            target_size_m: 0.01,
            reason: Some("structural.stress_gradient".to_string()),
        }],
        ..MeshSizingField::default()
    };

    let mesh = generate_analysis_mesh_with_sizing(&geometry, options, &sizing)
        .expect("quality-guarded local sizing mesh should generate");

    validate_analysis_mesh(&mesh, Default::default())
        .expect("quality-guarded local mesh should validate");
    assert!(mesh.quality.min_scaled_jacobian >= QualityThresholds::default().min_scaled_jacobian);
    assert!(mesh.sizing.rejected_samples.iter().any(|rejection| {
        rejection.status == "skipped_quality"
            && rejection.reason.as_deref() == Some("structural.stress_gradient")
    }));
    let min_spacing = unique_axis_coordinates(&mesh, 0)
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(f64::INFINITY, f64::min);
    assert!(min_spacing > 0.01);
}
#[test]
fn sizing_field_refinement_respects_element_budget() {
    let geometry = cube_geometry();
    let sizing = MeshSizingField {
        samples: vec![SizingSample {
            position_m: [0.5, 0.5, 0.5],
            target_size_m: 0.01,
            reason: Some("structural.stress_gradient".to_string()),
        }],
        ..MeshSizingField::default()
    };

    let mesh = generate_analysis_mesh_with_sizing(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredTetFallback,
            target_size: MeshTargetSize::LengthM(1.0),
            max_elements: 48,
            ..VolumeMeshingOptions::default()
        },
        &sizing,
    )
    .expect("budgeted sizing-driven mesh should generate");

    assert!(mesh.volume_elements.len() <= 48);
    assert_eq!(mesh.volume_elements.len(), 48);
    assert!(mesh
        .sizing
        .rejected_samples
        .iter()
        .any(|rejection| rejection.status == "skipped_budget"));
}
#[test]
fn curvature_focus_adds_geometry_sizing_samples() {
    let geometry = cube_geometry();
    let mut coarse_options = VolumeMeshingOptions {
        backend: MeshBackendKind::StructuredTetFallback,
        target_size: MeshTargetSize::LengthM(1.0),
        max_elements: 10_000,
        ..VolumeMeshingOptions::default()
    };
    coarse_options.refinement.focus.curvature = false;
    coarse_options.refinement.focus.small_features = false;
    let coarse = generate_analysis_mesh(&geometry, coarse_options.clone())
        .expect("coarse mesh should generate");

    coarse_options.refinement.focus.curvature = true;
    let focused = generate_analysis_mesh(&geometry, coarse_options)
        .expect("curvature-focused mesh should generate");

    assert!(focused.volume_elements.len() > coarse.volume_elements.len());
    assert!(focused
        .sizing
        .samples
        .iter()
        .any(|sample| sample.reason.as_deref() == Some("geometry.curvature")));
}
#[test]
fn small_feature_focus_adds_geometry_sizing_samples() {
    let geometry = thin_box_geometry();
    let mut coarse_options = VolumeMeshingOptions {
        backend: MeshBackendKind::StructuredTetFallback,
        target_size: MeshTargetSize::LengthM(1.0),
        max_elements: 10_000,
        ..VolumeMeshingOptions::default()
    };
    coarse_options.refinement.focus.curvature = false;
    coarse_options.refinement.focus.small_features = false;
    let coarse = generate_analysis_mesh(&geometry, coarse_options.clone())
        .expect("coarse thin-box mesh should generate");

    coarse_options.refinement.focus.small_features = true;
    let focused = generate_analysis_mesh(&geometry, coarse_options)
        .expect("small-feature-focused mesh should generate");

    assert!(focused.volume_elements.len() > coarse.volume_elements.len());
    assert!(focused
        .sizing
        .samples
        .iter()
        .any(|sample| sample.reason.as_deref() == Some("geometry.small_features")));
}
