use super::*;
use futures::executor::block_on;
fn cooperation() -> ScalarCooperation {
    ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "fseminf" },
        None,
    )
}
#[test]
fn off_grid_quadratic_peak_is_not_replaced_by_sample_maximum() {
    let values = (0..4)
        .map(|i| {
            let w = i as f64;
            0.04 - (w - 0.4).powi(2)
        })
        .collect::<Vec<_>>();
    assert!(values.iter().all(|v| *v < 0.0));
    let samples = Arc::new(
        Samples::new(
            values,
            1,
            4,
            Spacing {
                column: 1.0,
                row: None,
            },
        )
        .unwrap(),
    );
    let result = block_on(analyze(
        samples,
        FamilyId(0),
        MeshRevision(1),
        1e-8,
        &mut cooperation(),
    ))
    .unwrap();
    let peak = result
        .peaks
        .iter()
        .max_by(|a, b| a.value.total_cmp(&b.value))
        .unwrap();
    assert!((peak.value - 0.04).abs() < 1e-8);
    assert!((peak.location[0] - 0.4).abs() < 1e-3);
    assert!(peak.upper >= peak.value - 1e-12);
    assert_eq!(result.recommendation, Recommendation::Keep);
    assert_eq!(result.recommended.column, 1.0);
}
#[test]
fn rectangle_peak_and_frozen_stencil_preserve_column_major_axes() {
    let mut values = Vec::new();
    for column in 0..4 {
        for row in 0..4 {
            values.push(0.2 - (column as f64 - 0.3).powi(2) - 2.0 * (row as f64 - 0.7).powi(2));
        }
    }
    let samples = Arc::new(
        Samples::new(
            values,
            4,
            4,
            Spacing {
                column: 0.5,
                row: Some(0.25),
            },
        )
        .unwrap(),
    );
    let result = block_on(analyze(
        samples.clone(),
        FamilyId(2),
        MeshRevision(9),
        1e-7,
        &mut cooperation(),
    ))
    .unwrap();
    let peak = result
        .peaks
        .iter()
        .max_by(|a, b| a.value.total_cmp(&b.value))
        .unwrap();
    assert!((peak.value - 0.2).abs() < 1e-6);
    assert!((peak.location[0] - 0.15).abs() < 1e-3);
    assert!((peak.location[1] - 0.175).abs() < 1e-3);
    let shifted = samples.values.iter().map(|v| v + 0.75).collect::<Vec<_>>();
    assert!((peak.stencil.evaluate(&shifted).unwrap() - peak.value - 0.75).abs() < 1e-10);
}

#[test]
fn far_resolved_dense_mesh_proposes_fewer_samples_not_an_arbitrary_fixed_grid() {
    let spacing = 1.0 / 16.0;
    let samples = Arc::new(
        Samples::new(
            (0..17)
                .map(|i| -10.0 - (i as f64 * spacing - 0.3).powi(2))
                .collect(),
            1,
            17,
            Spacing {
                column: spacing,
                row: None,
            },
        )
        .unwrap(),
    );
    let analyzed = block_on(analyze(
        samples,
        FamilyId(0),
        MeshRevision(1),
        1e-6,
        &mut cooperation(),
    ))
    .unwrap();
    assert_eq!(analyzed.recommendation, Recommendation::Coarsen);
    assert_eq!(analyzed.recommended.column, spacing * 2.0);
}
