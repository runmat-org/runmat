#![cfg(feature = "occt-native")]
//! The preview measure of imported CAD: the volume and area its preview surface encloses, and
//! the curvature, wall and gap spectrum a check-time mesh estimate reads.
//!
//! Fixtures: `fixtures/camera_bracket.step` (three 3 mm plates: a 15 x 8 mm mount with two
//! 2.2 mm bores, a 20 x 17.9 mm web and a 26 x 26 mm camera plate with four 2.2 mm bores) and
//! `fixtures/named-faces.step` (a 100 x 20 x 10 mm bar). Set `RUNMAT_PREVIEW_MEASURE_DIR` to
//! write each measure as JSON, the input of the mesh estimate's calibration fixtures.

use runmat_geometry_core::{LengthUnit, PreviewMeasure};
use runmat_geometry_io::{import_geometry, GeometryImportOptions};

fn measure(name: &str, bytes: &[u8]) -> PreviewMeasure {
    let imported = import_geometry(
        name,
        bytes,
        GeometryImportOptions {
            units: LengthUnit::Millimeter,
            source_units: Some(LengthUnit::Millimeter),
            ..GeometryImportOptions::default()
        },
    )
    .unwrap();
    let measure = PreviewMeasure::measure(&imported.asset).expect("closed CAD preview");
    if let Some(directory) = std::env::var_os("RUNMAT_PREVIEW_MEASURE_DIR") {
        let path = std::path::Path::new(&directory).join(format!("{name}.measure.json"));
        std::fs::write(path, serde_json::to_vec_pretty(&measure).unwrap()).unwrap();
    }
    measure
}

/// The bracket's 3 mm walls and 1.1 mm bores, and the bar's volume, read from their previews.
#[test]
fn imported_previews_measure_volume_walls_and_bores() {
    let bracket = measure(
        "camera_bracket.step",
        include_bytes!("fixtures/camera_bracket.step"),
    );
    // 337.19 + 1074 + 1982.38 mm^3 from the part's sketches.
    let volume = 3393.57e-9;
    assert!(
        (bracket.volume_m3 - volume).abs() < 0.01 * volume,
        "{bracket:?}"
    );
    let area = |keep: &dyn Fn(&runmat_geometry_core::PreviewSurfacePatch) -> bool| -> f64 {
        bracket
            .patches
            .iter()
            .filter(|p| keep(p))
            .map(|p| p.area_m2)
            .sum()
    };
    let walls = area(&|p| p.thickness_m.is_some_and(|t| (t - 0.003).abs() < 3.0e-4));
    assert!(walls > 0.6 * bracket.area_m2, "{bracket:?}");
    // Six bores of radius 1.1 mm through 3 mm plates.
    let bores = area(&|p| p.radius_m.is_some_and(|r| (r - 0.0011).abs() < 2.0e-4));
    let bore_area = 6.0 * std::f64::consts::TAU * 0.0011 * 0.003;
    assert!(
        (bores - bore_area).abs() < 0.2 * bore_area,
        "{bores} vs {bore_area}: {bracket:?}"
    );

    let bar = measure(
        "named-faces.step",
        include_bytes!("fixtures/named-faces.step"),
    );
    let volume = 100.0 * 20.0 * 10.0 * 1.0e-9;
    assert!((bar.volume_m3 - volume).abs() < 1.0e-3 * volume, "{bar:?}");
    assert!(bar.patches.iter().all(|p| p.radius_m.is_none()), "{bar:?}");
}
