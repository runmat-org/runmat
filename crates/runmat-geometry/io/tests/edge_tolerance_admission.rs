#![cfg(feature = "occt-native")]
//! Import settles every edge whose curve/pcurve gap exceeds the tolerance its file wrote: it
//! rebuilds the edge's pcurves against the 3D curve, measures the gap again, and admits the
//! remaining gap up to the request's geometric tolerance (`requested_deviation_m`) by raising the
//! edge's tolerance. A larger gap stays a defect, and the import records each edge either way.
//!
//! Fixture: `fixtures/drone_arm_study.step` is the drone arm study export of 2026-11-05 (the arm
//! with its motor pod), imported with shared interfaces as the arm studies do. Several of its
//! edges carry pcurve gaps of 1 nm to 0.2 um a little above the tolerances the file writes.

use runmat_geometry_core::LengthUnit;
use runmat_geometry_io::{
    import_exact_cad, ExactCadImportOptions, GeometryFormat, GeometryImportContext,
    ImportedExactCad,
};

const DRONE_ARM: &[u8] = include_bytes!("fixtures/drone_arm_study.step");

fn import(requested_deviation_m: f64) -> ImportedExactCad {
    let mut options = ExactCadImportOptions {
        source_units: Some(LengthUnit::Millimeter),
        ..ExactCadImportOptions::default()
    };
    options.analysis.requested_deviation_m =
        runmat_types::physical::Length::new(requested_deviation_m);
    options.analysis.healing.shared_interfaces = true;
    import_exact_cad(
        "drone_arm_study.step",
        DRONE_ARM,
        GeometryFormat::Step,
        &options,
        &GeometryImportContext::new(),
    )
    .unwrap()
}

fn edge_tolerance(
    imported: &ImportedExactCad,
    id: &runmat_geometry_core::PersistentEntityId,
) -> f64 {
    imported
        .topology
        .edges
        .iter()
        .find(|edge| &edge.id == id)
        .unwrap()
        .tolerance_m
        .canonical_value()
}

#[test]
fn gaps_within_the_request_tolerance_are_admitted_and_recorded() {
    let imported = import(1.0e-4);
    assert!(!imported.edge_tolerance_admissions.is_empty());
    for admission in &imported.edge_tolerance_admissions {
        eprintln!("{}", admission.describe());
        assert!(admission.admitted, "{}", admission.describe());
        assert_eq!(admission.bound_m, 1.0e-4);
        assert!(admission.gap_before_m * 1.05 > admission.written_tolerance_m);
        assert!(admission.gap_after_m <= admission.gap_before_m);
        let expected = admission
            .written_tolerance_m
            .max(1.05 * admission.gap_after_m);
        assert!(
            (admission.admitted_tolerance_m - expected).abs() <= 1.0e-6 * expected,
            "{}",
            admission.describe()
        );
        // The record names the imported edge, and that edge carries the admitted tolerance.
        let id = admission
            .edge_id
            .as_ref()
            .unwrap_or_else(|| panic!("edge not identified: {}", admission.describe()));
        let tolerance = edge_tolerance(&imported, id);
        assert!(
            (tolerance - admission.admitted_tolerance_m).abs()
                <= 1.0e-6 * admission.admitted_tolerance_m,
            "edge tolerance {tolerance:e} m: {}",
            admission.describe()
        );
    }
    assert_eq!(imported.edge_tolerance_defects().count(), 0);
}

#[test]
fn gaps_over_the_request_tolerance_stay_defects_with_their_records() {
    let bound = 1.0e-7;
    let imported = import(bound);
    let defects: Vec<_> = imported.edge_tolerance_defects().collect();
    assert!(
        !defects.is_empty(),
        "the 0.2 um gaps exceed a 0.1 um request tolerance"
    );
    for defect in defects {
        assert!(defect.gap_after_m > bound);
        assert_eq!(defect.admitted_tolerance_m, defect.written_tolerance_m);
        let description = defect.describe();
        assert!(
            description.contains("exceeds the request's geometric tolerance 1e-7 m"),
            "{description}"
        );
        let id = defect
            .edge_id
            .as_ref()
            .expect("the defect edge is identified");
        let tolerance = edge_tolerance(&imported, id);
        assert!(
            (tolerance - defect.written_tolerance_m).abs() <= 1.0e-6 * defect.written_tolerance_m,
            "edge tolerance {tolerance:e} m: {description}"
        );
    }
}
