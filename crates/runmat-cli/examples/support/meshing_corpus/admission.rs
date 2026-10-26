use std::path::Path;

use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_runner_native::{
    prepare_exact_geometry_admission, ExactGeometryAdmissionError,
};
use runmat_geometry_core::LengthUnit;
use runmat_geometry_io::{ExactCadImportOptions, GeometryImportContext};

use super::contract::{
    AdmissionExpectation, AdmissionFailureKind, AdmissionScenario, HealingSettings, SourceUnits,
};
use super::report::{AdmissionReport, ObservedFailure};

pub fn import_options(units: SourceUnits, healing: &HealingSettings) -> ExactCadImportOptions {
    let mut options = ExactCadImportOptions {
        source_units: Some(match units {
            SourceUnits::Meter => LengthUnit::Meter,
            SourceUnits::Millimeter => LengthUnit::Millimeter,
            SourceUnits::Inch => LengthUnit::Inch,
        }),
        ..ExactCadImportOptions::default()
    };
    options.analysis.healing.sew = healing.sew;
    options.analysis.healing.repair_orientation = healing.repair_orientation;
    options.analysis.healing.consolidate_duplicates = healing.consolidate_duplicates;
    options.analysis.healing.repair_tolerance_scale_gaps = healing.repair_tolerance_scale_gaps;
    options
        .analysis
        .healing
        .simplify_short_edges_and_sliver_faces = healing.simplify_short_edges_and_sliver_faces;
    if let Some(limit) = healing.maximum_displacement_m {
        options.analysis.maximum_healing_displacement_m =
            runmat_types::physical::Length::new(limit);
    }
    options
}

pub fn execute(
    source: &Path,
    bytes: &[u8],
    units: SourceUnits,
    scenario: &AdmissionScenario,
) -> AdmissionReport {
    let source_name = source.to_string_lossy();
    let result = prepare_exact_geometry_admission(
        &source_name,
        bytes,
        &import_options(units, &scenario.healing),
        &[],
        &GeometryImportContext::new(),
        ObjectInventoryLimits::default(),
    );
    match (&scenario.expected, result) {
        (
            AdmissionExpectation::Success {
                solid_regions,
                healing_operations,
            },
            Ok(prepared),
        ) => {
            let actual_regions = prepared.topology().regions.len();
            let actual_healing = prepared
                .healing_report()
                .map(|report| report.operations.len());
            let passed = actual_regions == *solid_regions
                && healing_operations.is_none_or(|expected| actual_healing == Some(expected));
            AdmissionReport {
                id: scenario.id.clone(),
                outcome: "success".into(),
                solid_regions: Some(actual_regions),
                healing_operations: actual_healing,
                failure: None,
                passed,
                detail: (!passed).then(|| {
                    format!(
                        "expected {solid_regions} solid regions and {healing_operations:?} healing operations"
                    )
                }),
            }
        }
        (AdmissionExpectation::Failure { kind }, Err(error)) => {
            let actual = classify(&error);
            let passed = actual == Some(*kind);
            AdmissionReport {
                id: scenario.id.clone(),
                outcome: "failure".into(),
                solid_regions: None,
                healing_operations: None,
                failure: actual.map(|category| ObservedFailure::Admission { category }),
                passed,
                detail: (!passed).then(|| error.to_string()),
            }
        }
        (AdmissionExpectation::Success { .. }, Err(error)) => AdmissionReport {
            id: scenario.id.clone(),
            outcome: "failure".into(),
            solid_regions: None,
            healing_operations: None,
            failure: classify(&error).map(|category| ObservedFailure::Admission { category }),
            passed: false,
            detail: Some(error.to_string()),
        },
        (AdmissionExpectation::Failure { .. }, Ok(prepared)) => AdmissionReport {
            id: scenario.id.clone(),
            outcome: "success".into(),
            solid_regions: Some(prepared.topology().regions.len()),
            healing_operations: prepared
                .healing_report()
                .map(|report| report.operations.len()),
            failure: None,
            passed: false,
            detail: Some("admission succeeded but a typed failure was required".into()),
        },
    }
}

pub fn classify(error: &ExactGeometryAdmissionError) -> Option<AdmissionFailureKind> {
    match error {
        ExactGeometryAdmissionError::UnsupportedFormat => {
            Some(AdmissionFailureKind::UnsupportedFormat)
        }
        ExactGeometryAdmissionError::Import(_) => Some(AdmissionFailureKind::Import),
        ExactGeometryAdmissionError::Contract(_) => Some(AdmissionFailureKind::Contract),
        ExactGeometryAdmissionError::Artifact(_) => Some(AdmissionFailureKind::Artifact),
        ExactGeometryAdmissionError::Store(_) => Some(AdmissionFailureKind::Store),
    }
}
