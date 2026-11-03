use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use super::*;

#[test]
fn unique_entity_validation_preserves_arbitrary_declared_order() {
    for ids in [
        vec![1],
        vec![1, 7, 100],
        vec![100, 1, 7],
        vec![u64::MAX, 0, 3],
    ] {
        let original = ids.clone();
        assert!(unique_nonempty(&ids).unwrap());
        assert_eq!(ids, original);
    }
    for ids in [
        vec![],
        vec![1, 1],
        vec![7, 1, 7],
        vec![u64::MAX, 0, u64::MAX],
    ] {
        assert!(!unique_nonempty(&ids).unwrap());
    }
    let mut ids: Vec<_> = (0..100_000).rev().collect();
    assert!(unique_nonempty(&ids).unwrap());
    ids[99_999] = ids[0];
    assert!(!unique_nonempty(&ids).unwrap());
}

#[test]
fn ordinal_history_checks_reject_duplicates_reordering_and_invalid_storage() {
    let fields: Vec<_> = (0..3)
        .map(|index| {
            AnalysisField::host_f64(
                fea_thermal_temperature_field_id(index), AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node).step(index),
                vec![2],
                vec![300.0; 2],
            )
        })
        .collect();
    let check = |fields: &[AnalysisField]| {
        validate_history(fields, 3, &[2], fea_thermal_temperature_field_id)
    };
    check(&fields).unwrap();
    let mut duplicate = fields.clone();
    duplicate[2].field_id = duplicate[0].field_id.clone();
    assert!(check(&duplicate).is_err());
    let mut reordered = fields.clone();
    reordered.swap(0, 1);
    assert!(check(&reordered).is_err());
    assert!(check(&fields[..2]).is_err());
    let mut invalid = fields.clone();
    invalid[1] = AnalysisField::host_f64(
        fea_thermal_temperature_field_id(1), AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node).step(1),
        vec![2],
        vec![f64::NAN; 2],
    );
    assert!(check(&invalid).is_err());
    invalid[1] = AnalysisField::host_f64(
        fea_thermal_temperature_field_id(1), AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node).step(1),
        vec![1, 2],
        vec![300.0; 2],
    );
    assert!(check(&invalid).is_err());
    invalid[1] = AnalysisField::host_f64(fea_thermal_temperature_field_id(1), AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node).step(1), vec![2], vec![300.0]);
    assert!(check(&invalid).is_err());
}
