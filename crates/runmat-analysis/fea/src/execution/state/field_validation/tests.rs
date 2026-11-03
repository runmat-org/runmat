use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use super::*;
use runmat_analysis_core::DeviceFieldRef;

#[test]
fn portable_inventory_preserves_arbitrary_order_and_exact_id_uniqueness() {
    let fields = vec![
        AnalysisField::host_f64("z", AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global), vec![2], vec![1.0, -1.0]),
        AnalysisField::host_f64("a", AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global), vec![], vec![0.0]),
        AnalysisField::host_f64(" a", AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global), vec![1, 1], vec![2.0]),
    ];
    let before = fields.clone();
    validate_portable_fields(&fields).unwrap();
    assert_eq!(fields, before);
    validate_portable_fields(&[]).unwrap();
    let duplicate = [fields[0].clone(), fields[1].clone(), fields[0].clone()];
    assert!(duplicate
        .iter()
        .all(|field| validate_portable_field(field).is_ok()));
    assert!(validate_portable_fields(&duplicate).is_err());
}

#[test]
fn single_field_and_inventory_reject_the_same_invalid_payloads() {
    for field in [
        AnalysisField::host_f64(" \t", AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global), vec![1], vec![1.0]),
        AnalysisField::host_f64("temperature", AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node), vec![2], vec![1.0]),
        AnalysisField::host_f64("temperature", AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node), vec![0], vec![]),
        AnalysisField::host_f64("temperature", AnalysisFieldLayout::components(ObservationQuantity::Temperature, FieldLocation::Node, 2), vec![usize::MAX, 2], vec![1.0]),
        AnalysisField::host_f64("temperature", AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node), vec![1], vec![f64::NAN]),
        AnalysisField::host_f64("temperature", AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node), vec![1], vec![f64::INFINITY]),
        AnalysisField::host_f64("temperature", AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node), vec![1], vec![f64::NEG_INFINITY]),
        AnalysisField {
            field_id: "temperature".into(),
            layout: runmat_analysis_core::AnalysisFieldLayout::scalar(runmat_analysis_core::ObservationQuantity::Temperature, runmat_analysis_core::FieldLocation::Node),
            shape: vec![1],
            values: AnalysisFieldValues::DeviceRef(DeviceFieldRef {
                backend: "device".into(),
                token: "resident".into(),
                element_count: 1,
            }),
        },
    ] {
        let individual = validate_portable_field(&field).unwrap_err();
        let inventory = validate_portable_fields(std::slice::from_ref(&field)).unwrap_err();
        assert_eq!(individual.to_string(), inventory.to_string());
    }
}
