use super::*;
use crate::simple_provider::InProcessProvider;

fn value(
    element_type: ProviderElementType,
    residency: ProviderResidency,
) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type,
        residency,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![1, 8],
    }
}

#[test]
fn output_cast_and_resident_widening_are_both_in_peak_and_dispatch_estimates() {
    let provider = InProcessProvider::new(); // F64 physical geometry; not an execution witness.
    let query = ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![value(ProviderElementType::F32, ProviderResidency::Device)],
        outputs: vec![
            value(ProviderElementType::F32, ProviderResidency::Device),
            value(ProviderElementType::F64, ProviderResidency::Device),
        ],
        workload: Default::default(),
    };
    let ProviderFeasibility::Supported { estimate } = validate(&provider, true, &query).unwrap()
    else {
        panic!("typed physical plan should be feasible")
    };
    // Borrowed F32 input32 + widened64 + two raw outputs128 + finalSingle32.
    assert_eq!(estimate.transient_bytes, Some(256));
    assert_eq!(estimate.output_bytes, Some(96));
    assert_eq!(estimate.dispatches, Some(3));
    assert!(matches!(
        validate(&provider, false, &query),
        Some(ProviderFeasibility::Rejected { .. })
    ));
}

#[test]
fn host_single_upload_is_counted_at_the_actual_native_f64_buffer_width() {
    let provider = InProcessProvider::new();
    let query = ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![value(ProviderElementType::F32, ProviderResidency::Host)],
        outputs: vec![value(ProviderElementType::F64, ProviderResidency::Device)],
        workload: Default::default(),
    };
    let ProviderFeasibility::Supported { estimate } = validate(&provider, true, &query).unwrap()
    else {
        panic!("native staging should be feasible")
    };
    assert_eq!(estimate.transient_bytes, Some(128));
    assert_eq!(estimate.output_bytes, Some(64));
    assert_eq!(estimate.dispatches, Some(1));
}

#[test]
fn broadcast_inputs_are_charged_at_source_size_but_incompatible_geometry_rejects() {
    let provider = InProcessProvider::new();
    let mut row = value(ProviderElementType::F64, ProviderResidency::Host);
    row.shape = vec![1, 8];
    let mut column = row.clone();
    column.shape = vec![4, 1];
    let mut output = value(ProviderElementType::F32, ProviderResidency::Device);
    output.shape = vec![4, 8];
    let mut query = ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![row, column],
        outputs: vec![output],
        workload: Default::default(),
    };
    let ProviderFeasibility::Supported { estimate } = validate(&provider, true, &query).unwrap()
    else {
        panic!("broadcast geometry should be feasible")
    };
    assert_eq!(estimate.transient_bytes, Some(64 + 32 + 256 + 128));
    assert_eq!(estimate.output_bytes, Some(128));
    query.inputs[0].shape = vec![2, 8];
    assert!(matches!(
        validate(&provider, true, &query),
        Some(ProviderFeasibility::Rejected { .. })
    ));
}
