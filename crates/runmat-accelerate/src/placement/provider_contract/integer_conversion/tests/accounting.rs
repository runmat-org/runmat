use super::*;
use crate::backend::wgpu_storage::checked_allocation_bytes;

fn estimate(result: ProviderFeasibility) -> ProviderResourceEstimate {
    match result {
        ProviderFeasibility::Supported { estimate } => estimate,
        other => panic!("expected feasible allocation: {other:?}"),
    }
}

#[test]
fn wgpu_packed_bytes_do_not_change_semantic_types_or_native_widths() {
    let provider = InProcessProvider::new();
    for (entry, target) in OPERATIONS {
        for residency in [ProviderResidency::Host, ProviderResidency::Device] {
            let ty = ProviderElementType::from(*target);
            let mut request = query(entry, ty, &[2, 3]);
            request.inputs[0].residency = residency;
            let original = request.clone();
            let native = estimate(validate(&provider, false, &request).unwrap());
            let packed = estimate(validate(&provider, true, &request).unwrap());
            let physical_width =
                if matches!(target, IntegerElementType::I64 | IntegerElementType::U64) {
                    8
                } else {
                    4
                };
            assert_eq!(native.output_bytes, Some(6 * ty.byte_width()));
            assert_eq!(native.transient_bytes, Some(12 * ty.byte_width()));
            assert_eq!(packed.output_bytes, Some(6 * physical_width));
            assert_eq!(packed.transient_bytes, Some(12 * physical_width));
            assert_eq!(request, original);
        }
    }
}

#[test]
fn floating_source_and_narrow_destination_use_distinct_allocation_widths() {
    let provider = InProcessProvider::new();
    let request = query(&UINT8_CATALOG_ENTRY, ProviderElementType::F64, &[7]);
    let result = estimate(validate(&provider, true, &request).unwrap());
    assert_eq!(result.output_bytes, Some(28));
    assert_eq!(result.transient_bytes, Some(56 + 28));
}

#[test]
fn empty_shapes_have_zero_registered_allocation_and_overflow_is_rejected() {
    let provider = InProcessProvider::new();
    for (entry, target) in OPERATIONS {
        let request = query(entry, ProviderElementType::from(*target), &[2, 0, 3]);
        let result = estimate(validate(&provider, true, &request).unwrap());
        assert_eq!(result.output_bytes, Some(0));
        assert_eq!(result.transient_bytes, Some(0));
    }
    for shape in [vec![u64::MAX, 2], vec![u64::MAX / 4 + 1]] {
        let request = query(&UINT8_CATALOG_ENTRY, ProviderElementType::U8, &shape);
        assert_eq!(checked_allocation_bytes(&request.inputs[0]), None);
        assert!(matches!(
            validate(&provider, true, &request),
            Some(ProviderFeasibility::Rejected { .. })
        ));
    }
}

#[test]
fn physical_allocation_limit_and_total_overflow_use_packed_allocations() {
    let provider = InProcessProvider::new();
    let mut snapshot = dense_capability_snapshot(&provider, 4, &[]);
    append(&mut snapshot);
    let mut request = query(&UINT8_CATALOG_ENTRY, ProviderElementType::U8, &[4]);
    request.outputs = request.inputs.clone();
    snapshot.max_allocation_bytes = Some(8);
    assert!(matches!(
        dense_feasibility(&snapshot, &request),
        ProviderFeasibility::Supported { .. }
    ));
    assert!(matches!(
        dense_feasibility_with_bytes(&snapshot, &request, checked_allocation_bytes),
        ProviderFeasibility::Rejected { .. }
    ));

    // Each packed allocation is 16 bytes. The per-allocation limit does not
    // constrain their combined 32-byte demand; placement owns total capacity.
    snapshot.max_allocation_bytes = Some(16);
    let packed = estimate(dense_feasibility_with_bytes(
        &snapshot,
        &request,
        checked_allocation_bytes,
    ));
    assert_eq!(packed.output_bytes, Some(16));
    assert_eq!(packed.transient_bytes, Some(32));

    snapshot.max_allocation_bytes = None;
    request.inputs[0].shape = vec![u64::MAX / 4];
    request.outputs = request.inputs.clone();
    assert!(checked_allocation_bytes(&request.inputs[0]).is_some());
    assert!(matches!(
        dense_feasibility_with_bytes(&snapshot, &request, checked_allocation_bytes),
        ProviderFeasibility::Rejected { .. }
    ));
}

#[test]
fn logical_class_without_physical_precision_is_not_a_byte_width_proof() {
    let request = query(&UINT8_CATALOG_ENTRY, ProviderElementType::Logical, &[4]);
    assert_eq!(checked_allocation_bytes(&request.inputs[0]), None);
}
