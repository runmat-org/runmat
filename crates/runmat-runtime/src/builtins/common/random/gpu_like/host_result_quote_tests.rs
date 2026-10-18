use super::*;

#[test]
fn selected_host_mode_quotes_its_actual_shape_and_payload_owner() {
    let shape = [2, 3];
    let class = GpuLikeClass::Floating {
        precision: ProviderPrecision::F32,
        complex: true,
    };
    let managed = host_result_bytes("randn", &shape, class, ExactHostRandomMode::Managed)
        .expect("managed quote");
    let unbounded = host_result_bytes("randn", &shape, class, ExactHostRandomMode::Unbounded)
        .expect("unbounded quote");
    assert_eq!(
        managed,
        ResourceByteBound::Known(
            HostAllocationPlan::retained_complex_tensor_bytes(NumericDType::F32, &shape)
                .expect("managed tensor quote")
        )
    );
    assert_eq!(
        unbounded,
        ResourceByteBound::Known(
            ExactHostComplexTensorAllocation::unmanaged_requested_bytes(NumericDType::F32, &shape)
                .expect("unmanaged tensor quote") as u64
        )
    );

    let logical = GpuLikeClass::Logical;
    assert_eq!(
        host_result_bytes("randi", &shape, logical, ExactHostRandomMode::Unbounded)
            .expect("unbounded logical quote"),
        ResourceByteBound::Known(
            ExactHostLogicalTensorAllocation::unmanaged_requested_bytes(&shape)
                .expect("unmanaged logical quote") as u64
        )
    );
}
