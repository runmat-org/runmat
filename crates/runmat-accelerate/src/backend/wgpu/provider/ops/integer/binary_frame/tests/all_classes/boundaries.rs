use super::*;

async fn check_wide(
    provider: &WgpuProvider,
    operation: IntegerBinaryOperation,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    class: IntegerElementType,
    shape: &[usize],
    expected: HostIntegerDataOwned,
) {
    let output = public_hook(provider, operation, lhs, rhs).await.unwrap();
    assert_eq!(output.shape(), shape);
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(class)
    );
    let entry = provider.get_entry_raw(&output).unwrap();
    let bytes = (shape.iter().product::<usize>() * 8) as u64;
    assert_eq!(entry.buffer.size(), bytes);
    assert_eq!(entry.allocated_bytes, bytes);
    assert!(!output.shares_allocation_with(lhs));
    assert!(!output.shares_allocation_with(rhs));
    assert_eq!(
        provider.download_integer(&output).await.unwrap().data,
        expected
    );
}

pub(super) async fn actual_boundaries(provider: &WgpuProvider) {
    let signed_lhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[i64::MAX, i64::MIN, 3, -3]),
            shape: &[1, 4],
        })
        .unwrap();
    let signed_rhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[1, -1, 2, 2]),
            shape: &[1, 4],
        })
        .unwrap();
    check_wide(
        provider,
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Add),
        &signed_lhs,
        &signed_rhs,
        IntegerElementType::I64,
        &[1, 4],
        HostIntegerDataOwned::I64(vec![i64::MAX, i64::MIN, 5, -1]),
    )
    .await;
    check_wide(
        provider,
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Div),
        &signed_lhs,
        &signed_rhs,
        IntegerElementType::I64,
        &[1, 4],
        HostIntegerDataOwned::I64(vec![i64::MAX, i64::MAX, 2, -2]),
    )
    .await;
    let rem_lhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[-7, 7]),
            shape: &[2, 1],
        })
        .unwrap();
    let rem_rhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[4, -4, 3]),
            shape: &[1, 3],
        })
        .unwrap();
    check_wide(
        provider,
        IntegerBinaryOperation::Remainder,
        &rem_lhs,
        &rem_rhs,
        IntegerElementType::I64,
        &[2, 3],
        HostIntegerDataOwned::I64(vec![-3, 3, -3, 3, -1, 1]),
    )
    .await;
    check_wide(
        provider,
        IntegerBinaryOperation::Modulus,
        &rem_lhs,
        &rem_rhs,
        IntegerElementType::I64,
        &[2, 3],
        HostIntegerDataOwned::I64(vec![1, 3, -3, -1, 2, 1]),
    )
    .await;
    assert_eq!(
        provider.download_integer(&signed_lhs).await.unwrap().data,
        HostIntegerDataOwned::I64(vec![i64::MAX, i64::MIN, 3, -3])
    );
    assert_eq!(
        provider.download_integer(&rem_lhs).await.unwrap().data,
        HostIntegerDataOwned::I64(vec![-7, 7])
    );
    assert_eq!(
        provider.download_integer(&signed_rhs).await.unwrap().data,
        HostIntegerDataOwned::I64(vec![1, -1, 2, 2])
    );
    assert_eq!(
        provider.download_integer(&rem_rhs).await.unwrap().data,
        HostIntegerDataOwned::I64(vec![4, -4, 3])
    );

    let len = provider.linear_launch.workgroup() as usize + 1;
    let mut values = vec![2u64; len];
    values[len - 1] = u64::MAX;
    let tail_lhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&values),
            shape: &[1, len],
        })
        .unwrap();
    let tail_rhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[2]),
            shape: &[1, 1],
        })
        .unwrap();
    let mut expected = vec![4u64; len];
    expected[len - 1] = u64::MAX;
    check_wide(
        provider,
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Mul),
        &tail_lhs,
        &tail_rhs,
        IntegerElementType::U64,
        &[1, len],
        HostIntegerDataOwned::U64(expected),
    )
    .await;
    assert_eq!(
        provider.download_integer(&tail_lhs).await.unwrap().data,
        HostIntegerDataOwned::U64(values)
    );
    assert_eq!(
        provider.download_integer(&tail_rhs).await.unwrap().data,
        HostIntegerDataOwned::U64(vec![2])
    );
    for input in [
        &signed_lhs,
        &signed_rhs,
        &rem_lhs,
        &rem_rhs,
        &tail_lhs,
        &tail_rhs,
    ] {
        input.allocation_lease().validate_live().unwrap();
    }
}
