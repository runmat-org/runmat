use super::*;

const OPERATIONS: [IntegerBinaryOperation; 9] = [
    IntegerBinaryOperation::Ordinary(BinaryOpCode::Add),
    IntegerBinaryOperation::Ordinary(BinaryOpCode::Sub),
    IntegerBinaryOperation::Ordinary(BinaryOpCode::Mul),
    IntegerBinaryOperation::Ordinary(BinaryOpCode::Div),
    IntegerBinaryOperation::Ordinary(BinaryOpCode::Pow),
    IntegerBinaryOperation::Remainder,
    IntegerBinaryOperation::Modulus,
    IntegerBinaryOperation::Ordinary(BinaryOpCode::Min),
    IntegerBinaryOperation::Ordinary(BinaryOpCode::Max),
];

async fn public_hook(
    provider: &WgpuProvider,
    operation: IntegerBinaryOperation,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
    match operation {
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Add) => provider.elem_add(lhs, rhs).await,
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Sub) => provider.elem_sub(lhs, rhs).await,
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Mul) => provider.elem_mul(lhs, rhs).await,
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Div) => provider.elem_div(lhs, rhs).await,
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Pow) => provider.elem_pow(lhs, rhs).await,
        IntegerBinaryOperation::Remainder => provider.elem_rem(lhs, rhs).await,
        IntegerBinaryOperation::Modulus => provider.elem_mod(lhs, rhs).await,
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Min) => provider.elem_min(lhs, rhs).await,
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Max) => provider.elem_max(lhs, rhs).await,
        _ => unreachable!("nine typed integer operations only"),
    }
}

struct ClassCase<'a> {
    class: IntegerElementType,
    words_per_element: u64,
    left_data: HostIntegerDataView<'a>,
    scalar_data: HostIntegerDataView<'a>,
    empty_data: HostIntegerDataView<'a>,
    left_expected: HostIntegerDataOwned,
    scalar_expected: HostIntegerDataOwned,
    empty_expected: HostIntegerDataOwned,
    expected: [HostIntegerDataOwned; 9],
    first_class: bool,
}

async fn check_class(provider: &WgpuProvider, case: ClassCase<'_>) {
    let ClassCase {
        class,
        words_per_element,
        left_data,
        scalar_data,
        empty_data,
        left_expected,
        scalar_expected,
        empty_expected,
        expected,
        first_class,
    } = case;
    let lhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: left_data,
            shape: &[1, 3],
        })
        .unwrap();
    let rhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: scalar_data,
            shape: &[1, 1],
        })
        .unwrap();
    let empty = provider
        .upload_integer(&HostIntegerTensorView {
            data: empty_data,
            shape: &[0, 3],
        })
        .unwrap();
    let initial_cache = provider.validated_fusion_cache.lock().unwrap().len();
    let initial_misses = provider.telemetry_snapshot().fusion_cache_misses;
    let initial_hits = provider.telemetry_snapshot().fusion_cache_hits;
    for (index, (operation, oracle)) in OPERATIONS.into_iter().zip(expected).enumerate() {
        let output = public_hook(provider, operation, &lhs, &rhs).await.unwrap();
        assert_eq!(output.shape(), [1, 3]);
        assert_eq!(output.device_id(), provider.device_id());
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(class)
        );
        let entry = provider.get_entry_raw(&output).unwrap();
        let bytes = 3 * words_per_element * 4;
        assert_eq!(
            entry.buffer.size(),
            bytes,
            "class {class:?}, operation {index}"
        );
        assert_eq!(
            entry.allocated_bytes, bytes,
            "class {class:?}, operation {index}"
        );
        assert!(!output.shares_allocation_with(&lhs));
        assert!(!output.shares_allocation_with(&rhs));
        assert_eq!(
            provider.download_integer(&output).await.unwrap().data,
            oracle,
            "class {class:?}, operation {index}"
        );
        if first_class && index == 0 {
            assert_eq!(
                provider.validated_fusion_cache.lock().unwrap().len(),
                initial_cache + 1
            );
            assert_eq!(
                provider.telemetry_snapshot().fusion_cache_misses,
                initial_misses + 1
            );
        }
        if first_class && index == 1 {
            assert!(provider.telemetry_snapshot().fusion_cache_hits > initial_hits);
        }
    }
    for (index, operation) in OPERATIONS.into_iter().enumerate() {
        let operation_receipt = receipt(provider);
        let checkpoint = operation_receipt.recorder().submission_checkpoint();
        let cache_before = provider.validated_fusion_cache.lock().unwrap().len();
        let metrics_before = provider.telemetry_snapshot();
        let output = observe_provider_future(
            Some(operation_receipt.recorder()),
            public_hook(provider, operation, &empty, &rhs),
        )
        .await
        .unwrap();
        let signed_power_readback = matches!(
            operation,
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Pow)
        ) && matches!(
            class,
            IntegerElementType::I8
                | IntegerElementType::I16
                | IntegerElementType::I32
                | IntegerElementType::I64
        );
        if signed_power_readback {
            assert!(
                operation_receipt.recorder().submission_checkpoint() > checkpoint,
                "signed empty power validates its nonempty scalar exponent by readback"
            );
        } else {
            assert_eq!(
                operation_receipt.recorder().submission_checkpoint(),
                checkpoint,
                "empty operation {index} has no readback or numerical submission"
            );
        }
        // Neither branch prepares the numerical integer binary recipe. The
        // signed branch's observed submission is the exponent readback; the
        // empty-output source route itself contains no numerical dispatch.
        assert_eq!(
            provider.validated_fusion_cache.lock().unwrap().len(),
            cache_before
        );
        let metrics_after = provider.telemetry_snapshot();
        assert_eq!(
            metrics_after.fusion_cache_misses,
            metrics_before.fusion_cache_misses
        );
        assert_eq!(
            metrics_after.fusion_cache_hits,
            metrics_before.fusion_cache_hits
        );
        assert_eq!(output.shape(), [0, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(class)
        );
        let entry = provider.get_entry_raw(&output).unwrap();
        assert_eq!(
            entry.buffer.size(),
            4,
            "empty class {class:?}, operation {index}"
        );
        assert_eq!(
            entry.allocated_bytes, 4,
            "empty class {class:?}, operation {index}"
        );
        assert!(!output.shares_allocation_with(&empty));
        assert!(!output.shares_allocation_with(&rhs));
        assert_eq!(
            provider.download_integer(&output).await.unwrap().data,
            empty_expected,
            "empty class {class:?}, operation {index}"
        );
    }
    assert_eq!(
        provider.download_integer(&lhs).await.unwrap().data,
        left_expected
    );
    assert_eq!(
        provider.download_integer(&rhs).await.unwrap().data,
        scalar_expected
    );
    assert_eq!(
        provider.download_integer(&empty).await.unwrap().data,
        empty_expected
    );
    lhs.allocation_lease().validate_live().unwrap();
    rhs.allocation_lease().validate_live().unwrap();
    empty.allocation_lease().validate_live().unwrap();
}

async fn actual_all_eight_classes_nine_public_hooks() {
    let provider = owner().await;
    macro_rules! one_class {
        ($variant:ident, $ty:ty, $class:expr, $words:expr, $first:expr) => {
            check_class(
                provider,
                ClassCase {
                    class: $class,
                    words_per_element: $words,
                    left_data: HostIntegerDataView::$variant(&[2 as $ty, 3 as $ty, 4 as $ty]),
                    scalar_data: HostIntegerDataView::$variant(&[2 as $ty]),
                    empty_data: HostIntegerDataView::$variant(&[]),
                    left_expected: HostIntegerDataOwned::$variant(vec![
                        2 as $ty, 3 as $ty, 4 as $ty,
                    ]),
                    scalar_expected: HostIntegerDataOwned::$variant(vec![2 as $ty]),
                    empty_expected: HostIntegerDataOwned::$variant(vec![]),
                    expected: [
                        HostIntegerDataOwned::$variant(vec![4 as $ty, 5 as $ty, 6 as $ty]),
                        HostIntegerDataOwned::$variant(vec![0 as $ty, 1 as $ty, 2 as $ty]),
                        HostIntegerDataOwned::$variant(vec![4 as $ty, 6 as $ty, 8 as $ty]),
                        HostIntegerDataOwned::$variant(vec![1 as $ty, 2 as $ty, 2 as $ty]),
                        HostIntegerDataOwned::$variant(vec![4 as $ty, 9 as $ty, 16 as $ty]),
                        HostIntegerDataOwned::$variant(vec![0 as $ty, 1 as $ty, 0 as $ty]),
                        HostIntegerDataOwned::$variant(vec![0 as $ty, 1 as $ty, 0 as $ty]),
                        HostIntegerDataOwned::$variant(vec![2 as $ty, 2 as $ty, 2 as $ty]),
                        HostIntegerDataOwned::$variant(vec![2 as $ty, 3 as $ty, 4 as $ty]),
                    ],
                    first_class: $first,
                },
            )
            .await;
        };
    }
    one_class!(I8, i8, IntegerElementType::I8, 1, true);
    one_class!(I16, i16, IntegerElementType::I16, 1, false);
    one_class!(I32, i32, IntegerElementType::I32, 1, false);
    one_class!(I64, i64, IntegerElementType::I64, 2, false);
    one_class!(U8, u8, IntegerElementType::U8, 1, false);
    one_class!(U16, u16, IntegerElementType::U16, 1, false);
    one_class!(U32, u32, IntegerElementType::U32, 1, false);
    one_class!(U64, u64, IntegerElementType::U64, 2, false);
    boundaries::actual_boundaries(provider).await;
}

#[path = "all_classes/boundaries.rs"]
mod boundaries;

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual WGPU adapter; no adapter skip"]
async fn native_all_eight_classes_nine_public_hooks() {
    actual_all_eight_classes_nine_public_hooks().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_all_eight_classes_nine_public_hooks() {
    actual_all_eight_classes_nine_public_hooks().await;
}
