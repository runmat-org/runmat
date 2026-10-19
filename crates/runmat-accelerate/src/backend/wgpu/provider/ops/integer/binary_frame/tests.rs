use super::*;
use crate::backend::wgpu::types::BinaryOpCode;
use runmat_accelerate_api::observation::{observe_provider_future, ProviderOperationReceipt};
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView,
};

#[path = "tests/all_classes.rs"]
mod all_classes;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn typed_integer_binary_recipes_cover_selected_geometry_without_float_capability() {
    let operations = [
        (IntegerBinaryOperation::Ordinary(BinaryOpCode::Add), 0),
        (IntegerBinaryOperation::Ordinary(BinaryOpCode::Sub), 1),
        (IntegerBinaryOperation::Ordinary(BinaryOpCode::Mul), 2),
        (IntegerBinaryOperation::Ordinary(BinaryOpCode::Div), 3),
        (IntegerBinaryOperation::Ordinary(BinaryOpCode::Pow), 4),
        (IntegerBinaryOperation::Remainder, 5),
        (IntegerBinaryOperation::Modulus, 6),
        (IntegerBinaryOperation::Ordinary(BinaryOpCode::Max), 0),
        (IntegerBinaryOperation::Ordinary(BinaryOpCode::Min), 1),
    ];
    for (index, (operation, control)) in operations.into_iter().enumerate() {
        let kind = operation.kind().unwrap();
        assert_eq!(operation.shader_control(), control);
        for workgroup in [64, 256, 512] {
            let shader = kind.shader(workgroup);
            assert!(shader.contains(&format!("@workgroup_size({workgroup})")));
            assert!(
                !shader.contains("f64"),
                "native word kernels have no f64 dependency"
            );
            crate::backend::wgpu::shaders::validation_tests::validate_generated(
                &format!("integer-binary-{index}-wg{workgroup}"),
                &shader,
                workgroup,
                1,
                crate::backend::wgpu::shaders::validation_tests::RequiredCaps::None,
            );
        }
    }
    assert!(IntegerBinaryOperation::Ordinary(BinaryOpCode::Hypot)
        .kind()
        .is_err());
    assert!(IntegerBinaryOperation::Ordinary(BinaryOpCode::Atan2)
        .kind()
        .is_err());
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: 8,
        ..wgpu::Limits::default()
    };
    let launch = FrozenLinearLaunch::new(64, limits.max_compute_workgroups_per_dimension).unwrap();
    assert!(super::super::super::elementwise::check_kernel_limits(
        "integer binary",
        &limits,
        3,
        4,
        std::mem::size_of::<super::super::IntegerBinaryParams>() as u64,
        &[16, 16, 16],
        launch,
    )
    .is_err());
}

async fn owner() -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let mut provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let mut provider = WgpuProvider::new_async(Default::default())
        .await
        .expect("actual browser WebGPU adapter");
    let workgroup = provider.linear_launch.workgroup();
    assert!(workgroup > 0);
    provider.linear_launch = FrozenLinearLaunch::new(
        workgroup,
        provider.adapter_limits.max_compute_workgroups_per_dimension,
    )
    .unwrap();
    let provider = Box::leak(Box::new(provider));
    // SAFETY: the actual provider remains live for this process or browser worker.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn receipt(provider: &WgpuProvider) -> ProviderOperationReceipt {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ProviderOperationReceipt::with_wgpu_progress(
            provider.observation_budget.clone(),
            provider
                .observation_progress
                .as_ref()
                .expect("native progress owner")
                .clone(),
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        ProviderOperationReceipt::with_budget(provider.observation_budget.clone())
    }
}

async fn actual_cold_warm_binary_words_and_tail() {
    let provider = owner().await;
    let len = provider.linear_launch.workgroup() as usize + 1;
    let lhs: Vec<u64> = (0..len)
        .map(|index| {
            if index % 3 == 0 {
                u64::MAX - 1
            } else {
                index as u64
            }
        })
        .collect();
    let rhs: Vec<u64> = (0..len)
        .map(|index| if index % 3 == 0 { 7 } else { 2 })
        .collect();
    let expected_add: Vec<u64> = lhs
        .iter()
        .zip(&rhs)
        .map(|(&left, &right)| left.saturating_add(right))
        .collect();
    let expected_min: Vec<u64> = lhs
        .iter()
        .zip(&rhs)
        .map(|(&left, &right)| left.min(right))
        .collect();
    let left = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&lhs),
            shape: &[1, len],
        })
        .unwrap();
    let right = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&rhs),
            shape: &[1, len],
        })
        .unwrap();
    let wrong_class = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[1]),
            shape: &[1, 1],
        })
        .unwrap();
    let no_effects = receipt(provider);
    let checkpoint = no_effects.recorder().submission_checkpoint();
    let record_count = provider.allocations.record_count().unwrap();
    let cache_before_rejection = provider.validated_fusion_cache.lock().unwrap().len();
    let wrong = observe_provider_future(
        Some(no_effects.recorder()),
        provider.elem_add(&left, &wrong_class),
    )
    .await
    .unwrap_err();
    assert!(!wrong.is_after_effect());
    assert!(wrong.to_string().contains("same class"));
    assert_eq!(provider.allocations.record_count().unwrap(), record_count);
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        cache_before_rejection
    );
    assert_eq!(no_effects.recorder().submission_checkpoint(), checkpoint);
    wrong_class.allocation_lease().validate_live().unwrap();
    let before_signed_pow = receipt(provider);
    let signed_pow_checkpoint = before_signed_pow.recorder().submission_checkpoint();
    let signed_exponent = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[-1]),
            shape: &[1, 1],
        })
        .unwrap();
    let incompatible_pow = observe_provider_future(
        Some(before_signed_pow.recorder()),
        provider.elem_pow(&left, &signed_exponent),
    )
    .await
    .unwrap_err();
    assert!(!incompatible_pow.is_after_effect());
    assert_eq!(
        before_signed_pow.recorder().submission_checkpoint(),
        signed_pow_checkpoint
    );
    let negative_receipt = receipt(provider);
    let negative_checkpoint = negative_receipt.recorder().submission_checkpoint();
    let negative = observe_provider_future(
        Some(negative_receipt.recorder()),
        provider.elem_pow(&wrong_class, &signed_exponent),
    )
    .await
    .unwrap_err();
    assert!(negative.is_after_effect());
    assert!(negative.to_string().contains("nonnegative"));
    assert!(negative_receipt.recorder().submission_checkpoint() > negative_checkpoint);
    signed_exponent.allocation_lease().validate_live().unwrap();
    let frame = IntegerBinaryFrame::plan(
        provider,
        "elem_add",
        &left,
        &right,
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Add),
    )
    .unwrap();
    assert_eq!(frame.recipe.workgroup, provider.linear_launch.workgroup());
    assert_eq!(frame.output_bytes, len as u64 * 8);
    // This forged source fails canonical recipe identity before entering the
    // driver scope. Shared scoped-owner tests separately prove real WGSL
    // validation-error capture; this witness proves route preflight only.
    let invalid_recipe = super::super::super::pipeline_recipe::PipelineRecipe {
        shader: "@compute @workgroup_size(1) fn main() { let invalid = ; }".into(),
        layout: frame.kind.layout_tag().into(),
        workgroup: frame.launch.workgroup(),
    };
    let invalid_receipt = receipt(provider);
    let invalid_checkpoint = invalid_receipt.recorder().submission_checkpoint();
    let records_before_invalid = provider.allocations.record_count().unwrap();
    let cache_before_invalid = provider.validated_fusion_cache.lock().unwrap().len();
    let metrics_before_invalid = provider.telemetry_snapshot();
    let invalid = match observe_provider_future(
        Some(invalid_receipt.recorder()),
        provider.prepare_validated_integer_binary_pipeline(&invalid_recipe, frame.kind),
    )
    .await
    {
        Ok(_) => panic!("forged integer binary recipe passed canonical source preflight"),
        Err(error) => error,
    };
    assert!(!invalid.is_after_effect());
    assert_eq!(
        invalid_receipt.recorder().submission_checkpoint(),
        invalid_checkpoint
    );
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        records_before_invalid
    );
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        cache_before_invalid
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        metrics_before_invalid.fusion_cache_misses
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = provider.pipeline_cache_dir.as_ref().unwrap();
        assert!(!dir
            .join(format!("{:016x}.wgsl", invalid_recipe.key()))
            .exists());
        assert!(!dir
            .join(format!("{:016x}.json", invalid_recipe.key()))
            .exists());
    }
    let initial = provider.validated_fusion_cache.lock().unwrap().len();
    let before_metrics = provider.telemetry_snapshot();
    let positive = receipt(provider);
    let positive_checkpoint = positive.recorder().submission_checkpoint();
    let add = observe_provider_future(Some(positive.recorder()), provider.elem_add(&left, &right))
        .await
        .unwrap();
    assert!(positive.recorder().submission_checkpoint() > positive_checkpoint);
    assert_eq!(add.device_id(), provider.device_id());
    assert_eq!(add.shape(), [1, len]);
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&add),
        Some(IntegerElementType::U64)
    );
    assert_eq!(
        provider.get_entry_raw(&add).unwrap().buffer.size(),
        frame.output_bytes
    );
    assert!(!add.shares_allocation_with(&left));
    assert!(!add.shares_allocation_with(&right));
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        before_metrics.fusion_cache_misses + 1
    );
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        initial + 1
    );
    let add_warm = provider.elem_add(&left, &right).await.unwrap();
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        initial + 1
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_hits,
        before_metrics.fusion_cache_hits + 1
    );
    let min = provider.elem_min(&left, &right).await.unwrap();
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        initial + 2
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        before_metrics.fusion_cache_misses + 2
    );
    let failure_receipt = receipt(provider);
    let failure_checkpoint = failure_receipt.recorder().submission_checkpoint();
    let records_before_failure = provider.allocations.record_count().unwrap();
    let injected = observe_provider_future(
        Some(failure_receipt.recorder()),
        provider.integer_binary_exec(
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Add),
            "elem_add",
            &left,
            &right,
            |_| {
                Err(anyhow!(
                    "injected integer output allocation boundary failure"
                ))
            },
        ),
    )
    .await
    .unwrap_err();
    assert!(injected.is_after_effect());
    assert!(!injected.is_unsupported());
    assert!(injected
        .to_string()
        .contains("injected integer output allocation boundary failure"));
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        records_before_failure
    );
    assert_eq!(
        failure_receipt.recorder().submission_checkpoint(),
        failure_checkpoint
    );
    assert_eq!(
        provider.download_integer(&add).await.unwrap().data,
        HostIntegerDataOwned::U64(expected_add.clone())
    );
    assert_eq!(
        provider.download_integer(&add_warm).await.unwrap().data,
        HostIntegerDataOwned::U64(expected_add)
    );
    assert_eq!(
        provider.download_integer(&min).await.unwrap().data,
        HostIntegerDataOwned::U64(expected_min)
    );
    assert_eq!(
        provider.download_integer(&left).await.unwrap().data,
        HostIntegerDataOwned::U64(lhs)
    );
    let small = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[2, 3, 4]),
            shape: &[1, 3],
        })
        .unwrap();
    let scalar = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[2]),
            shape: &[1, 1],
        })
        .unwrap();
    for (operation, expected) in [
        (
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Add),
            [4, 5, 6],
        ),
        (
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Sub),
            [0, 1, 2],
        ),
        (
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Mul),
            [4, 6, 8],
        ),
        (
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Div),
            [1, 2, 2],
        ),
        (
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Pow),
            [4, 9, 16],
        ),
        (IntegerBinaryOperation::Remainder, [0, 1, 0]),
        (IntegerBinaryOperation::Modulus, [0, 1, 0]),
        (
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Min),
            [2, 2, 2],
        ),
        (
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Max),
            [2, 3, 4],
        ),
    ] {
        let (label, result) = match operation {
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Add) => {
                ("add", provider.elem_add(&small, &scalar).await)
            }
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Sub) => {
                ("sub", provider.elem_sub(&small, &scalar).await)
            }
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Mul) => {
                ("mul", provider.elem_mul(&small, &scalar).await)
            }
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Div) => {
                ("div", provider.elem_div(&small, &scalar).await)
            }
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Pow) => {
                ("pow", provider.elem_pow(&small, &scalar).await)
            }
            IntegerBinaryOperation::Remainder => ("rem", provider.elem_rem(&small, &scalar).await),
            IntegerBinaryOperation::Modulus => ("mod", provider.elem_mod(&small, &scalar).await),
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Min) => {
                ("min", provider.elem_min(&small, &scalar).await)
            }
            IntegerBinaryOperation::Ordinary(BinaryOpCode::Max) => {
                ("max", provider.elem_max(&small, &scalar).await)
            }
            _ => unreachable!(),
        };
        let output = result.unwrap();
        assert_eq!(output.shape(), [1, 3], "{label} output shape");
        assert_eq!(output.device_id(), provider.device_id());
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(IntegerElementType::U64),
            "{label} physical class"
        );
        assert!(!output.shares_allocation_with(&small));
        assert!(!output.shares_allocation_with(&scalar));
        assert_eq!(
            provider.download_integer(&output).await.unwrap().data,
            HostIntegerDataOwned::U64(expected.to_vec()),
            "{label} scalar broadcast numerics"
        );
    }
    left.allocation_lease().validate_live().unwrap();
    right.allocation_lease().validate_live().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual WGPU adapter"]
async fn native_integer_binary_cold_warm_words_and_tail() {
    actual_cold_warm_binary_words_and_tail().await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires an actual SHADER_F64 adapter and RUNMAT_WGPU_FORCE_PRECISION=f64"]
async fn native_f64_precision_empty_integer_binary_uses_one_u32_word() {
    use runmat_accelerate_api::ProviderPrecision;

    let provider = owner().await;
    assert_eq!(
        provider.precision(),
        ProviderPrecision::F64,
        "run with RUNMAT_WGPU_FORCE_PRECISION=f64 on a SHADER_F64 adapter"
    );
    let left = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[]),
            shape: &[0, 3],
        })
        .unwrap();
    let right = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[2]),
            shape: &[1, 1],
        })
        .unwrap();
    let frame = IntegerBinaryFrame::plan(
        provider,
        "elem_add",
        &left,
        &right,
        IntegerBinaryOperation::Ordinary(BinaryOpCode::Add),
    )
    .unwrap();
    assert_eq!(frame.output_words, 0);
    assert_eq!(frame.output_bytes, 4);
    let operation = receipt(provider);
    let checkpoint = operation.recorder().submission_checkpoint();
    let output =
        observe_provider_future(Some(operation.recorder()), provider.elem_add(&left, &right))
            .await
            .unwrap();
    assert_eq!(operation.recorder().submission_checkpoint(), checkpoint);
    assert_eq!(output.shape(), [0, 3]);
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(IntegerElementType::U64)
    );
    let entry = provider.get_entry_raw(&output).unwrap();
    assert_eq!(entry.buffer.size(), 4);
    assert_eq!(entry.allocated_bytes, 4);
    assert!(!output.shares_allocation_with(&left));
    assert!(!output.shares_allocation_with(&right));
    assert_eq!(
        provider.download_integer(&output).await.unwrap().data,
        HostIntegerDataOwned::U64(Vec::new())
    );
    left.allocation_lease().validate_live().unwrap();
    right.allocation_lease().validate_live().unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_integer_binary_cold_warm_words_and_tail() {
    actual_cold_warm_binary_words_and_tail().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_cold_binary_promise_cancellation_keeps_borrowed_words() {
    use std::future::Future;
    let provider = owner().await;
    let len = provider.linear_launch.workgroup() as usize + 1;
    let lhs: Vec<u64> = (0..len).map(|index| index as u64 + 1).collect();
    let rhs = [2u64];
    let left = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&lhs),
            shape: &[1, len],
        })
        .unwrap();
    let right = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&rhs),
            shape: &[1, 1],
        })
        .unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let allocations = provider.allocations.record_count().unwrap();
    let operation = receipt(provider);
    let checkpoint = operation.recorder().submission_checkpoint();
    let mut future = Box::pin(observe_provider_future(
        Some(operation.recorder()),
        provider.elem_add(&left, &right),
    ));
    let pending = std::future::poll_fn(|context| {
        std::task::Poll::Ready(future.as_mut().poll(context).is_pending())
    })
    .await;
    assert!(
        pending,
        "browser pop_error_scope Promise must suspend cold validation"
    );
    drop(future);
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.allocations.record_count().unwrap(), allocations);
    assert_eq!(operation.recorder().submission_checkpoint(), checkpoint);
    left.allocation_lease().validate_live().unwrap();
    right.allocation_lease().validate_live().unwrap();
    let output = provider.elem_add(&left, &right).await.unwrap();
    let expected: Vec<u64> = lhs.iter().map(|&value| value + 2).collect();
    assert_eq!(
        provider.download_integer(&output).await.unwrap().data,
        HostIntegerDataOwned::U64(expected)
    );
    assert_eq!(
        provider.download_integer(&left).await.unwrap().data,
        HostIntegerDataOwned::U64(lhs)
    );
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_float_binary_precision_refusal_is_certified_before_emission() {
    use runmat_accelerate_api::{
        GpuTensorStorage, HostNumericDataView, HostNumericTensorView, ProviderPrecision,
    };
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let native_f64 = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.5, 2.5]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let records = provider.allocations.record_count().unwrap();
    for operation in ["add", "rem", "mod"] {
        let receipt = receipt(provider);
        let checkpoint = receipt.recorder().submission_checkpoint();
        let future = match operation {
            "add" => provider.elem_add(&native_f64, &native_f64),
            "rem" => provider.elem_rem(&native_f64, &native_f64),
            "mod" => provider.elem_mod(&native_f64, &native_f64),
            _ => unreachable!(),
        };
        let error = match observe_provider_future(Some(receipt.recorder()), future).await {
            Ok(_) => panic!("opposite native precision executed the F32 kernel"),
            Err(error) => error,
        };
        assert!(error.is_unsupported(), "{operation} typed fallback");
        assert!(!error.is_after_effect());
        assert_eq!(receipt.recorder().submission_checkpoint(), checkpoint);
        assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
        assert_eq!(provider.allocations.record_count().unwrap(), records);
    }
    native_f64.allocation_lease().validate_live().unwrap();
}
