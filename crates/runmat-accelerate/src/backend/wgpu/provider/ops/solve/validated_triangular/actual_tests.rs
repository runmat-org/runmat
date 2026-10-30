use super::*;
use crate::wgsl::f64_math::PreparationError;
use runmat_accelerate_api::observation::{observe_provider_future, ProviderOperationReceipt};
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataView, HostIntegerTensorView, HostLogicalTensorView,
    HostTensorView, NumericElementType, ProviderLinsolveOptions,
};

fn receipt(provider: &WgpuProvider) -> ProviderOperationReceipt {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ProviderOperationReceipt::with_wgpu_progress(
            provider.observation_budget.clone(),
            provider
                .observation_progress
                .as_ref()
                .expect("actual WGPU provider has a native progress port")
                .clone(),
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        ProviderOperationReceipt::with_budget(provider.observation_budget.clone())
    }
}

async fn owner_with_scalar(scalar: Option<u32>) -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let mut provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let mut provider = WgpuProvider::new_async(Default::default())
        .await
        .expect("actual WebGPU adapter");
    if let Some(scalar) = scalar {
        assert!(scalar <= provider.adapter_limits.max_compute_workgroup_size_x);
        assert!(
            scalar
                <= provider
                    .adapter_limits
                    .max_compute_invocations_per_workgroup
        );
        provider.workgroup_config.scalar = scalar;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        provider.pipeline_cache_dir = Some(std::env::temp_dir().join(format!(
            "runmat-validated-triangular-{}-{nonce}",
            std::process::id()
        )));
    }
    let provider = Box::leak(Box::new(provider));
    assert_eq!(
        provider.precision,
        crate::backend::wgpu::types::NumericPrecision::F32,
        "the validated triangular device recipe requires its F32 shader ABI"
    );
    // SAFETY: the actual provider remains live for the process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn fixture(workgroup: u32, lower: bool) -> ([f64; 4], Vec<f64>, [usize; 2], Vec<f64>) {
    let columns = workgroup as usize + 1;
    let lhs = if lower {
        [2.0, 1.0, 0.0, 3.0]
    } else {
        [2.0, 0.0, 1.0, 3.0]
    };
    let shape = [2, columns];
    let mut rhs = Vec::with_capacity(2 * columns);
    let mut expected = Vec::with_capacity(2 * columns);
    for column in 0..columns {
        let b0 = 2.0 + column as f64 * 0.25;
        let b1 = 7.0 + column as f64 * 0.5;
        rhs.extend([b0, b1]);
        if lower {
            let x0 = b0 / 2.0;
            expected.extend([x0, (b1 - x0) / 3.0]);
        } else {
            let x1 = b1 / 3.0;
            expected.extend([(b0 - x1) / 2.0, x1]);
        }
    }
    (lhs, rhs, shape, expected)
}

async fn cold_warm_and_invalid(provider: &'static WgpuProvider) {
    let (lhs_data, rhs_data, rhs_shape, _) = fixture(provider.workgroup_config.scalar, true);
    let lhs = provider
        .upload(&HostTensorView {
            data: &lhs_data,
            shape: &[2, 2],
        })
        .unwrap();
    let rhs = provider
        .upload(&HostTensorView {
            data: &rhs_data,
            shape: &rhs_shape,
        })
        .unwrap();
    let wrong = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 0, 0, 1],
            shape: &[2, 2],
        })
        .unwrap();
    let integer = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I32(&[2, 1, 0, 3]),
            shape: &[2, 2],
        })
        .unwrap();
    assert_eq!(integer.physical().element_type(), NumericElementType::I32);
    let options = ProviderLinsolveOptions {
        lower: true,
        ..Default::default()
    };
    let allocations = provider.allocations.record_count().unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let misses = provider.telemetry_snapshot().fusion_cache_misses;
    for invalid_lhs in [&wrong, &integer] {
        let refused = receipt(provider);
        let checkpoint = refused.recorder().submission_checkpoint();
        let error = observe_provider_future(
            Some(refused.recorder()),
            provider.linsolve(invalid_lhs, &rhs, &options),
        )
        .await
        .expect_err("non-floating triangular owner must be refused before preparation");
        assert!(error.is_unsupported());
        assert_eq!(refused.recorder().submission_checkpoint(), checkpoint);
        assert_eq!(provider.allocations.record_count().unwrap(), allocations);
        assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
        assert_eq!(provider.telemetry_snapshot().fusion_cache_misses, misses);
        invalid_lhs.allocation_lease().validate_live().unwrap();
    }
    let invalid = super::super::super::pipeline_recipe::PipelineRecipe {
        shader: "@compute @workgroup_size(1) fn main() { let bad = ; }".into(),
        layout: crate::backend::wgpu::bindings::TRIANGULAR_SOLVE_LAYOUT_TAG.into(),
        workgroup: provider.workgroup_config.scalar,
    };
    let failed_scope = receipt(provider);
    let checkpoint = failed_scope.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(failed_scope.recorder()),
        provider.prepare_validated_triangular_solve_pipeline(&invalid),
    )
    .await
    .err()
    .expect("invalid WGSL must fail typed compiler preparation before driver effects");
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    while !cause.is::<PreparationError>() {
        cause = cause
            .source()
            .expect("original typed shader compiler cause");
    }
    assert!(matches!(
        cause.downcast_ref::<PreparationError>().unwrap(),
        PreparationError::Parse(_)
    ));
    assert_eq!(failed_scope.recorder().submission_checkpoint(), checkpoint);
    assert_eq!(provider.allocations.record_count().unwrap(), allocations);
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.telemetry_snapshot().fusion_cache_misses, misses);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = provider.pipeline_cache_dir.as_ref().unwrap();
        assert!(!dir.join(format!("{:016x}.wgsl", invalid.key())).exists());
        assert!(!dir.join(format!("{:016x}.json", invalid.key())).exists());
    }
    let empty_lhs = provider
        .upload(&HostTensorView {
            data: &[],
            shape: &[0, 0],
        })
        .unwrap();
    let empty_rhs = provider
        .upload(&HostTensorView {
            data: &[],
            shape: &[0, 1],
        })
        .unwrap();
    let empty_logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[],
            shape: &[0, 0],
        })
        .unwrap();
    let empty_allocations = provider.allocations.record_count().unwrap();
    let empty_cache = provider.validated_fusion_cache.lock().unwrap().len();
    let empty_misses = provider.telemetry_snapshot().fusion_cache_misses;
    let empty_refused = receipt(provider);
    let empty_checkpoint = empty_refused.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(empty_refused.recorder()),
        provider.linsolve(&empty_logical, &empty_rhs, &options),
    )
    .await
    .expect_err("logical empty triangular owner must be refused");
    assert!(error.is_unsupported());
    assert_eq!(
        empty_refused.recorder().submission_checkpoint(),
        empty_checkpoint
    );
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        empty_allocations
    );
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        empty_cache
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        empty_misses
    );
    let empty = provider
        .linsolve(&empty_lhs, &empty_rhs, &options)
        .await
        .unwrap();
    assert_eq!(empty.solution.shape(), &[0, 1]);
    assert_eq!(empty.solution.device_id(), provider.device_id());
    assert_eq!(
        empty.solution.physical().element_type(),
        empty_rhs.physical().element_type()
    );
    assert!(!empty.solution.shares_allocation_with(&empty_rhs));
    let empty_host = provider.download(&empty.solution).await.unwrap();
    assert_eq!(empty_host.shape, [0, 1]);
    assert!(empty_host.data.is_empty());
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        empty_cache
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        empty_misses
    );
    empty_lhs.allocation_lease().validate_live().unwrap();
    empty_rhs.allocation_lease().validate_live().unwrap();
    empty_logical.allocation_lease().validate_live().unwrap();
    let tolerance = if provider.precision == crate::backend::wgpu::types::NumericPrecision::F64 {
        1.0e-10
    } else {
        1.0e-5
    };
    for lower in [true, false] {
        let (matrix, values, shape, expected) = fixture(provider.workgroup_config.scalar, lower);
        let matrix = provider
            .upload(&HostTensorView {
                data: &matrix,
                shape: &[2, 2],
            })
            .unwrap();
        let values = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .unwrap();
        let options = ProviderLinsolveOptions {
            lower,
            upper: !lower,
            ..Default::default()
        };
        let before_misses = provider.telemetry_snapshot().fusion_cache_misses;
        let before_hits = provider.telemetry_snapshot().fusion_cache_hits;
        for attempt in 0..2 {
            let observed = receipt(provider);
            let checkpoint = observed.recorder().submission_checkpoint();
            let result = observe_provider_future(
                Some(observed.recorder()),
                provider.linsolve(&matrix, &values, &options),
            )
            .await
            .unwrap();
            assert!(observed.recorder().submission_checkpoint() > checkpoint);
            assert!(result.reciprocal_condition.is_nan());
            let output = result.solution;
            assert_eq!(output.shape(), shape);
            assert_eq!(output.device_id(), provider.device_id());
            assert_eq!(
                output.physical().element_type(),
                matrix.physical().element_type()
            );
            assert!(!output.shares_allocation_with(&matrix));
            assert!(!output.shares_allocation_with(&values));
            output.allocation_lease().validate_live().unwrap();
            let host = provider.download(&output).await.unwrap();
            assert_eq!(host.shape, shape);
            assert_eq!(host.data.len(), expected.len());
            for (actual, expected) in host.data.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() < tolerance,
                    "got {actual}, expected {expected}"
                );
            }
            assert_eq!(
                provider.telemetry_snapshot().fusion_cache_misses,
                before_misses + 1
            );
            assert_eq!(
                provider.telemetry_snapshot().fusion_cache_hits,
                before_hits + attempt as u64
            );
        }
        matrix.allocation_lease().validate_live().unwrap();
        values.allocation_lease().validate_live().unwrap();
    }
    if provider.precision == crate::backend::wgpu::types::NumericPrecision::F32 {
        let matrix = provider
            .upload(&HostTensorView {
                data: &[4.0, 0.0, 0.0, 9.0],
                shape: &[2, 2],
            })
            .unwrap();
        let values = provider
            .upload(&HostTensorView {
                data: &[8.0, 18.0],
                shape: &[2, 1],
            })
            .unwrap();
        for options in [
            ProviderLinsolveOptions {
                posdef: true,
                ..Default::default()
            },
            ProviderLinsolveOptions::default(),
        ] {
            let observed = receipt(provider);
            let checkpoint = observed.recorder().submission_checkpoint();
            let result = observe_provider_future(
                Some(observed.recorder()),
                provider.linsolve(&matrix, &values, &options),
            )
            .await
            .unwrap();
            assert!(observed.recorder().submission_checkpoint() > checkpoint);
            assert_eq!(result.solution.shape(), &[2, 1]);
            assert!(!result.solution.shares_allocation_with(&matrix));
            assert!(!result.solution.shares_allocation_with(&values));
            let host = provider.download(&result.solution).await.unwrap();
            assert_eq!(host.shape, [2, 1]);
            assert_eq!(host.data.len(), 2);
            for (actual, expected) in host.data.iter().zip([2.0, 2.0]) {
                assert!((actual - expected).abs() < 1.0e-4);
            }
            matrix.allocation_lease().validate_live().unwrap();
            values.allocation_lease().validate_live().unwrap();
        }
    }
    lhs.allocation_lease().validate_live().unwrap();
    rhs.allocation_lease().validate_live().unwrap();
    wrong.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual F32 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f32"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_triangular_validated_cold_warm_and_invalid_scope() {
    cold_warm_and_invalid(owner_with_scalar(None).await).await;
    cold_warm_and_invalid(owner_with_scalar(Some(256)).await).await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn pending_triangular_preparation_cancel_retains_borrowed_inputs() {
    use std::future::Future;
    let provider = owner_with_scalar(None).await;
    let (matrix, values, shape, expected) = fixture(provider.workgroup_config.scalar, true);
    let matrix = provider
        .upload(&HostTensorView {
            data: &matrix,
            shape: &[2, 2],
        })
        .unwrap();
    let values = provider
        .upload(&HostTensorView {
            data: &values,
            shape: &shape,
        })
        .unwrap();
    let options = ProviderLinsolveOptions {
        lower: true,
        ..Default::default()
    };
    let allocations = provider.allocations.record_count().unwrap();
    let mut future = Box::pin(provider.linsolve(&matrix, &values, &options));
    let pending = std::future::poll_fn(|context| {
        std::task::Poll::Ready(future.as_mut().poll(context).is_pending())
    })
    .await;
    assert!(pending);
    drop(future);
    assert_eq!(provider.allocations.record_count().unwrap(), allocations);
    assert!(provider.validated_fusion_cache.lock().unwrap().is_empty());
    matrix.allocation_lease().validate_live().unwrap();
    values.allocation_lease().validate_live().unwrap();
    let result = provider.linsolve(&matrix, &values, &options).await.unwrap();
    let host = provider.download(&result.solution).await.unwrap();
    assert_eq!(host.shape, shape);
    assert_eq!(host.data.len(), expected.len());
    for (actual, expected) in host.data.iter().zip(&expected) {
        assert!((actual - expected).abs() < 1.0e-5);
    }
    matrix.allocation_lease().validate_live().unwrap();
    values.allocation_lease().validate_live().unwrap();
}
