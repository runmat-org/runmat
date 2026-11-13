use super::*;
use runmat_accelerate_api::{AccelProvider, HostNumericDataView, HostNumericTensorView};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pure_two_stage_geometry_and_limits() {
    let limits = wgpu::Limits::default();
    for (precision, bytes) in [(NumericPrecision::F32, 160), (NumericPrecision::F64, 320)] {
        for enabled in [false, true] {
            let snapshot = FrozenAnySnapshot { enabled };
            let plan = ProbeGeometry::plan(40, precision, snapshot, &limits, 64).unwrap();
            assert_eq!(plan.flag_bytes, bytes);
            assert_eq!(plan.scalar_bytes, bytes / 40);
            assert_eq!(plan.stages[0].input_bytes, bytes);
            assert_eq!(plan.stages[1].input_bytes, bytes / 40);
            assert_eq!(plan.stages[0].output_bytes, bytes / 40);
            assert_eq!(plan.stages[1].output_bytes, bytes / 40);
            assert_eq!(plan.stages[0].snapshot_copy_bytes, enabled.then_some(bytes));
            assert_eq!(
                plan.stages[1].snapshot_copy_bytes,
                enabled.then_some(bytes / 40)
            );
            assert_eq!(plan.stages[0].uniform_bytes, 16);
            assert_eq!(plan.stages[1].uniform_bytes, 16);
            assert_eq!(plan.stages[0].workgroups_x, 1);
            assert_eq!(plan.stages[1].workgroups_x, 1);
            assert_eq!(plan.readback_staging_bytes, bytes / 40);
        }
    }
    let snapshot = FrozenAnySnapshot { enabled: true };
    assert!(ProbeGeometry::plan(0, NumericPrecision::F32, snapshot, &limits, 64).is_err());
    assert!(ProbeGeometry::plan(usize::MAX, NumericPrecision::F32, snapshot, &limits, 64).is_err());
    assert!(ProbeGeometry::plan(40, NumericPrecision::F32, snapshot, &limits, 0).is_err());
    let mut constrained = limits.clone();
    constrained.max_storage_buffer_binding_size = 319;
    assert!(ProbeGeometry::plan(40, NumericPrecision::F64, snapshot, &constrained, 64).is_err());
    constrained = limits.clone();
    constrained.max_uniform_buffer_binding_size = 15;
    assert!(ProbeGeometry::plan(40, NumericPrecision::F32, snapshot, &constrained, 64).is_err());
    constrained = limits.clone();
    constrained.max_compute_workgroups_per_dimension = 0;
    assert!(ProbeGeometry::plan(40, NumericPrecision::F32, snapshot, &constrained, 64).is_err());
    constrained = limits.clone();
    constrained.max_compute_workgroup_size_x = 63;
    assert!(ProbeGeometry::plan(40, NumericPrecision::F32, snapshot, &constrained, 64).is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pure_stage_observation_rejects_missing_or_wrong_work() {
    let snapshot = FrozenAnySnapshot { enabled: true };
    let plan = ProbeGeometry::plan(
        40,
        NumericPrecision::F64,
        snapshot,
        &wgpu::Limits::default(),
        64,
    )
    .unwrap();
    let demand = plan.stages[0];
    let facts = StageFacts {
        output_bytes: 8,
        snapshot_copy_bytes: Some(320),
        uniform_bytes: Some(16),
        workgroups_x: Some(1),
        snapshot_copy_submitted: true,
        dispatch_submitted: true,
        snapshot_fence_bytes: Some(8),
        dispatch_fence_bytes: None,
    };
    assert!(demand
        .require_facts(ProbeStep::AnyRows, snapshot, facts)
        .is_ok());
    assert!(demand
        .require_facts(
            ProbeStep::AnyRows,
            snapshot,
            StageFacts {
                output_bytes: 4,
                ..facts
            }
        )
        .is_err());
    assert!(demand
        .require_facts(
            ProbeStep::AnyRows,
            snapshot,
            StageFacts {
                snapshot_copy_bytes: None,
                ..facts
            }
        )
        .is_err());
    assert!(demand
        .require_facts(
            ProbeStep::AnyRows,
            snapshot,
            StageFacts {
                uniform_bytes: None,
                ..facts
            }
        )
        .is_err());
    assert!(demand
        .require_facts(
            ProbeStep::AnyRows,
            snapshot,
            StageFacts {
                workgroups_x: None,
                ..facts
            }
        )
        .is_err());
    assert!(demand
        .require_facts(
            ProbeStep::AnyRows,
            snapshot,
            StageFacts {
                dispatch_submitted: false,
                ..facts
            }
        )
        .is_err());
    assert!(demand
        .require_facts(
            ProbeStep::AnyRows,
            snapshot,
            StageFacts {
                snapshot_fence_bytes: Some(4),
                ..facts
            }
        )
        .is_err());
    assert!(demand
        .require_facts(
            ProbeStep::AnyRows,
            snapshot,
            StageFacts {
                dispatch_fence_bytes: Some(16),
                ..facts
            }
        )
        .is_err());
}

async fn owner() -> &'static WgpuProvider {
    let options = crate::backend::wgpu::provider::WgpuProviderOptions::default();
    #[cfg(not(target_arch = "wasm32"))]
    let provider = WgpuProvider::new(options).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(options)
        .await
        .expect("actual WebGPU adapter");
    let provider = Box::leak(Box::new(provider));
    // SAFETY: independently created owner lives for this process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn inputs(provider: &'static WgpuProvider, complex: bool) -> (GpuTensorHandle, GpuTensorHandle) {
    let base_values = if complex {
        [2.0_f32, -2.0, 0.0, 5.0]
    } else {
        [2.0_f32, 3.0, 0.0, 5.0]
    };
    let exponent_values = [0.5_f32];
    let base = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&base_values),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let exponent = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&exponent_values),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    (base, exponent)
}

async fn numerical_and_unpolled(provider: &'static WgpuProvider) {
    for (complex, domain) in [
        (false, PowerResultDomain::Real),
        (true, PowerResultDomain::Complex),
    ] {
        let (base, exponent) = inputs(provider, complex);
        let before = provider.telemetry_snapshot();
        let unpolled = provider.classify_real_power_exec(&base, &exponent);
        drop(unpolled);
        assert_eq!(provider.telemetry_snapshot(), before);
        for _ in 0..2 {
            assert_eq!(
                provider
                    .classify_real_power_exec(&base, &exponent)
                    .await
                    .unwrap(),
                domain
            );
            base.allocation_lease().validate_live().unwrap();
            exponent.allocation_lease().validate_live().unwrap();
        }
        let (observed_domain, record) = provider
            .classify_real_power_observed_exec(&base, &exponent)
            .await
            .unwrap();
        assert_eq!(observed_domain, domain);
        assert_eq!(
            record
                .expect("nonempty probe ticket transcript")
                .completion_state(),
            ProbeCompletionState::Absent,
            "no outer recorder must not mint an optional completion fence"
        );
    }
    let wg = usize::try_from(provider.linear_launch.workgroup()).unwrap();
    let len = wg.checked_mul(2).unwrap().checked_add(1).unwrap();
    let mut base_values = vec![2.0_f32; len];
    *base_values.last_mut().unwrap() = -2.0;
    let base = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&base_values),
            shape: &[len, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let exponent = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[0.5]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert_eq!(
        provider
            .classify_real_power_exec(&base, &exponent)
            .await
            .unwrap(),
        PowerResultDomain::Complex
    );
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual WGPU adapter; Naga is not numerical or resource proof"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_single_probe_numerical_and_unpolled() {
    numerical_and_unpolled(owner().await).await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires RUNMAT_WGPU_FORCE_PRECISION=f64 and actual SHADER_F64/SHADER_INT64"]
async fn actual_f32_sources_on_f64_owner_probe() {
    assert_eq!(
        std::env::var("RUNMAT_WGPU_FORCE_PRECISION").as_deref(),
        Ok("f64")
    );
    let provider = owner().await;
    assert_eq!(provider.precision, NumericPrecision::F64);
    assert!(provider
        .enabled_features
        .contains(wgpu::Features::SHADER_F64 | wgpu::Features::SHADER_INT64));
    numerical_and_unpolled(provider).await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn actual_browser_pending_probe_cancellation_retains_borrowed_sources() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    let provider = owner().await;
    let (base, exponent) = inputs(provider, true);
    let mut future = Box::pin(provider.classify_real_power_exec(&base, &exponent));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    drop(future);
    base.allocation_lease().validate_live().unwrap();
    exponent.allocation_lease().validate_live().unwrap();
}

#[path = "tests/resource_scope.rs"]
mod resource_scope;
