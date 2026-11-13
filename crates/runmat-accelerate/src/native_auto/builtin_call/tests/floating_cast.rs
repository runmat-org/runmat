//! Actual dispatcher/body/provider path. The test placement service chooses a
//! candidate deterministically; it is not evidence of profitable auto selection.
use super::*;

mod fallback;
#[cfg(not(target_arch = "wasm32"))]
mod submitted;
#[test]
#[ignore = "requires actual WGPU device; execute explicitly for direct receipt qualification"]
fn actual_single_host_staging_body_and_publication_produce_one_completed_receipt() {
    use runmat_accelerate_api::{HostNumericDataOwned, NumericElementType};
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(Default::default())
        .expect("actual WGPU provider required");
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let probe = Rc::new(Probe::default());
    probe.provider.set(true);
    let context = runtime(&probe);
    let context = context.clone().with_service_ports(
        context
            .service_ports()
            .clone()
            .with_acceleration(Rc::new(Adapter(provider, 1))),
    );
    let input = Value::Tensor(
        Tensor::from_f32(vec![1.25, -0.0, 16_777_216.0, 3.5], vec![1, 2, 2]).unwrap(),
    );
    let output = futures::executor::block_on(context.scope(async {
        runmat_runtime::dispatcher::call_builtin_async_in_context(&context, "single", &[input])
            .await
    }))
    .unwrap();
    let Value::GpuTensor(resident_output) = output.value() else {
        panic!(
            "expected resident output, got {:?}; selected graphs={}, feedback={}",
            output.value(),
            probe.requests.borrow().len(),
            probe.feedback.borrow().len()
        );
    };
    assert_eq!(
        resident_output.descriptor().element_type,
        Some(NumericElementType::F32)
    );
    assert_eq!(resident_output.shape(), &[1, 2, 2]);
    assert_eq!(probe.requests.borrow().len(), 1);
    let feedback = probe.feedback.borrow();
    assert_eq!(
        feedback.len(),
        1,
        "mapped upload plus same-type cast needs no deferred queue completion"
    );
    assert!(feedback[0].succeeded);
    drop(feedback);
    let HostNumericDataOwned::F32(values) =
        futures::executor::block_on(provider.download_numeric(resident_output))
            .unwrap()
            .data
    else {
        panic!("F32 physical output");
    };
    assert_eq!(
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [1.25_f32, -0.0, 16_777_216.0, 3.5].map(f32::to_bits)
    );
    drop(output);
}

#[test]
#[ignore = "requires actual WGPU device; execute explicitly for template adaptation qualification"]
fn actual_single_like_adaptation_has_no_plain_cast_candidate_or_completion_receipt() {
    use runmat_accelerate_api::{
        GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    };
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(Default::default())
        .expect("actual WGPU provider required");
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _compatibility = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let data = [1.25_f32, -0.0, 16_777_216.0, 3.5];
    for device_like in [false, true] {
        let probe = Rc::new(Probe::default());
        probe.provider.set(true);
        let context = runtime(&probe);
        let context = context.clone().with_service_ports(
            context
                .service_ports()
                .clone()
                .with_acceleration(Rc::new(Adapter(provider, 2))),
        );
        context.set_runmat_extensions_enabled(true);
        let actual = futures::executor::block_on(context.scope(async {
            // Allocate and read back under the same invocation authority as
            // execution. Standalone allocations are not session imports.
            let input = provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&data),
                    shape: &[1, 2, 2],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            let prototype = provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&[0.0]),
                    shape: &[1, 1],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            // Host-like downloads; device-like uploads. Neither is the form1
            // no-adaptation physical route or a completed plain-cast sample.
            let args = [
                if device_like {
                    Value::Tensor(Tensor::from_f32(data.to_vec(), vec![1, 2, 2]).unwrap())
                } else {
                    Value::GpuTensor(input)
                },
                Value::from("like"),
                if device_like {
                    Value::GpuTensor(prototype)
                } else {
                    Value::Num(0.0)
                },
            ];
            let result = runmat_runtime::dispatcher::call_builtin_async_in_context(
                &context, "single", &args,
            )
            .await
            .unwrap();
            match result.value() {
                Value::Tensor(value) if !device_like => {
                    assert_eq!(value.shape(), [1, 2, 2]);
                    value.as_f32_slice().unwrap().to_vec()
                }
                Value::GpuTensor(value) if device_like => {
                    let downloaded = provider.download_numeric(value).await.unwrap();
                    assert_eq!(downloaded.shape, vec![1, 2, 2]);
                    let HostNumericDataOwned::F32(data) = downloaded.data else {
                        panic!("Single result")
                    };
                    data
                }
                other => panic!("wrong template residency: {other:?}"),
            }
        }));
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            data.map(f32::to_bits)
        );
        assert!(
            probe.requests.borrow().is_empty(),
            "unresolved adaptation must not enter profitability comparison"
        );
        assert!(
            probe.feedback.borrow().is_empty(),
            "adaptation is not a completed plain-cast sample"
        );
    }
}

#[test]
#[ignore = "requires actual WGPU device; execute explicitly for double host-like qualification"]
fn actual_double_host_like_keeps_domain_gather_without_plain_cast_feedback() {
    use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView, HostNumericTensorView};
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(Default::default())
        .expect("actual WGPU provider required");
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let probe = Rc::new(Probe::default());
    probe.provider.set(true);
    let context = runtime(&probe);
    let context = context.clone().with_service_ports(
        context
            .service_ports()
            .clone()
            .with_acceleration(Rc::new(Adapter(provider, 2))),
    );
    context.set_runmat_extensions_enabled(true);
    let data = [1.25_f32, -0.0, 16_777_216.0, 3.5];
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&data),
            shape: &[1, 2, 2],
            storage: GpuTensorStorage::Real,
        })
        .expect("resident single input");
    let output = futures::executor::block_on(context.scope(async {
        runmat_runtime::dispatcher::call_builtin_async_in_context(
            &context,
            "double",
            &[
                Value::GpuTensor(input.clone()),
                Value::from("like"),
                Value::Num(0.0),
            ],
        )
        .await
    }))
    .expect("host-like double conversion");
    let Value::Tensor(output) = output.value() else {
        panic!("host prototype must select dense host double output: {output:?}");
    };
    assert_eq!(output.shape(), &[1, 2, 2]);
    assert_eq!(
        output
            .as_f64_slice()
            .expect("double storage")
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        data.map(|value| f64::from(value).to_bits())
    );
    assert!(probe.requests.borrow().is_empty());
    assert!(probe.feedback.borrow().is_empty());
    drop(input);
}
