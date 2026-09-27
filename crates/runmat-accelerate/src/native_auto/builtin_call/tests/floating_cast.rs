//! Actual dispatcher/body/provider path. The test placement service chooses a
//! candidate deterministically; it is not evidence of profitable auto selection.
use super::*;
use runmat_runtime::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use runmat_runtime::context::{RuntimeAccelerationService, RuntimeServiceFuture};

mod fallback;

struct Adapter(&'static dyn AccelProvider, u16);
impl RuntimeAccelerationService for Adapter {
    fn supports_operation(&self, _: &str) -> bool {
        true
    }
    fn prepare_admitted(
        &self,
        context: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        let provider = self.0;
        let expected_form = self.1;
        Box::pin(async move {
            assert_eq!(
                request.form().id,
                runmat_builtins::BuiltinPreparationFormId(expected_form)
            );
            let auto = NativeAutoOffload::new(provider, ThresholdConfig::default());
            let prepared = super::super::admitted::prepare_with(&auto, &context, request)
                .map_err(|error| *error)?;
            let (operands, guard) = prepared.into_execution();
            Ok(PreparedBuiltinOperands::new(
                operands,
                Some(Box::new(guard)),
            ))
        })
    }
}

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
    let output =
        futures::executor::block_on(context.scope(async {
            runmat_runtime::dispatcher::call_builtin_async("single", &[input]).await
        }))
        .unwrap();
    let Value::GpuTensor(output) = output else {
        panic!("actual resident output");
    };
    assert_eq!(
        output.descriptor.element_type,
        Some(NumericElementType::F32)
    );
    assert_eq!(output.shape, vec![1, 2, 2]);
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
        futures::executor::block_on(provider.download_numeric(&output))
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
    provider.free(&output).unwrap();
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
    for device_like in [false, true] {
        let probe = Rc::new(Probe::default());
        probe.provider.set(true);
        let context = runtime(&probe);
        let context = context.clone().with_service_ports(
            context
                .service_ports()
                .clone()
                .with_acceleration(Rc::new(Adapter(provider, 0))),
        );
        // Host-like must download the resident result; device-like must upload
        // the host result. Neither is the form1 no-adaptation physical route.
        let args = [
            if device_like {
                Value::Tensor(Tensor::from_f32(data.to_vec(), vec![1, 2, 2]).unwrap())
            } else {
                Value::GpuTensor(input.clone())
            },
            Value::from("like"),
            if device_like {
                Value::GpuTensor(prototype.clone())
            } else {
                Value::Num(0.0)
            },
        ];
        let result = futures::executor::block_on(context.scope(async {
            runmat_runtime::dispatcher::call_builtin_async("single", &args).await
        }))
        .unwrap();
        let actual = match result {
            Value::Tensor(value) if !device_like => {
                assert_eq!(value.shape, vec![1, 2, 2]);
                value.as_f32_slice().unwrap().to_vec()
            }
            Value::GpuTensor(value) if device_like => {
                let downloaded =
                    futures::executor::block_on(provider.download_numeric(&value)).unwrap();
                assert_eq!(downloaded.shape, vec![1, 2, 2]);
                let HostNumericDataOwned::F32(data) = downloaded.data else {
                    panic!("Single result")
                };
                provider.free(&value).unwrap();
                data
            }
            other => panic!("wrong template residency: {other:?}"),
        };
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
    provider.free(&input).unwrap();
    provider.free(&prototype).unwrap();
}
