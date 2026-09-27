//! Actual dispatcher/body/provider path. The test placement service chooses a
//! candidate deterministically; it is not evidence of profitable auto selection.
use super::*;
use runmat_runtime::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use runmat_runtime::context::{RuntimeAccelerationService, RuntimeServiceFuture};

struct Adapter(&'static dyn AccelProvider);
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
        Box::pin(async move {
            assert_eq!(
                request.form().id,
                runmat_builtins::BuiltinPreparationFormId(1)
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
            .with_acceleration(Rc::new(Adapter(provider))),
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
