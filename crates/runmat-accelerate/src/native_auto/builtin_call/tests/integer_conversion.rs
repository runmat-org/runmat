use super::*;
use runmat_runtime::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};

#[test]
fn integer_family_real_provider_preparation_and_execution_use_existing_observer() {
    let provider = super::admitted_owners::owner();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for name in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        for use_provider in [false, true] {
            let probe = Rc::new(Probe::default());
            probe.provider.set(use_provider);
            let context = runtime(&probe);
            futures::executor::block_on(context.scope(async {
                let auto = NativeAutoOffload::new(provider, ThresholdConfig::default());
                let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name).unwrap();
                let request = AdmittedBuiltinPreparation::new(
                    BuiltinPreparationAdmission {
                        binding: runmat_builtins::BuiltinBindingIdentity {
                            builtin: entry.identity,
                            variant: "default",
                        },
                        form: runmat_builtins::BuiltinPreparationFormId(0),
                        signature: 0,
                        input_count: 1,
                        requested_outputs: Some(1),
                    },
                    vec![Value::Tensor(
                        Tensor::new(vec![1.0, 3.5, 0.0, 7.0], vec![1, 2, 2]).unwrap(),
                    )],
                )
                .unwrap();
                let prepared =
                    super::super::admitted::prepare_with(&auto, &context, request).unwrap();
                assert_eq!(probe.requests.borrow().len(), 1);
                assert_eq!(probe.requests.borrow()[0].signature.operation, name);
                assert_eq!(
                    matches!(prepared.args()[0], Value::GpuTensor(_)),
                    use_provider
                );
                let output = prepared
                    .execute(|args| async move {
                        runmat_runtime::dispatcher::call_builtin_async(name, &args).await
                    })
                    .await
                    .unwrap();
                if use_provider {
                    let Value::GpuTensor(handle) = output.value() else {
                        panic!("typed resident output")
                    };
                    assert_eq!(handle.shape(), &[1, 2, 2]);
                    assert!(runmat_accelerate_api::handle_integer_type(handle).is_some());
                    // Hook return proves submission, not device completion. Do not
                    // train a completed provider sample from this elapsed time.
                    assert!(probe.feedback.borrow().is_empty());
                    drop(output);
                } else {
                    let Value::Tensor(tensor) = output.value() else {
                        panic!("host integer array")
                    };
                    assert_eq!(tensor.shape(), &[1, 2, 2]);
                    assert!(tensor.integer_storage().is_some());
                    assert_eq!(probe.feedback.borrow().len(), 1);
                    assert!(probe.feedback.borrow()[0].succeeded);
                }
            }));
        }
    }
}
