use super::*;
use runmat_runtime::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};

#[test]
fn pol2cart_actual_nary_preparation_preserves_all_outputs_and_provider_feedback() {
    let provider = super::admitted_owners::owner();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for use_provider in [false, true] {
        for outputs in 0..=3 {
            let probe = Rc::new(Probe::default());
            probe.provider.set(use_provider);
            let context = runtime(&probe);
            futures::executor::block_on(context.scope(async {
                let auto = NativeAutoOffload::new(provider, ThresholdConfig::default());
                let request = AdmittedBuiltinPreparation::new(
                    BuiltinPreparationAdmission {
                        binding: runmat_builtins::BuiltinBindingIdentity {
                            builtin: runmat_builtins::POL2CART_CATALOG_ENTRY.identity,
                            variant: "default",
                        },
                        form: runmat_builtins::BuiltinPreparationFormId(1),
                        signature: 1,
                        input_count: 3,
                        requested_outputs: Some(outputs),
                    },
                    vec![
                        Value::Tensor(Tensor::new(vec![0.0; 2], vec![2, 1]).unwrap()),
                        Value::Tensor(Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap()),
                        Value::Tensor(Tensor::new(vec![7.0, 8.0], vec![1, 1, 2]).unwrap()),
                    ],
                )
                .unwrap();
                let prepared =
                    super::super::admitted::prepare_with(&auto, &context, request).unwrap();
                assert_eq!(prepared.args().len(), 3);
                assert!(prepared
                    .args()
                    .iter()
                    .all(|value| matches!(value, Value::GpuTensor(_)) == use_provider));
                let output = prepared
                    .execute(|args| async move {
                        runmat_runtime::dispatcher::call_builtin_async_with_outputs(
                            "pol2cart", &args, outputs,
                        )
                        .await
                    })
                    .await
                    .unwrap();
                let Value::OutputList(values) = output.value() else {
                    panic!("requested coordinate outputs")
                };
                assert_eq!(values.len(), outputs);
                for (index, value) in values.iter().enumerate() {
                    let (shape, data) = match value {
                        Value::GpuTensor(handle) => {
                            assert!(use_provider);
                            assert_eq!(
                                runmat_accelerate_api::handle_provenance(handle),
                                Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                            );
                            let host = provider.download(handle).await.unwrap();
                            (host.shape, host.data)
                        }
                        Value::Tensor(tensor) => {
                            assert!(!use_provider);
                            (tensor.shape().to_vec(), tensor.materialize_f64())
                        }
                        _ => panic!("dense coordinate"),
                    };
                    assert_eq!(shape, vec![2, 3, 2]);
                    let expected = match index {
                        0 => [2.0, 2.0, 3.0, 3.0, 4.0, 4.0].repeat(2),
                        1 => vec![0.0; 12],
                        _ => [vec![7.0; 6], vec![8.0; 6]].concat(),
                    };
                    assert_eq!(data, expected);
                }
                assert_eq!(probe.feedback.borrow().len(), usize::from(!use_provider));
            }));
        }
    }
}
