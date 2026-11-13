use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{
    GpuHandleProvenance, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
    HostTensorView,
};
use runmat_value::{IntValue, IntegerStorage};
use std::error::Error;

#[test]
fn adam_operational_provider_error_retains_cause_without_gather_retry() {
    let error = adam_provider_error(anyhow::anyhow!(
        "unsupported-looking device allocation failed"
    ));
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    assert!(error.source().is_some());
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}

#[test]
fn adam_provider_result_admission_rejects_borrowed_alias_before_publication() {
    test_support::with_test_provider(|provider| {
        let shape = [1usize, 1usize];
        let parameter = provider
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &shape,
            })
            .expect("parameter");
        let gradient = provider
            .upload(&HostTensorView {
                data: &[0.1],
                shape: &shape,
            })
            .expect("gradient");
        let request = ProviderAdamUpdateRequest {
            parameters: &parameter,
            gradient: &gradient,
            average_grad: None,
            average_sq_grad: None,
            iteration: 1,
            learn_rate: 0.001,
            gradient_decay_factor: 0.9,
            squared_gradient_decay_factor: 0.999,
            epsilon: 1.0e-8,
        };
        let result = ProviderAdamUpdateResult {
            parameters: parameter.clone(),
            average_grad: provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &shape,
                })
                .expect("moment"),
            average_sq_grad: provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &shape,
                })
                .expect("squared moment"),
        };
        let err = validate_adam_gpu_result(&result, &request, provider)
            .expect_err("borrowed parameter alias must reject");
        assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        drop(result);
        assert!(parameter.allocation_lease().validate_live().is_ok());
    });
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-10,
        "got {actual}, expected {expected}"
    );
}

#[test]
fn adamupdate_iteration_parser_preserves_typed_integer_bounds() {
    assert_eq!(
        positive_iteration(&Value::Int(IntValue::U16(7))).unwrap(),
        7
    );
    assert!(positive_iteration(&Value::Int(IntValue::I8(-1))).is_err());

    let exact = (1_u64 << 53) + 1;
    let typed =
        Tensor::new_integer(IntegerStorage::U64(vec![exact]), vec![1, 1]).expect("iteration");
    let parsed = positive_iteration(&Value::Tensor(typed));
    if usize::BITS == 64 {
        assert_eq!(parsed.unwrap(), exact as usize);
    } else {
        assert!(parsed.is_err());
    }

    let boundary = positive_iteration(&Value::Num(usize::MAX as f64));
    if usize::BITS == 64 {
        assert!(boundary.is_err());
    } else {
        assert_eq!(boundary.unwrap(), usize::MAX);
    }
    assert!(positive_iteration(&Value::Num((usize::MAX as f64) + 1.0)).is_err());
}

#[test]
fn adamupdate_payload_preserves_native_single_representation() {
    let input = Tensor::from_f32(vec![0.1, f32::MAX], vec![1, 2]).expect("single input");
    let payload =
        NumericPayload::parse(&Value::Tensor(input), "parameters", true).expect("single payload");
    assert_eq!(payload.data, vec![f64::from(0.1_f32), f64::from(f32::MAX)]);
    let output = payload.materialize(vec![0.25, 0.5]).expect("single output");
    let Value::Tensor(output) = output else {
        panic!("expected tensor output");
    };
    assert_eq!(
        output.into_numeric_storage().expect("single storage"),
        runmat_value::NumericStorage::F32(vec![0.25, 0.5])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn adamupdate_admission_preserves_exact_rank_zero_and_one_payload_shapes() {
    let empty = || Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
    for (shape, data) in [(vec![], vec![1.0]), (vec![2], vec![1.0, 2.0])] {
        let values = vec![
            Value::Tensor(Tensor::new(data.clone(), shape.clone()).unwrap()),
            Value::Tensor(Tensor::new(vec![0.1; data.len()], shape.clone()).unwrap()),
            empty(),
            empty(),
            Value::Num(1.0),
        ];
        let args = AdamUpdateArgs::parse(
            crate::call::BuiltinArguments::from_borrowed(&values, None).unwrap(),
        )
        .unwrap();
        assert_eq!(args.parameters.shape, shape);
        assert_eq!(args.gradient.shape, shape);
        assert_eq!(args.average_grad.shape, shape);
        assert_eq!(args.average_sq_grad.shape, shape);
    }
    let values = vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1, 0.2], vec![1, 2]).unwrap()),
        empty(),
        empty(),
        Value::Num(1.0),
    ];
    let error =
        AdamUpdateArgs::parse(crate::call::BuiltinArguments::from_borrowed(&values, None).unwrap())
            .err()
            .unwrap();
    assert!(error.message().contains("grad"));
}

#[test]
fn adamupdate_gpu_inputs_return_resident_outputs() {
    test_support::with_test_provider(|provider| {
        let shape = [1usize, 3usize];
        let parameters = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0],
                shape: &shape,
            })
            .expect("upload parameters");
        let gradient = provider
            .upload(&HostTensorView {
                data: &[0.1, -0.2, 0.3],
                shape: &shape,
            })
            .expect("upload gradient");
        let empty = Tensor::new(Vec::new(), vec![0, 0]).expect("empty");

        provider.reset_telemetry();
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let out = block_on(adamupdate_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(parameters),
                Value::GpuTensor(gradient),
                Value::Tensor(empty.clone()),
                Value::Tensor(empty),
                Value::Num(1.0),
                Value::Num(0.01),
            ]),
        ))
        .expect("adamupdate gpu");

        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), 3);
        assert!(matches!(values[0], Value::GpuTensor(_)));
        assert!(matches!(values[1], Value::GpuTensor(_)));
        assert!(matches!(values[2], Value::GpuTensor(_)));

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.download_bytes, 0);

        let updated = test_support::gather(values[0].clone()).expect("updated");
        let avg = test_support::gather(values[1].clone()).expect("avg");
        let avg_sq = test_support::gather(values[2].clone()).expect("avg sq");

        assert_eq!(updated.shape(), shape);
        assert_eq!(avg.shape(), shape);
        assert_eq!(avg_sq.shape(), shape);
        assert_close(updated.materialize_f64()[0], 0.990000001);
        assert_close(updated.materialize_f64()[1], 2.0099999995);
        assert_close(updated.materialize_f64()[2], 2.9900000003333334);
        assert_close(avg.materialize_f64()[0], 0.01);
        assert_close(avg.materialize_f64()[1], -0.02);
        assert_close(avg.materialize_f64()[2], 0.03);
        assert_close(avg_sq.materialize_f64()[0], 0.00001);
        assert_close(avg_sq.materialize_f64()[1], 0.00004);
        assert_close(avg_sq.materialize_f64()[2], 0.00009);
    });
}

#[test]
fn adamupdate_resident_results_inherit_explicit_input_intent() {
    test_support::with_test_provider(|provider| {
        let shape = [1usize, 1usize];
        let parameters = provider
            .upload(&HostTensorView {
                data: &[2.0],
                shape: &shape,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let gradient = provider
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &shape,
            })
            .unwrap();
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        provider.reset_telemetry();
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let result = block_on(adamupdate_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(parameters.clone()),
                Value::GpuTensor(gradient.clone()),
                empty.clone(),
                empty,
                Value::Num(1.0),
            ]),
        ))
        .unwrap();
        let Value::OutputList(values) = result else {
            panic!("expected three results")
        };
        for value in &values {
            let Value::GpuTensor(handle) = value else {
                panic!("expected resident result")
            };
            assert!(runmat_accelerate_api::handle_is_explicit(handle));
        }
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let updated = test_support::gather(values[0].clone()).unwrap();
        assert_close(updated.materialize_f64()[0], 1.99900000001);
        assert!(parameters.allocation_lease().validate_live().is_ok());
        assert!(gradient.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn adamupdate_explicit_single_rejects_unsupported_provider_without_download() {
    test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let parameters = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[1.0, 2.0]),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let gradient = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[0.1, -0.2]),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        provider.reset_telemetry();
        let error = block_on(adamupdate_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(parameters.clone()),
                Value::GpuTensor(gradient.clone()),
                empty.clone(),
                empty,
                Value::Num(1.0),
            ]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:Unsupported"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(parameters.allocation_lease().validate_live().is_ok());
        assert!(gradient.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn adamupdate_single_resident_inputs_fall_back_without_widening_class() {
    test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let parameters = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[1.0, 2.0]),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .expect("single parameters");
        let gradient = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[0.1, -0.2]),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .expect("single gradient");
        let empty = Tensor::new(Vec::new(), vec![0, 0]).expect("empty state");

        provider.reset_telemetry();
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let result = block_on(adamupdate_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(parameters),
                Value::GpuTensor(gradient),
                Value::Tensor(empty.clone()),
                Value::Tensor(empty),
                Value::Num(1.0),
            ]),
        ))
        .expect("single storage host fallback");
        let Value::OutputList(values) = result else {
            panic!("expected three optimizer outputs");
        };
        assert_eq!(values.len(), 3);
        for value in values.try_into_owned().expect("unique adam outputs") {
            let Value::Tensor(tensor) = value else {
                panic!("the fallback returns host tensors");
            };
            assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
            assert_eq!(tensor.shape(), shape);
        }
        assert!(provider.telemetry_snapshot().download_bytes > 0);
    });
}

#[test]
fn adamupdate_gpu_single_output_frees_state_outputs() {
    test_support::with_test_provider(|provider| {
        let shape = [1usize, 1usize];
        let parameters = provider
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &shape,
            })
            .expect("upload parameters");
        let gradient = provider
            .upload(&HostTensorView {
                data: &[0.1],
                shape: &shape,
            })
            .expect("upload gradient");
        let average_grad = provider
            .upload(&HostTensorView {
                data: &[0.0],
                shape: &shape,
            })
            .expect("upload averageGrad");
        let average_sq_grad = provider
            .upload(&HostTensorView {
                data: &[0.0],
                shape: &shape,
            })
            .expect("upload averageSqGrad");

        let _guard = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let out = block_on(adamupdate_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(parameters),
                Value::GpuTensor(gradient),
                Value::GpuTensor(average_grad),
                Value::GpuTensor(average_sq_grad),
                Value::Num(1.0),
            ]),
        ))
        .expect("adamupdate gpu");

        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), 1);
        assert!(matches!(values[0], Value::GpuTensor(_)));
    });
}
