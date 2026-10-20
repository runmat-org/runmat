use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::BuiltinOutputMode;
use runmat_value::IntegerStorage;

#[test]
fn single_xy_and_elevation_keep_independent_classes() {
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let outputs = output_list(
        call(
            Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
            Value::Tensor(tensor(vec![2.0, 3.0], vec![1, 2])),
            vec![Value::Tensor(tensor(vec![7.0, 8.0], vec![1, 1, 2]))],
        )
        .unwrap(),
    );
    for value in &outputs[..2] {
        let Value::Tensor(t) = value else {
            panic!("tensor")
        };
        assert_eq!(t.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(t.shape(), &[1, 2, 2]);
    }
    let Value::Tensor(z) = &outputs[2] else {
        panic!("elevation tensor")
    };
    assert_eq!(z.numeric_dtype(), runmat_value::NumericDType::F64);
    assert_eq!(z.materialize_f64(), vec![7.0, 7.0, 8.0, 8.0]);
    assert_eq!(data(&outputs[0]), vec![2.0, 3.0, 2.0, 3.0]);
}

#[test]
fn one_dimensional_storage_is_aligned_as_row_and_empty_nd_stays_empty() {
    let outputs = call(
        Value::Tensor(tensor(vec![0.0; 3], vec![3])),
        Value::Tensor(tensor(vec![2.0, 4.0], vec![2, 1])),
        Vec::new(),
    )
    .unwrap();
    let Value::Tensor(out) = outputs else {
        panic!("tensor")
    };
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.materialize_f64(), vec![2.0, 4.0, 2.0, 4.0, 2.0, 4.0]);
    let output = call(
        Value::Tensor(Tensor::from_f32(Vec::new(), vec![0, 1, 2]).unwrap()),
        Value::Tensor(tensor(vec![1.0; 3], vec![1, 3])),
        Vec::new(),
    )
    .unwrap();
    let Value::Tensor(out) = output else {
        panic!("empty tensor")
    };
    assert_eq!(out.shape(), &[0, 3, 2]);
    assert_eq!(out.numeric_dtype(), runmat_value::NumericDType::F32);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn low_rank_coordinates_preserve_requested_outputs_and_row_broadcasting() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let scalar = output_list(
        call(
            Value::Tensor(Tensor::new(vec![0.0], Vec::new()).unwrap()),
            Value::Num(2.0),
            Vec::new(),
        )
        .unwrap(),
    );
    assert_eq!(scalar, vec![Value::Num(2.0), Value::Num(0.0)]);

    let outputs = output_list(
        call(
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![2]).unwrap()),
            Value::Num(2.0),
            Vec::new(),
        )
        .unwrap(),
    );
    assert_eq!(outputs.len(), 2);
    for (output, expected) in outputs.into_iter().zip([vec![2.0, 2.0], vec![0.0, 0.0]]) {
        let Value::Tensor(output) = output else {
            panic!("expected coordinate tensor");
        };
        assert_eq!(output.shape(), &[1, 2]);
        assert_eq!(output.materialize_f64(), expected);
    }
}

use crate::builtins::common::test_support;

const EPS: f64 = 1.0e-12;

fn call(theta: Value, rho: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::pol2cart_builtin(
        None,
        theta,
        rho,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
    .map(|owned| owned.value().clone())
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::new(data, shape).expect("tensor")
}

fn output_list(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique output list")
            .try_into_unbounded_vec()
            .expect("unbounded output list"),
        other => panic!("expected output list, got {other:?}"),
    }
}

fn data(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(n) => vec![*n],
        Value::Tensor(tensor) => tensor.materialize_f64().clone(),
        Value::GpuTensor(handle) => test_support::gather(Value::GpuTensor(handle.clone()))
            .expect("gather gpu output")
            .materialize_f64(),
        other => panic!("expected numeric output, got {other:?}"),
    }
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() <= EPS, "index {idx}: expected {e}, got {a}");
    }
}

#[test]
fn descriptor_covers_polar_and_cylindrical_forms() {
    let labels: Vec<&str> = POL2CART_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"[x, y] = pol2cart(theta, rho)"));
    assert!(labels.contains(&"[x, y, z] = pol2cart(theta, rho, z)"));
    assert_eq!(
        POL2CART_DESCRIPTOR.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
}

#[test]
fn scalar_no_output_context_returns_x() {
    let result = call(
        Value::Num(std::f64::consts::FRAC_PI_2),
        Value::Num(2.0),
        Vec::new(),
    )
    .expect("pol2cart");
    let Value::Num(x) = result else {
        panic!("expected scalar x");
    };
    assert!(x.abs() <= EPS);
}

#[test]
fn polar_two_outputs_for_row_vectors() {
    let theta = tensor(
        vec![
            0.0,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
        ],
        vec![1, 4],
    );
    let rho = tensor(vec![5.0, 5.0, 10.0, 10.0], vec![1, 4]);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let outputs = output_list(call(Value::Tensor(theta), Value::Tensor(rho), Vec::new()).unwrap());
    assert_close(&data(&outputs[0]), &[5.0, 3.5355339059327378, 0.0, -10.0]);
    assert_close(&data(&outputs[1]), &[0.0, 3.5355339059327373, 10.0, 0.0]);
}

#[test]
fn cylindrical_expands_all_three_inputs() {
    let theta = tensor(vec![0.0, std::f64::consts::FRAC_PI_2], vec![2, 1]);
    let rho = tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let outputs = output_list(
        call(
            Value::Tensor(theta),
            Value::Tensor(rho),
            vec![Value::Num(7.0)],
        )
        .expect("pol2cart"),
    );
    assert_close(&data(&outputs[0]), &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    assert_close(&data(&outputs[1]), &[0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
    assert_close(&data(&outputs[2]), &[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);
}

#[test]
fn cylindrical_two_outputs_omits_z() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let outputs = output_list(
        call(Value::Num(0.0), Value::Num(3.0), vec![Value::Num(9.0)]).expect("pol2cart"),
    );
    assert_eq!(outputs.len(), 2);
    assert_close(&data(&outputs[0]), &[3.0]);
    assert_close(&data(&outputs[1]), &[0.0]);
}

#[test]
fn cylindrical_broadcasts_missing_trailing_dimension() {
    let theta = tensor(vec![0.0; 12], vec![3, 4]);
    let rho = tensor((1..=24).map(|value| value as f64).collect(), vec![3, 4, 2]);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let outputs = output_list(
        call(
            Value::Tensor(theta),
            Value::Tensor(rho),
            vec![Value::Num(5.0)],
        )
        .expect("trailing singleton expansion"),
    );
    match &outputs[0] {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), &[3, 4, 2]);
            assert_close(
                &out.materialize_f64(),
                &(1..=24).map(|value| value as f64).collect::<Vec<_>>(),
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    match &outputs[1] {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), &[3, 4, 2]);
            assert_close(&out.materialize_f64(), &[0.0; 24]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn rejects_trailing_aligned_non_matlab_shapes() {
    let theta = tensor(vec![0.0; 12], vec![3, 4]);
    let rho = tensor(vec![1.0; 12], vec![1, 3, 4]);
    let err = call(Value::Tensor(theta), Value::Tensor(rho), Vec::new())
        .expect_err("shape should not be MATLAB-compatible");
    assert_eq!(err.identifier(), ERROR_SIZE_MISMATCH.identifier);
}

#[test]
fn polar_rejects_three_outputs() {
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let err = call(Value::Num(0.0), Value::Num(1.0), Vec::new()).expect_err("too many outputs");
    assert_eq!(err.identifier(), ERROR_OUTPUT_COUNT.identifier);
}

#[test]
fn rejects_complex_inputs() {
    let err = call(Value::Complex(1.0, 1.0), Value::Num(1.0), Vec::new())
        .expect_err("complex should error");
    assert_eq!(err.identifier(), ERROR_COMPLEX_UNSUPPORTED.identifier);
}

#[test]
fn rejects_non_floating_inputs() {
    let err = call(Value::Bool(true), Value::Num(1.0), Vec::new()).expect_err("bool should error");
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn rejects_incompatible_shapes() {
    let theta = tensor(vec![0.0, 1.0], vec![2, 1]);
    let rho = tensor(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let err =
        call(Value::Tensor(theta), Value::Tensor(rho), Vec::new()).expect_err("size mismatch");
    assert_eq!(err.identifier(), ERROR_SIZE_MISMATCH.identifier);
}

#[test]
fn empty_inputs_preserve_broadcast_shape() {
    let theta = tensor(Vec::new(), vec![0, 3]);
    let rho = tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let outputs = output_list(call(Value::Tensor(theta), Value::Tensor(rho), Vec::new()).unwrap());
    match &outputs[0] {
        Value::Tensor(out) => assert_eq!(out.shape(), &[0, 3]),
        other => panic!("expected empty tensor, got {other:?}"),
    }
    match &outputs[1] {
        Value::Tensor(out) => assert_eq!(out.shape(), &[0, 3]),
        other => panic!("expected empty tensor, got {other:?}"),
    }
}

#[test]
fn typed_integer_host_inputs_read_exact_storage() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let theta = Tensor::new_integer(IntegerStorage::I16(vec![0, 0]), vec![1, 2]).expect("theta");
    let rho = Tensor::new_integer(IntegerStorage::U16(vec![2, 3]), vec![1, 2]).expect("rho");
    let z = Tensor::new_integer(IntegerStorage::U16(vec![7, 8]), vec![1, 2]).expect("z");

    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let outputs = output_list(
        call(
            Value::Tensor(theta),
            Value::Tensor(rho),
            vec![Value::Tensor(z)],
        )
        .expect("pol2cart"),
    );

    assert_close(&data(&outputs[0]), &[2.0, 3.0]);
    assert_close(&data(&outputs[1]), &[0.0, 0.0]);
    assert_close(&data(&outputs[2]), &[7.0, 8.0]);
}

#[test]
fn integer_roles_validate_exact_wide_values_and_compatibility_independently() {
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let exact = runmat_value::IntValue::U64(1_u64 << 63);
    let values = output_list(
        call(
            Value::Num(0.0),
            Value::Int(exact.clone()),
            vec![Value::Int(exact)],
        )
        .unwrap(),
    );
    assert_eq!(data(&values[0]), vec![(1_u64 << 63) as f64]);
    assert_eq!(data(&values[2]), vec![(1_u64 << 63) as f64]);
    for position in 0..3 {
        let mut args = [Value::Num(0.0), Value::Num(1.0), Value::Num(2.0)];
        args[position] = Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993));
        assert!(call(args[0].clone(), args[1].clone(), vec![args[2].clone()]).is_err());
        args[position] = Value::Int(runmat_value::IntValue::U8(1));
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call(args[0].clone(), args[1].clone(), vec![args[2].clone()]).unwrap_err();
        assert!(error
            .identifier()
            .unwrap()
            .starts_with("RunMat:compatibility:Pol2cartInteger"));
    }
}

#[test]
fn gpu_inputs_keep_resident_outputs() {
    test_support::with_test_provider(|provider| {
        let theta_view = HostTensorView {
            data: &[0.0, std::f64::consts::FRAC_PI_2],
            shape: &[1, 2],
        };
        let rho_view = HostTensorView {
            data: &[2.0, 3.0],
            shape: &[1, 2],
        };
        let theta = provider.upload(&theta_view).expect("upload theta");
        let rho = provider.upload(&rho_view).expect("upload rho");
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let outputs = output_list(
            call(Value::GpuTensor(theta), Value::GpuTensor(rho), Vec::new()).expect("pol2cart"),
        );
        assert!(matches!(outputs[0], Value::GpuTensor(_)));
        assert!(matches!(outputs[1], Value::GpuTensor(_)));
        assert_close(&data(&outputs[0]), &[2.0, 0.0]);
        assert_close(&data(&outputs[1]), &[0.0, 3.0]);
    });
}

#[test]
fn mixed_gpu_typed_integer_inputs_upload_exact_storage() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let theta_view = HostTensorView {
            data: &[0.0, 0.0],
            shape: &[1, 2],
        };
        let theta = provider.upload(&theta_view).expect("upload theta");

        let rho = Tensor::new_integer(IntegerStorage::U16(vec![2, 3]), vec![1, 2]).expect("rho");
        let z = Tensor::new_integer(IntegerStorage::U16(vec![7, 8]), vec![1, 2]).expect("z");

        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let outputs = output_list(
            call(
                Value::GpuTensor(theta),
                Value::Tensor(rho),
                vec![Value::Tensor(z)],
            )
            .expect("pol2cart"),
        );

        assert!(matches!(outputs[0], Value::GpuTensor(_)));
        assert!(matches!(outputs[1], Value::GpuTensor(_)));
        assert!(matches!(outputs[2], Value::GpuTensor(_)));
        assert_close(&data(&outputs[0]), &[2.0, 3.0]);
        assert_close(&data(&outputs[1]), &[0.0, 0.0]);
        assert_close(&data(&outputs[2]), &[7.0, 8.0]);
    });
}

#[test]
fn mixed_gpu_typed_integer_theta_upload_reads_exact_storage() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let rho_view = HostTensorView {
            data: &[2.0, 3.0],
            shape: &[1, 2],
        };
        let rho = provider.upload(&rho_view).expect("upload rho");

        let theta =
            Tensor::new_integer(IntegerStorage::I16(vec![0, 0]), vec![1, 2]).expect("theta");

        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let outputs = output_list(
            call(Value::Tensor(theta), Value::GpuTensor(rho), Vec::new()).expect("pol2cart"),
        );

        assert!(matches!(outputs[0], Value::GpuTensor(_)));
        assert!(matches!(outputs[1], Value::GpuTensor(_)));
        assert_close(&data(&outputs[0]), &[2.0, 3.0]);
        assert_close(&data(&outputs[1]), &[0.0, 0.0]);
    });
}

#[test]
fn gpu_mixed_cylindrical_broadcasts_z_output() {
    test_support::with_test_provider(|provider| {
        let theta_view = HostTensorView {
            data: &[0.0, std::f64::consts::FRAC_PI_2],
            shape: &[2, 1],
        };
        let theta = provider.upload(&theta_view).expect("upload theta");
        let rho = tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let outputs = output_list(
            call(
                Value::GpuTensor(theta),
                Value::Tensor(rho),
                vec![Value::Num(7.0)],
            )
            .expect("pol2cart"),
        );
        assert!(matches!(outputs[0], Value::GpuTensor(_)));
        assert!(matches!(outputs[1], Value::GpuTensor(_)));
        assert!(matches!(outputs[2], Value::GpuTensor(_)));
        assert_close(&data(&outputs[0]), &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
        assert_close(&data(&outputs[1]), &[0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
        assert_close(&data(&outputs[2]), &[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);
    });
}
