use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_builtins::{
    LOG2_ERROR_COMPLEX_DISSECTION, LOG2_ERROR_GPU_DISSECTION, LOG2_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_value::{NumericDType, Tensor};

#[test]
fn dissection_preserves_shape_class_and_special_values() {
    let values = vec![
        0.0,
        -0.0,
        1.0,
        std::f64::consts::PI,
        -3.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let input = Tensor::new(values, vec![2, 4]).expect("input");
    let Value::OutputList(outputs) =
        call_with_outputs(Value::Tensor(input), 2).expect("dissection")
    else {
        panic!("expected two outputs")
    };
    let [Value::Tensor(fraction), Value::Tensor(exponent)] = outputs.as_slice() else {
        panic!("expected tensor outputs")
    };
    assert_eq!(fraction.shape, vec![2, 4]);
    assert_eq!(exponent.shape, vec![2, 4]);
    assert_eq!(fraction.numeric_dtype(), NumericDType::F64);
    assert_eq!(fraction.materialize_f64()[..3], [0.0, -0.0, 0.5]);
    assert_eq!(exponent.materialize_f64()[..3], [0.0, 0.0, 1.0]);
    assert_eq!(fraction.materialize_f64()[5], f64::INFINITY);
    assert!(fraction.materialize_f64()[7].is_nan());
}

#[test]
fn dissection_preserves_single_class_and_table_identity() {
    let input = Tensor::from_f32(vec![1.0, -3.0], vec![1, 2]).expect("single input");
    let Value::OutputList(outputs) =
        call_with_outputs(Value::Tensor(input), 2).expect("dissection")
    else {
        panic!("expected outputs")
    };
    let [Value::Tensor(fraction), Value::Tensor(exponent)] = outputs.as_slice() else {
        panic!("expected tensors")
    };
    assert_eq!(fraction.as_f32_slice(), Some(&[0.5, -0.75][..]));
    assert_eq!(exponent.as_f32_slice(), Some(&[1.0, 2.0][..]));

    let table = crate::builtins::table::table_from_columns(
        vec!["X".into()],
        vec![Value::Tensor(
            Tensor::new(vec![1.0, -3.0], vec![2, 1]).unwrap(),
        )],
    )
    .expect("table");
    let Value::OutputList(outputs) = call_with_outputs(table, 2).expect("table dissection") else {
        panic!("expected table outputs")
    };
    assert!(outputs.iter().all(|output| matches!(output, Value::Object(object) if crate::builtins::table::is_tabular_object(object))));
}

#[test]
fn output_count_and_dissection_rejections_are_stable() {
    assert_eq!(
        call_with_outputs(Value::Num(8.0), 0).expect("zero outputs"),
        Value::OutputList(Vec::new())
    );
    assert_eq!(
        call_with_outputs(Value::Num(8.0), 3)
            .expect_err("too many outputs")
            .identifier(),
        LOG2_ERROR_TOO_MANY_OUTPUTS.identifier
    );
    assert_eq!(
        call_with_outputs(Value::Complex(2.0, 0.0), 2)
            .expect_err("complex dissection")
            .identifier(),
        LOG2_ERROR_COMPLEX_DISSECTION.identifier
    );

    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        assert_eq!(
            call_with_outputs(Value::GpuTensor(handle), 2)
                .expect_err("GPU dissection")
                .identifier(),
            LOG2_ERROR_GPU_DISSECTION.identifier
        );
    });
}
