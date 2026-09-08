use super::*;

#[test]
fn returns_permutation_as_second_output() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let input = structure(&[("z", 1.0.into()), ("x", 2.0.into()), ("y", 3.0.into())]);
    let Value::OutputList(outputs) = call(Value::Struct(input), Vec::new()).unwrap() else {
        panic!("expected output list")
    };
    assert_eq!(outputs.len(), 2);
    let Value::Tensor(permutation) = &outputs[1] else {
        panic!("expected permutation tensor")
    };
    assert_eq!(permutation.shape, [3, 1]);
    assert_eq!(permutation.materialize_f64(), [2.0, 3.0, 1.0]);
}

#[test]
fn zero_outputs_returns_empty_output_list() {
    let _outputs = crate::output_count::push_output_count(Some(0));
    let input = structure(&[("a", 1.0.into())]);
    assert_eq!(
        call(Value::Struct(input), Vec::new()).unwrap(),
        Value::OutputList(Vec::new())
    );
}

#[test]
fn one_field_permutation_uses_the_double_scalar_carrier() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let input = structure(&[("only", 1.0.into())]);
    let Value::OutputList(outputs) = call(Value::Struct(input), Vec::new()).unwrap() else {
        panic!("expected output list")
    };
    assert_eq!(outputs[1], Value::Num(1.0));
}
