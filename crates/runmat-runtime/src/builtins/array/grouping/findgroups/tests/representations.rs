use super::*;
use runmat_value::{ComplexTensor, IntegerStorage, StringArray, Tensor};

#[test]
fn sorts_text_and_preserves_orientation() {
    let input = Value::StringArray(
        StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![1, 3]).unwrap(),
    );
    let _outputs = crate::output_count::push_output_count(Some(2));
    let outputs = output_list(call(input, Vec::new()).unwrap());
    let Value::Tensor(groups) = &outputs[0] else {
        panic!("expected group numbers");
    };
    assert_eq!(groups.shape, vec![1, 3]);
    assert_eq!(groups.materialize_f64(), vec![2.0, 1.0, 2.0]);
    assert!(matches!(&outputs[1], Value::StringArray(ids) if ids.data == ["a", "b"]));
}

#[test]
fn preserves_exact_integer_identifiers() {
    let base = 1_u64 << 53;
    let input = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![base + 1, base, base + 1]),
            vec![3, 1],
        )
        .unwrap(),
    );
    let _outputs = crate::output_count::push_output_count(Some(2));
    let outputs = output_list(call(input, Vec::new()).unwrap());
    assert!(
        matches!(&outputs[1], Value::Tensor(ids) if ids.integer_storage() == Some(&IntegerStorage::U64(vec![base, base + 1])))
    );
}

#[test]
fn missing_values_do_not_form_groups() {
    let input = Value::Tensor(Tensor::new(vec![3.0, f64::NAN, 1.0, 3.0], vec![4, 1]).unwrap());
    let Value::Tensor(groups) = call(input, Vec::new()).unwrap() else {
        panic!("expected group numbers");
    };
    assert_eq!(groups.materialize_f64()[0], 2.0);
    assert!(groups.materialize_f64()[1].is_nan());
    assert_eq!(groups.materialize_f64()[2..], [1.0, 2.0]);
}

#[test]
fn complex_grouping_values_reject() {
    let complex =
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 1.0), (2.0, 0.0)], vec![2, 1]).unwrap());
    let error = call(complex, Vec::new()).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:findgroups:InvalidInput"));
}
