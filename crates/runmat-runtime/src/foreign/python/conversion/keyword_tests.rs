use super::*;
use runmat_value::{CellArray, ObjectInstance};

#[test]
fn manually_constructed_keyword_bundle_cannot_coerce_nontext_names() {
    for name in [
        Value::Num(1.0),
        Value::Int(runmat_value::IntValue::I64(1)),
        Value::Bool(true),
        Value::String(String::new().into()),
    ] {
        let mut bundle = ObjectInstance::new(runmat_types::standard::PYTHON_ARGUMENTS);
        bundle.replace_property_unbounded(
            "Names".into(),
            Value::Cell(CellArray::new(vec![name], 1, 1).unwrap()),
        );
        bundle.replace_property_unbounded(
            "Values".into(),
            Value::Cell(CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap()),
        );
        let error = keyword_bundle(bundle).unwrap_err();
        assert!(error.message().contains("pyargs names must be"));
    }
}

#[test]
fn keyword_bundle_accepts_effective_character_row_without_coercing_values() {
    let mut bundle = ObjectInstance::new(runmat_types::standard::PYTHON_ARGUMENTS);
    let row =
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['x'], vec![1, 1, 1]).unwrap();
    bundle.replace_property_unbounded(
        "Names".into(),
        Value::Cell(CellArray::new(vec![Value::CharArray(row)], 1, 1).unwrap()),
    );
    bundle.replace_property_unbounded(
        "Values".into(),
        Value::Cell(CellArray::new(vec![Value::Bool(true)], 1, 1).unwrap()),
    );
    let converted = keyword_bundle(bundle).expect("strict text name");
    let PythonValue::Keywords(values) = converted else {
        panic!("keywords payload");
    };
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].0, "x");
    assert!(matches!(&values[0].1, PythonValue::Bool(true)));
}
