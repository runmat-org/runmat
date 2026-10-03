use super::*;
use runmat_value::{CellArray, ObjectInstance};

#[test]
fn manually_constructed_keyword_bundle_cannot_coerce_nontext_names() {
    for name in [
        Value::Num(1.0),
        Value::Int(runmat_value::IntValue::I64(1)),
        Value::Bool(true),
        Value::String(String::new()),
    ] {
        let mut bundle = ObjectInstance::new(runmat_types::standard::PYTHON_ARGUMENTS);
        bundle.properties.insert(
            "Names".into(),
            Value::Cell(CellArray::new(vec![name], 1, 1).unwrap()),
        );
        bundle.properties.insert(
            "Values".into(),
            Value::Cell(CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap()),
        );
        let error = keyword_bundle(bundle).unwrap_err();
        assert!(error.message().contains("pyargs names must be"));
    }
}
