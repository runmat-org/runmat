use runmat_value::{IntValue, NumericScalar, Value};
#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_object_restoration_and_field_reordering_preserve_wide_integers() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let values = execute_source(
        "w=uint64(9007199254740992)+uint64(1); s=struct(); s.z=w; s.a=int64(-7); [o,p]=orderfields(s,uint8([2 1])); b=loadobj(o); n=numel(uint32([1 2 3])); c=class(b.z);",
    )
    .expect("compiled persistence and structural introspection");

    assert!(values
        .iter()
        .any(|value| matches!(value, Value::Int(IntValue::U64(9_007_199_254_740_993)))));
    assert!(values.iter().any(|value| value == &Value::Num(3.0)));
    assert!(values
        .iter()
        .any(|value| value == &Value::String("uint64".to_string())));
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.numeric_value_at(0) == Some(NumericScalar::F64(2.0)) && tensor.numeric_value_at(1) == Some(NumericScalar::F64(1.0)))
    }));
}

#[test]
fn matlab_mode_rejects_plain_loadobj_and_numel_dimension_extensions() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "b=loadobj(uint64(7));",
            "RunMat:compatibility:LoadobjPlainPayloadPassthroughExtension",
        ),
        (
            "n=numel(uint16([1 2]),uint8(1));",
            "RunMat:compatibility:NumelDimensionSelectorsExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict compatibility gate");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_orderfields_uses_documented_ascii_name_order() {
    let values = execute_source("s=struct(); s.a=uint8(1); s.B=uint8(2); o=orderfields(s);")
        .expect("compiled ASCII field ordering");
    let ordered = values
        .iter()
        .find_map(|value| match value {
            Value::Struct(value)
                if value.fields.len() == 2
                    && value.fields.keys().next().is_some_and(|name| name == "B") =>
            {
                Some(value)
            }
            _ => None,
        })
        .expect("ordered structure");
    assert_eq!(
        ordered
            .fields
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["B", "a"]
    );
}
