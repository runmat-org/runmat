#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::{IntegerStorage, Value};
use test_helpers::execute_source;

#[test]
fn compiled_datetime_metadata_and_nonzero_forms_cover_every_integer_class() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "serial = {constructor}(42); h = hour(serial); mi = minute(serial); mo = month(serial); a = {constructor}([0 1 2]); n = nnz(a); v = nonzeros(a); d = ndims(reshape(a, [1 3 1 1])); s = struct('Exact', {constructor}(2)); c = namedargs2cell(s); z = missing({constructor}(1), {constructor}(2));"
        );
        let values = execute_source(&source).expect("compiled metadata/nonzero semantics");
        assert!(
            values
                .iter()
                .filter(|value| matches!(value, Value::Num(number) if *number == 2.0))
                .count()
                >= 2,
            "{constructor} nnz and ndims outputs"
        );
        assert!(values.iter().any(|value| {
            matches!(value, Value::Tensor(tensor) if tensor.shape == vec![2, 1] && tensor.integer_storage().is_some_and(|storage| storage.exact_values().iter().map(|value| value.to_f64()).collect::<Vec<_>>() == vec![1.0, 2.0]))
        }), "{constructor} nonzeros output");
        assert!(values.iter().any(|value| {
            matches!(value, Value::Cell(cell) if cell.data.get(1).is_some_and(|value| matches!(value, Value::Int(integer) if integer.to_f64() == 2.0)))
        }), "{constructor} namedargs2cell payload");
        assert!(values.iter().any(|value| {
            matches!(value, Value::StringArray(array) if array.shape == vec![1, 2] && array.data.len() == 2)
        }), "{constructor} shaped missing output");
    }
}

#[test]
fn compiled_uint64_nonzeros_preserves_values_above_flintmax() {
    let values = execute_source(
        "base = uint64(9007199254740992); a = [uint64(0) base+uint64(1) intmax('uint64')]; v = nonzeros(a); n = nnz(a);",
    )
    .expect("compiled wide uint64 nonzeros");
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX])))
    }));
    assert!(values
        .iter()
        .any(|value| matches!(value, Value::Num(number) if *number == 2.0)));
}

#[test]
fn compiled_legacy_nargoutchk_integer_forms_return_text_and_structures() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "text = nargoutchk({constructor}(0), {constructor}(1), {constructor}(2)); info = nargoutchk({constructor}(0), {constructor}(1), {constructor}(2), 'struct');"
        );
        let values = execute_source(&source).expect("compiled legacy nargoutchk");
        assert!(values.iter().any(|value| {
            matches!(value, Value::CharArray(array) if array.data.iter().collect::<String>().contains("Too many"))
        }));
        assert!(values.iter().any(|value| {
            matches!(value, Value::Struct(structure) if structure.fields.contains_key("message") && structure.fields.contains_key("identifier"))
        }));
    }
}

#[test]
fn matlab_compatibility_mode_rejects_runmat_only_structural_forms() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "x = hour(uint16(42));",
            "RunMat:compatibility:HourTypedLegacySerialExtension",
        ),
        (
            "x = minute(uint16(42));",
            "RunMat:compatibility:MinuteTypedLegacySerialExtension",
        ),
        (
            "x = month(uint16(42));",
            "RunMat:compatibility:MonthTypedLegacySerialExtension",
        ),
        (
            "x = missing(uint16(2));",
            "RunMat:compatibility:MissingShapedArrayExtension",
        ),
        (
            "x = nnz(uint16([0 1]), uint16(2));",
            "RunMat:compatibility:NnzDimensionExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("RunMat-only form must reject");
        assert_eq!(error.identifier(), Some(identifier));
    }
}
