#[path = "support/mod.rs"]
mod test_helpers;

use futures::executor::block_on;
use runmat_builtins::{IntegerStorage, NumericDType, Tensor, Value};
use test_helpers::execute_source;

fn unique_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "runmat_vm_matfile_{name}_{}.mat",
        std::process::id()
    ))
}

fn tensor(data: &[f64], shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor {
        data: data.to_vec(),
        integer_data: None,
        shape: shape.clone(),
        rows: *shape.first().unwrap_or(&1),
        cols: *shape.get(1).unwrap_or(&data.len()),
        dtype: NumericDType::F64,
    })
}

fn write_sample(path: &std::path::Path) {
    let bytes = block_on(
        runmat_runtime::builtins::io::mat::save::encode_workspace_to_mat_bytes(&[(
            "A".to_string(),
            tensor(&[1.0, 2.0, 3.0], vec![1, 3]),
        )]),
    )
    .expect("encode MAT");
    std::fs::write(path, bytes).expect("write MAT");
}

#[test]
fn matfile_dot_reads_properties_and_writes_whole_variables() {
    let path = unique_path("dot_round_trip");
    write_sample(&path);
    let source_path = path.to_string_lossy().replace('\'', "''");
    let input = format!(
        "\
        m = matfile('{source_path}'); \
        A = m.A; \
        props = m.Properties; \
        if A(2) ~= 2; error('matfile read mismatch'); end; \
        if ~strcmp(props.Source, '{source_path}'); error('matfile source mismatch'); end; \
        if props.Writable; error('matfile should default to read-only'); end; \
        mw = matfile('{source_path}', 'Writable', true); \
        mw.B = 42; \
        m2 = matfile('{source_path}'); \
        out = m2.B;"
    );
    let vars = execute_source(&input).expect("execute matfile source");
    assert!(vars.iter().any(|value| value == &Value::Num(42.0)));
    let _ = std::fs::remove_file(path);
}

#[test]
fn save_load_roundtrip_preserves_typed_complex_uint64_components() {
    let path = unique_path("typed_complex_uint64");
    let source_path = path.to_string_lossy().replace('\'', "''");
    let input = format!(
        "z = complex(uint64([9223372036854775808 18446744073709551615]), uint64([18446744073709551615 7])); save('{source_path}', 'z'); S = load('{source_path}'); loaded = S.z;"
    );

    let vars = execute_source(&input).expect("save/load typed complex uint64");
    assert!(vars.iter().any(|value| {
        matches!(
            value,
            Value::ComplexTensor(tensor)
                if tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                        &IntegerStorage::U64(vec![u64::MAX, 7]),
                    ))
        )
    }));
    let _ = std::fs::remove_file(path);
}

#[test]
fn save_load_script_surface_preserves_every_integer_class() {
    let path = unique_path("all_integer_classes");
    let source_path = path.to_string_lossy().replace('\'', "''");
    let input = format!(
        "\
        i8 = int8([-128 127]); \
        u8 = uint8([0 255]); \
        i16 = int16([-32768 32767]); \
        u16 = uint16([0 65535]); \
        i32 = int32([-2147483648 2147483647]); \
        u32 = uint32([0 4294967295]); \
        i64 = int64([-9223372036854775808 9223372036854775807]); \
        u64 = uint64([9223372036854775808 18446744073709551615]); \
        save('{source_path}', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64'); \
        S = load('{source_path}'); \
        li8 = S.i8; lu8 = S.u8; li16 = S.i16; lu16 = S.u16; \
        li32 = S.i32; lu32 = S.u32; li64 = S.i64; lu64 = S.u64; \
        ci8 = class(li8); cu8 = class(lu8); ci16 = class(li16); cu16 = class(lu16); \
        ci32 = class(li32); cu32 = class(lu32); ci64 = class(li64); cu64 = class(lu64);"
    );

    let vars = execute_source(&input).expect("save/load every integer class");
    let expected = [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::U8(vec![0, u8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::U16(vec![0, u16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
    ];
    for storage in expected {
        assert!(
            vars.iter().any(|value| matches!(
                value,
                Value::Tensor(tensor) if tensor.integer_storage() == Some(&storage)
            )),
            "expected loaded tensor storage {storage:?}; vars={vars:?}"
        );
    }
    for expected_class in [
        "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    ] {
        assert!(
            vars.iter().any(
                |value| matches!(value, Value::String(class_name) if class_name == expected_class)
            ),
            "expected loaded class {expected_class:?}; vars={vars:?}"
        );
    }
    let _ = std::fs::remove_file(path);
}
