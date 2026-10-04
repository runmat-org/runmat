use super::super::load::read_mat_file_for_builtin;
use super::*;
use runmat_filesystem::{self as fs, FsProvider};
use std::sync::Arc;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_invalid_writable_pair_does_not_download_resident_filename() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = Box::leak(Box::new(super::super::tests::CountingProvider::new()));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let resident = owner.resident();
    let error = futures::executor::block_on(matfile_builtin(
        Value::GpuTensor(resident),
        vec![Value::from("Other"), Value::Bool(true)],
    ))
    .expect_err("known invalid option");
    assert_eq!(error.identifier(), Some("RunMat:matfile:InvalidOption"));
    assert_eq!(owner.downloads(), 0);
}

async fn matfile_builtin(filename: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    super::matfile_builtin(None, filename, rest).await
}

async fn matfile_subsref(obj: Value, kind: String, payload: Value) -> BuiltinResult<Value> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::matfile_subsref(None, obj, subscript).await
}

async fn matfile_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::matfile_subsasgn(None, obj, subscript, rhs).await
}
use crate::builtins::io::mat::save::encode_workspace_to_mat_bytes;
use futures::executor::block_on;
use runmat_value::{CharArray, IntegerStorage, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_FILE_ID: AtomicU64 = AtomicU64::new(0);

fn unique_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "runmat_matfile_{name}_{}_{}.mat",
        std::process::id(),
        NEXT_TEMP_FILE_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn tensor(data: &[f64], shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data.to_vec(), shape).expect("test tensor"))
}

fn write_sample(path: &Path) {
    let bytes = block_on(encode_workspace_to_mat_bytes(&[
        ("A".to_string(), tensor(&[1.0, 2.0, 3.0], vec![1, 3])),
        ("flag".to_string(), Value::Bool(true)),
    ]))
    .expect("encode MAT");
    std::fs::write(path, bytes).expect("write MAT");
}

fn string_scalar(value: &str) -> Value {
    Value::StringArray(StringArray {
        data: vec![value.to_string()],
        shape: vec![1, 1],
        rows: 1,
        cols: 1,
    })
}

fn char_row(value: &str) -> Value {
    Value::CharArray(CharArray::new_row(value))
}

#[test]
fn matfile_constructs_properties_and_reads_whole_variables() {
    let path = unique_path("read");
    write_sample(&path);
    let object = block_on(matfile_builtin(
        Value::String(path.to_string_lossy().into_owned()),
        vec![],
    ))
    .expect("matfile");

    let props = block_on(matfile_subsref(
        object.clone(),
        ".".to_string(),
        Value::String("Properties".into()),
    ))
    .expect("properties");
    let Value::Struct(props) = props else {
        panic!("expected Properties struct");
    };
    assert_eq!(props.fields.get("Writable"), Some(&Value::Bool(false)));

    let value = block_on(matfile_subsref(
        object,
        ".".to_string(),
        Value::String("A".into()),
    ))
    .expect("A");
    assert_eq!(value, tensor(&[1.0, 2.0, 3.0], vec![1, 3]));
    let _ = std::fs::remove_file(path);
}

#[test]
fn matfile_accepts_text_forms_and_normalises_extension() {
    let path_without_ext =
        std::env::temp_dir().join(format!("runmat_matfile_text_{}", std::process::id()));
    let expected_path = path_without_ext.with_extension("mat");
    let _ = std::fs::remove_file(&expected_path);

    let object = block_on(matfile_builtin(
        string_scalar(&path_without_ext.to_string_lossy()),
        vec![char_row("Writable"), Value::Num(1.0)],
    ))
    .expect("string filename and char option");
    let Value::Object(object) = object else {
        panic!("expected matfile object");
    };
    assert_eq!(
        object
            .properties
            .get(PROPERTIES_FIELD)
            .and_then(|value| match value {
                Value::Struct(props) => props.fields.get(SOURCE_FIELD),
                _ => None,
            }),
        Some(&Value::String(expected_path.to_string_lossy().into_owned()))
    );
    assert_eq!(
        object
            .properties
            .get(PROPERTIES_FIELD)
            .and_then(|value| match value {
                Value::Struct(props) => props.fields.get(WRITABLE_FIELD),
                _ => None,
            }),
        Some(&Value::Bool(true))
    );
    assert!(object.matfile_authority().is_some());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn repeated_writable_pairs_are_prepared_once_and_last_value_wins() {
    let object = block_on(matfile_builtin(
        Value::String("repeated.mat".into()),
        vec![
            Value::String("Writable".into()),
            Value::Bool(true),
            Value::String("wRiTaBlE".into()),
            Value::Int(runmat_value::IntValue::I8(0)),
        ],
    ))
    .expect("repeated complete pairs");
    let Value::Object(object) = object else {
        panic!("MAT object");
    };
    let Some(Value::Struct(properties)) = object.properties.get(PROPERTIES_FIELD) else {
        panic!("Properties");
    };
    assert_eq!(
        properties.fields.get(WRITABLE_FIELD),
        Some(&Value::Bool(false))
    );
}

#[test]
fn matfile_writable_assignment_replaces_and_appends_variables() {
    let path = unique_path("write");
    write_sample(&path);
    let mut object = block_on(matfile_builtin(
        Value::String(path.to_string_lossy().into_owned()),
        vec![Value::String("Writable".into()), Value::Bool(true)],
    ))
    .expect("matfile writable");

    object = block_on(matfile_subsasgn(
        object,
        ".".to_string(),
        Value::String("A".into()),
        tensor(&[9.0, 8.0], vec![1, 2]),
    ))
    .expect("replace A");
    block_on(matfile_subsasgn(
        object,
        ".".to_string(),
        Value::String("B".into()),
        Value::Num(42.0),
    ))
    .expect("append B");

    let entries = block_on(read_mat_file_for_builtin(&path, "matfile")).expect("read back");
    let by_name: HashMap<_, _> = entries.into_iter().collect();
    assert_eq!(by_name.get("A"), Some(&tensor(&[9.0, 8.0], vec![1, 2])));
    assert_eq!(by_name.get("B"), Some(&Value::Num(42.0)));
    assert_eq!(by_name.get("flag"), Some(&Value::Bool(true)));
    let _ = std::fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn matfile_receiver_keeps_original_provider_and_cwd_after_swap() {
    let _lock = fs::provider_override_lock();
    let selected = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
    let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir("/replacement"));
    let selected_bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "A".into(),
        Value::Num(7.0),
    )]))
    .expect("selected MAT");
    let replacement_bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "A".into(),
        Value::Num(9.0),
    )]))
    .expect("replacement MAT");
    selected
        .write_project_path("/selected/fixture.mat", &selected_bytes)
        .expect("selected file");
    replacement
        .write_project_path("/replacement/fixture.mat", &replacement_bytes)
        .expect("replacement file");
    let _selected = fs::replace_provider(selected.clone());
    fs::set_current_dir("/selected").expect("selected cwd");
    let object = block_on(matfile_builtin(
        Value::String("fixture.mat".into()),
        vec![Value::String("Writable".into()), Value::Bool(true)],
    ))
    .expect("selected receiver");
    let Value::Object(ref presented) = object else {
        panic!("MAT object");
    };
    let Some(Value::Struct(properties)) = presented.properties.get(PROPERTIES_FIELD) else {
        panic!("MAT Properties");
    };
    assert_eq!(
        properties.fields.get(SOURCE_FIELD),
        Some(&Value::String("fixture.mat".into()))
    );
    let _replacement = fs::replace_provider(replacement.clone());
    fs::set_current_dir("/replacement").expect("replacement cwd");
    let value = block_on(matfile_subsref(
        object.clone(),
        ".".into(),
        Value::String("A".into()),
    ))
    .expect("read original owner");
    assert_eq!(value, Value::Num(7.0));
    block_on(matfile_subsasgn(
        object,
        ".".into(),
        Value::String("A".into()),
        Value::Num(11.0),
    ))
    .expect("write original owner");
    let original =
        block_on(selected.read(Path::new("/selected/fixture.mat"))).expect("read selected bytes");
    let competing = block_on(replacement.read(Path::new("/replacement/fixture.mat")))
        .expect("read replacement bytes");
    let original = crate::builtins::io::mat::load::decode_workspace_from_mat_bytes(&original)
        .expect("decode original");
    let competing = crate::builtins::io::mat::load::decode_workspace_from_mat_bytes(&competing)
        .expect("decode replacement");
    assert_eq!(original, vec![("A".into(), Value::Num(11.0))]);
    assert_eq!(competing, vec![("A".into(), Value::Num(9.0))]);
}

#[test]
fn matfile_writable_assignment_preserves_exact_integer_storage() {
    let path = unique_path("integer_write");
    write_sample(&path);
    let object = block_on(matfile_builtin(
        Value::String(path.to_string_lossy().into_owned()),
        vec![Value::String("Writable".into()), Value::Bool(true)],
    ))
    .expect("matfile writable");
    let tensor = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![1, 2])
        .expect("integer tensor");
    let input = Value::Tensor(tensor);

    let object = block_on(matfile_subsasgn(
        object,
        ".".to_string(),
        Value::String("limits".into()),
        input,
    ))
    .expect("write integer variable");
    let value = block_on(matfile_subsref(
        object,
        ".".to_string(),
        Value::String("limits".into()),
    ))
    .expect("read integer variable");

    assert!(matches!(value, Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))));
    let _ = std::fs::remove_file(path);
}

#[test]
fn matfile_rejects_readonly_assignment_and_partial_indexing() {
    let path = unique_path("readonly");
    write_sample(&path);
    let object = block_on(matfile_builtin(
        Value::String(path.to_string_lossy().into_owned()),
        vec![],
    ))
    .expect("matfile readonly");

    let err = block_on(matfile_subsasgn(
        object.clone(),
        ".".to_string(),
        Value::String("A".into()),
        Value::Num(1.0),
    ))
    .expect_err("readonly assignment should fail");
    assert!(err.message().contains("not writable"));

    let err = block_on(matfile_subsref(
        object,
        "()".to_string(),
        Value::Cell(runmat_value::CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
    ))
    .expect_err("partial indexing should fail");
    assert!(err.message().contains("whole-variable"));
    let _ = std::fs::remove_file(path);
}
