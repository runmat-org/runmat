use super::*;
use runmat_filesystem::MemoryFsProvider;
use runmat_value::{CharArray, IntValue, Value};
use std::sync::Arc;

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(what_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(result) = value else {
        panic!("expected structure")
    };
    result.fields.get(name).expect("field")
}

fn names(value: &Value) -> Vec<String> {
    let Value::Cell(values) = value else {
        panic!("expected cell")
    };
    values
        .data()
        .iter()
        .map(|value| {
            let Value::CharArray(chars) = value else {
                panic!("expected character row")
            };
            runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 fixture")
        })
        .collect()
}

#[test]
fn classifies_and_sorts_direct_children_through_the_filesystem_provider() {
    let _lock = runmat_filesystem::provider_override_lock();
    let filesystem = MemoryFsProvider::with_current_dir("/workspace");
    filesystem
        .write_project_path("/workspace/project/zeta.m", b"")
        .unwrap();
    filesystem
        .write_project_path("/workspace/project/alpha.m", b"")
        .unwrap();
    filesystem
        .write_project_path("/workspace/project/state.mat", b"")
        .unwrap();
    filesystem
        .write_project_path("/workspace/project/native.mexa64", b"")
        .unwrap();
    filesystem
        .write_project_path("/workspace/project/not.mexnotes.txt", b"")
        .unwrap();
    filesystem
        .write_project_path("/workspace/project/native.mexa64.backup", b"")
        .unwrap();
    filesystem
        .create_dir_project_path("/workspace/project/@Vehicle", true)
        .unwrap();
    filesystem
        .create_dir_project_path("/workspace/project/+geometry", true)
        .unwrap();
    filesystem
        .write_project_path("/workspace/project/nested/hidden.m", b"")
        .unwrap();

    let result = runmat_filesystem::with_provider_override(Arc::new(filesystem), || {
        run(vec![Value::from("/workspace/project")]).expect("inventory")
    });
    assert_eq!(names(field(&result, "m")), vec!["alpha.m", "zeta.m"]);
    assert_eq!(names(field(&result, "mat")), vec!["state.mat"]);
    assert_eq!(names(field(&result, "mex")), vec!["native.mexa64"]);
    assert_eq!(names(field(&result, "classes")), vec!["Vehicle"]);
    assert_eq!(names(field(&result, "packages")), vec!["geometry"]);
}

#[test]
fn default_input_uses_the_provider_current_directory() {
    let _lock = runmat_filesystem::provider_override_lock();
    let filesystem = MemoryFsProvider::with_current_dir("/workspace");
    filesystem
        .write_project_path("/workspace/current.m", b"")
        .unwrap();
    let result = runmat_filesystem::with_provider_override(Arc::new(filesystem), || {
        run(Vec::new()).unwrap()
    });
    assert_eq!(names(field(&result, "m")), vec!["current.m"]);
}

#[test]
fn accepts_character_rows_and_rejects_numeric_inputs_without_gathering() {
    let missing =
        run(vec![Value::CharArray(CharArray::new_row("missing"))]).expect_err("missing folder");
    assert_eq!(
        missing.identifier(),
        runmat_builtins::WHAT_ERROR_FILESYSTEM.identifier
    );
    let numeric = run(vec![Value::Int(IntValue::U64(u64::MAX))]).expect_err("numeric folder");
    assert_eq!(
        numeric.identifier(),
        runmat_builtins::WHAT_ERROR_FOLDER.identifier
    );
}

#[test]
fn rejects_excess_arguments_with_the_catalog_error() {
    let error = run(vec![Value::from("one"), Value::from("two")]).expect_err("arity");
    assert_eq!(
        error.identifier(),
        runmat_builtins::WHAT_ERROR_ARITY.identifier
    );
}

#[test]
fn output_and_character_shape_admission_precede_enumeration() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = run(vec![Value::Num(1.0)]).unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::WHAT_ERROR_TOO_MANY_OUTPUTS.identifier
    );
    drop(_outputs);
    let pages = CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap();
    assert_eq!(
        run(vec![Value::CharArray(pages)]).unwrap_err().identifier(),
        runmat_builtins::WHAT_ERROR_FOLDER.identifier
    );
}

#[test]
fn empty_folder_has_empty_column_groups_even_without_outputs() {
    let _lock = runmat_filesystem::provider_override_lock();
    let filesystem = MemoryFsProvider::with_current_dir("/workspace");
    filesystem
        .create_dir_project_path("/workspace/empty", true)
        .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let result = runmat_filesystem::with_provider_override(Arc::new(filesystem), || {
        run(vec![Value::from("/workspace/empty")]).unwrap()
    });
    for name in ["m", "mat", "mex", "classes", "packages"] {
        let Value::Cell(values) = field(&result, name) else {
            panic!("cell group")
        };
        assert_eq!(values.shape, vec![0, 1]);
    }
}
