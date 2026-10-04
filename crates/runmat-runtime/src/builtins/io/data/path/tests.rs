use super::*;
use crate::dispatcher::call_builtin;
use runmat_filesystem::{self as fs, FsProvider};
use std::sync::Arc;

#[test]
fn later_invalid_schema_dtype_rejects_before_first_array_write() {
    let _serial = fs::provider_override_lock();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
    let _provider = fs::replace_provider(provider.clone() as Arc<dyn FsProvider>);

    let mut first = StructValue::new();
    first
        .fields
        .insert("dtype".into(), Value::String("uint64".into()));
    first.fields.insert("shape".into(), Value::Num(1.0));
    let mut second = StructValue::new();
    second
        .fields
        .insert("dtype".into(), Value::String("not-a-dtype".into()));
    second.fields.insert("shape".into(), Value::Num(1.0));
    let mut arrays = StructValue::new();
    arrays.fields.insert("first".into(), Value::Struct(first));
    arrays.fields.insert("second".into(), Value::Struct(second));
    let mut schema = StructValue::new();
    schema.fields.insert("arrays".into(), Value::Struct(arrays));

    let error = call_builtin(
        "data.create",
        &[Value::String("late-bad.data".into()), Value::Struct(schema)],
    )
    .expect_err("late invalid dtype must reject");
    assert_eq!(error.identifier.as_deref(), Some("RUNMAT:Data:Error"));
    assert!(error.to_string().contains("unsupported"));
    assert!(
        provider
            .list_project_path("/selected/late-bad.data")
            .is_err(),
        "even the first array directory must not be created"
    );
}

#[test]
fn recursive_tree_destinations_reject_before_creating_any_child() {
    let _serial = fs::provider_override_lock();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
    let _provider = fs::replace_provider(provider.clone() as Arc<dyn FsProvider>);
    let mut schema = StructValue::new();
    schema
        .fields
        .insert("arrays".into(), Value::Struct(StructValue::new()));
    call_builtin(
        "data.create",
        &[Value::String("source.data".into()), Value::Struct(schema)],
    )
    .expect("source fixture");

    for destination in [
        "source.data",
        "source.data/nested.data",
        "source.data/new/../nested.data",
    ] {
        for (operation, arguments) in [
            (
                "data.copy",
                vec![
                    Value::String("source.data".into()),
                    Value::String(destination.into()),
                ],
            ),
            (
                "data.move",
                vec![
                    Value::String("source.data".into()),
                    Value::String(destination.into()),
                ],
            ),
            (
                "data.import",
                vec![
                    Value::String(destination.into()),
                    Value::String("data".into()),
                    Value::String("source.data".into()),
                ],
            ),
            (
                "data.export",
                vec![
                    Value::String("source.data".into()),
                    Value::String("data".into()),
                    Value::String(destination.into()),
                ],
            ),
        ] {
            let error = call_builtin(operation, &arguments).expect_err("overlap must reject");
            assert!(
                error.to_string().contains("destination must not"),
                "{operation}: {error}"
            );
            assert_eq!(error.identifier.as_deref(), Some("RUNMAT:Data:Error"));
        }
    }
    assert!(provider
        .read_project_path("/selected/source.data/manifest.json")
        .is_ok());
    assert!(provider
        .list_project_path("/selected/source.data/nested.data")
        .is_err());
    assert!(provider
        .list_project_path("/selected/source.data/new")
        .is_err());
}

#[cfg(unix)]
#[test]
fn symlink_alias_of_source_rejects_before_copy_effect() {
    let _serial = fs::provider_override_lock();
    let _provider = fs::replace_provider(Arc::new(fs::NativeFsProvider));
    let root = tempfile::tempdir().expect("fixture root");
    let source = root.path().join("source.data");
    std::fs::create_dir(&source).expect("source directory");
    std::fs::write(source.join("manifest.json"), b"source").expect("source manifest");
    let alias = root.path().join("alias.data");
    std::os::unix::fs::symlink(&source, &alias).expect("source alias");
    let destination = alias.join("nested.data");
    let error = call_builtin(
        "data.copy",
        &[
            Value::String(source.to_str().expect("fixture path").into()),
            Value::String(destination.to_str().expect("fixture path").into()),
        ],
    )
    .expect_err("symlink alias must reject");
    assert!(error.to_string().contains("destination must not"));
    assert!(!source.join("nested.data").exists());
    assert_eq!(
        std::fs::read(source.join("manifest.json")).unwrap(),
        b"source"
    );
}
