use super::*;
use crate::dispatcher::call_builtin;
use runmat_filesystem::{self as fs, FsProvider};
use runmat_value::StructValue;
use std::sync::Arc;

#[test]
fn path_and_manifest_failures_retain_typed_source_causes() {
    use std::error::Error;

    let _serial = fs::provider_override_lock();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
    let _provider = fs::replace_provider(provider.clone() as Arc<dyn FsProvider>);
    let missing = call_builtin("data.open", &[Value::String("missing.data".into())])
        .expect_err("missing manifest");
    assert_eq!(missing.identifier.as_deref(), Some("RUNMAT:Data:Error"));
    assert!(missing
        .source()
        .is_some_and(|cause| cause.downcast_ref::<std::io::Error>().is_some()));

    provider
        .write_project_path("/selected/bad.data/manifest.json", b"{not-json")
        .expect("malformed manifest fixture");
    let malformed = call_builtin("data.open", &[Value::String("bad.data".into())])
        .expect_err("malformed manifest");
    assert!(malformed
        .source()
        .is_some_and(|cause| cause.downcast_ref::<serde_json::Error>().is_some()));
}

#[path = "tests/receipt.rs"]
mod receipt;

fn empty_schema() -> Value {
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".into(), Value::Struct(StructValue::new()));
    Value::Struct(schema.finish())
}

#[test]
fn finite_uncertified_create_rejects_before_filesystem_effect() {
    use crate::context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
    };
    use std::rc::Rc;

    struct Finite;
    impl RuntimeParallelService for Finite {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            RuntimeParallelResources {
                memory_available_bytes: Some(1_024),
                ..Default::default()
            }
        }
    }

    let _serial = fs::provider_override_lock();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
    let _provider = fs::replace_provider(provider.clone() as Arc<dyn FsProvider>);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_parallel(Rc::new(Finite)));
    let error = futures::executor::block_on(data_create_builtin(
        Some(runtime),
        Value::String("not-created.data".into()),
        empty_schema(),
    ))
    .expect_err("finite quota requires certified whole-call peak");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert!(provider
        .list_project_path("/selected/not-created.data")
        .is_err());
}

#[test]
fn cancelled_create_does_not_publish_any_array_or_manifest() {
    use crate::context::RuntimeContext;
    use runmat_execution::cancellation::CancellationSource;
    use std::rc::Rc;

    let _serial = fs::provider_override_lock();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
    let _provider = fs::replace_provider(provider.clone() as Arc<dyn FsProvider>);
    let cancellation = CancellationSource::new();
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(cancellation.token());
    let error = futures::executor::block_on(data_create_builtin(
        Some(runtime),
        Value::String("cancelled.data".into()),
        empty_schema(),
    ))
    .expect_err("cancelled call must not write");
    assert_eq!(error.identifier.as_deref(), Some("RunMat:Cancelled"));
    assert!(provider
        .list_project_path("/selected/cancelled.data")
        .is_err());
}

#[test]
fn immediately_ready_recursive_copy_cooperates_before_next_entry() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::path::PathBuf;
    use std::rc::Rc;

    let provider = fs::MemoryFsProvider::with_current_dir("/data");
    provider
        .write_project_path("/data/source.data/a.bin", b"first")
        .unwrap();
    provider
        .write_project_path("/data/source.data/b.bin", b"second")
        .unwrap();
    let cancellation = Rc::new(CancellationSource::new());
    let cancel = cancellation.clone();
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let mut budget = Some(CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        cancellation.token(),
        move || {
            counted.set(counted.get() + 1);
            cancel.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    ));
    let error = futures::executor::block_on(copy_dir_recursive_on(
        &provider,
        &PathBuf::from("/data/source.data"),
        &PathBuf::from("/data/copied.data"),
        &mut budget,
    ))
    .expect_err("cancelled traversal must not continue through immediately-ready entries");
    assert_eq!(error.identifier.as_deref(), Some("RunMat:Cancelled"));
    assert_eq!(turns.get(), 1);
    assert!(provider
        .read_project_path("/data/copied.data/a.bin")
        .is_err());
    assert!(provider
        .read_project_path("/data/copied.data/b.bin")
        .is_err());
    assert_eq!(
        provider
            .read_project_path("/data/source.data/a.bin")
            .unwrap(),
        b"first"
    );
}

#[test]
fn later_invalid_schema_dtype_rejects_before_first_array_write() {
    let _serial = fs::provider_override_lock();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
    let _provider = fs::replace_provider(provider.clone() as Arc<dyn FsProvider>);

    let mut first = runmat_value::UnboundedStructValueBuilder::new();
    first
        .fields
        .insert("dtype".into(), Value::String("uint64".into()));
    first.fields.insert("shape".into(), Value::Num(1.0));
    let mut second = runmat_value::UnboundedStructValueBuilder::new();
    second
        .fields
        .insert("dtype".into(), Value::String("not-a-dtype".into()));
    second.fields.insert("shape".into(), Value::Num(1.0));
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays
        .fields
        .insert("first".into(), Value::Struct(first.finish()));
    arrays
        .fields
        .insert("second".into(), Value::Struct(second.finish()));
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".into(), Value::Struct(arrays.finish()));

    let error = call_builtin(
        "data.create",
        &[
            Value::String("late-bad.data".into()),
            Value::Struct(schema.finish()),
        ],
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
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".into(), Value::Struct(StructValue::new()));
    call_builtin(
        "data.create",
        &[
            Value::String("source.data".into()),
            Value::Struct(schema.finish()),
        ],
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

    let through_missing_parent = root
        .path()
        .join("missing")
        .join("..")
        .join("alias.data")
        .join("nested.data");
    let error = call_builtin(
        "data.copy",
        &[
            Value::String(source.to_str().expect("fixture path").into()),
            Value::String(
                through_missing_parent
                    .to_str()
                    .expect("fixture path")
                    .into(),
            ),
        ],
    )
    .expect_err("missing/../symlink alias must not evade preflight");
    assert!(error.to_string().contains("destination must not"));
    assert!(!source.join("nested.data").exists());
    assert!(!root.path().join("missing").exists());
}

#[cfg(unix)]
#[test]
fn directory_symlink_inside_source_is_not_recursively_descended() {
    let _serial = fs::provider_override_lock();
    let _provider = fs::replace_provider(Arc::new(fs::NativeFsProvider));
    let root = tempfile::tempdir().expect("fixture root");
    let source = root.path().join("source.data");
    std::fs::create_dir(&source).expect("source directory");
    std::os::unix::fs::symlink(&source, source.join("cycle"))
        .expect("directory symlink into source");
    let destination = root.path().join("copied.data");
    let error = call_builtin(
        "data.copy",
        &[
            Value::String(source.to_str().expect("fixture path").into()),
            Value::String(destination.to_str().expect("fixture path").into()),
        ],
    )
    .expect_err("directory symlink must not trigger recursive traversal");
    assert_eq!(error.identifier.as_deref(), Some("RUNMAT:Data:Error"));
    assert!(!destination.join("cycle").exists());
}
