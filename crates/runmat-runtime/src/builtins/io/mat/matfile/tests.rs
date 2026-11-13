use super::super::load::read_mat_file_for_builtin;
use super::*;
use runmat_filesystem::{self as fs, FsProvider};
use runmat_value::StringArray;
use std::path::Path;
use std::sync::Arc;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelled_assignment_does_not_open_or_publish_mat_bytes() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::rc::Rc;

    let (directory, filename) = ("/selected", "/selected/cancelled.mat");
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir(directory));
    let _selected = fs::replace_provider(provider.clone());
    let object_owner = block_on(matfile_builtin(
        Value::from(filename),
        vec![Value::from("Writable"), Value::Bool(true)],
    ))
    .expect("writable receiver");
    let object = object_owner.value().clone();
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let context = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let subscript =
        crate::object::indexing::standard_substruct_fixture_from_parts(".", Value::from("answer"))
            .expect("member selector");
    let error = block_on(super::matfile_subsasgn(
        Some(context),
        object,
        subscript,
        Value::Num(42.0),
    ))
    .expect_err("cancel before metadata and write");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(
        block_on(provider.read(Path::new(filename)))
            .expect_err("no partial MAT publication")
            .kind(),
        std::io::ErrorKind::NotFound
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_invalid_writable_pair_does_not_download_resident_filename() {
    let _state = crate::builtins::io::mat::tests::state_guard();
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

fn finite_runtime() -> crate::context::RuntimeContext {
    struct Finite;
    impl crate::context::RuntimeParallelService for Finite {
        fn supports(&self, _: crate::context::ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
            crate::context::RuntimeParallelResources {
                memory_available_bytes: Some(1024),
                ..Default::default()
            }
        }
    }
    crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        crate::context::RuntimeServicePorts::default().with_parallel(std::rc::Rc::new(Finite)),
    )
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn finite_matfile_constructor_rejects_before_resident_writable_download() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let owner = Box::leak(Box::new(super::super::tests::CountingProvider::new()));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let error = futures::executor::block_on(super::matfile_builtin(
        Some(finite_runtime()),
        Value::from("blocked.mat"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Writable"),
            Value::GpuTensor(owner.resident()),
        ]),
    ))
    .expect_err("finite constructor has no complete resource producer");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(owner.downloads(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn finite_nested_matfile_assignment_rejects_before_read_or_rhs_gather() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let (directory, filename) = ("/mat", "/mat/missing.mat");
    let selected = Arc::new(fs::MemoryFsProvider::with_current_dir(directory));
    let _provider = fs::replace_provider(selected.clone());
    fs::set_current_dir(directory).expect("virtual cwd");
    let object_owner = futures::executor::block_on(matfile_builtin(
        Value::from("missing.mat"),
        vec![Value::from("Writable"), Value::Bool(true)],
    ))
    .expect("context-free writable handle");
    let object = object_owner.value().clone();
    let owner = Box::leak(Box::new(super::super::tests::CountingProvider::new()));
    let _gpu = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let subscript = crate::object::indexing::ObjectSubscriptPath::new(vec![
        crate::object::indexing::ObjectSubscript::member("A"),
        crate::object::indexing::ObjectSubscript::parentheses(
            crate::object::indexing::ObjectIndexSelector::IndexValues {
                components: runmat_value::ExactArrayStorage::shared_unbounded(vec![Value::Num(
                    1.0,
                )
                .into()]),
            },
        ),
    ])
    .expect("nested path")
    .to_standard_substruct_value()
    .expect("protocol value");
    let error = futures::executor::block_on(super::matfile_subsasgn(
        Some(finite_runtime()),
        object,
        subscript,
        Value::GpuTensor(owner.resident()),
    ))
    .expect_err("finite path must reject before first MAT read");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(owner.downloads(), 0);
    let missing =
        futures::executor::block_on(selected.read(Path::new(filename))).expect_err("no write");
    assert_eq!(missing.kind(), std::io::ErrorKind::NotFound);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn finite_matfile_member_read_rejects_before_missing_file_is_opened() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let directory = "/mat";
    let selected = Arc::new(fs::MemoryFsProvider::with_current_dir(directory));
    let _provider = fs::replace_provider(selected);
    fs::set_current_dir(directory).expect("virtual cwd");
    let object_owner =
        futures::executor::block_on(matfile_builtin(Value::from("missing.mat"), vec![]))
            .expect("context-free receiver");
    let object = object_owner.value().clone();
    let subscript =
        crate::object::indexing::standard_substruct_fixture_from_parts(".", Value::from("A"))
            .expect("member selector");
    let error = futures::executor::block_on(super::matfile_subsref(
        Some(finite_runtime()),
        object,
        subscript,
    ))
    .expect_err("finite member read needs complete resource evidence before open");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selected_outer_member_read_reports_open_failure_once() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    use super::super::tests::receipt::Witness;
    use runmat_execution::WholeWorkObservationOutcome;

    let directory = "/mat";
    let selected = Arc::new(fs::MemoryFsProvider::with_current_dir(directory));
    let _provider = fs::replace_provider(selected.clone());
    fs::set_current_dir(directory).expect("virtual cwd");
    let object_owner = block_on(matfile_builtin(Value::from("missing.mat"), vec![]))
        .expect("context-free MAT receiver");
    let object = object_owner.value().clone();
    let witness = Witness::new(selected);
    let subscript =
        crate::object::indexing::standard_substruct_fixture_from_parts(".", Value::from("A"))
            .expect("member selector");
    let error = block_on(super::matfile_subsref(
        Some(witness.runtime.clone()),
        object,
        subscript,
    ))
    .expect_err("missing source must preserve open failure");
    assert_eq!(error.identifier(), Some("RunMat:load:Io"));
    assert!(error.source.is_some(), "retain MAT open cause chain");
    assert_eq!(witness.selections(), 1);
    assert_eq!(witness.probes(), 0);
    assert_eq!(
        witness.outcomes(),
        vec![WholeWorkObservationOutcome::OperationalFailure]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selected_outer_assignment_pending_drop_releases_without_publication_or_feedback() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    use super::super::tests::receipt::{PendingWriteFs, Witness};

    let directory = "/mat";
    let selected = Arc::new(PendingWriteFs::new(directory));
    let _provider = fs::replace_provider(selected.clone());
    fs::set_current_dir(directory).expect("virtual cwd");
    let object_owner = block_on(matfile_builtin(
        Value::from("pending.mat"),
        vec![Value::from("Writable"), Value::Bool(true)],
    ))
    .expect("context-free writable receiver");
    let object = object_owner.value().clone();
    let witness = Witness::new(selected.clone());
    let subscript =
        crate::object::indexing::standard_substruct_fixture_from_parts(".", Value::from("A"))
            .expect("member selector");
    let mut future = Box::pin(super::matfile_subsasgn(
        Some(witness.runtime.clone()),
        object,
        subscript,
        Value::Num(42.0),
    ));
    assert!(matches!(
        super::super::tests::receipt::poll_once(future.as_mut()),
        std::task::Poll::Pending
    ));
    assert_eq!(selected.writes(), 1);
    assert_eq!(witness.selections(), 1);
    assert_eq!(witness.probes(), 1);
    drop(future);
    assert_eq!(selected.abandoned(), 1);
    assert_eq!(witness.probes(), 0);
    assert!(witness.outcomes().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_read_index_precedes_detached_receiver_authority() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(
        "()",
        Value::Cell(
            runmat_value::CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("selector cell"),
        ),
    )
    .expect("fixture");
    let path =
        crate::object::indexing::parse_standard_substruct(&subscript).expect("parsed fixture");
    let step = path
        .into_steps()
        .into_iter()
        .next()
        .expect("one fixture step");
    let detached = Value::Object(ObjectInstance::new(MATFILE_CLASS.owned()));
    let error = futures::executor::block_on(super::matfile_read_step(None, detached, step, None))
        .expect_err("partial indexing is unsupported before authority lookup");
    assert_eq!(error.identifier(), Some("RunMat:matfile:Unsupported"));
}

async fn matfile_builtin(
    filename: Value,
    rest: Vec<Value>,
) -> BuiltinResult<crate::execution::RootedResult<Value>> {
    super::matfile_builtin(
        None,
        filename,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

async fn matfile_subsref(
    obj: Value,
    kind: String,
    payload: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::matfile_subsref(None, obj, subscript).await
}

async fn matfile_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::matfile_subsasgn(None, obj, subscript, rhs).await
}
use crate::builtins::io::mat::encoding::encode_workspace_to_mat_bytes;
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
    Value::StringArray(
        StringArray::new(vec![value.into()], vec![1, 1]).expect("string scalar fixture"),
    )
}

fn char_row(value: &str) -> Value {
    Value::CharArray(CharArray::new_row(value))
}

#[test]
fn matfile_constructs_properties_and_reads_whole_variables() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let path = unique_path("read");
    write_sample(&path);
    let object_owner = block_on(matfile_builtin(
        Value::String(path.to_string_lossy().into_owned().into()),
        vec![],
    ))
    .expect("matfile");
    let object = object_owner.value().clone();

    let props = block_on(matfile_subsref(
        object.clone(),
        ".".to_string(),
        Value::String("Properties".into()),
    ))
    .expect("properties");
    let Value::Struct(props) = props.value() else {
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
    let _state = crate::builtins::io::mat::tests::state_guard();
    let path_without_ext =
        std::env::temp_dir().join(format!("runmat_matfile_text_{}", std::process::id()));
    let expected_path = path_without_ext.with_extension("mat");
    let _ = std::fs::remove_file(&expected_path);

    let object_owner = block_on(matfile_builtin(
        string_scalar(&path_without_ext.to_string_lossy()),
        vec![char_row("Writable"), Value::Num(1.0)],
    ))
    .expect("string filename and char option");
    let object = object_owner.value().clone();
    let Value::Object(object) = object else {
        panic!("expected matfile object");
    };
    assert_eq!(
        object
            .properties()
            .get(PROPERTIES_FIELD)
            .and_then(|value| match value {
                Value::Struct(props) => props.fields.get(SOURCE_FIELD),
                _ => None,
            }),
        Some(&Value::String(
            expected_path.to_string_lossy().into_owned().into()
        ))
    );
    assert_eq!(
        object
            .properties()
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
    let _state = crate::builtins::io::mat::tests::state_guard();
    let object_owner = block_on(matfile_builtin(
        Value::String("repeated.mat".into()),
        vec![
            Value::String("Writable".into()),
            Value::Bool(true),
            Value::String("wRiTaBlE".into()),
            Value::Int(runmat_value::IntValue::I8(0)),
        ],
    ))
    .expect("repeated complete pairs");
    let object = object_owner.value().clone();
    let Value::Object(object) = object else {
        panic!("MAT object");
    };
    let Some(Value::Struct(properties)) = object.properties().get(PROPERTIES_FIELD) else {
        panic!("Properties");
    };
    assert_eq!(
        properties.fields.get(WRITABLE_FIELD),
        Some(&Value::Bool(false))
    );
}

#[test]
fn matfile_writable_assignment_replaces_and_appends_variables() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let path = unique_path("write");
    write_sample(&path);
    let object_owner = block_on(matfile_builtin(
        Value::String(path.to_string_lossy().into_owned().into()),
        vec![Value::String("Writable".into()), Value::Bool(true)],
    ))
    .expect("matfile writable");
    let object = object_owner.value().clone();

    let object = block_on(matfile_subsasgn(
        object,
        ".".to_string(),
        Value::String("A".into()),
        tensor(&[9.0, 8.0], vec![1, 2]),
    ))
    .expect("replace A");
    object
        .with_owned(|object| {
            block_on(matfile_subsasgn(
                object,
                ".".to_string(),
                Value::String("B".into()),
                Value::Num(42.0),
            ))
        })
        .expect("append B");

    let entries = block_on(read_mat_file_for_builtin(&path, "matfile")).expect("read back");
    let by_name: HashMap<_, _> = entries.value().0.iter().cloned().collect();
    assert_eq!(by_name.get("A"), Some(&tensor(&[9.0, 8.0], vec![1, 2])));
    assert_eq!(by_name.get("B"), Some(&Value::Num(42.0)));
    assert_eq!(by_name.get("flag"), Some(&Value::Bool(true)));
    let _ = std::fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn matfile_receiver_keeps_original_provider_and_cwd_after_swap() {
    let _state = crate::builtins::io::mat::tests::state_guard();
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
    let object_owner = block_on(matfile_builtin(
        Value::String("fixture.mat".into()),
        vec![Value::String("Writable".into()), Value::Bool(true)],
    ))
    .expect("selected receiver");
    let object = object_owner.value().clone();
    let Value::Object(ref presented) = object else {
        panic!("MAT object");
    };
    let Some(Value::Struct(properties)) = presented.properties().get(PROPERTIES_FIELD) else {
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
    assert_eq!(original.value().0, vec![("A".into(), Value::Num(11.0))]);
    assert_eq!(competing.value().0, vec![("A".into(), Value::Num(9.0))]);
}

#[test]
fn matfile_writable_assignment_preserves_exact_integer_storage() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let path = unique_path("integer_write");
    write_sample(&path);
    let object_owner = block_on(matfile_builtin(
        Value::String(path.to_string_lossy().into_owned().into()),
        vec![Value::String("Writable".into()), Value::Bool(true)],
    ))
    .expect("matfile writable");
    let object = object_owner.value().clone();
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
    let value = object
        .with_owned(|object| {
            block_on(matfile_subsref(
                object,
                ".".to_string(),
                Value::String("limits".into()),
            ))
        })
        .expect("read integer variable");

    assert!(matches!(value.value(), Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))));
    let _ = std::fs::remove_file(path);
}

#[test]
fn matfile_rejects_readonly_assignment_and_partial_indexing() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let path = unique_path("readonly");
    write_sample(&path);
    let object_owner = block_on(matfile_builtin(
        Value::String(path.to_string_lossy().into_owned().into()),
        vec![],
    ))
    .expect("matfile readonly");
    let object = object_owner.value().clone();

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
