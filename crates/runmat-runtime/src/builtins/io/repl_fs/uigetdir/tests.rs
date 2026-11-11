use super::*;
use runmat_value::Tensor;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

fn call(args: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
    let _guard = crate::output_count::push_output_count(outputs).expect("output-count scope");
    futures::executor::block_on(uigetdir_builtin(
        crate::context::legacy::active(),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

fn text(value: &Value) -> String {
    match value {
        Value::CharArray(chars) => {
            runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 fixture")
        }
        other => panic!("expected char array, got {other:?}"),
    }
}

fn output_list(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique test output list")
            .try_into_unbounded_vec()
            .expect("ordinary unbounded test output list"),
        other => panic!("expected output list, got {other:?}"),
    }
}

#[derive(Clone)]
struct DialogProvider {
    selection: Option<DirectoryDialogSelection>,
    request: Arc<Mutex<Option<DirectoryDialogRequest>>>,
}

impl crate::context::RuntimeHostService for DialogProvider {
    fn console(&self, _: crate::console::ConsoleStream, _: crate::console::ConsoleText) {}

    fn warning(&self, _: crate::warning_store::RuntimeWarning) {}

    fn interact(
        &self,
        _: crate::context::HostInteraction,
    ) -> crate::context::RuntimeServiceFuture<
        Result<crate::interaction::InteractionResponse, RuntimeError>,
    > {
        panic!("dialog fixture must not request terminal interaction")
    }

    fn select_directory(
        &self,
        request: DirectoryDialogRequest,
    ) -> crate::context::RuntimeServiceFuture<Result<Option<DirectoryDialogSelection>, RuntimeError>>
    {
        *self.request.lock().unwrap() = Some(request);
        let selection = self.selection.clone();
        Box::pin(async move { Ok(selection) })
    }
}

fn rooted_tmp_path() -> PathBuf {
    #[cfg(windows)]
    {
        std::env::temp_dir()
    }
    #[cfg(not(windows))]
    {
        let mut path = PathBuf::from(std::path::MAIN_SEPARATOR.to_string());
        path.push("tmp");
        path
    }
}

fn rooted_tmp_path_text() -> String {
    rooted_tmp_path().to_string_lossy().into_owned()
}

fn with_dialog_provider(
    selection: Option<DirectoryDialogSelection>,
    body: impl FnOnce(Arc<Mutex<Option<DirectoryDialogRequest>>>),
) {
    let _lock = runmat_filesystem::provider_override_lock();
    let request = Arc::new(Mutex::new(None));
    let provider = Rc::new(DialogProvider {
        selection,
        request: request.clone(),
    });
    let context = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(crate::context::RuntimeServicePorts::default().with_host(provider));
    let _selected = context.enter();
    body(request);
}

#[test]
fn validates_arguments_and_output_count_before_dialog_or_provider_access() {
    with_dialog_provider(None, |request| {
        let resident = Value::GpuTensor(
            crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
                .handle,
        );
        let error = call(vec![resident], Some(1)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:uigetdir:InvalidArgument"));
        assert!(request.lock().unwrap().is_none());
        let error = call(vec![], Some(2)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:uigetdir:TooManyOutputs"));
        assert!(request.lock().unwrap().is_none());
        let pages = Value::CharArray(
            runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
                .unwrap(),
        );
        assert_eq!(
            call(vec![pages], Some(1)).unwrap_err().identifier(),
            Some("RunMat:uigetdir:InvalidArgument")
        );
        assert!(request.lock().unwrap().is_none());
        assert!(
            matches!(call(vec![], Some(0)).unwrap(), Value::OutputList(values) if values.is_empty())
        );
        assert!(request.lock().unwrap().is_some());
    });
}

#[test]
fn cancel_returns_zero() {
    with_dialog_provider(None, |_| {
        assert_eq!(call(vec![], None).expect("uigetdir"), Value::Num(0.0));
        let outputs = output_list(call(vec![], Some(1)).expect("uigetdir"));
        assert_eq!(outputs, vec![Value::Num(0.0)]);
    });
}

#[test]
fn parses_start_path_title_and_returns_selected_folder() {
    let selection = DirectoryDialogSelection {
        path: rooted_tmp_path(),
    };
    with_dialog_provider(Some(selection), |request| {
        let output = call(
            vec![
                Value::CharArray(CharArray::new_row(&rooted_tmp_path_text())),
                Value::CharArray(CharArray::new_row("Select input folder")),
            ],
            None,
        )
        .expect("uigetdir");
        assert_eq!(text(&output), rooted_tmp_path_text());

        let request = request.lock().unwrap().clone().expect("request");
        assert_eq!(request.title.as_deref(), Some("Select input folder"));
        assert_eq!(request.default_path, Some(rooted_tmp_path()));
    });
}

#[test]
fn accepts_backslash_separated_provider_path() {
    let selection = DirectoryDialogSelection {
        path: PathBuf::from(r"C:\data\images"),
    };
    with_dialog_provider(Some(selection), |_| {
        let output = call(vec![], None).expect("uigetdir");
        assert_eq!(text(&output), r"C:\data\images");
    });
}

#[test]
fn rejects_empty_provider_path() {
    let selection = DirectoryDialogSelection {
        path: PathBuf::new(),
    };
    with_dialog_provider(Some(selection), |_| {
        let err = call(vec![], None).expect_err("expected invalid selection");
        assert_eq!(err.identifier(), Some("RunMat:uigetdir:InvalidSelection"));
        assert!(err.message().contains("empty folder path"));
    });
}

#[test]
fn rejects_numeric_tensor_text_arguments() {
    with_dialog_provider(None, |_| {
        let tensor = Tensor::new(vec![42.0], vec![1, 1]).expect("tensor");
        let err =
            call(vec![Value::Tensor(tensor)], Some(1)).expect_err("expected invalid argument");
        assert_eq!(err.identifier(), Some("RunMat:uigetdir:InvalidArgument"));
    });
}

#[test]
fn rejects_too_many_inputs_and_outputs() {
    with_dialog_provider(None, |_| {
        let err = call(
            vec![
                Value::CharArray(CharArray::new_row("a")),
                Value::CharArray(CharArray::new_row("b")),
                Value::CharArray(CharArray::new_row("c")),
            ],
            None,
        )
        .expect_err("expected too many inputs");
        assert_eq!(err.identifier(), Some("RunMat:uigetdir:InvalidArgument"));

        let err = call(vec![], Some(2)).expect_err("expected too many outputs");
        assert_eq!(err.identifier(), Some("RunMat:uigetdir:TooManyOutputs"));
    });
}
