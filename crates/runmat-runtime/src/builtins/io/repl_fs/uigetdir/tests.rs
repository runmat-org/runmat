use super::*;
use async_trait::async_trait;
use runmat_filesystem::HostDialogProvider;
use runmat_value::Tensor;
use std::io;
use std::sync::{Arc, Mutex};

fn call(args: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
    let _guard = crate::output_count::push_output_count(outputs);
    futures::executor::block_on(uigetdir_builtin(args))
}

fn text(value: &Value) -> String {
    match value {
        Value::CharArray(chars) => chars.data.iter().collect(),
        other => panic!("expected char array, got {other:?}"),
    }
}

fn output_list(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values,
        other => panic!("expected output list, got {other:?}"),
    }
}

#[derive(Clone)]
struct DialogProvider {
    selection: Option<DirectoryDialogSelection>,
    request: Arc<Mutex<Option<DirectoryDialogRequest>>>,
}

#[async_trait(?Send)]
impl HostDialogProvider for DialogProvider {
    async fn select_directory(
        &self,
        request: &DirectoryDialogRequest,
    ) -> io::Result<Option<DirectoryDialogSelection>> {
        *self.request.lock().unwrap() = Some(request.clone());
        Ok(self.selection.clone())
    }
}

fn rooted_tmp_path() -> PathBuf {
    let mut path = PathBuf::from(std::path::MAIN_SEPARATOR.to_string());
    path.push("tmp");
    path
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
    let provider = Arc::new(DialogProvider {
        selection,
        request: request.clone(),
    });
    let _guard = runmat_filesystem::replace_host_dialog_provider(Some(provider));
    body(request);
}

#[test]
fn validates_arguments_and_output_count_before_dialog_or_provider_access() {
    with_dialog_provider(None, |request| {
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
            descriptor: Default::default(),
        });
        let error = call(vec![resident], Some(1)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:uigetdir:InvalidArgument"));
        assert!(request.lock().unwrap().is_none());
        let error = call(vec![], Some(2)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:uigetdir:TooManyOutputs"));
        assert!(request.lock().unwrap().is_none());
        let pages = Value::CharArray(
            runmat_value::CharArray::new_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap(),
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
