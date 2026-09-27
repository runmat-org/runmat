use super::*;
use async_trait::async_trait;
use runmat_filesystem::HostDialogProvider;
use runmat_value::{CharArray, Tensor};
use std::io;
use std::sync::{Arc, Mutex};

fn call(args: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
    let _guard = crate::output_count::push_output_count(outputs);
    futures::executor::block_on(uiputfile_builtin(args))
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
    selection: Option<SaveFileDialogSelection>,
    request: Arc<Mutex<Option<SaveFileDialogRequest>>>,
}

#[async_trait(?Send)]
impl HostDialogProvider for DialogProvider {
    async fn select_file_save(
        &self,
        request: &SaveFileDialogRequest,
    ) -> io::Result<Option<SaveFileDialogSelection>> {
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

fn rooted_tmp_dir_text() -> String {
    let mut text = rooted_tmp_path_text();
    if !text.ends_with(std::path::MAIN_SEPARATOR) {
        text.push(std::path::MAIN_SEPARATOR);
    }
    text
}

fn rooted_tmp_file(file_name: &str) -> PathBuf {
    let mut path = rooted_tmp_path();
    path.push(file_name);
    path
}

fn with_dialog_provider(
    selection: Option<SaveFileDialogSelection>,
    body: impl FnOnce(Arc<Mutex<Option<SaveFileDialogRequest>>>),
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
        let resident = Value::GpuTensor(
            crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
                .handle,
        );
        let error = call(vec![resident], Some(1)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:uiputfile:InvalidArgument"));
        assert!(request.lock().unwrap().is_none());
        let error = call(vec![], Some(4)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:uiputfile:TooManyOutputs"));
        assert!(request.lock().unwrap().is_none());
        let pages = Value::CharArray(
            runmat_value::CharArray::new_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap(),
        );
        assert_eq!(
            call(vec![pages], Some(1)).unwrap_err().identifier(),
            Some("RunMat:uiputfile:InvalidArgument")
        );
        assert!(request.lock().unwrap().is_none());
        assert!(
            matches!(call(vec![], Some(0)).unwrap(), Value::OutputList(values) if values.is_empty())
        );
        assert!(request.lock().unwrap().is_some());
    });
}

#[test]
fn cancel_returns_zero_outputs() {
    with_dialog_provider(None, |_| {
        let outputs = output_list(call(vec![], Some(2)).expect("uiputfile"));
        assert_eq!(outputs, vec![Value::Num(0.0), Value::Num(0.0)]);
    });
}

#[test]
fn parses_filter_title_default_and_selection_outputs() {
    let selection = SaveFileDialogSelection {
        path: rooted_tmp_file("scores.xlsx"),
        filter_index: Some(1),
    };
    with_dialog_provider(Some(selection), |request| {
        let outputs = output_list(
            call(
                vec![
                    Value::CharArray(CharArray::new_row("*.xlsx;*.xls")),
                    Value::CharArray(CharArray::new_row("Save spreadsheet")),
                    Value::CharArray(CharArray::new_row(&rooted_tmp_path_text())),
                ],
                Some(3),
            )
            .expect("uiputfile"),
        );
        assert_eq!(text(&outputs[0]), "scores.xlsx");
        assert_eq!(text(&outputs[1]), rooted_tmp_dir_text());
        assert_eq!(outputs[2], Value::Num(1.0));

        let request = request.lock().unwrap().clone().expect("request");
        assert_eq!(request.title.as_deref(), Some("Save spreadsheet"));
        assert_eq!(request.default_path, Some(rooted_tmp_path()));
        assert_eq!(request.filters[0].patterns, vec!["*.xlsx", "*.xls"]);
    });
}

#[test]
fn accepts_backslash_separated_provider_path() {
    let selection = SaveFileDialogSelection {
        path: PathBuf::from(r"C:\data\scores.xlsx"),
        filter_index: Some(1),
    };
    with_dialog_provider(Some(selection), |_| {
        let outputs = output_list(call(vec![], Some(2)).expect("uiputfile"));
        assert_eq!(text(&outputs[0]), "scores.xlsx");
        assert_eq!(text(&outputs[1]), r"C:\data\");
    });
}

#[test]
fn rejects_invalid_filter_index() {
    let selection = SaveFileDialogSelection {
        path: rooted_tmp_file("scores.xlsx"),
        filter_index: Some(2),
    };
    with_dialog_provider(Some(selection), |_| {
        let err = call(vec![], Some(3)).expect_err("expected invalid selection");
        assert_eq!(err.identifier(), Some("RunMat:uiputfile:InvalidSelection"));
        assert!(err.message().contains("filter index"));
    });
}

#[test]
fn rejects_numeric_tensor_text_arguments() {
    with_dialog_provider(None, |_| {
        let tensor = Tensor::new(vec![42.0], vec![1, 1]).expect("tensor");
        let err =
            call(vec![Value::Tensor(tensor)], Some(1)).expect_err("expected invalid argument");
        assert_eq!(err.identifier(), Some("RunMat:uiputfile:InvalidArgument"));
    });
}

#[test]
fn rejects_too_many_outputs() {
    with_dialog_provider(None, |_| {
        let err = call(vec![], Some(4)).expect_err("expected too many outputs");
        assert_eq!(err.identifier(), Some("RunMat:uiputfile:TooManyOutputs"));
    });
}
