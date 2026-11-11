use super::*;
use runmat_value::Tensor;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

fn call(args: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
    let _guard = crate::output_count::push_output_count(outputs).expect("output-count scope");
    futures::executor::block_on(uigetfile_builtin(
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
    selection: Option<OpenFileDialogSelection>,
    request: Arc<Mutex<Option<OpenFileDialogRequest>>>,
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

    fn select_file_open(
        &self,
        request: OpenFileDialogRequest,
    ) -> crate::context::RuntimeServiceFuture<Result<Option<OpenFileDialogSelection>, RuntimeError>>
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
    selection: Option<OpenFileDialogSelection>,
    body: impl FnOnce(Arc<Mutex<Option<OpenFileDialogRequest>>>),
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
        assert_eq!(error.identifier(), Some("RunMat:uigetfile:InvalidArgument"));
        assert!(request.lock().unwrap().is_none());
        let error = call(vec![], Some(4)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:uigetfile:TooManyOutputs"));
        assert!(request.lock().unwrap().is_none());
        let pages = Value::CharArray(
            runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
                .unwrap(),
        );
        assert_eq!(
            call(vec![pages], Some(1)).unwrap_err().identifier(),
            Some("RunMat:uigetfile:InvalidArgument")
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
        let outputs = output_list(call(vec![], Some(2)).expect("uigetfile"));
        assert_eq!(outputs, vec![Value::Num(0.0), Value::Num(0.0)]);
    });
}

#[test]
fn multiselect_controls_preserve_exact_integer_truth_values() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    use runmat_value::IntValue;
    for control in [
        IntValue::I8(i8::MIN),
        IntValue::I16(i16::MIN),
        IntValue::I32(i32::MIN),
        IntValue::I64(i64::MIN),
        IntValue::U8(u8::MAX),
        IntValue::U16(u16::MAX),
        IntValue::U32(u32::MAX),
        IntValue::U64(u64::MAX),
    ] {
        assert!(parse_multiselect(&Value::Int(control)).unwrap());
    }
    for control in [
        IntValue::I8(0),
        IntValue::I16(0),
        IntValue::I32(0),
        IntValue::I64(0),
        IntValue::U8(0),
        IntValue::U16(0),
        IntValue::U32(0),
        IntValue::U64(0),
    ] {
        assert!(!parse_multiselect(&Value::Int(control)).unwrap());
    }
    assert!(parse_multiselect(&Value::Bool(true)).unwrap());
    assert!(!parse_multiselect(&Value::Num(0.0)).unwrap());
    assert!(parse_multiselect(&Value::Num(f64::NAN)).is_err());
    assert!(parse_multiselect(&Value::Num(f64::INFINITY)).is_err());
}

#[test]
fn strict_multiselect_extensions_reject_before_dialog() {
    with_dialog_provider(None, |request| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for control in [
            Value::Bool(true),
            Value::Num(1.0),
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            Value::from("yes"),
            Value::from("false"),
        ] {
            let error = call(
                vec![Value::from("*.txt"), Value::from("MultiSelect"), control],
                Some(1),
            )
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:UigetfileMultiSelectExtension")
            );
            assert!(request.lock().unwrap().is_none());
        }
        assert!(parse_multiselect(&Value::from("on")).unwrap());
        assert!(!parse_multiselect(&Value::from("off")).unwrap());
    });
}

#[test]
fn parses_filter_title_default_and_selection_outputs() {
    let selection = OpenFileDialogSelection {
        paths: vec![rooted_tmp_file("scores.xlsx")],
        filter_index: Some(1),
    };
    with_dialog_provider(Some(selection), |request| {
        let outputs = output_list(
            call(
                vec![
                    Value::CharArray(CharArray::new_row("*.xlsx;*.xls")),
                    Value::CharArray(CharArray::new_row("Select spreadsheet")),
                    Value::CharArray(CharArray::new_row(&rooted_tmp_path_text())),
                ],
                Some(3),
            )
            .expect("uigetfile"),
        );
        assert_eq!(text(&outputs[0]), "scores.xlsx");
        assert_eq!(text(&outputs[1]), rooted_tmp_dir_text());
        assert_eq!(outputs[2], Value::Num(1.0));

        let request = request.lock().unwrap().clone().expect("request");
        assert_eq!(request.title.as_deref(), Some("Select spreadsheet"));
        assert_eq!(request.default_path, Some(rooted_tmp_path()));
        assert!(!request.multiselect);
        assert_eq!(request.filters[0].patterns, vec!["*.xlsx", "*.xls"]);
    });
}

#[test]
fn multiselect_returns_cell_array_for_multiple_files() {
    let selection = OpenFileDialogSelection {
        paths: vec![rooted_tmp_file("a.m"), rooted_tmp_file("b.m")],
        filter_index: Some(1),
    };
    with_dialog_provider(Some(selection), |_| {
        let outputs = output_list(
            call(
                vec![
                    Value::CharArray(CharArray::new_row("*.m")),
                    Value::CharArray(CharArray::new_row("Open")),
                    Value::CharArray(CharArray::new_row(&rooted_tmp_path_text())),
                    Value::CharArray(CharArray::new_row("MultiSelect")),
                    Value::CharArray(CharArray::new_row("on")),
                ],
                Some(3),
            )
            .expect("uigetfile"),
        );
        match &outputs[0] {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 2);
                assert_eq!(text(&cell.get(0, 0).expect("first file")), "a.m");
                assert_eq!(text(&cell.get(0, 1).expect("second file")), "b.m");
            }
            other => panic!("expected cell array, got {other:?}"),
        }
        assert_eq!(text(&outputs[1]), rooted_tmp_dir_text());
        assert_eq!(outputs[2], Value::Num(1.0));
    });
}

#[test]
fn rejects_multiple_single_select_provider_paths() {
    let selection = OpenFileDialogSelection {
        paths: vec![rooted_tmp_file("a.m"), rooted_tmp_file("b.m")],
        filter_index: Some(1),
    };
    with_dialog_provider(Some(selection), |_| {
        let err = call(vec![], Some(3)).expect_err("expected invalid selection");
        assert_eq!(err.identifier(), Some("RunMat:uigetfile:InvalidSelection"));
        assert!(err.message().contains("single-selection"));
    });
}

#[test]
fn accepts_backslash_separated_provider_paths() {
    let selection = OpenFileDialogSelection {
        paths: vec![PathBuf::from(r"C:\data\scores.xlsx")],
        filter_index: Some(1),
    };
    with_dialog_provider(Some(selection), |_| {
        let outputs = output_list(call(vec![], Some(2)).expect("uigetfile"));
        assert_eq!(text(&outputs[0]), "scores.xlsx");
        assert_eq!(text(&outputs[1]), r"C:\data\");
    });
}

#[test]
fn rejects_numeric_tensor_text_arguments() {
    with_dialog_provider(None, |_| {
        let tensor = Tensor::new(vec![42.0], vec![1, 1]).expect("tensor");
        let err =
            call(vec![Value::Tensor(tensor)], Some(1)).expect_err("expected invalid argument");
        assert_eq!(err.identifier(), Some("RunMat:uigetfile:InvalidArgument"));
    });
}

#[test]
fn rejects_too_many_outputs() {
    with_dialog_provider(None, |_| {
        let err = call(vec![], Some(4)).expect_err("expected too many outputs");
        assert_eq!(err.identifier(), Some("RunMat:uigetfile:TooManyOutputs"));
    });
}
