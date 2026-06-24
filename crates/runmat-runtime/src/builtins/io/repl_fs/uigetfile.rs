//! MATLAB-compatible `uigetfile` builtin.

use std::path::PathBuf;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, Value,
};
use runmat_filesystem::{OpenFileDialogRequest, OpenFileDialogSelection};
use runmat_macros::runtime_builtin;

use super::file_dialog::{
    default_filters, ensure_same_directory, parse_filter_spec, scalar_text, selected_path_parts,
    try_scalar_text,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "uigetfile";

const UIGETFILE_OUTPUT_FILE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "file",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Selected filename, cell array of filenames for multiple selections, or 0 when cancelled.",
}];

const UIGETFILE_OUTPUT_FILE_PATH: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected filename, cell array of filenames for multiple selections, or 0 when cancelled.",
    },
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Directory containing the selected file, including a trailing separator, or 0 when cancelled.",
    },
];

const UIGETFILE_OUTPUT_FILE_PATH_INDEX: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected filename, cell array of filenames for multiple selections, or 0 when cancelled.",
    },
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Directory containing the selected file, including a trailing separator, or 0 when cancelled.",
    },
    BuiltinParamDescriptor {
        name: "index",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based selected filter index, or 0 when cancelled.",
    },
];

const UIGETFILE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const UIGETFILE_INPUTS_FILTER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filter",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("\"*.*\""),
    description: "File extension pattern, semicolon-delimited patterns, or an N-by-1/N-by-2 cell array of patterns and descriptions.",
}];

const UIGETFILE_INPUTS_FILTER_TITLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"*.*\""),
        description: "File extension pattern, semicolon-delimited patterns, or an N-by-1/N-by-2 cell array of patterns and descriptions.",
    },
    BuiltinParamDescriptor {
        name: "title",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Dialog title.",
    },
];

const UIGETFILE_INPUTS_FILTER_TITLE_DEFAULT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"*.*\""),
        description: "File extension pattern, semicolon-delimited patterns, or an N-by-1/N-by-2 cell array of patterns and descriptions.",
    },
    BuiltinParamDescriptor {
        name: "title",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Dialog title.",
    },
    BuiltinParamDescriptor {
        name: "defaultName",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Initial file or directory path for the dialog.",
    },
];

const UIGETFILE_INPUTS_MULTISELECT: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "filter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"*.*\""),
        description: "File extension pattern, semicolon-delimited patterns, or an N-by-1/N-by-2 cell array of patterns and descriptions.",
    },
    BuiltinParamDescriptor {
        name: "title",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Dialog title.",
    },
    BuiltinParamDescriptor {
        name: "defaultName",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Initial file or directory path for the dialog.",
    },
    BuiltinParamDescriptor {
        name: "optionName",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"MultiSelect\""),
        description: "Name-value option name. `MultiSelect` is supported.",
    },
    BuiltinParamDescriptor {
        name: "optionValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"off\""),
        description: "`\"on\"`, `\"off\"`, or a scalar logical/numeric value.",
    },
];

const UIGETFILE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "file = uigetfile()",
        inputs: &UIGETFILE_INPUTS_NONE,
        outputs: &UIGETFILE_OUTPUT_FILE,
    },
    BuiltinSignatureDescriptor {
        label: "file = uigetfile(filter)",
        inputs: &UIGETFILE_INPUTS_FILTER,
        outputs: &UIGETFILE_OUTPUT_FILE,
    },
    BuiltinSignatureDescriptor {
        label: "[file, path] = uigetfile(filter, title)",
        inputs: &UIGETFILE_INPUTS_FILTER_TITLE,
        outputs: &UIGETFILE_OUTPUT_FILE_PATH,
    },
    BuiltinSignatureDescriptor {
        label: "[file, path, index] = uigetfile(filter, title, defaultName)",
        inputs: &UIGETFILE_INPUTS_FILTER_TITLE_DEFAULT,
        outputs: &UIGETFILE_OUTPUT_FILE_PATH_INDEX,
    },
    BuiltinSignatureDescriptor {
        label:
            "[file, path, index] = uigetfile(filter, title, defaultName, \"MultiSelect\", value)",
        inputs: &UIGETFILE_INPUTS_MULTISELECT,
        outputs: &UIGETFILE_OUTPUT_FILE_PATH_INDEX,
    },
];

const UIGETFILE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETFILE.INVALID_ARGUMENT",
    identifier: Some("RunMat:uigetfile:InvalidArgument"),
    when: "A filter, title, default path, or name-value option has an unsupported type or shape.",
    message: "uigetfile: invalid argument",
};

const UIGETFILE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETFILE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:uigetfile:TooManyOutputs"),
    when: "More than three output arguments are requested.",
    message: "uigetfile: too many output arguments",
};

const UIGETFILE_ERROR_HOST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETFILE.HOST",
    identifier: Some("RunMat:uigetfile:HostError"),
    when: "The active filesystem provider fails while opening the host file-selection UI.",
    message: "uigetfile: file selection failed",
};

const UIGETFILE_ERROR_INVALID_SELECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETFILE.INVALID_SELECTION",
    identifier: Some("RunMat:uigetfile:InvalidSelection"),
    when: "The active filesystem provider returns a malformed or internally inconsistent file selection.",
    message: "uigetfile: invalid file selection",
};

const UIGETFILE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    UIGETFILE_ERROR_INVALID_ARGUMENT,
    UIGETFILE_ERROR_TOO_MANY_OUTPUTS,
    UIGETFILE_ERROR_HOST,
    UIGETFILE_ERROR_INVALID_SELECTION,
];

pub const UIGETFILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UIGETFILE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UIGETFILE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::uigetfile")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "`uigetfile` is a host UI/filesystem interaction. GPU-resident textual arguments are gathered before dispatching to the provider.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::uigetfile")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "`uigetfile` depends on host UI state and terminates fusion plans.",
};

#[derive(Clone, Debug)]
struct UigetfileOptions {
    request: OpenFileDialogRequest,
}

fn uigetfile_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    uigetfile_error(&UIGETFILE_ERROR_INVALID_ARGUMENT, detail)
}

fn invalid_selection(detail: impl AsRef<str>) -> RuntimeError {
    uigetfile_error(&UIGETFILE_ERROR_INVALID_SELECTION, detail)
}

fn too_many_outputs() -> RuntimeError {
    uigetfile_error(&UIGETFILE_ERROR_TOO_MANY_OUTPUTS, "")
}

fn host_error(detail: impl AsRef<str>) -> RuntimeError {
    uigetfile_error(&UIGETFILE_ERROR_HOST, detail)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{NAME}: {}", err.message()))
        .with_builtin(NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "uigetfile",
    category = "io/repl_fs",
    summary = "Open a host file-selection dialog and return the selected file name and path.",
    keywords = "uigetfile,file picker,file dialog,open file,filesystem,ui",
    accel = "sink",
    type_resolver(crate::builtins::io::type_resolvers::uigetfile_type),
    descriptor(crate::builtins::io::repl_fs::uigetfile::UIGETFILE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::uigetfile"
)]
async fn uigetfile_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_arguments(&args).await?;
    let options = parse_options(&gathered)?;
    let selection = runmat_filesystem::select_file_open_async(&options.request)
        .await
        .map_err(|err| host_error(err.to_string()))?;
    outputs_for_selection(selection, &options)
}

async fn gather_arguments(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(gathered)
}

fn parse_options(args: &[Value]) -> BuiltinResult<UigetfileOptions> {
    let mut end = args.len();
    let mut multiselect = false;
    while end >= 2 {
        let option_name = match try_scalar_text(&args[end - 2]) {
            Some(name) => name,
            None => break,
        };
        if !option_name.eq_ignore_ascii_case("MultiSelect") {
            break;
        }
        multiselect = parse_multiselect(&args[end - 1])?;
        end -= 2;
    }
    if end > 3 {
        return Err(invalid_argument(
            "expected filter, title, defaultName, and optional 'MultiSelect' name-value pair",
        ));
    }

    let filters = if end >= 1 {
        parse_filter_spec(&args[0], invalid_argument)?
    } else {
        default_filters()
    };
    let title = if end >= 2 {
        Some(scalar_text(&args[1], "title", invalid_argument)?)
    } else {
        None
    };
    let default_path = if end >= 3 {
        Some(PathBuf::from(scalar_text(
            &args[2],
            "defaultName",
            invalid_argument,
        )?))
    } else {
        None
    };

    Ok(UigetfileOptions {
        request: OpenFileDialogRequest {
            title,
            default_path,
            filters,
            multiselect,
        },
    })
}

fn parse_multiselect(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(enabled) => Ok(*enabled),
        Value::Num(number) if number.is_finite() => Ok(*number != 0.0),
        Value::Int(int) => Ok(!int.is_zero()),
        other => {
            let text = scalar_text(other, "MultiSelect value", invalid_argument)?;
            match text.trim().to_ascii_lowercase().as_str() {
                "on" | "true" | "yes" => Ok(true),
                "off" | "false" | "no" => Ok(false),
                _ => Err(invalid_argument(
                    "MultiSelect must be 'on', 'off', true, or false",
                )),
            }
        }
    }
}

fn outputs_for_selection(
    selection: Option<OpenFileDialogSelection>,
    options: &UigetfileOptions,
) -> BuiltinResult<Value> {
    let outputs = match selection {
        Some(selection) => selected_outputs(selection, options)?,
        None => vec![Value::Num(0.0), Value::Num(0.0), Value::Num(0.0)],
    };

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count > 3 {
            return Err(too_many_outputs());
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }

    Ok(outputs.into_iter().next().unwrap_or(Value::Num(0.0)))
}

fn selected_outputs(
    selection: OpenFileDialogSelection,
    options: &UigetfileOptions,
) -> BuiltinResult<Vec<Value>> {
    if selection.paths.is_empty() {
        return Err(invalid_selection("provider returned no selected paths"));
    }

    let filter_index = selection.filter_index.unwrap_or(1);
    if filter_index == 0 || filter_index > options.request.filters.len() {
        return Err(invalid_selection(
            "provider returned an invalid filter index",
        ));
    }
    if !options.request.multiselect && selection.paths.len() > 1 {
        return Err(invalid_selection(
            "provider returned multiple paths for a single-selection request",
        ));
    }

    let first_path = selected_path_parts(&selection.paths[0], invalid_selection)?;
    let directory = first_path.directory;
    if options.request.multiselect && selection.paths.len() > 1 {
        ensure_same_directory(&selection.paths, &directory, invalid_selection)?;
        let mut names = Vec::with_capacity(selection.paths.len());
        for path in &selection.paths {
            names.push(Value::CharArray(CharArray::new_row(
                &selected_path_parts(path, invalid_selection)?.file_name,
            )));
        }
        let file_count = names.len();
        let files = CellArray::new(names, 1, file_count).map_err(invalid_selection)?;
        return Ok(vec![
            Value::Cell(files),
            Value::CharArray(CharArray::new_row(&directory)),
            Value::Num(filter_index as f64),
        ]);
    }

    Ok(vec![
        Value::CharArray(CharArray::new_row(&first_path.file_name)),
        Value::CharArray(CharArray::new_row(&directory)),
        Value::Num(filter_index as f64),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use runmat_builtins::Tensor;
    use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags};
    use std::io::{self, ErrorKind};
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    fn call(args: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
        let _guard = crate::output_count::push_output_count(outputs);
        futures::executor::block_on(uigetfile_builtin(args))
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
        selection: Option<OpenFileDialogSelection>,
        request: Arc<Mutex<Option<OpenFileDialogRequest>>>,
    }

    #[async_trait(?Send)]
    impl FsProvider for DialogProvider {
        fn open(&self, _path: &Path, _flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
            Err(unsupported())
        }

        async fn read(&self, _path: &Path) -> io::Result<Vec<u8>> {
            Err(unsupported())
        }

        async fn write(&self, _path: &Path, _data: &[u8]) -> io::Result<()> {
            Err(unsupported())
        }

        async fn remove_file(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
            Err(unsupported())
        }

        async fn symlink_metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
            Err(unsupported())
        }

        async fn read_dir(&self, _path: &Path) -> io::Result<Vec<DirEntry>> {
            Err(unsupported())
        }

        async fn canonicalize(&self, _path: &Path) -> io::Result<PathBuf> {
            Err(unsupported())
        }

        async fn create_dir(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn create_dir_all(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn remove_dir(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn remove_dir_all(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn rename(&self, _from: &Path, _to: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn set_readonly(&self, _path: &Path, _readonly: bool) -> io::Result<()> {
            Err(unsupported())
        }

        async fn select_file_open(
            &self,
            request: &OpenFileDialogRequest,
        ) -> io::Result<Option<OpenFileDialogSelection>> {
            *self.request.lock().unwrap() = Some(request.clone());
            Ok(self.selection.clone())
        }
    }

    fn unsupported() -> io::Error {
        io::Error::new(ErrorKind::Unsupported, "unsupported")
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
        selection: Option<OpenFileDialogSelection>,
        body: impl FnOnce(Arc<Mutex<Option<OpenFileDialogRequest>>>),
    ) {
        let _lock = runmat_filesystem::provider_override_lock();
        let request = Arc::new(Mutex::new(None));
        let provider = Arc::new(DialogProvider {
            selection,
            request: request.clone(),
        });
        let _guard = runmat_filesystem::replace_provider(provider);
        body(request);
    }

    #[test]
    fn cancel_returns_zero_outputs() {
        with_dialog_provider(None, |_| {
            let outputs = output_list(call(vec![], Some(2)).expect("uigetfile"));
            assert_eq!(outputs, vec![Value::Num(0.0), Value::Num(0.0)]);
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
}
