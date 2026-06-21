//! MATLAB-compatible `uiputfile` builtin.

use std::path::PathBuf;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_filesystem::{SaveFileDialogRequest, SaveFileDialogSelection};
use runmat_macros::runtime_builtin;

use super::file_dialog::{default_filters, parse_filter_spec, scalar_text, selected_path_parts};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "uiputfile";

const UIPUTFILE_OUTPUT_FILE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "file",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected filename, or 0 when cancelled.",
}];

const UIPUTFILE_OUTPUT_FILE_PATH: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected filename, or 0 when cancelled.",
    },
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Directory containing the selected file, including a trailing separator, or 0 when cancelled.",
    },
];

const UIPUTFILE_OUTPUT_FILE_PATH_INDEX: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected filename, or 0 when cancelled.",
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

const UIPUTFILE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const UIPUTFILE_INPUTS_FILTER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filter",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("\"*.*\""),
    description: "File extension pattern, semicolon-delimited patterns, or an N-by-1/N-by-2 cell array of patterns and descriptions.",
}];

const UIPUTFILE_INPUTS_FILTER_TITLE: [BuiltinParamDescriptor; 2] = [
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

const UIPUTFILE_INPUTS_FILTER_TITLE_DEFAULT: [BuiltinParamDescriptor; 3] = [
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

const UIPUTFILE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "file = uiputfile()",
        inputs: &UIPUTFILE_INPUTS_NONE,
        outputs: &UIPUTFILE_OUTPUT_FILE,
    },
    BuiltinSignatureDescriptor {
        label: "file = uiputfile(filter)",
        inputs: &UIPUTFILE_INPUTS_FILTER,
        outputs: &UIPUTFILE_OUTPUT_FILE,
    },
    BuiltinSignatureDescriptor {
        label: "[file, path] = uiputfile(filter, title)",
        inputs: &UIPUTFILE_INPUTS_FILTER_TITLE,
        outputs: &UIPUTFILE_OUTPUT_FILE_PATH,
    },
    BuiltinSignatureDescriptor {
        label: "[file, path, index] = uiputfile(filter, title, defaultName)",
        inputs: &UIPUTFILE_INPUTS_FILTER_TITLE_DEFAULT,
        outputs: &UIPUTFILE_OUTPUT_FILE_PATH_INDEX,
    },
];

const UIPUTFILE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIPUTFILE.INVALID_ARGUMENT",
    identifier: Some("RunMat:uiputfile:InvalidArgument"),
    when: "A filter, title, or default path has an unsupported type or shape.",
    message: "uiputfile: invalid argument",
};

const UIPUTFILE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIPUTFILE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:uiputfile:TooManyOutputs"),
    when: "More than three output arguments are requested.",
    message: "uiputfile: too many output arguments",
};

const UIPUTFILE_ERROR_HOST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIPUTFILE.HOST",
    identifier: Some("RunMat:uiputfile:HostError"),
    when: "The active filesystem provider fails while opening the host save-file UI.",
    message: "uiputfile: file selection failed",
};

const UIPUTFILE_ERROR_INVALID_SELECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIPUTFILE.INVALID_SELECTION",
    identifier: Some("RunMat:uiputfile:InvalidSelection"),
    when: "The active filesystem provider returns a malformed save-file selection.",
    message: "uiputfile: invalid file selection",
};

const UIPUTFILE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    UIPUTFILE_ERROR_INVALID_ARGUMENT,
    UIPUTFILE_ERROR_TOO_MANY_OUTPUTS,
    UIPUTFILE_ERROR_HOST,
    UIPUTFILE_ERROR_INVALID_SELECTION,
];

pub const UIPUTFILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UIPUTFILE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UIPUTFILE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::uiputfile")]
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
    notes: "`uiputfile` is a host UI/filesystem interaction. GPU-resident textual arguments are gathered before dispatching to the provider.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::uiputfile")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "`uiputfile` depends on host UI state and terminates fusion plans.",
};

#[derive(Clone, Debug)]
struct UiputfileOptions {
    request: SaveFileDialogRequest,
}

fn uiputfile_error(
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

fn invalid_argument(detail: String) -> RuntimeError {
    uiputfile_error(&UIPUTFILE_ERROR_INVALID_ARGUMENT, detail)
}

fn invalid_selection(detail: String) -> RuntimeError {
    uiputfile_error(&UIPUTFILE_ERROR_INVALID_SELECTION, detail)
}

fn too_many_outputs() -> RuntimeError {
    uiputfile_error(&UIPUTFILE_ERROR_TOO_MANY_OUTPUTS, "")
}

fn host_error(detail: impl AsRef<str>) -> RuntimeError {
    uiputfile_error(&UIPUTFILE_ERROR_HOST, detail)
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
    name = "uiputfile",
    category = "io/repl_fs",
    summary = "Open a host save-file dialog and return the selected file name and path.",
    keywords = "uiputfile,file picker,file dialog,save file,filesystem,ui",
    accel = "sink",
    type_resolver(crate::builtins::io::type_resolvers::uiputfile_type),
    descriptor(crate::builtins::io::repl_fs::uiputfile::UIPUTFILE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::uiputfile"
)]
async fn uiputfile_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_arguments(&args).await?;
    let options = parse_options(&gathered)?;
    let selection = runmat_filesystem::select_file_save_async(&options.request)
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

fn parse_options(args: &[Value]) -> BuiltinResult<UiputfileOptions> {
    if args.len() > 3 {
        return Err(invalid_argument(
            "expected filter, title, and defaultName".to_string(),
        ));
    }

    let filters = if !args.is_empty() {
        parse_filter_spec(&args[0], invalid_argument)?
    } else {
        default_filters()
    };
    let title = if args.len() >= 2 {
        Some(scalar_text(&args[1], "title", invalid_argument)?)
    } else {
        None
    };
    let default_path = if args.len() >= 3 {
        Some(PathBuf::from(scalar_text(
            &args[2],
            "defaultName",
            invalid_argument,
        )?))
    } else {
        None
    };

    Ok(UiputfileOptions {
        request: SaveFileDialogRequest {
            title,
            default_path,
            filters,
        },
    })
}

fn outputs_for_selection(
    selection: Option<SaveFileDialogSelection>,
    options: &UiputfileOptions,
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
    selection: SaveFileDialogSelection,
    options: &UiputfileOptions,
) -> BuiltinResult<Vec<Value>> {
    let filter_index = selection.filter_index.unwrap_or(1);
    if filter_index == 0 || filter_index > options.request.filters.len() {
        return Err(invalid_selection(
            "provider returned an invalid filter index".to_string(),
        ));
    }

    let selected = selected_path_parts(&selection.path, invalid_selection)?;
    Ok(vec![
        Value::CharArray(runmat_builtins::CharArray::new_row(&selected.file_name)),
        Value::CharArray(runmat_builtins::CharArray::new_row(&selected.directory)),
        Value::Num(filter_index as f64),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use runmat_builtins::{CharArray, Tensor};
    use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags};
    use std::io::{self, ErrorKind};
    use std::path::Path;
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

        async fn select_file_save(
            &self,
            request: &SaveFileDialogRequest,
        ) -> io::Result<Option<SaveFileDialogSelection>> {
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
        selection: Option<SaveFileDialogSelection>,
        body: impl FnOnce(Arc<Mutex<Option<SaveFileDialogRequest>>>),
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
}
