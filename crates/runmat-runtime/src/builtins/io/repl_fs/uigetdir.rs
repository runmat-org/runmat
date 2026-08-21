//! MATLAB-compatible `uigetdir` builtin.

use std::path::PathBuf;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_filesystem::{DirectoryDialogRequest, DirectoryDialogSelection};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};

use super::file_dialog::scalar_text;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "uigetdir";

const UIGETDIR_OUTPUT_DIR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "folder",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected folder path as a character vector, or 0 when cancelled.",
}];

const UIGETDIR_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const UIGETDIR_INPUTS_START_PATH: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "startPath",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Initial folder shown by the dialog.",
}];

const UIGETDIR_INPUTS_START_PATH_TITLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "startPath",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Initial folder shown by the dialog.",
    },
    BuiltinParamDescriptor {
        name: "title",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Dialog title.",
    },
];

const UIGETDIR_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "folder = uigetdir()",
        inputs: &UIGETDIR_INPUTS_NONE,
        outputs: &UIGETDIR_OUTPUT_DIR,
    },
    BuiltinSignatureDescriptor {
        label: "folder = uigetdir(startPath)",
        inputs: &UIGETDIR_INPUTS_START_PATH,
        outputs: &UIGETDIR_OUTPUT_DIR,
    },
    BuiltinSignatureDescriptor {
        label: "folder = uigetdir(startPath, title)",
        inputs: &UIGETDIR_INPUTS_START_PATH_TITLE,
        outputs: &UIGETDIR_OUTPUT_DIR,
    },
];

const UIGETDIR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETDIR.INVALID_ARGUMENT",
    identifier: Some("RunMat:uigetdir:InvalidArgument"),
    when: "The start path or title has an unsupported type or shape.",
    message: "uigetdir: invalid argument",
};

const UIGETDIR_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETDIR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:uigetdir:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "uigetdir: too many output arguments",
};

const UIGETDIR_ERROR_HOST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETDIR.HOST",
    identifier: Some("RunMat:uigetdir:HostError"),
    when: "The active filesystem provider fails while opening the host folder-selection UI.",
    message: "uigetdir: folder selection failed",
};

const UIGETDIR_ERROR_INVALID_SELECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETDIR.INVALID_SELECTION",
    identifier: Some("RunMat:uigetdir:InvalidSelection"),
    when: "The active filesystem provider returns a malformed folder selection.",
    message: "uigetdir: invalid folder selection",
};

const UIGETDIR_ERRORS: [BuiltinErrorDescriptor; 4] = [
    UIGETDIR_ERROR_INVALID_ARGUMENT,
    UIGETDIR_ERROR_TOO_MANY_OUTPUTS,
    UIGETDIR_ERROR_HOST,
    UIGETDIR_ERROR_INVALID_SELECTION,
];

pub const UIGETDIR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UIGETDIR_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UIGETDIR_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::uigetdir")]
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
    notes: "`uigetdir` is a host UI/filesystem interaction. GPU-resident textual arguments are gathered before dispatching to the provider.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::uigetdir")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "`uigetdir` depends on host UI state and terminates fusion plans.",
};

#[derive(Clone, Debug)]
struct UigetdirOptions {
    request: DirectoryDialogRequest,
}

fn uigetdir_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
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
    uigetdir_error(&UIGETDIR_ERROR_INVALID_ARGUMENT, detail)
}

fn invalid_selection(detail: impl AsRef<str>) -> RuntimeError {
    uigetdir_error(&UIGETDIR_ERROR_INVALID_SELECTION, detail)
}

fn too_many_outputs() -> RuntimeError {
    uigetdir_error(&UIGETDIR_ERROR_TOO_MANY_OUTPUTS, "")
}

fn host_error(detail: impl AsRef<str>) -> RuntimeError {
    uigetdir_error(&UIGETDIR_ERROR_HOST, detail)
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
    name = "uigetdir",
    category = "io/repl_fs",
    summary = "Open a host folder-selection dialog and return the selected folder path.",
    keywords = "uigetdir,folder picker,directory picker,file dialog,filesystem,ui",
    accel = "sink",
    type_resolver(crate::builtins::io::type_resolvers::uigetdir_type),
    descriptor(crate::builtins::io::repl_fs::uigetdir::UIGETDIR_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::uigetdir"
)]
async fn uigetdir_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_arguments(&args).await?;
    let options = parse_options(&gathered)?;
    let selection = runmat_filesystem::select_directory_async(&options.request)
        .await
        .map_err(|err| host_error(err.to_string()))?;
    outputs_for_selection(selection)
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

fn parse_options(args: &[Value]) -> BuiltinResult<UigetdirOptions> {
    if args.len() > 2 {
        return Err(invalid_argument("expected startPath and title"));
    }
    let default_path = if !args.is_empty() {
        Some(PathBuf::from(scalar_text(
            &args[0],
            "startPath",
            invalid_argument,
        )?))
    } else {
        None
    };
    let title = if args.len() >= 2 {
        Some(scalar_text(&args[1], "title", invalid_argument)?)
    } else {
        None
    };

    Ok(UigetdirOptions {
        request: DirectoryDialogRequest {
            title,
            default_path,
        },
    })
}

fn outputs_for_selection(selection: Option<DirectoryDialogSelection>) -> BuiltinResult<Value> {
    let output = match selection {
        Some(selection) => selected_output(selection)?,
        None => Value::Num(0.0),
    };

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count > 1 {
            return Err(too_many_outputs());
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![output],
        ));
    }

    Ok(output)
}

fn selected_output(selection: DirectoryDialogSelection) -> BuiltinResult<Value> {
    if selection.path.as_os_str().is_empty() {
        return Err(invalid_selection("provider returned an empty folder path"));
    }
    Ok(Value::CharArray(CharArray::new_row(
        &selection.path.to_string_lossy(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags};
    use runmat_value::Tensor;
    use std::io::{self, ErrorKind};
    use std::path::Path;
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

        async fn select_directory(
            &self,
            request: &DirectoryDialogRequest,
        ) -> io::Result<Option<DirectoryDialogSelection>> {
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
        let _guard = runmat_filesystem::replace_provider(provider);
        body(request);
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
}
