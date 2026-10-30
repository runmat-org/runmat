//! MATLAB-compatible `which` builtin for RunMat.
//!
//! Resolves the entity associated with a name, mirroring MATLAB search
//! semantics (workspace variables, builtins, classes, scripts, and folders)
//! and supporting `-all`, `-builtin`, `-var`, and `-file` options.

use std::collections::HashSet;
use std::path::Path;

use runmat_builtins::{
    which_option_token, BuiltinErrorDescriptor, WhichOptionToken, WHICH_ERROR_ARG_COUNT_MAX,
    WHICH_ERROR_ARG_COUNT_MIN, WHICH_ERROR_NAME_ARG_TYPE, WHICH_ERROR_OPTION_ARG_TYPE,
    WHICH_ERROR_OPTION_CONFLICT, WHICH_ERROR_OPTION_UNRECOGNIZED,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};

use crate::builtins::common::fs::path_to_string;
use crate::user_functions::{
    observe_runtime_name, RuntimeNameCandidate, RuntimeNameFilter, RuntimeNameQuery,
    RuntimeNameResolutionError, RuntimeNameRoute,
};
use crate::{build_runtime_error, make_cell, BuiltinResult, RuntimeError};

fn which_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    which_error_with_message(error.message, error)
}

fn which_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("which");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn which_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    which_error_with_message(format!("{} {detail}", error.message), error)
}

fn which_runtime_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("which").build()
}

fn which_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(|id| id.to_string());
    let retry = err.gpu_gather_retry;
    let mut builder = build_runtime_error(err.message().to_string())
        .with_builtin("which")
        .with_gpu_gather_retry(retry)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "which",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::which"
)]
async fn which_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(which_error(&WHICH_ERROR_ARG_COUNT_MIN));
    }

    let mut name: Option<String> = None;
    let mut options = WhichOptions::default();

    for arg in args {
        // GPU handles carry only numeric element/storage descriptors, never a
        // character or string representation. Decode the source-language
        // argument role before touching an acceleration provider.
        let text = value_to_string_scalar(&arg, || {
            if name.is_none() {
                which_error(&WHICH_ERROR_NAME_ARG_TYPE)
            } else {
                which_error(&WHICH_ERROR_OPTION_ARG_TYPE)
            }
        })?
        .ok_or_else(|| {
            if name.is_none() {
                which_error(&WHICH_ERROR_NAME_ARG_TYPE)
            } else {
                which_error(&WHICH_ERROR_OPTION_ARG_TYPE)
            }
        })?;

        if let Some(option) = which_option_token(&text) {
            options.apply(option, &text)?;
        } else if name.is_none() {
            name = Some(text);
        } else {
            return Err(which_error(&WHICH_ERROR_ARG_COUNT_MAX));
        }
    }

    let name = name.ok_or_else(|| which_error(&WHICH_ERROR_ARG_COUNT_MIN))?;
    let matches = search_matches(&name, &options).await?;
    if matches.is_empty() {
        return Ok(Value::CharArray(CharArray::new_row(&format!(
            "'{name}' not found."
        ))));
    }

    if options.all {
        let mut cell_values = Vec::with_capacity(matches.len());
        for entry in &matches {
            cell_values.push(Value::CharArray(CharArray::new_row(entry)));
        }
        return make_cell(cell_values, matches.len(), 1).map_err(|err| {
            build_runtime_error(err)
                .with_builtin("which")
                .build()
                .into()
        });
    }

    Ok(Value::CharArray(CharArray::new_row(
        matches.first().expect("non-empty result"),
    )))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum WhichMode {
    #[default]
    Any,
    Builtin,
    Variable,
    File,
}

impl WhichMode {
    fn label(self) -> &'static str {
        match self {
            Self::Any => "",
            Self::Builtin => "-builtin",
            Self::Variable => "-var",
            Self::File => "-file",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WhichOption {
    All,
    Mode(WhichMode),
}

impl WhichOption {
    fn decode(token: WhichOptionToken, text: &str) -> BuiltinResult<Self> {
        match token {
            WhichOptionToken::All => Ok(Self::All),
            WhichOptionToken::Builtin => Ok(Self::Mode(WhichMode::Builtin)),
            WhichOptionToken::Variable => Ok(Self::Mode(WhichMode::Variable)),
            WhichOptionToken::File => Ok(Self::Mode(WhichMode::File)),
            WhichOptionToken::Unrecognized => Err(which_error_with_detail(
                &WHICH_ERROR_OPTION_UNRECOGNIZED,
                format!("'{}'", text.trim().to_ascii_lowercase()),
            )),
        }
    }
}

#[derive(Default, Debug)]
struct WhichOptions {
    all: bool,
    mode: WhichMode,
}

impl WhichOptions {
    fn apply(&mut self, option: WhichOptionToken, text: &str) -> BuiltinResult<()> {
        match WhichOption::decode(option, text)? {
            WhichOption::All => self.all = true,
            WhichOption::Mode(mode) if self.mode == WhichMode::Any || self.mode == mode => {
                self.mode = mode;
            }
            WhichOption::Mode(mode) => {
                return Err(which_error_with_detail(
                    &WHICH_ERROR_OPTION_CONFLICT,
                    format!(
                        "'{}'; cannot combine with {}",
                        mode.label(),
                        self.mode.label()
                    ),
                ));
            }
        }
        Ok(())
    }
}

async fn search_matches(name: &str, options: &WhichOptions) -> BuiltinResult<Vec<String>> {
    let filter = if options.mode == WhichMode::Variable {
        RuntimeNameFilter::Workspace
    } else if options.mode == WhichMode::Builtin {
        RuntimeNameFilter::Builtin
    } else if options.mode == WhichMode::File {
        RuntimeNameFilter::File
    } else {
        RuntimeNameFilter::Any
    };
    let resolution = observe_runtime_name(
        name,
        RuntimeNameQuery {
            route: RuntimeNameRoute::DynamicCall,
            filter,
            all: options.all,
        },
    )
    .await
    .map_err(|error| match error {
        RuntimeNameResolutionError::Workspace(error)
        | RuntimeNameResolutionError::Control(error) => which_flow(*error),
        RuntimeNameResolutionError::WasmRegistryNotReady
        | RuntimeNameResolutionError::SearchPathChanged
        | RuntimeNameResolutionError::ClassRegistryChanged => {
            which_runtime_error(error.to_string())
        }
        RuntimeNameResolutionError::Search(error) => error.into_runtime_error("which"),
    })?;
    let mut results = Vec::new();
    let mut seen = HashSet::new();
    let mut seen_semantic_owners = HashSet::new();
    let mut seen_semantic_origins = HashSet::new();
    for candidate in &resolution.candidates {
        let admitted = match candidate {
            RuntimeNameCandidate::WorkspaceVariable => {
                options.mode != WhichMode::Builtin && options.mode != WhichMode::File
            }
            RuntimeNameCandidate::Builtin(_) => {
                options.mode != WhichMode::Variable && options.mode != WhichMode::File
            }
            RuntimeNameCandidate::Native(_)
            | RuntimeNameCandidate::RegisteredConstructor(_)
            | RuntimeNameCandidate::RegisteredStaticMethod { .. }
            | RuntimeNameCandidate::Semantic { .. }
            | RuntimeNameCandidate::Source(_)
            | RuntimeNameCandidate::ClassFolder(_)
            | RuntimeNameCandidate::ClassFile(_)
            | RuntimeNameCandidate::File(_)
            | RuntimeNameCandidate::Directory(_) => {
                options.mode != WhichMode::Variable && options.mode != WhichMode::Builtin
            }
        };
        if !admitted {
            continue;
        }
        let text = match candidate {
            RuntimeNameCandidate::WorkspaceVariable => format!("'{name}' is a variable."),
            RuntimeNameCandidate::Builtin(identity) => {
                format!("built-in (RunMat builtin: {})", identity.0)
            }
            RuntimeNameCandidate::Native(native) => canonical_path(&native.path).await,
            RuntimeNameCandidate::RegisteredConstructor(class) => {
                format!("class constructor: {class}")
            }
            RuntimeNameCandidate::RegisteredStaticMethod { resolution, source } => {
                match &resolution.target {
                    crate::dispatcher::RegisteredStaticMethodTarget::Bound(_)
                    | crate::dispatcher::RegisteredStaticMethodTarget::Semantic(_) => {
                        match source
                            .as_ref()
                            .and_then(|source| source.source_file.as_ref())
                        {
                            Some(path) => canonical_path(Path::new(path)).await,
                            None => format!(
                                "static method: {}.{}",
                                resolution.registration.class, resolution.registration.method_name
                            ),
                        }
                    }
                    crate::dispatcher::RegisteredStaticMethodTarget::Builtin(identity) => {
                        format!("built-in (RunMat builtin: {})", identity.0)
                    }
                }
            }
            RuntimeNameCandidate::Semantic { source, .. } => match source {
                Some(source) => match source.source_file.as_ref() {
                    Some(path) => canonical_path(Path::new(path)).await,
                    None => format!("function {}", source.name),
                },
                None => format!("function {name}"),
            },
            RuntimeNameCandidate::Source(file) | RuntimeNameCandidate::File(file) => {
                canonical_path(file.path()).await
            }
            RuntimeNameCandidate::ClassFolder(folder) => {
                format!("class folder: {}", canonical_path(folder.path()).await)
            }
            RuntimeNameCandidate::ClassFile(file) => {
                format!("classdef file: {}", canonical_path(file.path()).await)
            }
            RuntimeNameCandidate::Directory(folder) => canonical_path(folder.path()).await,
        };
        // Identical display paths do not make two executable definitions the
        // same candidate: a retained body and its replacement can share the
        // path while carrying different exact definition authority.
        let distinct_semantic_definition = match candidate {
            RuntimeNameCandidate::Semantic {
                definition_owner: Some(owner),
                ..
            } => seen_semantic_owners.insert(std::sync::Arc::as_ptr(owner)),
            RuntimeNameCandidate::Semantic {
                source: Some(source),
                ..
            } => source
                .binding_origin
                .is_some_and(|origin| seen_semantic_origins.insert(origin)),
            _ => false,
        };
        if !seen.insert(text.clone()) && !distinct_semantic_definition {
            continue;
        }
        results.push(text);
        if !options.all {
            break;
        }
    }
    Ok(results)
}

async fn canonical_path(path: &Path) -> String {
    vfs::canonicalize_async(path)
        .await
        .map(|p| path_to_string(&p))
        .unwrap_or_else(|_| path_to_string(path))
}

fn value_to_string_scalar<F>(value: &Value, error: F) -> crate::BuiltinResult<Option<String>>
where
    F: FnOnce() -> crate::RuntimeError,
{
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(Some(text.as_str().to_owned()))
        }
        Value::CharArray(array) if array.rows == 1 => {
            super::text::decode_utf16_with_error(array.data(), error).map(Some)
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            ))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
#[path = "which/tests.rs"]
pub(crate) mod tests;
