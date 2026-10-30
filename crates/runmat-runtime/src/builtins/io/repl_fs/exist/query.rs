use super::{exist_error_row, BUILTIN_NAME};
use crate::builtins::common::fs::contains_wildcards;
use crate::builtins::common::path_search::{
    class_file_exists as path_class_file_exists,
    class_folder_candidates as path_class_folder_candidates,
    directory_candidates as path_directory_candidates,
    find_file_with_extensions as path_find_file_with_extensions, path_is_directory,
    CLASS_M_FILE_EXTENSIONS, GENERAL_FILE_EXTENSIONS, LIB_EXTENSIONS, MEX_EXTENSIONS,
    PCODE_EXTENSIONS, SIMULINK_EXTENSIONS, THUNK_EXTENSIONS,
};
use crate::BuiltinResult;
use runmat_builtins::{
    builtin_name_is_known, EXIST_ERROR_INVALID_TYPE, EXIST_ERROR_TYPE_ARG,
    EXIST_SEARCH_TYPE_EXTENSION,
};
use runmat_value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExistQuery {
    Any,
    Var,
    File,
    Dir,
    Builtin,
    Class,
    Mex,
    Pcode,
    Method,
    Handle,
    Simulink,
    Thunk,
    Lib,
    Java,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExistResultKind {
    NotFound,
    Variable,
    File,
    Mex,
    Simulink,
    Builtin,
    Pcode,
    Directory,
    Class,
}

impl ExistResultKind {
    pub(super) fn code(self) -> f64 {
        match self {
            ExistResultKind::NotFound => 0.0,
            ExistResultKind::Variable => 1.0,
            ExistResultKind::File => 2.0,
            ExistResultKind::Mex => 3.0,
            ExistResultKind::Simulink => 4.0,
            ExistResultKind::Builtin => 5.0,
            ExistResultKind::Pcode => 6.0,
            ExistResultKind::Directory => 7.0,
            ExistResultKind::Class => 8.0,
        }
    }
}

pub(super) fn parse_type_argument(value: &Value) -> BuiltinResult<ExistQuery> {
    let text = value_to_string(value)?.ok_or_else(|| exist_error_row(&EXIST_ERROR_TYPE_ARG))?;
    match text.trim() {
        "var" => Ok(ExistQuery::Var),
        "file" => Ok(ExistQuery::File),
        "dir" => Ok(ExistQuery::Dir),
        "builtin" => Ok(ExistQuery::Builtin),
        "class" => Ok(ExistQuery::Class),
        raw => parse_extension_type(raw),
    }
}

fn parse_extension_type(raw: &str) -> BuiltinResult<ExistQuery> {
    let query = match raw.to_ascii_lowercase().as_str() {
        "" => ExistQuery::Any,
        "var" | "variable" => ExistQuery::Var,
        "file" => ExistQuery::File,
        "dir" | "directory" | "folder" => ExistQuery::Dir,
        "builtin" | "built-in" => ExistQuery::Builtin,
        "class" => ExistQuery::Class,
        "mex" => ExistQuery::Mex,
        "pcode" | "p" => ExistQuery::Pcode,
        "handle" => ExistQuery::Handle,
        "method" => ExistQuery::Method,
        "simulink" => ExistQuery::Simulink,
        "thunk" => ExistQuery::Thunk,
        "lib" | "library" => ExistQuery::Lib,
        "java" => ExistQuery::Java,
        _ => return Err(exist_error_row(&EXIST_ERROR_INVALID_TYPE)),
    };
    crate::compatibility::ensure_builtin_extension_enabled(
        &EXIST_SEARCH_TYPE_EXTENSION,
        BUILTIN_NAME,
    )?;
    Ok(query)
}

pub(super) async fn exist_for_query(
    name: &str,
    query: ExistQuery,
) -> BuiltinResult<ExistResultKind> {
    if contains_wildcards(name) {
        return Ok(ExistResultKind::NotFound);
    }

    match query {
        ExistQuery::Any => evaluate_default(name).await,
        ExistQuery::Var => Ok(if variable_exists(name)? {
            ExistResultKind::Variable
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::Builtin => Ok(if builtin_exists(name) {
            ExistResultKind::Builtin
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::Class => Ok(if class_exists(name).await? {
            ExistResultKind::Class
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::Dir => Ok(if directory_exists(name).await? {
            ExistResultKind::Directory
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::File => {
            if directory_exists(name).await? {
                Ok(ExistResultKind::Directory)
            } else {
                Ok(detect_file_kind(name)
                    .await?
                    .unwrap_or(ExistResultKind::NotFound))
            }
        }
        ExistQuery::Mex => Ok(
            if path_find_file_with_extensions(name, MEX_EXTENSIONS, "exist")
                .await
                .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
                .is_some()
            {
                ExistResultKind::Mex
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Pcode => Ok(
            if path_find_file_with_extensions(name, PCODE_EXTENSIONS, "exist")
                .await
                .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
                .is_some()
            {
                ExistResultKind::Pcode
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Method => Ok(if method_exists(name) {
            ExistResultKind::Builtin
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::Simulink => Ok(
            if path_find_file_with_extensions(name, SIMULINK_EXTENSIONS, "exist")
                .await
                .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
                .is_some()
            {
                ExistResultKind::Simulink
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Thunk => Ok(
            if path_find_file_with_extensions(name, THUNK_EXTENSIONS, "exist")
                .await
                .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
                .is_some()
            {
                ExistResultKind::File
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Lib => Ok(
            if path_find_file_with_extensions(name, LIB_EXTENSIONS, "exist")
                .await
                .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
                .is_some()
            {
                ExistResultKind::File
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Java => Ok(ExistResultKind::NotFound),
        ExistQuery::Handle => unreachable!("handle queries handled separately"),
    }
}

async fn evaluate_default(name: &str) -> BuiltinResult<ExistResultKind> {
    if variable_exists(name)? {
        return Ok(ExistResultKind::Variable);
    }
    if builtin_exists(name) {
        return Ok(ExistResultKind::Builtin);
    }
    if directory_exists(name).await? {
        return Ok(ExistResultKind::Directory);
    }
    if class_exists(name).await? {
        return Ok(ExistResultKind::Class);
    }
    if let Some(kind) = detect_file_kind(name).await? {
        return Ok(kind);
    }
    Ok(ExistResultKind::NotFound)
}

pub(super) fn exist_handle(value: &Value) -> ExistResultKind {
    match value {
        Value::HandleObject(handle) => {
            if crate::is_handle_valid(handle) {
                ExistResultKind::Variable
            } else {
                ExistResultKind::NotFound
            }
        }
        Value::Listener(listener) => {
            if listener.valid {
                ExistResultKind::Variable
            } else {
                ExistResultKind::NotFound
            }
        }
        _ => ExistResultKind::NotFound,
    }
}

fn variable_exists(name: &str) -> BuiltinResult<bool> {
    crate::workspace::try_lookup(name)
        .map(|value| value.is_some())
        .map_err(exist_workspace_flow)
}

fn exist_workspace_flow(mut error: crate::RuntimeError) -> crate::RuntimeError {
    error.context = error.context.with_builtin(BUILTIN_NAME);
    error
}

fn builtin_exists(name: &str) -> bool {
    builtin_name_is_known(name)
}

async fn class_exists(name: &str) -> BuiltinResult<bool> {
    if runmat_types::ClassIdentity::new(name)
        .ok()
        .and_then(|identity| crate::class_registry::get_class(&identity))
        .is_some()
    {
        return Ok(true);
    }
    if class_folder_exists(name).await? {
        return Ok(true);
    }
    if class_file_exists(name).await? {
        return Ok(true);
    }
    Ok(false)
}

async fn class_folder_exists(name: &str) -> BuiltinResult<bool> {
    for path in path_class_folder_candidates(name, "exist")
        .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
    {
        if path_is_directory(&path)
            .await
            .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn class_file_exists(name: &str) -> BuiltinResult<bool> {
    path_class_file_exists(name, CLASS_M_FILE_EXTENSIONS, "classdef", "exist")
        .await
        .map_err(|error| error.into_runtime_error(BUILTIN_NAME))
}

fn method_exists(name: &str) -> bool {
    if let Some((class_name, method_name)) = split_method_name(name) {
        runmat_types::ClassIdentity::new(class_name)
            .ok()
            .and_then(|identity| {
                crate::class_registry::lookup_method(
                    &identity,
                    &runmat_types::MethodName::from(method_name.as_str()),
                )
            })
            .is_some()
    } else {
        false
    }
}

async fn directory_exists(name: &str) -> BuiltinResult<bool> {
    for path in path_directory_candidates(name, "exist")
        .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
    {
        if path_is_directory(&path)
            .await
            .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn detect_file_kind(name: &str) -> BuiltinResult<Option<ExistResultKind>> {
    if path_find_file_with_extensions(name, MEX_EXTENSIONS, "exist")
        .await
        .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
        .is_some()
    {
        return Ok(Some(ExistResultKind::Mex));
    }
    if path_find_file_with_extensions(name, PCODE_EXTENSIONS, "exist")
        .await
        .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
        .is_some()
    {
        return Ok(Some(ExistResultKind::Pcode));
    }
    if path_find_file_with_extensions(name, SIMULINK_EXTENSIONS, "exist")
        .await
        .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
        .is_some()
    {
        return Ok(Some(ExistResultKind::Simulink));
    }
    if path_find_file_with_extensions(name, GENERAL_FILE_EXTENSIONS, "exist")
        .await
        .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
        .is_some()
    {
        return Ok(Some(ExistResultKind::File));
    }
    Ok(None)
}

pub(super) fn value_to_string(value: &Value) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(Some(text.as_str().to_owned()))
        }
        Value::CharArray(array)
            if array.rows == 1
                && crate::builtins::common::shape::effective_rank(&array.shape) <= 2 =>
        {
            crate::builtins::io::utf16::decode_utf16_with_error(array.data(), || {
                exist_error_row(&EXIST_ERROR_TYPE_ARG)
            })
            .map(Some)
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

fn split_method_name(name: &str) -> Option<(String, String)> {
    let mut parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let method = parts.pop()?.to_string();
    if method.is_empty() {
        return None;
    }
    let class = parts.join(".");
    if class.is_empty() {
        return None;
    }
    Some((class, method))
}
