use super::{
    filewrite_error_with_detail, FILEWRITE_ERROR_INVALID_INPUT, FILEWRITE_ERROR_INVALID_OPTION,
};
use crate::BuiltinResult;
use runmat_builtins::FilewriteOptionName;
use runmat_value::Value;
use std::path::{Path, PathBuf};

pub(super) use runmat_builtins::{FilewriteEncoding as FileEncoding, FilewriteMode as WriteMode};

#[derive(Debug, Clone, Copy)]
pub(super) struct FilewriteOptions {
    pub(super) encoding: FileEncoding,
    pub(super) write_mode: WriteMode,
}

impl Default for FilewriteOptions {
    fn default() -> Self {
        Self {
            encoding: FileEncoding::Auto,
            write_mode: WriteMode::Overwrite,
        }
    }
}
pub(super) fn parse_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<FilewriteOptions> {
    if args.is_empty() {
        return Ok(FilewriteOptions::default());
    }

    let mut options = FilewriteOptions::default();
    let mut idx = 0usize;
    let mut encoding_specified = false;
    let mut write_mode_specified = false;

    if !args.is_empty() && !is_keyword(&args[0])? {
        match encoding_from_value(&args[0]) {
            Ok(enc) => {
                options.encoding = enc;
                encoding_specified = true;
                idx = 1;
            }
            Err(err) => {
                if args.len() == 1 {
                    return Err(err);
                }
            }
        }
    }

    if !(args.len() - idx).is_multiple_of(2) {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            "expected keyword/value argument pairs",
        ));
    }

    while idx < args.len() {
        let key = keyword_name(&args[idx])?;
        let value = &args[idx + 1];
        if FilewriteOptionName::from_label(&key) == Some(FilewriteOptionName::Encoding) {
            if encoding_specified {
                return Err(filewrite_error_with_detail(
                    &FILEWRITE_ERROR_INVALID_OPTION,
                    "duplicate 'Encoding' argument",
                ));
            }
            options.encoding = encoding_from_value(value)?;
            encoding_specified = true;
        } else if FilewriteOptionName::from_label(&key) == Some(FilewriteOptionName::WriteMode) {
            if write_mode_specified {
                return Err(filewrite_error_with_detail(
                    &FILEWRITE_ERROR_INVALID_OPTION,
                    "duplicate 'WriteMode' argument",
                ));
            }
            options.write_mode = write_mode_from_value(value)?;
            write_mode_specified = true;
        } else {
            return Err(filewrite_error_with_detail(
                &FILEWRITE_ERROR_INVALID_OPTION,
                format!("unrecognised option '{}'", key),
            ));
        }
        idx += 2;
    }

    Ok(options)
}

fn is_keyword(value: &Value) -> BuiltinResult<bool> {
    let text = keyword_name(value)?;
    Ok(FilewriteOptionName::from_label(&text).is_some())
}

fn keyword_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                filewrite_error_with_detail(
                    &FILEWRITE_ERROR_INVALID_OPTION,
                    "keyword name contains malformed UTF-16 text",
                )
            })
        }
        Value::CharArray(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            "keyword names must be 1-by-N character vectors",
        )),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        Value::StringArray(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            "keyword inputs must be scalar string arrays",
        )),
        other => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            format!("expected keyword as string scalar or character vector, got {other:?}"),
        )),
    }
}

fn encoding_from_value(value: &Value) -> BuiltinResult<FileEncoding> {
    let label = keyword_name(value)?;
    match FileEncoding::from_label(&label) {
        Some(enc) => Ok(enc),
        None if label.trim().is_empty() => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            "encoding name must not be empty",
        )),
        None => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            format!("unsupported encoding '{}'", label),
        )),
    }
}

fn write_mode_from_value(value: &Value) -> BuiltinResult<WriteMode> {
    let label = keyword_name(value)?;
    match WriteMode::from_label(&label) {
        Some(mode) => Ok(mode),
        None if label.trim().is_empty() => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            "write mode must not be empty",
        )),
        None => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            format!(
                "unsupported write mode '{}'; use 'overwrite' or 'append'",
                label
            ),
        )),
    }
}

pub(super) fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let path = crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                filewrite_error_with_detail(
                    &FILEWRITE_ERROR_INVALID_INPUT,
                    "filename contains malformed UTF-16 text",
                )
            })?;
            normalize_path(&path)
        }
        Value::CharArray(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_INPUT,
            "filename must be a 1-by-N character vector",
        )),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            normalize_path(
                sa.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string"),
            )
        }
        Value::StringArray(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_INPUT,
            "string array filename inputs must be scalar",
        )),
        other => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_INPUT,
            format!("expected filename as string scalar or character vector, got {other:?}"),
        )),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.is_empty() {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_INPUT,
            "filename must not be empty",
        ));
    }
    Ok(Path::new(raw).to_path_buf())
}
