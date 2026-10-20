use super::*;
use crate::builtins::common::fs::expand_user_path;
use std::path::PathBuf;

pub(super) fn preflight_extensions(
    data: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let requested_outputs = crate::output_count::current_output_count();
    if requested_outputs.is_some_and(|count| count > 1) {
        return Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_ARG_CONFIG,
            "writematrix: too many output arguments",
        ));
    }
    if requested_outputs.is_some_and(|count| count > 0) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WRITEMATRIX_BYTES_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(data)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WRITEMATRIX_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    for pair in rest.range(rest.len().min(1)..)?.chunks_exact(2) {
        let Ok(name) = option_name_from_value(&pair[0]) else {
            continue;
        };
        if name.eq_ignore_ascii_case("QuoteStrings") {
            let documented_value =
                value_to_string_scalar(&pair[1], "QuoteStrings").is_ok_and(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "minimal" | "all" | "none"
                    )
                });
            if !documented_value {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &WRITEMATRIX_QUOTE_ALIAS_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
        }
        if name.eq_ignore_ascii_case("DecimalSeparator") || name.eq_ignore_ascii_case("LineEnding")
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &WRITEMATRIX_TEXT_OPTIONS_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    Ok(())
}

pub(super) fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    if let Some(text) = crate::builtins::common::validation::value_text_scalar_string(value) {
        return normalize_path(&text);
    }
    match value {
        Value::CharArray(_) => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_FILENAME,
            "writematrix: expected a 1-by-N character vector for the filename",
        )),
        Value::StringArray(_) => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_FILENAME,
            "writematrix: filename string array inputs must be scalar",
        )),
        other => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_FILENAME,
            format!(
                "writematrix: expected filename as string scalar or character vector, got {other:?}"
            ),
        )),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_FILENAME,
            "writematrix: filename must not be empty",
        ));
    }
    let expanded = expand_user_path(raw, BUILTIN_NAME)
        .map_err(|msg| writematrix_error_with(&WRITEMATRIX_ERROR_FILENAME, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}
