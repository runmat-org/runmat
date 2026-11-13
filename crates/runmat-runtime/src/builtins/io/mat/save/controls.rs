use super::*;

pub(super) fn preflight_known_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for (index, value) in args.iter().enumerate() {
        let Some(token) = value_to_string_scalar(value)? else {
            continue;
        };
        if !token.starts_with('-') {
            continue;
        }
        match runmat_builtins::MatSaveOption::parse(&token) {
            Some(
                runmat_builtins::MatSaveOption::Struct | runmat_builtins::MatSaveOption::Regexp,
            ) if index + 1 == args.len() => {
                let message = if token.eq_ignore_ascii_case("-struct") {
                    "save: '-struct' requires a struct variable name"
                } else {
                    "save: '-regexp' requires at least one pattern"
                };
                return Err(save_error_with(&SAVE_ERROR_INVALID_OPTION, message));
            }
            Some(runmat_builtins::MatSaveOption::Regexp)
                if value_to_string_scalar(
                    args.get(index + 1).expect("validated following argument"),
                )?
                .is_some_and(|next| next.starts_with('-')) =>
            {
                return Err(save_error_with(
                    &SAVE_ERROR_INVALID_OPTION,
                    "save: '-regexp' requires at least one pattern",
                ));
            }
            None => {
                return Err(save_error_with(
                    &SAVE_ERROR_INVALID_OPTION,
                    format!("save: unsupported option '{}'", token.to_ascii_lowercase()),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) struct StructRequest {
    pub(super) source: String,
    pub(super) fields: Option<Vec<String>>,
}

pub(super) struct SaveRequest {
    pub(super) variables: Vec<String>,
    pub(super) structs: Vec<StructRequest>,
    pub(super) regex_patterns: Vec<String>,
    pub(super) append: bool,
}

pub(super) async fn parse_arguments(
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<SaveRequest> {
    let mut variables = Vec::new();
    let mut structs = Vec::new();
    let mut regex_patterns = Vec::new();
    let mut append = false;

    let mut idx = 0;
    while idx < values.len() {
        if let Some(flag) = option_token(&values[idx])? {
            match runmat_builtins::MatSaveOption::parse(&flag) {
                Some(runmat_builtins::MatSaveOption::Append) => append = true,
                Some(runmat_builtins::MatSaveOption::Struct) => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err(save_error_with(
                            &SAVE_ERROR_INVALID_OPTION,
                            "save: '-struct' requires a struct variable name",
                        ));
                    }
                    let struct_names = extract_names(&values[idx]).await?;
                    if struct_names.len() != 1 {
                        return Err(save_error_with(
                            &SAVE_ERROR_INVALID_OPTION,
                            "save: '-struct' requires a single struct variable name",
                        ));
                    }
                    let source = struct_names.into_iter().next().unwrap();
                    idx += 1;
                    let mut field_names = Vec::new();
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let names = extract_names(&values[idx]).await?;
                        if names.is_empty() {
                            break;
                        }
                        field_names.extend(names);
                        idx += 1;
                    }
                    idx -= 1; // compensate for loop increment
                    let fields = if field_names.is_empty() {
                        None
                    } else {
                        Some(field_names)
                    };
                    structs.push(StructRequest { source, fields });
                }
                Some(runmat_builtins::MatSaveOption::Regexp) => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err(save_error_with(
                            &SAVE_ERROR_INVALID_OPTION,
                            "save: '-regexp' requires at least one pattern",
                        ));
                    }
                    let mut patterns = Vec::new();
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let names = extract_names(&values[idx]).await?;
                        if names.is_empty() {
                            return Err(save_error_with(
                                &SAVE_ERROR_INVALID_OPTION,
                                "save: '-regexp' requires pattern strings or character rows",
                            ));
                        }
                        patterns.extend(names);
                        idx += 1;
                    }
                    if patterns.is_empty() {
                        return Err(save_error_with(
                            &SAVE_ERROR_INVALID_OPTION,
                            "save: '-regexp' requires at least one pattern",
                        ));
                    }
                    idx -= 1;
                    regex_patterns.extend(patterns);
                }
                None => {
                    return Err(save_error_with(
                        &SAVE_ERROR_INVALID_OPTION,
                        format!("save: unsupported option '{flag}'"),
                    ));
                }
            }
        } else {
            let names = extract_names(&values[idx]).await?;
            variables.extend(names);
        }
        idx += 1;
    }

    Ok(SaveRequest {
        variables,
        structs,
        regex_patterns,
        append,
    })
}

pub(super) fn option_token(value: &Value) -> BuiltinResult<Option<String>> {
    if let Some(token) = value_to_string_scalar(value)? {
        if token.starts_with('-') {
            return Ok(Some(token.to_ascii_lowercase()));
        }
    }
    Ok(None)
}

async fn extract_names(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(vec![s.as_str().to_owned()]),
        Value::CharArray(ca) => {
            let rows = char_array_rows_as_strings(ca)?;
            if rows.is_empty() && ca.rows > 0 {
                return Err(save_error_with(
                    &SAVE_ERROR_INVALID_ARGUMENT,
                    "save: character arrays used for variable names must contain non-empty rows",
                ));
            }
            Ok(rows)
        }
        Value::StringArray(sa) if sa.data().iter().all(|text| !text.is_missing()) => {
            let mut names = Vec::with_capacity(sa.data().len());
            for s in sa.data() {
                names.push(s.as_str().expect("checked nonmissing string").to_owned());
            }
            Ok(names)
        }
        Value::Cell(ca) => {
            let mut names = Vec::with_capacity(ca.data().len());
            for handle in ca.data() {
                let inner = handle;
                let text = value_to_string_scalar(inner)?.ok_or_else(|| {
                    save_error_with(
                        &SAVE_ERROR_INVALID_ARGUMENT,
                        "save: cell arrays must contain string scalars when specifying variable names",
                    )
                })?;
                names.push(text);
            }
            Ok(names)
        }
        other => {
            // Gather once, then require a string-like scalar to avoid infinite recursion.
            let gathered = gather_if_needed_async(other).await?;
            if let Some(text) = value_to_string_scalar(&gathered)? {
                return Ok(vec![text]);
            }
            Err(save_error_with(
                &SAVE_ERROR_INVALID_ARGUMENT,
                "save: variable names must be strings, character arrays, string arrays, or cell arrays of strings",
            ))
        }
    }
}

fn value_to_string_scalar(value: &Value) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(Some(s.as_str().to_owned())),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                save_error_with(&SAVE_ERROR_INVALID_ARGUMENT, "save: malformed UTF-16 text")
            })
            .map(Some)
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(Some(
            sa.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned(),
        )),
        _ => Ok(None),
    }
}

fn char_array_rows_as_strings(ca: &CharArray) -> BuiltinResult<Vec<String>> {
    let mut rows = Vec::with_capacity(ca.rows);
    for row in 0..ca.rows {
        let start = row * ca.cols;
        let buffer = crate::builtins::io::utf16::decode_utf16_with_error(
            &ca.data()[start..start + ca.cols],
            || save_error_with(&SAVE_ERROR_INVALID_ARGUMENT, "save: malformed UTF-16 text"),
        )?;
        let trimmed = buffer.trim_end_matches([' ', '\0']).to_string();
        if !trimmed.is_empty() {
            rows.push(trimmed);
        }
    }
    Ok(rows)
}

pub(super) fn normalise_path(path: &Value) -> BuiltinResult<PathBuf> {
    let raw = value_to_string_scalar(path)?.ok_or_else(|| {
        save_error_with(
            &SAVE_ERROR_FILENAME,
            "save: filename must be a character vector or string scalar",
        )
    })?;
    let mut path = PathBuf::from(raw);
    if path.extension().is_none() {
        path.set_extension("mat");
    }
    Ok(path)
}
