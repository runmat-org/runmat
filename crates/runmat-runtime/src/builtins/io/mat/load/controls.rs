use super::*;
use crate::gather_if_needed_async;

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
        match runmat_builtins::MatLoadOption::parse(&token) {
            Some(runmat_builtins::MatLoadOption::Regexp)
                if index + 1 == args.len()
                    || value_to_string_scalar(
                        args.get(index + 1).expect("validated following argument"),
                    )?
                    .is_some_and(|next| next.starts_with('-')) =>
            {
                return Err(load_error_with(
                    &LOAD_ERROR_INVALID_OPTION,
                    "load: '-regexp' requires at least one pattern",
                ));
            }
            None => {
                return Err(load_error_with(
                    &LOAD_ERROR_INVALID_OPTION,
                    format!("load: unsupported option '{}'", token.to_ascii_lowercase()),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) struct ParsedInvocation {
    pub(super) path_value: Option<Value>,
    pub(super) path_was_default: bool,
    pub(super) variables: Vec<String>,
    pub(super) regex_tokens: Vec<String>,
}

pub(super) async fn parse_invocation(
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ParsedInvocation> {
    let mut path_value = None;
    let mut path_was_default = false;
    let mut variables = Vec::new();
    let mut regex_tokens = Vec::new();
    let mut idx = 0usize;
    while idx < values.len() {
        if let Some(flag) = option_token(&values[idx])? {
            match runmat_builtins::MatLoadOption::parse(&flag) {
                Some(runmat_builtins::MatLoadOption::Mat) => {
                    idx += 1;
                    continue;
                }
                Some(runmat_builtins::MatLoadOption::Regexp) => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err(load_error_with(
                            &LOAD_ERROR_INVALID_OPTION,
                            "load: '-regexp' requires at least one pattern",
                        ));
                    }
                    let prior_patterns = regex_tokens.len();
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let names = extract_names(&values[idx]).await?;
                        if names.is_empty() {
                            return Err(load_error_with(
                                &LOAD_ERROR_INVALID_OPTION,
                                "load: '-regexp' requires non-empty pattern strings",
                            ));
                        }
                        regex_tokens.extend(names);
                        idx += 1;
                    }
                    if regex_tokens.len() == prior_patterns {
                        return Err(load_error_with(
                            &LOAD_ERROR_INVALID_OPTION,
                            "load: '-regexp' requires at least one pattern",
                        ));
                    }
                    continue;
                }
                None => {
                    return Err(load_error_with(
                        &LOAD_ERROR_INVALID_OPTION,
                        format!("load: unsupported option '{flag}'"),
                    ));
                }
            }
        } else {
            if path_value.is_none() {
                path_value = Some(values[idx].clone());
                idx += 1;
                continue;
            }
            let names = extract_names(&values[idx]).await?;
            variables.extend(names);
            idx += 1;
        }
    }

    if path_value.is_none() {
        path_was_default = true;
    }

    Ok(ParsedInvocation {
        path_value,
        path_was_default,
        variables,
        regex_tokens,
    })
}

pub(super) fn normalise_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw = value_to_string_scalar(value)?.ok_or_else(|| {
        load_error_with(
            &LOAD_ERROR_FILENAME,
            "load: filename must be a character vector or string scalar",
        )
    })?;
    let mut path = PathBuf::from(raw);
    if path.extension().is_none() {
        path.set_extension("mat");
    }
    Ok(path)
}

pub(super) fn select_variables(
    entries: &[(String, Value)],
    request: &LoadRequest,
) -> BuiltinResult<Vec<(String, Value)>> {
    if request.variables.is_empty() && request.regex_patterns.is_empty() {
        return Ok(entries.to_vec());
    }

    let mut by_name: HashMap<&str, &Value> = HashMap::with_capacity(entries.len());
    for (name, value) in entries {
        by_name.insert(name, value);
    }

    let mut selected = Vec::new();

    for name in &request.variables {
        let value = by_name.get(name.as_str()).ok_or_else(|| {
            load_error_with(
                &LOAD_ERROR_SELECTION,
                format!("load: variable '{name}' was not found in the file"),
            )
        })?;
        insert_or_replace(&mut selected, name, (*value).clone());
    }

    if !request.regex_patterns.is_empty() {
        let mut matched = 0usize;
        for (name, value) in entries {
            if request
                .regex_patterns
                .iter()
                .any(|regex| regex.is_match(name))
            {
                matched += 1;
                insert_or_replace(&mut selected, name, value.clone());
            }
        }
        if matched == 0 && request.variables.is_empty() {
            return Err(load_error_with(
                &LOAD_ERROR_SELECTION,
                "load: no variables matched '-regexp' patterns",
            ));
        }
    }

    if selected.is_empty() {
        return Err(load_error_with(
            &LOAD_ERROR_SELECTION,
            "load: no variables selected",
        ));
    }

    Ok(selected)
}

fn insert_or_replace(selected: &mut Vec<(String, Value)>, name: &str, value: Value) {
    if let Some(entry) = selected.iter_mut().find(|(existing, _)| existing == name) {
        entry.1 = value;
    } else {
        selected.push((name.to_string(), value));
    }
}

fn option_token(value: &Value) -> BuiltinResult<Option<String>> {
    if let Some(token) = value_to_string_scalar(value)? {
        if token.starts_with('-') {
            return Ok(Some(token.to_ascii_lowercase()));
        }
    }
    Ok(None)
}

#[async_recursion::async_recursion(?Send)]
async fn extract_names(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(vec![s.as_str().to_owned()]),
        Value::CharArray(ca) => char_array_rows_as_strings(ca),
        Value::StringArray(sa) if sa.data().iter().all(|text| !text.is_missing()) => Ok(sa
            .data()
            .iter()
            .map(|text| text.as_str().expect("checked nonmissing string").to_owned())
            .collect()),
        Value::Cell(ca) => {
            let mut names = Vec::with_capacity(ca.data().len());
            for handle in ca.data() {
                let inner = handle;
                let text = value_to_string_scalar(inner)?.ok_or_else(|| {
                    load_error(
                        "load: cell arrays used for variable selection must contain string scalars",
                    )
                })?;
                names.push(text);
            }
            Ok(names)
        }
        other => {
            let gathered = gather_if_needed_async(other).await?;
            extract_names(&gathered).await
        }
    }
}

fn value_to_string_scalar(value: &Value) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(Some(s.as_str().to_owned())),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                load_error_with(&LOAD_ERROR_INVALID_ARGUMENT, "load: malformed UTF-16 text")
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
    for r in 0..ca.rows {
        let start = r * ca.cols;
        let row = crate::builtins::io::utf16::decode_utf16_with_error(
            &ca.data()[start..start + ca.cols],
            || load_error_with(&LOAD_ERROR_INVALID_ARGUMENT, "load: malformed UTF-16 text"),
        )?;
        let trimmed = row.trim_end_matches([' ', '\0']).to_string();
        rows.push(trimmed);
    }
    Ok(rows)
}
