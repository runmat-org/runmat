use super::*;

pub(super) fn parse_saveas_args(args: &[Value]) -> BuiltinResult<SaveAsRequest> {
    if args.len() < 2 || args.len() > 3 {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            SAVEAS,
            "expected saveas(fig, filename) or saveas(fig, filename, formattype)",
        ));
    }
    let figure = parent_figure_from_value(&args[0], SAVEAS)?;
    let filename = scalar_string(&args[1], SAVEAS)?;
    validate_filename(&filename, SAVEAS)?;
    let explicit_format = args
        .get(2)
        .map(|value| parse_save_format(&scalar_string(value, SAVEAS)?, SAVEAS))
        .transpose()?;
    let inferred = infer_save_format(&filename, SAVEAS)?;
    let format = match explicit_format {
        Some(explicit) => {
            if let Some(inferred) = inferred {
                if inferred != explicit {
                    return Err(persistence_error(
                        &ERROR_UNSUPPORTED_FORMAT,
                        SAVEAS,
                        format!(
                            "saveas: filename extension '.{}' does not match requested format '{}'",
                            inferred.standard_extension(),
                            explicit.label()
                        ),
                    ));
                }
            }
            explicit
        }
        None => inferred.unwrap_or(SaveFormat::Fig),
    };
    let output_path = output_path_for_format(&filename, format, SAVEAS)?;
    Ok(SaveAsRequest {
        figure,
        output_path,
        format,
    })
}

pub(super) fn parse_savefig_args(
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<SaveFigRequest> {
    if args.is_empty() || args.len() > 3 {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: expected filename, optional handles, and optional version"),
        ));
    }

    if builtin == HGSAVE && args.iter().any(typed_integer_handle_value) {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            "hgsave: typed integer values are not graphics handles",
        ));
    }

    let mut idx = 0usize;
    let figures =
        if args.len() >= 2 && !is_version_token(&args[0]) && !is_probable_filename(&args[0]) {
            idx = 1;
            top_level_figures_from_value(&args[0], builtin)?
        } else {
            vec![current_figure_handle()]
        };

    if idx >= args.len() {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: filename is required"),
        ));
    }
    let filename = scalar_string(&args[idx], builtin)?;
    validate_filename(&filename, builtin)?;
    idx += 1;

    while idx < args.len() {
        validate_version_token(&args[idx], builtin)?;
        idx += 1;
    }

    let output_path = fig_path(&filename, true, builtin)?;
    Ok(SaveFigRequest {
        figures,
        output_path,
    })
}

pub(super) fn typed_integer_handle_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

pub(super) fn parse_openfig_args(
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<OpenFigRequest> {
    if args.is_empty() || args.len() > 3 {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: expected filename, optional copies, and optional visibility"),
        ));
    }
    let filename = scalar_string(&args[0], builtin)?;
    validate_filename(&filename, builtin)?;
    let mut visibility = None;
    let mut saw_copies = false;
    for value in &args[1..] {
        let token = scalar_string(value, builtin)?;
        match token.trim().to_ascii_lowercase().as_str() {
            "new" if !saw_copies => saw_copies = true,
            "reuse" if !saw_copies => {
                return Err(persistence_error(
                    &ERROR_UNSUPPORTED_FORMAT,
                    builtin,
                    format!(
                        "{builtin}: openfig reuse mode requires figure-file reuse tracking, which is not available yet"
                    ),
                ));
            }
            "visible" if visibility.replace(true).is_none() => {}
            "invisible" if visibility.replace(false).is_none() => {}
            "on" if visibility.replace(true).is_none() => {}
            "off" if visibility.replace(false).is_none() => {}
            other => {
                return Err(persistence_error(
                    &ERROR_INVALID_ARGUMENT,
                    builtin,
                    format!("{builtin}: unsupported option '{other}'"),
                ));
            }
        }
    }
    Ok(OpenFigRequest {
        path: fig_path(&filename, false, builtin)?,
        visibility,
    })
}

pub(super) fn parse_hgload_args(
    args: &[Value],
) -> BuiltinResult<(OpenFigRequest, Option<StructValue>)> {
    if args.is_empty() || args.len() > 2 {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            HGLOAD,
            "hgload: expected filename and optional property structure",
        ));
    }
    let request = parse_openfig_args(&args[..1], HGLOAD)?;
    let overrides = match args.get(1) {
        Some(Value::Struct(st)) => Some(st.clone()),
        Some(_) => {
            return Err(persistence_error(
                &ERROR_INVALID_ARGUMENT,
                HGLOAD,
                "hgload: property_structure must be a scalar struct",
            ));
        }
        None => None,
    };
    Ok((request, overrides))
}
