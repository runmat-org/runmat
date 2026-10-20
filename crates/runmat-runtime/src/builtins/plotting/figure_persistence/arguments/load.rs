use super::*;

pub(in super::super) fn parse_openfig_args(
    args: crate::call::FunctionArgumentView<'_>,
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
    let mut copies = OpenFigCopies::New;
    for value in args.iter().skip(1) {
        let token = scalar_string(value, builtin)?;
        match token.trim().to_ascii_lowercase().as_str() {
            "new" if !saw_copies => {
                saw_copies = true;
                copies = OpenFigCopies::New;
            }
            "reuse" if !saw_copies => {
                saw_copies = true;
                copies = OpenFigCopies::Reuse;
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
        copies,
        visibility,
    })
}

pub(in super::super) fn parse_hgload_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(OpenFigRequest, Option<StructValue>)> {
    if args.is_empty() || args.len() > 2 {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            HGLOAD,
            "hgload: expected filename and optional property structure",
        ));
    }
    let request = parse_openfig_args(args.range(..1)?, HGLOAD)?;
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
