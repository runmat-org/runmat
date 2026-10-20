use super::*;

pub(in super::super) fn parse_saveas_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<SaveAsRequest> {
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
        // The explicit formattype determines the bytes while an existing
        // filename extension remains unchanged.
        Some(explicit) => explicit,
        None => inferred.unwrap_or(SaveFormat::Fig),
    };
    Ok(SaveAsRequest {
        figure,
        output_path: output_path_for_format(&filename, format, SAVEAS)?,
        format,
    })
}

pub(in super::super) fn parse_savefig_args(
    args: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<SaveFigRequest> {
    if args.is_empty() {
        if builtin == SAVEFIG {
            return Ok(SaveFigRequest {
                selection: FigureSaveSelection::Figures(vec![current_figure_handle()]),
                output_path: PathBuf::from("Untitled.fig"),
                profile: FigureArchiveProfile::Default,
            });
        }
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: filename is required"),
        ));
    }
    if args.len() > 3 {
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
    let selection =
        if args.len() >= 2 && !is_version_token(&args[0]) && !is_probable_filename(&args[0]) {
            idx = 1;
            if builtin == HGSAVE {
                FigureSaveSelection::Hierarchy(hierarchy_roots_from_value(&args[0], builtin)?)
            } else {
                FigureSaveSelection::Figures(figure_handles_from_value(&args[0], builtin)?)
            }
        } else {
            FigureSaveSelection::Figures(vec![current_figure_handle()])
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
    let profile = match args.get(idx) {
        Some(value) => {
            idx += 1;
            parse_archive_profile(value, builtin)?
        }
        None => FigureArchiveProfile::Default,
    };
    if idx != args.len() {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: expected at most one version option"),
        ));
    }
    Ok(SaveFigRequest {
        selection,
        output_path: fig_path(&filename, true, builtin)?,
        profile,
    })
}
