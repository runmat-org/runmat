use super::support::validate_load_outputs;
use super::*;

pub(in super::super) async fn execute_openfig(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let selected_runtime = runtime.clone();
    in_selected_runtime(
        selected_runtime.as_ref(),
        execute_openfig_in_context(runtime.as_ref(), requested_outputs, args),
    )
    .await
}

async fn execute_openfig_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    validate_load_outputs(requested_outputs, 1, OPENFIG)?;
    let request = parse_openfig_args(args.view(), OPENFIG)?;
    let selected = FigureFsPath::capture(&request.path).map_err(|source| {
        persistence_error_with_source(
            &ERROR_IO,
            OPENFIG,
            "openfig: unable to capture input path",
            source,
        )
    })?;
    let identity = archive_identity(runtime, &selected, OPENFIG).await?;
    if request.copies == OpenFigCopies::Reuse {
        if let Some(handles) = reusable_figures(&identity) {
            apply_open_visibility(&handles, request.visibility, OPENFIG)?;
            if let Some(handle) = handles.first() {
                super::super::super::state::select_current_figure_if_exists(*handle).map_err(
                    |source| {
                        persistence_error_with_source(
                            &ERROR_IMPORT,
                            OPENFIG,
                            "openfig: unable to select reused figure",
                            source,
                        )
                    },
                )?;
            }
            return Ok(handles_value(&handles));
        }
    }
    let imported = open_figures(runtime, &selected, &request, OPENFIG).await?;
    let value = handles_value(imported.handles());
    let handles = imported.commit();
    remember_figures(runtime, identity, selected.provider_lease(), &handles);
    Ok(value)
}

fn apply_open_visibility(
    handles: &[FigureHandle],
    visibility: Option<bool>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let Some(visibility) = visibility else {
        return Ok(());
    };
    if handles
        .iter()
        .any(|handle| !super::super::super::state::figure_handle_exists(*handle))
    {
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: a reusable figure was closed during admission"),
        ));
    }
    for handle in handles {
        set_figure_visible(*handle, visibility).map_err(|source| {
            persistence_error_with_source(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: unable to update reused figure visibility"),
                source,
            )
        })?;
    }
    Ok(())
}

pub(in super::super) async fn execute_hgload(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<runmat_value::ValueSequence> {
    let selected_runtime = runtime.clone();
    in_selected_runtime(
        selected_runtime.as_ref(),
        execute_hgload_in_context(runtime.as_ref(), requested_outputs, args),
    )
    .await
}

async fn execute_hgload_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<runmat_value::ValueSequence> {
    validate_load_outputs(requested_outputs, 2, HGLOAD)?;
    let (request, overrides) = parse_hgload_args(args.view())?;
    let selected = FigureFsPath::capture(&request.path).map_err(|source| {
        persistence_error_with_source(
            &ERROR_IO,
            HGLOAD,
            "hgload: unable to capture input path",
            source,
        )
    })?;
    let imported = open_figures(runtime, &selected, &request, HGLOAD).await?;
    let old_values = apply_hgload_overrides(imported.handles(), overrides.as_ref())?;
    let handle_value = handles_value(imported.handles());
    let sequence = if let Some(out_count) = requested_outputs {
        let outputs = match out_count {
            0 => Vec::new(),
            1 => vec![handle_value],
            2 => vec![handle_value, old_values],
            _ => {
                return Err(persistence_error(
                    &ERROR_INVALID_ARGUMENT,
                    HGLOAD,
                    "hgload: expected at most two output values",
                ))
            }
        };
        runmat_value::ValueSequence::comma_separated(outputs)
            .map_err(crate::sequence::sequence_error_to_runtime)?
    } else {
        runmat_value::ValueSequence::single(handle_value)
            .map_err(crate::sequence::sequence_error_to_runtime)?
    };
    imported.commit();
    Ok(sequence)
}
