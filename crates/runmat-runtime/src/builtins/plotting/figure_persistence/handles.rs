use super::*;

pub(super) fn parent_figure_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<FigureHandle> {
    match resolve_plot_handle(value, builtin)? {
        PlotHandle::Figure(handle) => Ok(handle),
        PlotHandle::Axes(handle, _) => Ok(handle),
        PlotHandle::Ruler(handle, _, _) => Ok(handle),
        PlotHandle::Text(handle, _, _) => Ok(handle),
        PlotHandle::Legend(handle, _) => Ok(handle),
        PlotHandle::PlotChild(_, state) => Ok(state.figure_axes().0),
        PlotHandle::Root => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: root handle cannot be saved as a figure"),
        )),
    }
}

pub(super) fn top_level_figures_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<Vec<FigureHandle>> {
    match value {
        Value::Tensor(tensor) => {
            let values = tensor_utils::tensor_values_f64_cow(tensor);
            if values.is_empty() {
                return Err(persistence_error(
                    &ERROR_INVALID_ARGUMENT,
                    builtin,
                    format!("{builtin}: figure handle array cannot be empty"),
                ));
            }
            let mut seen = HashSet::with_capacity(values.len());
            let mut figures = Vec::with_capacity(values.len());
            for &handle in values.iter() {
                let value = Value::Num(handle);
                let figure = parent_figure_from_value(&value, builtin)?;
                if seen.insert(figure) {
                    figures.push(figure);
                }
            }
            Ok(figures)
        }
        _ => Ok(vec![parent_figure_from_value(value, builtin)?]),
    }
}

pub(super) fn is_probable_filename(value: &Value) -> bool {
    scalar_string(value, SAVEFIG).is_ok()
}

pub(super) fn is_version_token(value: &Value) -> bool {
    scalar_string(value, SAVEFIG)
        .map(|text| {
            matches!(
                text.trim().to_ascii_lowercase().as_str(),
                "-v6" | "v6" | "-v7" | "v7" | "-v7.3" | "v7.3" | "compact"
            )
        })
        .unwrap_or(false)
}

pub(super) fn validate_version_token(value: &Value, builtin: &'static str) -> BuiltinResult<()> {
    let token = scalar_string(value, builtin)?;
    match token.trim().to_ascii_lowercase().as_str() {
        "-v6" | "v6" | "-v7" | "v7" | "-v7.3" | "v7.3" | "compact" => Ok(()),
        other => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: unsupported FIG version option '{other}'"),
        )),
    }
}
