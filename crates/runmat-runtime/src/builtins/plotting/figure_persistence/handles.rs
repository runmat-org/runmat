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

pub(super) fn figure_handles_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<Vec<FigureHandle>> {
    resolve_handle_array(value, builtin)?
        .into_iter()
        .map(|handle| match handle {
            PlotHandle::Figure(figure) => Ok(figure),
            _ => Err(persistence_error(
                &ERROR_INVALID_ARGUMENT,
                builtin,
                format!("{builtin}: expected a Figure handle"),
            )),
        })
        .collect()
}

pub(super) fn hierarchy_roots_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<Vec<PlotHandle>> {
    let roots = resolve_handle_array(value, builtin)?;
    for (index, root) in roots.iter().enumerate() {
        for other in &roots[index + 1..] {
            if hierarchy_ancestor(root, other) || hierarchy_ancestor(other, root) {
                return Err(persistence_error(
                    &ERROR_INVALID_ARGUMENT,
                    builtin,
                    format!(
                        "{builtin}: a handle array cannot contain ancestor and descendant graphics objects"
                    ),
                ));
            }
        }
    }
    Ok(roots)
}

fn resolve_handle_array(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<PlotHandle>> {
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
            let mut handles = Vec::with_capacity(values.len());
            for &handle in values.iter() {
                let value = Value::Num(handle);
                let resolved = resolve_plot_handle(&value, builtin)?;
                if matches!(resolved, PlotHandle::Root) {
                    return Err(root_not_persistable(builtin));
                }
                handles.push(resolved);
            }
            Ok(handles)
        }
        _ => {
            let handle = resolve_plot_handle(value, builtin)?;
            if matches!(handle, PlotHandle::Root) {
                return Err(root_not_persistable(builtin));
            }
            Ok(vec![handle])
        }
    }
}

fn root_not_persistable(builtin: &'static str) -> RuntimeError {
    persistence_error(
        &ERROR_INVALID_ARGUMENT,
        builtin,
        format!("{builtin}: root handle cannot be saved"),
    )
}

fn hierarchy_ancestor(parent: &PlotHandle, child: &PlotHandle) -> bool {
    match (parent, child) {
        (PlotHandle::Figure(parent), child) => {
            handle_location(child).is_some_and(|(figure, _, depth)| figure == *parent && depth > 0)
        }
        (PlotHandle::Axes(parent_figure, parent_axes), child) => handle_location(child)
            .is_some_and(|(figure, axes, depth)| {
                figure == *parent_figure && axes == Some(*parent_axes) && depth > 1
            }),
        _ => false,
    }
}

fn handle_location(handle: &PlotHandle) -> Option<(FigureHandle, Option<usize>, u8)> {
    match handle {
        PlotHandle::Root => None,
        PlotHandle::Figure(figure) => Some((*figure, None, 0)),
        PlotHandle::Axes(figure, axes) => Some((*figure, Some(*axes), 1)),
        PlotHandle::Ruler(figure, axes, _)
        | PlotHandle::Text(figure, axes, _)
        | PlotHandle::Legend(figure, axes) => Some((*figure, Some(*axes), 2)),
        PlotHandle::PlotChild(_, child) => {
            let (figure, axes) = child.figure_axes();
            Some((figure, Some(axes), 2))
        }
    }
}

pub(super) fn selection_parent_figures(selection: &FigureSaveSelection) -> Vec<FigureHandle> {
    let roots = match selection {
        FigureSaveSelection::Figures(figures) => return figures.clone(),
        FigureSaveSelection::Hierarchy(roots) => roots,
    };
    let mut seen = HashSet::with_capacity(roots.len());
    let mut figures = Vec::with_capacity(roots.len());
    for root in roots {
        let figure = match root {
            PlotHandle::Figure(figure)
            | PlotHandle::Axes(figure, _)
            | PlotHandle::Ruler(figure, _, _)
            | PlotHandle::Text(figure, _, _)
            | PlotHandle::Legend(figure, _) => *figure,
            PlotHandle::PlotChild(_, child) => child.figure_axes().0,
            PlotHandle::Root => continue,
        };
        if seen.insert(figure) {
            figures.push(figure);
        }
    }
    figures
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
