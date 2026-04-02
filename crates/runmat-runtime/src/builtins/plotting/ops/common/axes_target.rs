use runmat_builtins::Value;

use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{select_axes_for_figure, FigureHandle};
use crate::builtins::plotting::{plotting_error, plotting_error_with_source};
use crate::BuiltinResult;

pub fn split_leading_axes_handle(
    args: Vec<Value>,
    builtin: &'static str,
) -> BuiltinResult<(Option<(FigureHandle, usize)>, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, builtin) {
        let rest: Vec<Value> = iter.collect();
        if rest.is_empty() {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: expected data after axes handle"),
            ));
        }
        return Ok((Some((handle, axes_index)), rest));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}

pub fn apply_axes_target(
    target: Option<(FigureHandle, usize)>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let Some((handle, axes_index)) = target else {
        return Ok(());
    };
    select_axes_for_figure(handle, axes_index)
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))
}
