use runmat_value::Value;

use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{select_axes_for_figure, FigureHandle};
use crate::builtins::plotting::{plotting_error, plotting_error_with_source};
use crate::BuiltinResult;

pub type AxesTarget = Option<(FigureHandle, usize)>;
pub type SplitAxesArgs = (AxesTarget, crate::call::BuiltinArguments);

#[path = "axes_target/prepared.rs"]
mod prepared;
pub(crate) use prepared::{
    prepare_axes_call, prepare_graphics_target, PreparedAxesTarget, PreparedGraphicsTarget,
};

pub fn split_leading_axes_arguments(
    args: crate::call::BuiltinArguments,
    builtin: &'static str,
) -> BuiltinResult<SplitAxesArgs> {
    let target = leading_axes_target(args.first(), args.len(), builtin)?;
    let skip = usize::from(target.is_some());
    Ok((target, args.into_range(skip..)?))
}

fn leading_axes_target(
    first: Option<&Value>,
    count: usize,
    builtin: &'static str,
) -> BuiltinResult<AxesTarget> {
    if let Some(first) = first {
        if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(first, builtin) {
            if count == 1 {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: expected data after axes handle"),
                ));
            }
            return Ok(Some((handle, axes_index)));
        }
    }
    Ok(None)
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
