use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use super::op_common::current_axes_target;
use super::state::set_view_for_axes;
use crate::builtins::plotting::type_resolvers::get_type;

fn parse_view_target(
    args: &[Value],
) -> crate::BuiltinResult<(
    (crate::builtins::plotting::state::FigureHandle, usize),
    &[Value],
)> {
    if let Some(first) = args.first() {
        if let Ok(handle) =
            crate::builtins::plotting::properties::resolve_plot_handle(first, "view")
        {
            if let crate::builtins::plotting::properties::PlotHandle::Axes(fig, axes) = handle {
                return Ok(((fig, axes), &args[1..]));
            }
        }
    }
    Ok((current_axes_target(), args))
}

fn parse_view_angles(args: &[Value]) -> crate::BuiltinResult<(f32, f32)> {
    match args.len() {
        1 => {
            let tensor = scalar_or_tensor(&args[0])?;
            if tensor.data.len() == 1 {
                match tensor.data[0] as i32 {
                    2 => Ok((0.0, 90.0)),
                    3 => Ok((-37.5, 30.0)),
                    _ => Err(crate::builtins::plotting::plotting_error(
                        "view",
                        "view: expected [az el], view(2), or view(3)",
                    )),
                }
            } else if tensor.data.len() == 2 {
                Ok((tensor.data[0] as f32, tensor.data[1] as f32))
            } else {
                Err(crate::builtins::plotting::plotting_error(
                    "view",
                    "view: expected [az el], view(2), or view(3)",
                ))
            }
        }
        2 => {
            let az = scalar_or_tensor(&args[0])?;
            let el = scalar_or_tensor(&args[1])?;
            if az.data.len() != 1 || el.data.len() != 1 {
                return Err(crate::builtins::plotting::plotting_error(
                    "view",
                    "view: azimuth and elevation must be scalars",
                ));
            }
            Ok((az.data[0] as f32, el.data[0] as f32))
        }
        _ => Err(crate::builtins::plotting::plotting_error(
            "view",
            "view: expected (az, el) or a 2-element vector",
        )),
    }
}

fn scalar_or_tensor(value: &Value) -> crate::BuiltinResult<Tensor> {
    match value {
        Value::Num(v) => Ok(Tensor {
            rows: 1,
            cols: 1,
            shape: vec![1, 1],
            data: vec![*v],
            dtype: runmat_builtins::NumericDType::F64,
        }),
        Value::Int(i) => Ok(Tensor {
            rows: 1,
            cols: 1,
            shape: vec![1, 1],
            data: vec![i.to_f64()],
            dtype: runmat_builtins::NumericDType::F64,
        }),
        other => Tensor::try_from(other)
            .map_err(|e| crate::builtins::plotting::plotting_error("view", format!("view: {e}"))),
    }
}

#[runtime_builtin(
    name = "view",
    category = "plotting",
    summary = "Set or query the current 3-D view angles.",
    keywords = "view,plotting,3d,camera",
    suppress_auto_output = true,
    type_resolver(get_type),
    builtin_path = "crate::builtins::plotting::view"
)]
pub fn view_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (target, rest) = parse_view_target(&args)?;
    if rest.is_empty() {
        let meta = crate::builtins::plotting::state::axes_metadata_snapshot(target.0, target.1)
            .map_err(|err| {
                crate::builtins::plotting::plotting_error_with_source(
                    "view",
                    format!("view: {err}"),
                    err,
                )
            })?;
        let az = meta.view_azimuth_deg.unwrap_or(-37.5) as f64;
        let el = meta.view_elevation_deg.unwrap_or(30.0) as f64;
        return Ok(Value::Tensor(Tensor {
            rows: 1,
            cols: 2,
            shape: vec![1, 2],
            data: vec![az, el],
            dtype: runmat_builtins::NumericDType::F64,
        }));
    }
    let (az, el) = parse_view_angles(rest)?;
    set_view_for_axes(target.0, target.1, az, el).map_err(|err| {
        crate::builtins::plotting::plotting_error_with_source("view", format!("view: {err}"), err)
    })?;
    Ok(Value::Tensor(Tensor {
        rows: 1,
        cols: 2,
        shape: vec![1, 2],
        data: vec![az as f64, el as f64],
        dtype: runmat_builtins::NumericDType::F64,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };

    #[test]
    fn view_sets_axes_local_angles() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let value = view_builtin(vec![Value::Num(45.0), Value::Num(20.0)]).unwrap();
        let t = Tensor::try_from(&value).unwrap();
        assert_eq!(t.data, vec![45.0, 20.0]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.view_azimuth_deg, Some(45.0));
        assert_eq!(meta.view_elevation_deg, Some(20.0));
    }

    #[test]
    fn view_is_subplot_local() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let _ = view_builtin(vec![Value::Num(10.0), Value::Num(15.0)]).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.axes_metadata(0).unwrap().view_azimuth_deg, None);
        assert_eq!(fig.axes_metadata(1).unwrap().view_azimuth_deg, Some(10.0));
    }
}
