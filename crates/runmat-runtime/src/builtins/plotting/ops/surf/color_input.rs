use runmat_plot::plots::{NumericPlotData, SurfacePlot};
use runmat_value::Value;

use super::{
    map_surf_invalid_argument, numeric_plot_data_from_value, parse_surface_call_args_matlab_xy,
    plotting_error, BUILTIN_NAME,
};
use crate::BuiltinResult;

pub(super) struct ParsedSurfaceInput {
    pub(super) x: Value,
    pub(super) y: Value,
    pub(super) z: Value,
    pub(super) color: Option<Value>,
    pub(super) rest: Vec<Value>,
}

pub(super) fn parse(mut args: Vec<Value>) -> BuiltinResult<ParsedSurfaceInput> {
    if args.len() >= 4 && args[..4].iter().all(is_color_data) {
        let color = args.remove(3);
        let (x, y, z, rest) = parse_surface_call_args_matlab_xy(args, BUILTIN_NAME)
            .map_err(map_surf_invalid_argument)?;
        return Ok(ParsedSurfaceInput {
            x,
            y,
            z,
            color: Some(color),
            rest,
        });
    }
    if args.len() >= 3 && args[..3].iter().all(is_color_data) {
        let (x, y, z, rest) = parse_surface_call_args_matlab_xy(args, BUILTIN_NAME)
            .map_err(map_surf_invalid_argument)?;
        return Ok(ParsedSurfaceInput {
            x,
            y,
            z,
            color: None,
            rest,
        });
    }
    if args.len() >= 2 && args[..2].iter().all(is_color_data) {
        let z = args.remove(0);
        let color = args.remove(0);
        let (x, y, z, _) = parse_surface_call_args_matlab_xy(vec![z], BUILTIN_NAME)
            .map_err(map_surf_invalid_argument)?;
        return Ok(ParsedSurfaceInput {
            x,
            y,
            z,
            color: Some(color),
            rest: args,
        });
    }
    if args.len() >= 3 && is_property_name(&args[1]) {
        let z = args.remove(0);
        let (x, y, z, _) = parse_surface_call_args_matlab_xy(vec![z], BUILTIN_NAME)
            .map_err(map_surf_invalid_argument)?;
        return Ok(ParsedSurfaceInput {
            x,
            y,
            z,
            color: None,
            rest: args,
        });
    }
    let (x, y, z, rest) =
        parse_surface_call_args_matlab_xy(args, BUILTIN_NAME).map_err(map_surf_invalid_argument)?;
    Ok(ParsedSurfaceInput {
        x,
        y,
        z,
        color: None,
        rest,
    })
}

fn is_color_data(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::GpuTensor(_) | Value::Num(_) | Value::Int(_)
    )
}

fn is_property_name(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::CharArray(_))
}

pub(super) async fn prepare(
    value: Value,
    rows: usize,
    cols: usize,
) -> BuiltinResult<NumericPlotData> {
    let shape = match &value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::GpuTensor(handle) => Some(handle.shape()),
        _ => None,
    };
    if shape.is_some_and(|shape| !matches!(shape, [m, n] | [m, n, 3] if *m == rows && *n == cols)) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "surf: CData must match the surface grid or be an m-by-n-by-3 truecolor array",
        ));
    }
    numeric_plot_data_from_value(&value, BUILTIN_NAME)
        .await
        .map_err(map_surf_invalid_argument)?
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "surf: CData must be numeric"))
}

pub(super) fn apply(surface: &mut SurfacePlot, color: NumericPlotData) -> BuiltinResult<()> {
    surface
        .set_color_source(color)
        .map_err(|error| plotting_error(BUILTIN_NAME, error))
}
