//! Cylinder coordinate grids and statement-form surface publication.

use runmat_builtins::{CYLINDER_ERRORS, CYLINDER_TYPED_N_EXTENSION};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::plotting::common::tensor_to_surface_grid_matlab_xy;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    render_active_plot, select_axes_for_figure, FigureHandle, PlotRenderOptions,
};
use crate::builtins::plotting::style::{parse_surface_style_args, SurfaceStyleDefaults};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[path = "cylinder/output.rs"]
mod output;
use output::{grid_size, make_coordinates};

const NAME: &str = "cylinder";
const DEFAULT_N: usize = 20;

fn error(index: usize, detail: impl AsRef<str>) -> RuntimeError {
    let descriptor = &CYLINDER_ERRORS[index];
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "cylinder",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::cylinder"
)]
async fn cylinder_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    if requested.is_some_and(|count| count > 3) {
        return Err(error(0, "requested more than [X,Y,Z]"));
    }
    let plot = requested == Some(0)
        || (requested.is_none() && crate::output_context::requested_output_count() == Some(0));
    let CylinderCall { target, args } = split_axes(args, plot)?;
    let (radius, n, options) = parse_arguments(args).await?;
    if plot && radius.is_empty() {
        return Err(error(1, "an empty radius profile cannot be plotted"));
    }
    if !plot && !options.is_empty() {
        return Err(error(
            1,
            "Name=Value arguments require statement-form plotting",
        ));
    }
    let style = if plot {
        Some(
            parse_surface_style_args(
                NAME,
                options.view(),
                SurfaceStyleDefaults::new(
                    ColorMap::Parula,
                    ShadingMode::Flat,
                    false,
                    1.0,
                    false,
                    true,
                ),
            )
            .map_err(|err| error(1, err.message))?,
        )
    } else {
        None
    };
    let (x, y, z) = make_coordinates(&radius, n, requested.unwrap_or(1), plot).await?;
    if let Some(style) = style {
        let rows = radius.len();
        let cols = n + 1;
        let x_grid = tensor_to_surface_grid_matlab_xy(x, rows, cols, NAME)
            .map_err(|err| error(3, err.message))?;
        let y_grid =
            tensor_to_surface_grid_matlab_xy(y.expect("plot requests Y"), rows, cols, NAME)
                .map_err(|err| error(3, err.message))?;
        let z_grid =
            tensor_to_surface_grid_matlab_xy(z.expect("plot requests Z"), rows, cols, NAME)
                .map_err(|err| error(3, err.message))?;
        let mut surface = SurfacePlot::from_coordinate_grids(x_grid, y_grid, z_grid)
            .map_err(|err| error(3, err))?;
        style.apply_to_plot(&mut surface);
        if let Some(target) = target {
            select_axes_for_figure(target.figure, target.axes)
                .map_err(|err| error(1, err.to_string()))?;
        }
        let mut surface = Some(surface);
        if let Err(err) = render_active_plot(
            NAME,
            PlotRenderOptions {
                axis_equal: true,
                ..Default::default()
            },
            move |figure, axes| {
                figure.add_surface_plot_on_axes(
                    surface.take().expect("cylinder surface consumed once"),
                    axes,
                );
                Ok(())
            },
        ) {
            if !crate::builtins::plotting::state::presentation_unavailable(&err) {
                return Err(error(3, err.message));
            }
        }
        return Ok(Value::output_list(Vec::new()));
    }
    let mut outputs = vec![Value::Tensor(x)];
    if let Some(y) = y {
        outputs.push(Value::Tensor(y));
    }
    if let Some(z) = z {
        outputs.push(Value::Tensor(z));
    }
    if requested.is_some() {
        Ok(Value::output_list(outputs))
    } else {
        Ok(outputs.remove(0))
    }
}

struct AxesTarget {
    figure: FigureHandle,
    axes: usize,
}

struct CylinderCall {
    target: Option<AxesTarget>,
    args: crate::call::BuiltinArguments,
}

fn split_axes(mut args: crate::call::BuiltinArguments, plot: bool) -> BuiltinResult<CylinderCall> {
    let Some(first) = args.first() else {
        return Ok(CylinderCall { target: None, args });
    };
    if let Ok(PlotHandle::Axes(figure, axes)) = resolve_plot_handle(first, NAME) {
        if !plot {
            return Err(error(1, "axes handles require statement-form plotting"));
        }
        args.pop_front().expect("validated argument prefix");
        return Ok(CylinderCall {
            target: Some(AxesTarget { figure, axes }),
            args,
        });
    }
    Ok(CylinderCall { target: None, args })
}

enum RadiusProfile {
    Constant(f64),
    Vector(Tensor),
}

impl RadiusProfile {
    fn len(&self) -> usize {
        match self {
            Self::Constant(_) => 2,
            Self::Vector(tensor) => tensor.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn at(&self, row: usize) -> f64 {
        match self {
            Self::Constant(value) => *value,
            Self::Vector(tensor) => tensor::tensor_value_f64(tensor, row),
        }
    }
}

async fn parse_arguments(
    mut args: crate::call::BuiltinArguments,
) -> BuiltinResult<(RadiusProfile, usize, crate::call::BuiltinArguments)> {
    let mut radius = RadiusProfile::Constant(1.0);
    if args
        .first()
        .is_some_and(|value| !matches!(value, Value::String(_) | Value::CharArray(_)))
    {
        radius = parse_radius(args.pop_front().expect("radius argument"))?;
    }
    let mut n = DEFAULT_N;
    if args
        .first()
        .is_some_and(|value| !matches!(value, Value::String(_) | Value::CharArray(_)))
    {
        n = parse_resolution(&args.pop_front().expect("resolution argument")).await?;
    }
    if !args.is_empty() && !args.len().is_multiple_of(2) {
        return Err(error(1, "plot property names and values must be paired"));
    }
    grid_size(radius.len(), n)?;
    Ok((radius, n, args))
}

fn parse_radius(value: Value) -> BuiltinResult<RadiusProfile> {
    match value {
        Value::Num(value) => Ok(RadiusProfile::Constant(value)),
        Value::Int(value) => Ok(RadiusProfile::Constant(value.to_f64())),
        Value::Tensor(tensor)
            if tensor.shape().len() <= 2 && (tensor.rows() <= 1 || tensor.cols() <= 1) =>
        {
            if tensor.len() == 1 {
                Ok(RadiusProfile::Constant(tensor::tensor_value_f64(
                    &tensor, 0,
                )))
            } else {
                Ok(RadiusProfile::Vector(tensor))
            }
        }
        Value::GpuTensor(_) => Err(error(
            1,
            "explicit GPU radius profiles are unsupported; cylinder does not gather them",
        )),
        _ => Err(error(1, "radius must be a real scalar or vector")),
    }
}

async fn parse_resolution(value: &Value) -> BuiltinResult<usize> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(&CYLINDER_TYPED_N_EXTENSION, NAME)?;
    }
    let exact = match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    };
    if let Some(integer) = exact {
        let n = integer
            .try_to_usize()
            .ok_or_else(|| error(1, "n must be a positive whole number"))?;
        return (n > 0)
            .then_some(n)
            .ok_or_else(|| error(1, "n must be positive"));
    }
    if matches!(value, Value::GpuTensor(_)) {
        return Err(error(1, "explicit GPU resolution is unsupported"));
    }
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| error(1, err))?
    else {
        return Err(error(1, "n must be a scalar"));
    };
    if !valid_resolution(raw) {
        return Err(error(1, "n must be a finite positive whole number"));
    }
    Ok(raw as usize)
}

fn valid_resolution(value: f64) -> bool {
    value.is_finite() && value >= 1.0 && value.fract() == 0.0 && value < usize::MAX as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    async fn cylinder_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        super::cylinder_builtin(crate::call::BuiltinArguments::from_borrowed(&args, None)?).await
    }

    #[test]
    fn default_and_profile_coordinates_have_height_and_closed_seam() {
        let (x, y, z) = block_on(make_coordinates(
            &RadiusProfile::Constant(1.0),
            20,
            3,
            false,
        ))
        .unwrap();
        let y = y.unwrap();
        let z = z.unwrap();
        assert_eq!(x.shape(), &[2, 21]);
        assert_eq!(y.shape(), &[2, 21]);
        assert_eq!(z.shape(), &[2, 21]);
        let profile = RadiusProfile::Vector(Tensor::new(vec![0.0, 2.0, 0.5], vec![1, 3]).unwrap());
        let (x, y, z) = block_on(make_coordinates(&profile, 4, 3, false)).unwrap();
        let y = y.unwrap();
        let z = z.unwrap();
        for row in 0..3 {
            let i = row;
            let seam = row + 4 * 3;
            assert_eq!(x.materialize_f64()[i], x.materialize_f64()[seam]);
            assert_eq!(y.materialize_f64()[i], y.materialize_f64()[seam]);
            assert!((z.materialize_f64()[i] - row as f64 / 2.0).abs() < 1e-12);
            for col in 0..5 {
                let p = row + col * 3;
                let radius = x.materialize_f64()[p].hypot(y.materialize_f64()[p]);
                assert!((radius - [0.0, 2.0, 0.5][row]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn output_count_and_empty_profile_preserve_shapes() {
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let result = block_on(cylinder_builtin(vec![Value::Num(2.0), Value::Num(4.0)])).unwrap();
        let Value::OutputList(outputs) = result else {
            panic!("expected XYZ")
        };
        assert_eq!(outputs.len(), 3);
        assert!(matches!(&outputs[0], Value::Tensor(tensor) if tensor.shape() == [2,5]));
        let empty = RadiusProfile::Vector(Tensor::new(Vec::new(), vec![1, 0]).unwrap());
        let (x, _, _) = block_on(make_coordinates(&empty, 4, 1, false)).unwrap();
        assert_eq!(x.shape(), &[0, 5]);
    }

    #[test]
    fn invalid_resolution_and_explicit_gpu_are_rejected() {
        for n in [0.0, -1.0, 2.5, f64::INFINITY] {
            assert!(block_on(parse_resolution(&Value::Num(n))).is_err());
        }
        assert!(grid_size(usize::MAX, 1).is_err());
    }

    #[test]
    fn statement_form_publishes_surface_without_outputs() {
        let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
        crate::builtins::plotting::reset_plot_state();
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let result = block_on(cylinder_builtin(vec![Value::Num(2.0), Value::Num(8.0)]))
            .expect("cylinder plot");
        assert!(matches!(result, Value::OutputList(outputs) if outputs.is_empty()));
        let figure = crate::builtins::plotting::clone_figure(
            crate::builtins::plotting::current_figure_handle(),
        )
        .expect("figure");
        assert!(figure.axis_equal);
    }

    #[test]
    fn typed_resolution_requires_extension() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let n = Tensor::new_integer(runmat_value::IntegerStorage::U8(vec![6]), vec![1, 1])
            .expect("typed resolution");
        let err = block_on(cylinder_builtin(vec![Value::Num(1.0), Value::Tensor(n)]))
            .expect_err("typed n is gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:CylinderTypedResolutionExtension")
        );
    }
}
