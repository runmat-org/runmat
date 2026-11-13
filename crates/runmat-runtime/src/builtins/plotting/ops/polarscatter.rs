//! MATLAB-compatible `polarscatter` builtin.
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{POLARSCATTER_ERROR_INTERNAL, POLARSCATTER_ERROR_INVALID_ARGUMENT};

use runmat_macros::runtime_builtin;
use runmat_plot::plots::AxesKind;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::line::handles_value;
use super::op_common::polar::{
    evaluate_theta_rho_tensors, is_real_numeric_value, polar_to_cartesian, real_tensor_from_value,
    tensor_cols, tensor_is_vector, tensor_rows, EvaluatedPolarData,
};
use super::op_common::{apply_axes_target, split_leading_axes_arguments};
use super::plotting_error;
use super::scatter::build_scatter_plot_from_args_async;
use super::state::{
    current_figure_handle, register_scatter_handle, render_active_plot, PlotRenderOptions,
};

const BUILTIN_NAME: &str = "polarscatter";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::polarscatter")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "polarscatter",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "polarscatter converts theta/rho inputs to cartesian scatter coordinates before rendering; GPU-resident real inputs are gathered for the conversion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::polarscatter")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "polarscatter",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "polarscatter performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "polarscatter",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::polarscatter"
)]
pub async fn polarscatter_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "polarscatter: too many output arguments",
        ));
    }
    let work =
        super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "polarscatter")?;
    let _arguments = args.fork();
    work.finish(polarscatter_impl(args, &work).await)
}
async fn polarscatter_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<Value> {
    let (axes_target, args) =
        split_leading_axes_arguments(args, BUILTIN_NAME).map_err(map_invalid_argument)?;
    work.check()?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_invalid_argument)?;
    let parsed = parse_polarscatter_args(args)
        .await
        .map_err(map_invalid_argument)?;
    render_polarscatter(work, parsed)
        .await
        .map_err(map_internal)
}

struct ParsedPolarScatter {
    series: Vec<EvaluatedPolarData>,
    size_arg: Option<Value>,
    color_arg: Option<Value>,
    tail_args: crate::call::BuiltinArguments,
}

async fn parse_polarscatter_args(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<ParsedPolarScatter> {
    if args.len() < 2 {
        return Err(polarscatter_err(
            "expected theta and rho data after optional axes handle",
        ));
    }
    if !is_real_numeric_value(&args[0]) || !is_real_numeric_value(&args[1]) {
        return Err(polarscatter_err(
            "theta and rho must be real numeric scalar, vector, or matrix data",
        ));
    }

    let theta = real_tensor_from_value(args[0].clone(), BUILTIN_NAME).await?;
    let rho = real_tensor_from_value(args[1].clone(), BUILTIN_NAME).await?;
    let series = evaluate_theta_rho_tensors(theta, rho, BUILTIN_NAME)?;
    if series.is_empty() || series.iter().any(|data| data.theta.is_empty()) {
        return Err(polarscatter_err("theta and rho data cannot be empty"));
    }

    let mut rest = args.into_range(2..)?;
    let size_arg = if rest.first().is_some_and(is_real_numeric_value) {
        rest.pop_front()
    } else {
        None
    };
    let color_arg = if rest.first().is_some_and(is_real_numeric_value) {
        rest.pop_front()
    } else {
        None
    };

    Ok(ParsedPolarScatter {
        series,
        size_arg,
        color_arg,
        tail_args: rest,
    })
}

async fn render_polarscatter(
    work: &super::coordinate_admission::CoordinateWork<'_>,
    parsed: ParsedPolarScatter,
) -> BuiltinResult<Value> {
    let ParsedPolarScatter {
        series,
        size_arg,
        color_arg,
        tail_args,
    } = parsed;
    let total_series = series.len();
    let mut plots = Vec::with_capacity(total_series);
    let mut max_radius = 1.0_f64;

    for (series_idx, data) in series.iter().enumerate() {
        max_radius = max_radius.max(
            data.rho
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| v.abs())
                .fold(0.0, f64::max),
        );
        let (x, y) = polar_to_cartesian(&data.theta, &data.rho, BUILTIN_NAME)?;
        let mut style_args = Vec::new();
        if let Some(size) = size_arg.as_ref() {
            style_args.push(series_style_value(
                size,
                series_idx,
                total_series,
                data.theta.len(),
                false,
            )?);
        }
        if let Some(color) = color_arg.as_ref() {
            style_args.push(series_style_value(
                color,
                series_idx,
                total_series,
                data.theta.len(),
                true,
            )?);
        }
        style_args.extend(tail_args.iter().cloned());
        let mut plot = build_scatter_plot_from_args_async(x, y, &style_args, BUILTIN_NAME)
            .await
            .map_err(map_invalid_argument)?;
        plot = plot.with_polar_data(data.theta.clone(), data.rho.clone());
        plots.push(plot);
    }

    let limit = max_radius.max(1.0);
    let figure_handle = current_figure_handle();
    let plot_indices_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_slot = std::rc::Rc::clone(&plot_indices_out);
    let mut plots_opt = Some(plots);
    let opts = PlotRenderOptions {
        axis_equal: true,
        grid: true,
        ..Default::default()
    };
    work.check()?;
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes_index| {
        figure.set_axes_kind(axes_index, AxesKind::Polar);
        figure.set_axes_axis_equal(axes_index, true);
        figure.set_axes_limits(axes_index, Some((-limit, limit)), Some((-limit, limit)));
        figure.set_axes_labels(axes_index, "", "");

        let plots = plots_opt
            .take()
            .expect("polarscatter plots consumed exactly once");
        let mut indices = plot_indices_slot.borrow_mut();
        for plot in plots {
            let plot_index = figure.add_scatter_plot_on_axes(plot, axes_index);
            indices.push((axes_index, plot_index));
        }
        Ok(())
    });

    let plot_indices = plot_indices_out.borrow();
    if plot_indices.is_empty() {
        return render_result.map(|_| Value::Num(f64::NAN));
    }
    let handles: Vec<f64> = plot_indices
        .iter()
        .map(|(axes_index, plot_index)| {
            register_scatter_handle(figure_handle, *axes_index, *plot_index)
        })
        .collect();
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handles_value(handles));
        }
        return Err(err);
    }
    Ok(handles_value(handles))
}

fn series_style_value(
    value: &Value,
    series_idx: usize,
    series_count: usize,
    point_count: usize,
    is_color_arg: bool,
) -> BuiltinResult<Value> {
    let Value::Tensor(tensor) = value else {
        return Ok(value.clone());
    };
    if tensor_utils::tensor_element_len(tensor) <= 1 || tensor_is_vector(tensor) {
        return Ok(value.clone());
    }
    let rows = tensor_rows(tensor);
    let cols = tensor_cols(tensor);
    if rows == point_count && cols == series_count {
        let start = series_idx * point_count;
        let indices: Vec<usize> = (start..start + point_count).collect();
        return super::op_common::line_inputs::select_native(
            tensor,
            &indices,
            vec![point_count, 1],
            BUILTIN_NAME,
        )
        .map(Value::Tensor);
    }
    if is_color_arg && (cols == 3 || cols == 4) && rows == point_count {
        return Ok(value.clone());
    }
    Ok(value.clone())
}

fn polarscatter_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped =
        polarscatter_error_with_detail(&POLARSCATTER_ERROR_INVALID_ARGUMENT, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = polarscatter_error_with_detail(&POLARSCATTER_ERROR_INTERNAL, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn polarscatter_err(msg: impl Into<String>) -> RuntimeError {
    plotting_error(BUILTIN_NAME, format!("polarscatter: {}", msg.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    async fn polarscatter_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        super::polarscatter_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::point::marker_area_points2_to_diameter_px;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clone_figure, current_figure_handle, reset_hold_state_for_run, reset_plot_state,
    };
    use runmat_value::{IntegerStorage, NumericDType, Tensor};

    fn tensor(data: &[f64], rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![rows, cols]).expect("polar scatter tensor"))
    }

    fn cleared_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        let tensor = Tensor::new_integer(storage, vec![rows, cols]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn series_style_value_reads_typed_integer_matrix_length_without_mirror() {
        let value = cleared_int_tensor(IntegerStorage::U8(vec![10, 20, 30, 40]), 2, 2);
        let series = series_style_value(&value, 1, 2, 2, false).unwrap();

        let Value::Tensor(tensor) = series else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.materialize_f64(), vec![30.0, 40.0]);
        assert_eq!(tensor.numeric_dtype(), NumericDType::U8);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U8(vec![30, 40]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn polarscatter_preserves_rank_zero_and_rank_one_style_boundaries() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();

        for (size, expected_sizes) in [
            (
                Value::Tensor(Tensor::new(vec![25.0], Vec::new()).unwrap()),
                Some(vec![
                    marker_area_points2_to_diameter_px(25.0),
                    marker_area_points2_to_diameter_px(25.0),
                ]),
            ),
            (
                Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2]).unwrap()),
                Some(vec![
                    marker_area_points2_to_diameter_px(10.0),
                    marker_area_points2_to_diameter_px(20.0),
                ]),
            ),
        ] {
            reset_plot_state();
            reset_hold_state_for_run();
            futures::executor::block_on(polarscatter_builtin(vec![
                tensor(&[0.0, 1.0], 1, 2),
                tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2),
                size,
            ]))
            .expect("polarscatter style rank boundary");

            let figure = clone_figure(current_figure_handle()).expect("figure");
            assert_eq!(figure.plots().count(), 2);
            for plot in figure.plots() {
                let runmat_plot::plots::PlotElement::Scatter(scatter) = plot else {
                    panic!("expected scatter plot");
                };
                let rendered_sizes: Vec<_> = (0..scatter.len())
                    .map(|index| {
                        scatter
                            .per_point_sizes
                            .as_ref()
                            .map_or(scatter.marker_size, |sizes| sizes[index])
                    })
                    .collect();
                assert_eq!(Some(rendered_sizes), expected_sizes);
            }
        }
    }

    #[test]
    fn polarscatter_sets_polar_axes_and_preserves_source_data() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();

        let handle = futures::executor::block_on(polarscatter_builtin(vec![
            tensor(&[0.0, std::f64::consts::FRAC_PI_2], 1, 2),
            tensor(&[1.0, 2.0], 2, 1),
            Value::Num(64.0),
            Value::String("filled".into()),
        ]))
        .expect("polarscatter");
        let Value::Num(handle) = handle else {
            panic!("expected scalar handle");
        };
        let figure = clone_figure(current_figure_handle()).expect("figure");
        let meta = figure.axes_metadata(0).expect("axes metadata");
        assert_eq!(meta.axes_kind, AxesKind::Polar);
        let theta = get_builtin(vec![Value::Num(handle), Value::String("ThetaData".into())])
            .expect("theta data");
        let Value::Tensor(theta) = theta else {
            panic!("expected theta tensor");
        };
        assert_eq!(
            theta.materialize_f64(),
            vec![0.0, std::f64::consts::FRAC_PI_2]
        );
        let r =
            get_builtin(vec![Value::Num(handle), Value::String("RData".into())]).expect("r data");
        let Value::Tensor(r) = r else {
            panic!("expected r tensor");
        };
        assert_eq!(r.materialize_f64(), vec![1.0, 2.0]);
    }

    #[test]
    fn matrix_inputs_return_one_handle_per_column_and_slice_sizes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();

        let out = futures::executor::block_on(polarscatter_builtin(vec![
            tensor(&[0.0, 1.0], 1, 2),
            tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2),
            tensor(&[10.0, 20.0, 30.0, 40.0], 2, 2),
        ]))
        .expect("polarscatter matrix");
        let Value::Tensor(handles) = out else {
            panic!("expected handle vector");
        };
        assert_eq!(handles.materialize_f64().len(), 2);
        let figure = clone_figure(current_figure_handle()).expect("figure");
        assert_eq!(figure.plots().count(), 2);
        let size_vectors: Vec<Vec<f32>> = figure
            .plots()
            .map(|plot| {
                let runmat_plot::plots::PlotElement::Scatter(scatter) = plot else {
                    panic!("expected scatter plot");
                };
                scatter.per_point_sizes.clone().expect("per-point sizes")
            })
            .collect();
        assert_eq!(
            size_vectors[0],
            vec![
                marker_area_points2_to_diameter_px(10.0),
                marker_area_points2_to_diameter_px(20.0)
            ]
        );
        assert_eq!(
            size_vectors[1],
            vec![
                marker_area_points2_to_diameter_px(30.0),
                marker_area_points2_to_diameter_px(40.0)
            ]
        );
    }

    #[test]
    fn matrix_color_argument_slices_per_series_before_rgb_interpretation() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();

        futures::executor::block_on(polarscatter_builtin(vec![
            tensor(&[0.0, 1.0], 1, 2),
            tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
            Value::Num(36.0),
            tensor(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3),
        ]))
        .expect("polarscatter matrix colors");

        let figure = clone_figure(current_figure_handle()).expect("figure");
        let color_values: Vec<Vec<f64>> = figure
            .plots()
            .map(|plot| {
                let runmat_plot::plots::PlotElement::Scatter(scatter) = plot else {
                    panic!("expected scatter plot");
                };
                scatter
                    .color_values
                    .clone()
                    .expect("per-point color values")
            })
            .collect();
        assert_eq!(
            color_values,
            vec![vec![10.0, 20.0], vec![30.0, 40.0], vec![50.0, 60.0]]
        );
    }

    #[test]
    fn setting_theta_or_r_data_updates_cartesian_data() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();

        let handle = futures::executor::block_on(polarscatter_builtin(vec![
            tensor(&[0.0], 1, 1),
            tensor(&[1.0], 1, 1),
        ]))
        .expect("polarscatter");
        let Value::Num(handle) = handle else {
            panic!("expected handle");
        };
        set_builtin(vec![
            Value::Num(handle),
            Value::String("ThetaData".into()),
            Value::Num(std::f64::consts::FRAC_PI_2),
        ])
        .expect("set theta");
        set_builtin(vec![
            Value::Num(handle),
            Value::String("RData".into()),
            Value::Num(2.0),
        ])
        .expect("set r");
        let x =
            get_builtin(vec![Value::Num(handle), Value::String("XData".into())]).expect("x data");
        let y =
            get_builtin(vec![Value::Num(handle), Value::String("YData".into())]).expect("y data");
        let Value::Tensor(x) = x else {
            panic!("expected x tensor");
        };
        let Value::Tensor(y) = y else {
            panic!("expected y tensor");
        };
        assert!(x.materialize_f64()[0].abs() < 1e-12);
        assert!((y.materialize_f64()[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn expanding_polar_data_clears_stale_per_point_style_vectors() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();

        let handle = futures::executor::block_on(polarscatter_builtin(vec![
            tensor(&[0.0], 1, 1),
            tensor(&[1.0], 1, 1),
            Value::Num(25.0),
            tensor(&[7.0], 1, 1),
        ]))
        .expect("polarscatter");
        let Value::Num(handle) = handle else {
            panic!("expected handle");
        };
        set_builtin(vec![
            Value::Num(handle),
            Value::String("ThetaData".into()),
            tensor(&[0.0, std::f64::consts::FRAC_PI_2], 1, 2),
            Value::String("RData".into()),
            tensor(&[1.0, 2.0], 1, 2),
        ])
        .expect("set expanded polar data");

        let figure = clone_figure(current_figure_handle()).expect("figure");
        let Some(runmat_plot::plots::PlotElement::Scatter(scatter)) = figure.plots().next() else {
            panic!("expected scatter plot");
        };
        assert_eq!(scatter.host_xy_f64().unwrap().unwrap().0.len(), 2);
        assert!(scatter.per_point_colors.is_none());
        assert!(scatter.color_values.is_none());
    }

    #[test]
    fn descriptor_covers_matlab_compatible_forms() {
        let labels: Vec<&str> = runmat_builtins::POLARSCATTER_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"ps = polarscatter(theta, rho)"));
        assert!(labels.contains(&"ps = polarscatter(theta, rho, sz, c)"));
        assert!(labels.contains(&"ps = polarscatter(___, 'filled', Name, Value)"));
        assert!(labels.contains(&"ps = polarscatter(pax, ___)"));
    }
}
