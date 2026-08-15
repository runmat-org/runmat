//! MATLAB-compatible `polarscatter` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::AxesKind;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::line::handles_value;
use super::op_common::polar::{
    evaluate_theta_rho_tensors, is_real_numeric_value, polar_to_cartesian, real_tensor_from_value,
    tensor_is_vector, EvaluatedPolarData,
};
use super::op_common::{apply_axes_target, split_leading_axes_handle};
use super::plotting_error;
use super::scatter::build_scatter_plot_from_args_async;
use super::state::{
    current_figure_handle, register_scatter_handle, render_active_plot, PlotRenderOptions,
};
use super::style::value_as_string;

const BUILTIN_NAME: &str = "polarscatter";

const POLARSCATTER_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ps",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scatter handle or row-vector of scatter handles.",
}];

const PARAM_THETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "theta",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Angular coordinates in radians.",
};

const PARAM_RHO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "rho",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Radial coordinates.",
};

const PARAM_SZ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("36"),
    description: "Marker area in points squared, as scalar, vector, or matrix.",
};

const PARAM_C: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Marker color as color name, RGB triplet, scalar vector, or RGB matrix.",
};

const PARAM_ARGS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional marker, filled flag, axes handle, and name/value properties.",
};

const INPUT_THETA_RHO: [BuiltinParamDescriptor; 2] = [PARAM_THETA, PARAM_RHO];
const INPUT_THETA_RHO_SZ: [BuiltinParamDescriptor; 3] = [PARAM_THETA, PARAM_RHO, PARAM_SZ];
const INPUT_THETA_RHO_SZ_C: [BuiltinParamDescriptor; 4] =
    [PARAM_THETA, PARAM_RHO, PARAM_SZ, PARAM_C];
const INPUT_ARGS: [BuiltinParamDescriptor; 1] = [PARAM_ARGS];

const POLARSCATTER_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(theta, rho)",
        inputs: &INPUT_THETA_RHO,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(theta, rho, sz)",
        inputs: &INPUT_THETA_RHO_SZ,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(theta, rho, sz, c)",
        inputs: &INPUT_THETA_RHO_SZ_C,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(___, mkr)",
        inputs: &INPUT_ARGS,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(___, 'filled', Name, Value)",
        inputs: &INPUT_ARGS,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(pax, ___)",
        inputs: &INPUT_ARGS,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
];

pub const POLARSCATTER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARSCATTER.INVALID_ARGUMENT",
    identifier: Some("RunMat:polarscatter:InvalidArgument"),
    when: "Input data, size/color/style arguments, or axes targeting are invalid.",
    message: "polarscatter: invalid argument",
};

pub const POLARSCATTER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARSCATTER.INTERNAL",
    identifier: Some("RunMat:polarscatter:Internal"),
    when: "Internal polar scatter construction or rendering fails unexpectedly.",
    message: "polarscatter: internal operation failed",
};

const POLARSCATTER_ERRORS: [BuiltinErrorDescriptor; 2] = [
    POLARSCATTER_ERROR_INVALID_ARGUMENT,
    POLARSCATTER_ERROR_INTERNAL,
];

pub const POLARSCATTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POLARSCATTER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POLARSCATTER_ERRORS,
};

const POLARSCATTER_INTEGER_COORDINATE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "theta and rho",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "R2026a explicitly lists all eight integer classes for theta and rho coordinate arrays and permits any numeric class in selected table variables.",
    }];
const POLARSCATTER_INTEGER_STYLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sz and numeric c",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "R2026a explicitly lists every integer class for marker areas and numeric color data; these values enter explicit renderer metadata boundaries.",
    }];
const POLARSCATTER_INTEGER_SELECTOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "thetavar, rhovar, or ColorVariable indices",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric table-variable indices are documented one-based structural selectors.",
    }];
pub const POLARSCATTER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "ps = polarscatter(integer_theta,integer_rho,___)",
        inputs: &POLARSCATTER_INTEGER_COORDINATE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Authoritative coordinates gather exactly; polar conversion and rendering are explicit floating geometry boundaries.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ps = polarscatter(theta,rho,integer_sz,integer_c,___)",
        inputs: &POLARSCATTER_INTEGER_STYLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Marker sizing, colormap normalization, and renderer buffers are intentional floating boundaries after typed values are parsed from authoritative storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ps = polarscatter(tbl,integer_thetavar,integer_rhovar,___)",
        inputs: &POLARSCATTER_INTEGER_SELECTOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Variable selection remains exact; broader table-form implementation is a general plotting surface concern.",
    },
];

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

fn polarscatter_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Unknown
}

#[runtime_builtin(
    name = "polarscatter",
    category = "plotting",
    summary = "Create scatter charts in polar coordinates.",
    keywords = "polarscatter,polar,scatter,markers,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(polarscatter_type),
    descriptor(crate::builtins::plotting::polarscatter::POLARSCATTER_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::plotting::polarscatter::POLARSCATTER_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::plotting::polarscatter"
)]
pub async fn polarscatter_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (axes_target, args) =
        split_leading_axes_handle(args, BUILTIN_NAME).map_err(map_invalid_argument)?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_invalid_argument)?;
    let parsed = parse_polarscatter_args(args)
        .await
        .map_err(map_invalid_argument)?;
    render_polarscatter(parsed).await.map_err(map_internal)
}

struct ParsedPolarScatter {
    series: Vec<EvaluatedPolarData>,
    size_arg: Option<Value>,
    color_arg: Option<Value>,
    tail_args: Vec<Value>,
}

async fn parse_polarscatter_args(args: Vec<Value>) -> BuiltinResult<ParsedPolarScatter> {
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

    let mut rest = args.into_iter().skip(2).collect::<Vec<_>>();
    let size_arg = if rest.first().is_some_and(is_real_numeric_value) {
        Some(rest.remove(0))
    } else {
        None
    };
    let color_arg = if rest.first().is_some_and(is_real_numeric_value) {
        Some(rest.remove(0))
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

async fn render_polarscatter(parsed: ParsedPolarScatter) -> BuiltinResult<Value> {
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
            ));
        }
        if let Some(color) = color_arg.as_ref() {
            style_args.push(series_style_value(
                color,
                series_idx,
                total_series,
                data.theta.len(),
                true,
            ));
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
        title: "Polar Scatter",
        x_label: "",
        y_label: "",
        axis_equal: true,
        grid: true,
    };
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
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
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
) -> Value {
    let Value::Tensor(tensor) = value else {
        return value.clone();
    };
    if tensor_utils::tensor_element_len(tensor) <= 1 || tensor_is_vector(tensor) {
        return value.clone();
    }
    if tensor.rows == point_count && tensor.cols == series_count {
        let start = series_idx * point_count;
        let indices: Vec<usize> = (start..start + point_count).collect();
        let storage = tensor
            .clone()
            .into_numeric_storage()
            .and_then(|storage| storage.gather(&indices))
            .expect("series style column is valid native storage");
        return Value::Tensor(
            Tensor::from_numeric_storage(storage, vec![point_count, 1])
                .expect("series style column shape"),
        );
    }
    if is_color_arg && (tensor.cols == 3 || tensor.cols == 4) && tensor.rows == point_count {
        return value.clone();
    }
    value.clone()
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
    polarscatter_error_with_detail(&POLARSCATTER_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    polarscatter_error_with_detail(&POLARSCATTER_ERROR_INTERNAL, err.message)
}

fn polarscatter_err(msg: impl Into<String>) -> RuntimeError {
    plotting_error(BUILTIN_NAME, format!("polarscatter: {}", msg.into()))
}

#[allow(dead_code)]
fn is_filled_token(value: &Value) -> bool {
    value_as_string(value)
        .map(|text| text.trim().eq_ignore_ascii_case("filled"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::point::marker_area_points2_to_diameter_px;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clone_figure, current_figure_handle, reset_hold_state_for_run, reset_plot_state,
    };
    use runmat_builtins::{IntegerStorage, NumericDType, Tensor};

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
        let series = series_style_value(&value, 1, 2, 2, false);

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
        let labels: Vec<&str> = POLARSCATTER_DESCRIPTOR
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
