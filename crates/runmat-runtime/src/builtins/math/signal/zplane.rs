//! MATLAB-compatible `zplane` zero-pole plot for digital filters.

use glam::Vec4;
use num_complex::Complex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::scatter::MarkerStyle;
use runmat_plot::plots::{LinePlot, LineStyle, ScatterPlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::poly::roots;
use crate::builtins::math::signal::common::{value_to_complex_vector, ComplexVectorInput};
use crate::builtins::math::signal::type_resolvers::zplane_type;
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_handle};
use crate::builtins::plotting::plotting_error;
use crate::builtins::plotting::state::{render_active_plot, PlotRenderOptions};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "zplane";
const UNIT_CIRCLE_POINTS: usize = 257;
const ZERO_COLOR: Vec4 = Vec4::new(0.10, 0.45, 0.85, 1.0);
const POLE_COLOR: Vec4 = Vec4::new(0.85, 0.18, 0.16, 1.0);
const GUIDE_COLOR: Vec4 = Vec4::new(0.55, 0.55, 0.55, 0.70);

const ZPLANE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Graphics handles for zeros, poles, unit circle, and axes guides.",
}];

const ZPLANE_INPUTS_BA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector or zero locations.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector or pole locations.",
    },
];

const ZPLANE_INPUTS_ZPK: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero locations.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pole locations.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "System gain; accepted for zpk compatibility and ignored for plotting.",
    },
];

const ZPLANE_INPUTS_SOS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sos",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second-order-section matrix with six columns [b0 b1 b2 a0 a1 a2].",
}];

const ZPLANE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "zplane(b, a)",
        inputs: &ZPLANE_INPUTS_BA,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "zplane(z, p, k)",
        inputs: &ZPLANE_INPUTS_ZPK,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "zplane(sos)",
        inputs: &ZPLANE_INPUTS_SOS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = zplane(...)",
        inputs: &ZPLANE_INPUTS_BA,
        outputs: &ZPLANE_OUTPUT,
    },
];

const ZPLANE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZPLANE.INVALID_ARGUMENT",
    identifier: Some("RunMat:zplane:InvalidArgument"),
    when: "Inputs cannot be interpreted as filter coefficients, zero/pole vectors, or SOS rows.",
    message: "zplane: invalid argument",
};

const ZPLANE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZPLANE.INTERNAL",
    identifier: Some("RunMat:zplane:Internal"),
    when: "Internal plotting state or tensor construction fails.",
    message: "zplane: internal operation failed",
};

const ZPLANE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ZPLANE_ERROR_INVALID_ARGUMENT, ZPLANE_ERROR_INTERNAL];

pub const ZPLANE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZPLANE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZPLANE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::zplane")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "zplane",
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
    notes: "zplane gathers coefficient inputs, computes roots on the host, and renders zero/pole markers.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::zplane")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "zplane",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "zplane is a plotting sink and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "zplane",
    category = "math/signal",
    summary = "Plot zeros and poles of a digital filter.",
    keywords = "zplane,zero pole plot,digital filter,signal processing",
    sink = true,
    suppress_auto_output = true,
    type_resolver(zplane_type),
    descriptor(crate::builtins::math::signal::zplane::ZPLANE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::zplane"
)]
async fn zplane_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(args).await
}

pub async fn evaluate(args: Vec<Value>) -> BuiltinResult<Value> {
    let (axes_target, args) =
        split_leading_axes_handle(args, BUILTIN_NAME).map_err(map_invalid_argument)?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_invalid_argument)?;
    let data = parse_zero_pole_data(args).await?;
    render_zero_pole_plot(data)
}

#[derive(Debug, Clone)]
struct ZeroPoleData {
    zeros: Vec<Complex<f64>>,
    poles: Vec<Complex<f64>>,
}

async fn parse_zero_pole_data(args: Vec<Value>) -> BuiltinResult<ZeroPoleData> {
    match args.len() {
        1 => zeros_poles_from_sos(args.into_iter().next().expect("one arg")).await,
        2 => {
            let mut iter = args.into_iter();
            let first = vector_input("first", iter.next().expect("first arg")).await?;
            let second = vector_input("second", iter.next().expect("second arg")).await?;
            if treat_as_explicit_zero_pole_vectors(&first, &second) {
                Ok(ZeroPoleData {
                    zeros: first.data,
                    poles: second.data,
                })
            } else {
                zeros_poles_from_coefficients(first, second).await
            }
        }
        3 => {
            let mut iter = args.into_iter();
            let zeros = vector_input("zeros", iter.next().expect("zeros arg")).await?;
            let poles = vector_input("poles", iter.next().expect("poles arg")).await?;
            validate_gain(&iter.next().expect("gain arg"))?;
            Ok(ZeroPoleData {
                zeros: zeros.data,
                poles: poles.data,
            })
        }
        _ => Err(zplane_error(
            &ZPLANE_ERROR_INVALID_ARGUMENT,
            "expected zplane(b,a), zplane(z,p,k), or zplane(sos)",
        )),
    }
}

async fn vector_input(label: &str, value: Value) -> BuiltinResult<ComplexVectorInput> {
    value_to_complex_vector(BUILTIN_NAME, label, value)
        .await
        .map_err(|err| zplane_error(&ZPLANE_ERROR_INVALID_ARGUMENT, err.message()))
}

fn treat_as_explicit_zero_pole_vectors(
    first: &ComplexVectorInput,
    second: &ComplexVectorInput,
) -> bool {
    // MATLAB disambiguates common row-vector transfer-function coefficients from
    // column-vector zero/pole lists. Scalars remain coefficients unless paired
    // with a non-scalar column vector because `zplane(1, [1 -0.8])` is a common
    // IIR form while `zplane([z1; z2], p)` is an explicit zero/pole form.
    is_non_scalar_column_vector_shape(&first.shape)
        || is_non_scalar_column_vector_shape(&second.shape)
}

fn is_non_scalar_column_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [rows, cols] => *cols == 1 && *rows > 1,
        _ => false,
    }
}

async fn zeros_poles_from_coefficients(
    numerator: ComplexVectorInput,
    denominator: ComplexVectorInput,
) -> BuiltinResult<ZeroPoleData> {
    Ok(ZeroPoleData {
        zeros: roots_from_coefficients(numerator).await?,
        poles: roots_from_coefficients(denominator).await?,
    })
}

async fn roots_from_coefficients(input: ComplexVectorInput) -> BuiltinResult<Vec<Complex<f64>>> {
    let shape = if input.data.is_empty() {
        vec![0, 1]
    } else {
        vec![1, input.data.len()]
    };
    let value = if input.is_complex {
        Value::ComplexTensor(
            ComplexTensor::new(
                input.data.into_iter().map(|z| (z.re, z.im)).collect(),
                shape,
            )
            .map_err(|e| zplane_error(&ZPLANE_ERROR_INTERNAL, e))?,
        )
    } else {
        Value::Tensor(
            Tensor::new(input.data.into_iter().map(|z| z.re).collect(), shape)
                .map_err(|e| zplane_error(&ZPLANE_ERROR_INTERNAL, e))?,
        )
    };
    roots_value_to_complex(roots::roots_value(value).await?)
}

async fn zeros_poles_from_sos(value: Value) -> BuiltinResult<ZeroPoleData> {
    let matrix = complex_matrix(value).await?;
    if matrix.cols != 6 {
        return Err(zplane_error(
            &ZPLANE_ERROR_INVALID_ARGUMENT,
            "SOS matrix must have exactly six columns",
        ));
    }
    let mut zeros = Vec::new();
    let mut poles = Vec::new();
    for row in 0..matrix.rows {
        let numerator = (0..3).map(|col| matrix.at(row, col)).collect::<Vec<_>>();
        let denominator = (3..6).map(|col| matrix.at(row, col)).collect::<Vec<_>>();
        zeros.extend(
            roots_from_coefficients(ComplexVectorInput {
                data: numerator,
                shape: vec![1, 3],
                is_complex: matrix.is_complex,
                gpu_handle: None,
            })
            .await?,
        );
        poles.extend(
            roots_from_coefficients(ComplexVectorInput {
                data: denominator,
                shape: vec![1, 3],
                is_complex: matrix.is_complex,
                gpu_handle: None,
            })
            .await?,
        );
    }
    Ok(ZeroPoleData { zeros, poles })
}

struct ComplexMatrixInput {
    data: Vec<Complex<f64>>,
    rows: usize,
    cols: usize,
    is_complex: bool,
}

impl ComplexMatrixInput {
    fn at(&self, row: usize, col: usize) -> Complex<f64> {
        self.data[row + self.rows * col]
    }
}

async fn complex_matrix(value: Value) -> BuiltinResult<ComplexMatrixInput> {
    let value = crate::gather_if_needed_async(&value)
        .await
        .map_err(|flow| {
            crate::builtins::common::map_control_flow_with_builtin(flow, BUILTIN_NAME)
        })?;
    match value {
        Value::Tensor(tensor) => {
            validate_sos_matrix_shape(&tensor.shape, tensor.data.len())?;
            Ok(ComplexMatrixInput {
                data: tensor
                    .data
                    .into_iter()
                    .map(|re| Complex::new(re, 0.0))
                    .collect(),
                rows: tensor.rows,
                cols: tensor.cols,
                is_complex: false,
            })
        }
        Value::ComplexTensor(tensor) => {
            validate_sos_matrix_shape(&tensor.shape, tensor.data.len())?;
            let rows = tensor.shape.first().copied().unwrap_or(0);
            let cols = tensor.shape.get(1).copied().unwrap_or(1);
            Ok(ComplexMatrixInput {
                data: tensor
                    .data
                    .into_iter()
                    .map(|(re, im)| Complex::new(re, im))
                    .collect(),
                rows,
                cols,
                is_complex: true,
            })
        }
        other => Err(zplane_error(
            &ZPLANE_ERROR_INVALID_ARGUMENT,
            format!("SOS input must be a numeric matrix, got {other:?}"),
        )),
    }
}

fn validate_sos_matrix_shape(shape: &[usize], data_len: usize) -> BuiltinResult<()> {
    let rows = shape.first().copied().unwrap_or(0);
    let cols = shape.get(1).copied().unwrap_or(1);
    if shape.len() > 2 {
        return Err(zplane_error(
            &ZPLANE_ERROR_INVALID_ARGUMENT,
            "SOS input must be a 2-D matrix",
        ));
    }
    if rows.checked_mul(cols) != Some(data_len) {
        return Err(zplane_error(
            &ZPLANE_ERROR_INVALID_ARGUMENT,
            "SOS matrix shape does not match its element count",
        ));
    }
    Ok(())
}

fn validate_gain(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(n) if n.is_finite() => Ok(()),
        Value::Int(_) | Value::Bool(_) => Ok(()),
        Value::Complex(re, im) if re.is_finite() && im.is_finite() => Ok(()),
        Value::Tensor(t) if t.data.len() == 1 && t.data[0].is_finite() => Ok(()),
        Value::ComplexTensor(t)
            if t.data.len() == 1 && t.data[0].0.is_finite() && t.data[0].1.is_finite() =>
        {
            Ok(())
        }
        _ => Err(zplane_error(
            &ZPLANE_ERROR_INVALID_ARGUMENT,
            "gain must be a finite numeric scalar",
        )),
    }
}

fn roots_value_to_complex(value: Value) -> BuiltinResult<Vec<Complex<f64>>> {
    match value {
        Value::Tensor(tensor) => Ok(tensor
            .data
            .into_iter()
            .map(|re| Complex::new(re, 0.0))
            .collect()),
        Value::ComplexTensor(tensor) => Ok(tensor
            .data
            .into_iter()
            .map(|(re, im)| Complex::new(re, im))
            .collect()),
        Value::Num(n) => Ok(vec![Complex::new(n, 0.0)]),
        other => Err(zplane_error(
            &ZPLANE_ERROR_INTERNAL,
            format!("unexpected roots output {other:?}"),
        )),
    }
}

fn render_zero_pole_plot(data: ZeroPoleData) -> BuiltinResult<Value> {
    let mut plots = build_zplane_plots(&data)?;
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let handles_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::<(bool, usize, usize)>::new()));
    let handles_slot = std::rc::Rc::clone(&handles_out);
    let opts = PlotRenderOptions {
        title: "Zero-Pole Plot",
        x_label: "Real Part",
        y_label: "Imaginary Part",
        grid: true,
        axis_equal: true,
    };
    let limits = zplane_limits(&data);
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        figure.set_axes_limits(axes, Some(limits), Some(limits));
        for plot in plots.lines.drain(..) {
            let plot_index = figure.add_line_plot_on_axes(plot, axes);
            handles_slot.borrow_mut().push((true, axes, plot_index));
        }
        for plot in plots.scatters.drain(..) {
            let plot_index = figure.add_scatter_plot_on_axes(plot, axes);
            handles_slot.borrow_mut().push((false, axes, plot_index));
        }
        Ok(())
    });

    let handles = handles_out
        .borrow()
        .iter()
        .map(|(is_line, axes, plot_index)| {
            if *is_line {
                crate::builtins::plotting::state::register_line_handle(
                    figure_handle,
                    *axes,
                    *plot_index,
                )
            } else {
                crate::builtins::plotting::state::register_scatter_handle(
                    figure_handle,
                    *axes,
                    *plot_index,
                )
            }
        })
        .collect::<Vec<_>>();
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if !(lower.contains("plotting is unavailable") || lower.contains("non-main thread")) {
            return Err(map_internal(err));
        }
    }
    handles_value(handles)
}

struct ZPlanePlots {
    lines: Vec<LinePlot>,
    scatters: Vec<ScatterPlot>,
}

fn build_zplane_plots(data: &ZeroPoleData) -> BuiltinResult<ZPlanePlots> {
    let limits = zplane_limits(data);
    let mut lines = vec![
        line_plot(
            unit_circle_x(),
            unit_circle_y(),
            GUIDE_COLOR,
            LineStyle::Dotted,
            "Unit Circle",
        )?,
        line_plot(
            vec![limits.0, limits.1],
            vec![0.0, 0.0],
            GUIDE_COLOR,
            LineStyle::Solid,
            "Real Axis",
        )?,
        line_plot(
            vec![0.0, 0.0],
            vec![limits.0, limits.1],
            GUIDE_COLOR,
            LineStyle::Solid,
            "Imaginary Axis",
        )?,
    ];
    for line in &mut lines[1..] {
        line.set_line_width(0.75);
    }
    let mut scatters = Vec::new();
    if !data.zeros.is_empty() {
        scatters.push(scatter_plot(
            &data.zeros,
            ZERO_COLOR,
            MarkerStyle::Circle,
            "Zeros",
            false,
        )?);
    }
    if !data.poles.is_empty() {
        scatters.push(scatter_plot(
            &data.poles,
            POLE_COLOR,
            MarkerStyle::Cross,
            "Poles",
            false,
        )?);
    }
    Ok(ZPlanePlots { lines, scatters })
}

fn line_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    color: Vec4,
    style: LineStyle,
    label: &str,
) -> BuiltinResult<LinePlot> {
    LinePlot::new(x, y)
        .map(|plot| plot.with_style(color, 1.25, style).with_label(label))
        .map_err(|err| plotting_error(BUILTIN_NAME, format!("zplane: {err}")))
}

fn scatter_plot(
    points: &[Complex<f64>],
    color: Vec4,
    marker: MarkerStyle,
    label: &str,
    filled: bool,
) -> BuiltinResult<ScatterPlot> {
    let x = points.iter().map(|z| z.re).collect::<Vec<_>>();
    let y = points.iter().map(|z| z.im).collect::<Vec<_>>();
    let mut plot = ScatterPlot::new(x, y)
        .map(|plot| plot.with_style(color, 9.0, marker).with_label(label))
        .map_err(|err| plotting_error(BUILTIN_NAME, format!("zplane: {err}")))?;
    plot.filled = filled;
    plot.set_edge_color(color);
    Ok(plot)
}

fn unit_circle_x() -> Vec<f64> {
    (0..UNIT_CIRCLE_POINTS)
        .map(|idx| {
            let theta = std::f64::consts::TAU * idx as f64 / (UNIT_CIRCLE_POINTS - 1) as f64;
            theta.cos()
        })
        .collect()
}

fn unit_circle_y() -> Vec<f64> {
    (0..UNIT_CIRCLE_POINTS)
        .map(|idx| {
            let theta = std::f64::consts::TAU * idx as f64 / (UNIT_CIRCLE_POINTS - 1) as f64;
            theta.sin()
        })
        .collect()
}

fn zplane_limits(data: &ZeroPoleData) -> (f64, f64) {
    let extent = data
        .zeros
        .iter()
        .chain(data.poles.iter())
        .flat_map(|z| [z.re.abs(), z.im.abs()])
        .filter(|v| v.is_finite())
        .fold(1.0_f64, f64::max);
    let pad = (extent * 0.12).max(0.15);
    let limit = (extent + pad).max(1.15);
    (-limit, limit)
}

fn handles_value(handles: Vec<f64>) -> BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![handles_tensor(handles)?]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![handles_tensor(handles)?],
        ));
    }
    handles_tensor(handles)
}

fn handles_tensor(handles: Vec<f64>) -> BuiltinResult<Value> {
    Tensor::new(handles.clone(), vec![1, handles.len()])
        .map(Value::Tensor)
        .map_err(|e| zplane_error(&ZPLANE_ERROR_INTERNAL, e))
}

fn zplane_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", error.message)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        zplane_error(&ZPLANE_ERROR_INVALID_ARGUMENT, err.message())
    }
}

fn map_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        zplane_error(&ZPLANE_ERROR_INTERNAL, err.message())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).unwrap())
    }

    fn col(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("zplane builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| sig.label == "zplane(b, a)"));
    }

    #[test]
    fn row_vectors_are_filter_coefficients() {
        let data = block_on(parse_zero_pole_data(vec![row(&[1.0]), row(&[1.0, -0.8])]))
            .expect("zero-pole data");
        assert!(data.zeros.is_empty());
        assert_eq!(data.poles.len(), 1);
        assert!((data.poles[0].re - 0.8).abs() < 1e-10);
        assert!(data.poles[0].im.abs() < 1e-10);
    }

    #[test]
    fn column_vectors_are_explicit_zero_pole_locations() {
        let data = block_on(parse_zero_pole_data(vec![col(&[0.2, 0.4]), col(&[0.8])]))
            .expect("zero-pole data");
        assert_eq!(data.zeros.len(), 2);
        assert_eq!(data.poles.len(), 1);
        assert!((data.zeros[0].re - 0.2).abs() < 1e-12);
        assert!((data.poles[0].re - 0.8).abs() < 1e-12);
    }

    #[test]
    fn sos_rows_expand_to_zeros_and_poles() {
        let sos =
            Value::Tensor(Tensor::new(vec![1.0, 1.0, 0.0, 1.0, -0.8, 0.0], vec![1, 6]).unwrap());
        let data = block_on(parse_zero_pole_data(vec![sos])).expect("sos");
        assert_eq!(data.zeros.len(), 2);
        assert_eq!(data.poles.len(), 2);
        assert!(data.poles.iter().any(|z| (z.re - 0.8).abs() < 1e-10));
    }

    #[test]
    fn sos_rejects_rank_greater_than_two() {
        let sos =
            Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 0.0); 12], vec![1, 6, 2]).unwrap());
        let err = block_on(parse_zero_pole_data(vec![sos])).expect_err("rank-3 sos rejected");
        assert!(err.message().contains("2-D matrix"));
    }

    #[test]
    fn explicit_zpk_accepts_complex_scalar_gain() {
        let data = block_on(parse_zero_pole_data(vec![
            col(&[0.2, 0.4]),
            col(&[0.8]),
            Value::Complex(1.0, 0.25),
        ]))
        .expect("complex gain accepted");
        assert_eq!(data.zeros.len(), 2);
        assert_eq!(data.poles.len(), 1);
    }

    #[test]
    fn zplane_limits_include_unit_circle_and_points() {
        let limits = zplane_limits(&ZeroPoleData {
            zeros: vec![Complex::new(2.0, -0.5)],
            poles: vec![Complex::new(0.0, -3.0)],
        });
        assert!(limits.0 < -3.0);
        assert!(limits.1 > 3.0);
        assert!((limits.0 + limits.1).abs() < 1e-12);
    }

    #[test]
    fn evaluate_returns_graphics_handles_without_interactive_backend() {
        let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
        let value =
            block_on(evaluate(vec![row(&[1.0]), row(&[1.0, -0.8])])).expect("zplane render");
        let Value::Tensor(handles) = value else {
            panic!("expected handle tensor");
        };
        assert_eq!(handles.shape, vec![1, 4]);
        assert!(handles.data.iter().all(|handle| handle.is_finite()));
    }

    #[test]
    fn evaluate_honors_zero_requested_outputs() {
        let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
        let _out_guard = crate::output_count::push_output_count(Some(0));
        let value =
            block_on(evaluate(vec![row(&[1.0]), row(&[1.0, -0.8])])).expect("zplane render");
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        assert!(values.is_empty());
    }
}
