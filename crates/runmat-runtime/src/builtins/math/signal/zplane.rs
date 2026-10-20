//! MATLAB-compatible `zplane` zero-pole plot for digital filters.

use glam::Vec4;
use num_complex::Complex;
use runmat_builtins::{
    BuiltinErrorDescriptor, ROOTS_ERROR_INVALID_INPUT, ZPLANE_ERROR_INTERNAL,
    ZPLANE_ERROR_INVALID_ARGUMENT, ZPLANE_NONFLOATING_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::scatter::MarkerStyle;
use runmat_plot::plots::{LinePlot, LineStyle, ScatterPlot};
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::poly::roots;
use crate::builtins::math::signal::common::{value_to_complex_vector, ComplexVectorInput};
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_arguments};
use crate::builtins::plotting::plotting_error;
use crate::builtins::plotting::state::{render_active_plot, PlotRenderOptions};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "zplane";
const UNIT_CIRCLE_POINTS: usize = 257;
const ZERO_COLOR: Vec4 = Vec4::new(0.10, 0.45, 0.85, 1.0);
const POLE_COLOR: Vec4 = Vec4::new(0.85, 0.18, 0.16, 1.0);
const GUIDE_COLOR: Vec4 = Vec4::new(0.55, 0.55, 0.55, 0.70);

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
    binding_variant = "default",
    builtin_path = "crate::builtins::math::signal::zplane"
)]
async fn zplane_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args).await
}

pub async fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    preflight_obvious_forms(args.view())?;
    // This includes an ambiguous leading axes candidate: complex integer
    // storage cannot encode an axes handle or enter the floating root kernel.
    // Check every authoritative class before exactness can download the first
    // resident integer value for axes decoding.
    for value in &args {
        if crate::builtins::common::validation::is_typed_complex_integer(value) {
            return Err(zplane_error(
                &ZPLANE_ERROR_INVALID_ARGUMENT,
                "typed complex integer input is not supported",
            ));
        }
    }
    if args
        .iter()
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ZPLANE_NONFLOATING_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    // The shared axes decoder accepts typed integer aliases through an f64
    // scalar projection. Certify the first value before that decoder can
    // round a wide handle or a first coefficient.
    if let Some(first) = args.first() {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            first,
            &ZPLANE_NONFLOATING_EXTENSION,
            BUILTIN_NAME,
            "leading input",
        )
        .await
        .map_err(map_invalid_argument)?;
    }
    let (axes_target, args) =
        split_leading_axes_arguments(args, BUILTIN_NAME).map_err(map_invalid_argument)?;
    preflight_zplane_inputs(args.view())?;
    for (index, value) in args.iter().enumerate() {
        crate::builtins::common::validation::reject_typed_complex_integer(value, BUILTIN_NAME)
            .map_err(map_invalid_argument)?;
        if index != 0 || axes_target.is_some() {
            crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                value,
                &ZPLANE_NONFLOATING_EXTENSION,
                BUILTIN_NAME,
                "numeric input",
            )
            .await
            .map_err(map_invalid_argument)?;
        }
    }
    let data = parse_zero_pole_data(args).await?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_invalid_argument)?;
    render_zero_pole_plot(data)
}

/// Reject malformed authoritative metadata before any numeric readback. A
/// three-argument call may be either ax/b/a or z/p/k, so only facts invalid
/// under both readings are checked before resolving the possible axes handle.
fn preflight_obvious_forms(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    match args.len() {
        1 => preflight_sos(args.first().expect("one zplane argument")),
        2 => {
            let first = args.get(0).expect("first zplane argument");
            let second = args.get(1).expect("second zplane argument");
            // Could be ax,SOS. Only prove a coefficient form when the first
            // carrier cannot encode a scalar axes handle.
            if value_shape(first).is_some_and(|shape| {
                shape
                    .iter()
                    .try_fold(1usize, |count, &dim| count.checked_mul(dim))
                    != Some(1)
            }) {
                preflight_vector(first)?;
                preflight_vector(second)?;
            }
            Ok(())
        }
        3 => {
            let first = args.get(0).expect("first zplane argument");
            let second = args.get(1).expect("second zplane argument");
            let third = args.get(2).expect("third zplane argument");
            preflight_vector(second)?;
            preflight_vector(third)?;
            if value_shape(first).is_some_and(|shape| shape.iter().any(|&n| n > 1)) {
                preflight_vector(first)?;
                preflight_gain(third)?;
            }
            Ok(())
        }
        4 => {
            let zeros = args.get(1).expect("zplane zeros argument");
            let poles = args.get(2).expect("zplane poles argument");
            let gain = args.get(3).expect("zplane gain argument");
            preflight_vector(zeros)?;
            preflight_vector(poles)?;
            preflight_gain(gain)
        }
        _ => Ok(()),
    }
}

fn preflight_zplane_inputs(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    match args.len() {
        1 => preflight_sos(args.first().expect("one zplane input")),
        2 => {
            let first = args.get(0).expect("first zplane input");
            let second = args.get(1).expect("second zplane input");
            preflight_vector(first)?;
            preflight_vector(second)
        }
        3 => {
            let zeros = args.get(0).expect("zeros input");
            let poles = args.get(1).expect("poles input");
            let gain = args.get(2).expect("gain input");
            preflight_vector(zeros)?;
            preflight_vector(poles)?;
            preflight_gain(gain)
        }
        _ => Ok(()),
    }
}

fn value_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::ComplexTensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(&array.shape),
        Value::GpuTensor(handle) => Some(handle.shape()),
        _ => None,
    }
}

fn preflight_sos(value: &Value) -> BuiltinResult<()> {
    let supported = matches!(value, Value::Tensor(_) | Value::ComplexTensor(_))
        || matches!(value, Value::GpuTensor(handle) if !runmat_accelerate_api::handle_is_logical(handle));
    if !supported {
        return Err(zplane_error(
            &ZPLANE_ERROR_INVALID_ARGUMENT,
            "SOS input must be a numeric matrix",
        ));
    }
    let shape = value_shape(value).expect("matrix carrier has shape");
    if shape.is_empty() || shape.len() > 2 {
        return Err(zplane_error(
            &ZPLANE_ERROR_INVALID_ARGUMENT,
            "SOS input must be a 2-D matrix",
        ));
    }
    let (_, cols) = sos_dense_dimensions(shape);
    if cols != 6 {
        return Err(zplane_error(
            &ZPLANE_ERROR_INVALID_ARGUMENT,
            "SOS matrix must have exactly six columns",
        ));
    }
    Ok(())
}

fn preflight_vector(value: &Value) -> BuiltinResult<()> {
    if let Some(shape) = value_shape(value) {
        if shape.iter().filter(|&&dimension| dimension > 1).count() > 1 {
            return Err(zplane_error(
                &ZPLANE_ERROR_INVALID_ARGUMENT,
                "filter data must be a vector",
            ));
        }
    }
    Ok(())
}

fn preflight_gain(value: &Value) -> BuiltinResult<()> {
    let supported = matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Complex(_, _)
            | Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::GpuTensor(_)
    );
    if !supported
        || value_shape(value).is_some_and(|shape| {
            shape
                .iter()
                .try_fold(1usize, |count, &dim| count.checked_mul(dim))
                != Some(1)
        })
    {
        return Err(zplane_error(
            &ZPLANE_ERROR_INVALID_ARGUMENT,
            "gain must be a finite numeric scalar",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ZeroPoleData {
    zeros: Vec<Complex<f64>>,
    poles: Vec<Complex<f64>>,
}

async fn parse_zero_pole_data(
    mut args: crate::call::BuiltinArguments,
) -> BuiltinResult<ZeroPoleData> {
    match args.len() {
        1 => zeros_poles_from_sos(args.pop_front().expect("one arg")).await,
        2 => {
            let first = vector_input("first", args.pop_front().expect("first arg")).await?;
            let second = vector_input("second", args.pop_front().expect("second arg")).await?;
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
            let zeros = vector_input("zeros", args.pop_front().expect("zeros arg")).await?;
            let poles = vector_input("poles", args.pop_front().expect("poles arg")).await?;
            validate_gain(args.pop_front().expect("gain arg")).await?;
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
    preflight_vector(&value)?;
    value_to_complex_vector(BUILTIN_NAME, label, value)
        .await
        .map_err(map_invalid_argument)
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
    // Retained L14 math/poly dependency: roots owns companion-matrix policy.
    // A domain-owned shared kernel extraction belongs to that closure, not to
    // a second zplane numerical implementation.
    roots_value_to_complex(roots::roots_value(value).await.map_err(|source| {
        let owner = if source.identifier() == ROOTS_ERROR_INVALID_INPUT.identifier {
            &ZPLANE_ERROR_INVALID_ARGUMENT
        } else {
            &ZPLANE_ERROR_INTERNAL
        };
        zplane_error_with_source(owner, source)
    })?)
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
    preflight_sos(&value)?;
    let value = crate::gather_if_needed_async(&value)
        .await
        .map_err(map_internal)?;
    match value {
        Value::Tensor(tensor) => {
            validate_sos_matrix_shape(tensor.shape(), tensor.len())?;
            let (rows, cols) = sos_dense_dimensions(tensor.shape());
            let values = tensor::tensor_into_values_f64(tensor);
            Ok(ComplexMatrixInput {
                rows,
                cols,
                data: values.into_iter().map(|re| Complex::new(re, 0.0)).collect(),
                is_complex: false,
            })
        }
        Value::ComplexTensor(tensor) => {
            validate_sos_matrix_shape(tensor.shape(), tensor.materialize_f64().len())?;
            let rows = tensor.shape().first().copied().unwrap_or(0);
            let cols = tensor.shape().get(1).copied().unwrap_or(1);
            Ok(ComplexMatrixInput {
                data: tensor::complex_tensor_into_values_complex64(tensor),
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

fn sos_dense_dimensions(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
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

async fn validate_gain(value: Value) -> BuiltinResult<()> {
    preflight_gain(&value)?;
    let value = crate::gather_if_needed_async(&value)
        .await
        .map_err(map_internal)?;
    match &value {
        Value::Num(n) if n.is_finite() => Ok(()),
        Value::Int(_) | Value::Bool(_) => Ok(()),
        Value::Complex(re, im) if re.is_finite() && im.is_finite() => Ok(()),
        Value::Tensor(t)
            if tensor::is_scalar_tensor(t) && tensor::tensor_value_f64(t, 0).is_finite() =>
        {
            Ok(())
        }
        Value::ComplexTensor(t)
            if t.materialize_f64().len() == 1
                && t.materialize_f64()[0].0.is_finite()
                && t.materialize_f64()[0].1.is_finite() =>
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
        Value::Tensor(tensor) => Ok(tensor::tensor_into_values_f64(tensor)
            .into_iter()
            .map(|re| Complex::new(re, 0.0))
            .collect()),
        Value::ComplexTensor(tensor) => Ok(tensor::complex_tensor_into_values_complex64(tensor)),
        Value::Num(n) => Ok(vec![Complex::new(n, 0.0)]),
        other => Err(zplane_error(
            &ZPLANE_ERROR_INTERNAL,
            format!("unexpected roots output {other:?}"),
        )),
    }
}

mod presentation;
use presentation::render_zero_pole_plot;

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

fn zplane_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    source: RuntimeError,
) -> RuntimeError {
    let retry = source.gpu_gather_retry;
    let span = source.span;
    let mut builder = build_runtime_error(format!("{}: {}", error.message, source.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .with_gpu_gather_retry(retry);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    let mut owner = builder.build();
    owner.span = span;
    owner
}

fn map_invalid_argument(err: RuntimeError) -> RuntimeError {
    zplane_error_with_source(&ZPLANE_ERROR_INVALID_ARGUMENT, err)
}

fn map_internal(err: RuntimeError) -> RuntimeError {
    zplane_error_with_source(&ZPLANE_ERROR_INTERNAL, err)
}

#[cfg(test)]
mod tests;
