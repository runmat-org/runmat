//! MATLAB-compatible legacy `interp1q` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::Tensor;

use super::pp::{
    evaluate_linear_or_nearest, query_points, series_from_values, Extrapolation, InterpMethod,
};

const NAME: &str = "interp1q";

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Linearly interpolated query values.",
}];

const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Strictly increasing sample locations.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query locations.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Vq = interp1q(X, V, Xq)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1Q.INVALID_INPUT",
    identifier: Some("RunMat:interp1q:InvalidInput"),
    when:
        "Arguments cannot be converted to the legacy one-dimensional linear interpolation domain.",
    message: "interp1q: invalid input",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1Q.INTERNAL",
    identifier: Some("RunMat:interp1q:Internal"),
    when: "Interpolation evaluation or output construction fails unexpectedly.",
    message: "interp1q: internal interpolation failure",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const INTERP1Q_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTERP1Q_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "interp1q-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interp1q with typed-integer X or V data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Interp1qIntegerDataExtension"),
};
const INTERP1Q_INTEGER_QUERY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "interp1q-integer-query",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interp1q with typed-integer query coordinates is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Interp1qIntegerQueryExtension"),
};
const INTERP1Q_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "interp1q-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interp1q with logical numeric input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Interp1qLogicalInputExtension"),
};
const INTERP1Q_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "interp1q-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interp1q with resident gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Interp1qGpuInputExtension"),
};
pub const INTERP1Q_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    INTERP1Q_INTEGER_DATA_EXTENSION,
    INTERP1Q_INTEGER_QUERY_EXTENSION,
    INTERP1Q_LOGICAL_INPUT_EXTENSION,
    INTERP1Q_GPU_INPUT_EXTENSION,
];

const INTERP1Q_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X or V",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer samples must be exactly representable before the binary64 interpolation boundary.",
    }];
const INTERP1Q_INTEGER_QUERY_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Xq",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer query coordinates must be exactly representable before interpolation.",
    }];
pub const INTERP1Q_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = interp1q(integer_X_or_V, V_or_X, Xq)",
        inputs: &INTERP1Q_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat retains this useful legacy extension through one checked binary64 boundary; MATLAB-compatible modes accept only documented single/double data.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = interp1q(X, V, integer_Xq)",
        inputs: &INTERP1Q_INTEGER_QUERY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer queries are checked before conversion; resident input is independently gated because interp1q has no documented gpuArray surface.",
    },
];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::interp1q")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("interpolation-1d-linear"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Legacy interp1q gathers GPU inputs and evaluates through the CPU linear interpolation path.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::interpolation::interp1q"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Interpolation is a runtime sink.",
};

fn interp1q_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.get(2) {
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        _ => Type::tensor(),
    }
}

#[runtime_builtin(
    name = "interp1q",
    category = "math/interpolation",
    summary = "Legacy quick one-dimensional linear interpolation.",
    keywords = "interp1q,interpolation,linear,legacy",
    accel = "sink",
    sink = true,
    type_resolver(interp1q_type),
    descriptor(crate::builtins::math::interpolation::interp1q::INTERP1Q_DESCRIPTOR),
    extensions(crate::builtins::math::interpolation::interp1q::INTERP1Q_EXTENSIONS),
    integer_capabilities(
        crate::builtins::math::interpolation::interp1q::INTERP1Q_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::math::interpolation::interp1q"
)]
async fn interp1q_builtin(x: Value, v: Value, xq: Value) -> BuiltinResult<Value> {
    preflight_interp1q(&x, &v, &xq).await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&x, "interp1q")?;
    crate::builtins::common::validation::reject_typed_complex_integer(&v, "interp1q")?;
    crate::builtins::common::validation::reject_typed_complex_integer(&xq, "interp1q")?;
    let x = gather_interp_input(x).await?;
    let v = gather_interp_input(v).await?;
    let xq = gather_interp_input(xq).await?;
    if value_is_complex(&x) || value_is_complex(&v) || value_is_complex(&xq) {
        let series = ComplexSeries::from_values(x, v).await?;
        let query = real_query_points(xq).await?;
        return evaluate_complex_linear(&series, &query);
    }
    let series = series_from_values(x, v, NAME)
        .await
        .map_err(|err| wrap_error(err, &ERROR_INVALID_INPUT))?;
    let query = query_points(xq, NAME)
        .await
        .map_err(|err| wrap_error(err, &ERROR_INVALID_INPUT))?;
    let result = evaluate_linear_or_nearest(
        &series,
        &query,
        InterpMethod::Linear,
        &Extrapolation::Nan,
        NAME,
    )
    .map_err(|err| wrap_error(err, &ERROR_INTERNAL))?;
    reshape_matrix_output(result, &series, &query)
}

async fn preflight_interp1q(x: &Value, v: &Value, xq: &Value) -> BuiltinResult<()> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64_async, value_has_logical_class,
        value_has_native_integer_class,
    };
    for value in [x, v] {
        if value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTERP1Q_INTEGER_DATA_EXTENSION,
                NAME,
            )?;
            if !crate::builtins::common::validation::is_typed_complex_integer(value)
                && !native_integer_value_is_exact_f64_async(value).await?
            {
                return Err(error_with_detail(
                    &ERROR_INVALID_INPUT,
                    "integer sample data must be exactly representable as double",
                ));
            }
        }
    }
    if value_has_native_integer_class(xq) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTERP1Q_INTEGER_QUERY_EXTENSION,
            NAME,
        )?;
        if !crate::builtins::common::validation::is_typed_complex_integer(xq)
            && !native_integer_value_is_exact_f64_async(xq).await?
        {
            return Err(error_with_detail(
                &ERROR_INVALID_INPUT,
                "integer query coordinates must be exactly representable as double",
            ));
        }
    }
    if [x, v, xq]
        .into_iter()
        .any(|value| value_has_logical_class(value))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTERP1Q_LOGICAL_INPUT_EXTENSION,
            NAME,
        )?;
    }
    if [x, v, xq]
        .into_iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTERP1Q_GPU_INPUT_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

fn wrap_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(err.message().to_string()).with_builtin(NAME);
    if let Some(identifier) = err.identifier().or(fallback.identifier) {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn reshape_matrix_output(
    result: Value,
    series: &super::pp::NumericSeries,
    query: &super::pp::QueryPoints,
) -> BuiltinResult<Value> {
    if series.series <= 1 {
        return Ok(result);
    }
    let Value::Tensor(tensor) = result else {
        return Ok(result);
    };
    let mut shape = vec![query.values.len()];
    if series.trailing_shape.is_empty() {
        shape.push(series.series);
    } else {
        shape.extend(series.trailing_shape.iter().copied());
    }
    let storage = tensor.into_numeric_storage().map_err(|err| {
        build_runtime_error(format!("interp1q: {err}"))
            .with_builtin(NAME)
            .build()
    })?;
    Tensor::from_numeric_storage(storage, shape)
        .map(Value::Tensor)
        .map_err(|err| {
            build_runtime_error(format!("interp1q: {err}"))
                .with_builtin(NAME)
                .build()
        })
}

async fn gather_interp_input(value: Value) -> BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(|err| wrap_error(err, &ERROR_INTERNAL))
}

#[derive(Clone, Debug)]
struct ComplexSeries {
    x: Vec<f64>,
    y: Vec<(f64, f64)>,
    series: usize,
    trailing_shape: Vec<usize>,
    complex_output: bool,
}

impl ComplexSeries {
    async fn from_values(x_value: Value, y_value: Value) -> BuiltinResult<Self> {
        let x = real_vector_from_value(x_value, "X").await?;
        let (y_data, y_shape, complex_output) = numeric_data_from_value(y_value, "V").await?;
        super::pp::validate_breaks(&x, NAME)
            .map_err(|err| wrap_error(err, &ERROR_INVALID_INPUT))?;
        let n = x.len();
        if y_data.len() != n && !y_data.len().is_multiple_of(n) {
            return Err(error_with_detail(
                &ERROR_INVALID_INPUT,
                "V length must match X or have X as its first dimension",
            ));
        }
        let y_shape = canonical_shape(&y_shape, y_data.len());
        let (series, trailing_shape) = if y_data.len() == n && super::pp::is_vector_shape(&y_shape)
        {
            (1, Vec::new())
        } else if y_shape.first().copied() == Some(n) {
            let series = y_data.len() / n;
            let trailing = if y_shape.len() > 1 {
                y_shape[1..].to_vec()
            } else {
                vec![series]
            };
            (series, trailing)
        } else if y_data.len() == n {
            (1, Vec::new())
        } else {
            return Err(error_with_detail(
                &ERROR_INVALID_INPUT,
                "size of V must be compatible with X",
            ));
        };
        Ok(Self {
            x,
            y: y_data,
            series,
            trailing_shape,
            complex_output,
        })
    }
}

#[derive(Clone, Debug)]
struct RealQuery {
    values: Vec<f64>,
    shape: Vec<usize>,
}

fn value_is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

async fn real_query_points(value: Value) -> BuiltinResult<RealQuery> {
    let (values, shape) = real_coordinate_data_from_value(value, "Xq").await?;
    let len = values.len();
    Ok(RealQuery {
        values,
        shape: canonical_shape(&shape, len),
    })
}

async fn real_vector_from_value(value: Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let (values, shape) = real_coordinate_data_from_value(value, label).await?;
    if !super::pp::is_vector_shape(&shape) && values.len() > 1 {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            format!("{label} must be a vector"),
        ));
    }
    Ok(values)
}

async fn real_coordinate_data_from_value(
    value: Value,
    label: &str,
) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    let gathered = crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(|err| wrap_error(err, &ERROR_INTERNAL))?;
    match gathered {
        Value::Num(value) => Ok((vec![value], vec![1, 1])),
        Value::Int(value) => Ok((vec![value.to_f64()], vec![1, 1])),
        Value::Bool(value) => Ok((vec![if value { 1.0 } else { 0.0 }], vec![1, 1])),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            Ok((tensor_utils::tensor_into_values_f64(tensor), shape))
        }
        Value::LogicalArray(array) => Ok((
            array
                .data
                .into_iter()
                .map(|bit| if bit != 0 { 1.0 } else { 0.0 })
                .collect(),
            array.shape,
        )),
        Value::Complex(re, im) => {
            validate_real_coordinate(label, im)?;
            Ok((vec![re], vec![1, 1]))
        }
        Value::ComplexTensor(tensor) => {
            for &(_, im) in &tensor.materialize_f64() {
                validate_real_coordinate(label, im)?;
            }
            Ok((
                tensor
                    .materialize_f64()
                    .into_iter()
                    .map(|(re, _)| re)
                    .collect(),
                tensor.shape,
            ))
        }
        other => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            format!("{label} must be numeric, got {other:?}"),
        )),
    }
}

async fn numeric_data_from_value(
    value: Value,
    label: &str,
) -> BuiltinResult<(Vec<(f64, f64)>, Vec<usize>, bool)> {
    let gathered = crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(|err| wrap_error(err, &ERROR_INTERNAL))?;
    match gathered {
        Value::Num(value) => Ok((vec![(value, 0.0)], vec![1, 1], false)),
        Value::Int(value) => Ok((vec![(value.to_f64(), 0.0)], vec![1, 1], false)),
        Value::Bool(value) => Ok((
            vec![(if value { 1.0 } else { 0.0 }, 0.0)],
            vec![1, 1],
            false,
        )),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            Ok((
                tensor_utils::tensor_into_values_f64(tensor)
                    .into_iter()
                    .map(|re| (re, 0.0))
                    .collect(),
                shape,
                false,
            ))
        }
        Value::LogicalArray(array) => Ok((
            array
                .data
                .into_iter()
                .map(|bit| (if bit != 0 { 1.0 } else { 0.0 }, 0.0))
                .collect(),
            array.shape,
            false,
        )),
        Value::Complex(re, im) => {
            validate_real_coordinate(label, im)?;
            Ok((vec![(re, im)], vec![1, 1], true))
        }
        Value::ComplexTensor(tensor) => {
            if label == "X" || label == "Xq" {
                for &(_, im) in &tensor.materialize_f64() {
                    validate_real_coordinate(label, im)?;
                }
            }
            Ok((tensor.materialize_f64(), tensor.shape, true))
        }
        other => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            format!("{label} must be numeric, got {other:?}"),
        )),
    }
}

fn validate_real_coordinate(label: &str, im: f64) -> BuiltinResult<()> {
    if label == "V" || im == 0.0 {
        return Ok(());
    }
    Err(error_with_detail(
        &ERROR_INVALID_INPUT,
        format!("{label} must contain real interpolation coordinates"),
    ))
}

fn evaluate_complex_linear(series: &ComplexSeries, query: &RealQuery) -> BuiltinResult<Value> {
    let len = query
        .values
        .len()
        .checked_mul(series.series)
        .ok_or_else(|| error_with_detail(&ERROR_INTERNAL, "output size overflow"))?;
    let mut out = vec![(0.0, 0.0); len];
    for s in 0..series.series {
        let y = &series.y[s * series.x.len()..(s + 1) * series.x.len()];
        for (q_index, &xq) in query.values.iter().enumerate() {
            out[q_index + s * query.values.len()] = eval_complex_linear(&series.x, y, xq);
        }
    }
    let shape = if series.series == 1 {
        query.shape.clone()
    } else {
        let mut shape = vec![query.values.len()];
        if series.trailing_shape.is_empty() {
            shape.push(series.series);
        } else {
            shape.extend(series.trailing_shape.iter().copied());
        }
        shape
    };
    if series.complex_output {
        return complex_value_from_data(out, shape);
    }
    let real = out.into_iter().map(|(re, _)| re).collect::<Vec<_>>();
    real_value_from_data(real, shape)
}

fn eval_complex_linear(x: &[f64], y: &[(f64, f64)], xq: f64) -> (f64, f64) {
    if !xq.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let Some(piece) = super::pp::interval_index(x, xq, false) else {
        return (f64::NAN, f64::NAN);
    };
    let h = x[piece + 1] - x[piece];
    let t = (xq - x[piece]) / h;
    let left = y[piece];
    let right = y[piece + 1];
    (
        left.0 + t * (right.0 - left.0),
        left.1 + t * (right.1 - left.1),
    )
}

fn real_value_from_data(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if data.len() == 1 {
        return Ok(Value::Num(data[0]));
    }
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
}

fn complex_value_from_data(data: Vec<(f64, f64)>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if data.len() == 1 {
        let (re, im) = data[0];
        return Ok(Value::Complex(re, im));
    }
    ComplexTensor::new(data, shape)
        .map(Value::ComplexTensor)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
}

fn canonical_shape(shape: &[usize], len: usize) -> Vec<usize> {
    crate::builtins::common::tensor::default_shape_for(shape, len)
}

fn error_with_detail(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(format!("interp1q: {}", detail.into()))
        .with_builtin(NAME)
        .with_identifier(descriptor.identifier.unwrap_or("RunMat:interp1q:Error"))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage, Tensor};

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    fn int_row(storage: IntegerStorage) -> Value {
        let len = storage.len();
        let tensor = Tensor::new_integer(storage, vec![1, len]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn interp1q_linearly_interpolates() {
        let out = block_on(interp1q_builtin(
            row(&[1.0, 2.0, 3.0]),
            row(&[10.0, 20.0, 40.0]),
            row(&[1.5, 2.5]),
        ))
        .expect("interp1q");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(tensor.materialize_f64(), vec![15.0, 30.0]);
    }

    #[test]
    fn interp1q_reads_typed_integer_x_v_and_query_exactly() {
        let args = || {
            (
                int_row(IntegerStorage::I16(vec![1, 2, 3])),
                int_row(IntegerStorage::U16(vec![10, 20, 40])),
                int_row(IntegerStorage::I16(vec![1, 2])),
            )
        };
        let (x, v, xq) = args();
        let error = block_on(interp1q_builtin(x, v, xq))
            .expect_err("compatible mode rejects integer samples");
        assert_eq!(
            error.identifier(),
            INTERP1Q_INTEGER_DATA_EXTENSION.error_identifier
        );
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let (x, v, xq) = args();
        let out = block_on(interp1q_builtin(x, v, xq)).expect("RunMat interp1q");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.materialize_f64(), vec![10.0, 20.0]);
    }

    #[test]
    fn interp1q_rejects_typed_complex_integer_inputs() {
        let typed_complex = || {
            Value::ComplexTensor(
                ComplexTensor::new_integer(
                    IntegerComplexStorage::new(
                        IntegerStorage::U64(vec![u64::MAX]),
                        IntegerStorage::U64(vec![1]),
                    )
                    .expect("storage"),
                    vec![1, 1],
                )
                .expect("tensor"),
            )
        };

        for (x, v, xq, extension) in [
            (
                typed_complex(),
                row(&[1.0]),
                row(&[1.0]),
                &INTERP1Q_INTEGER_DATA_EXTENSION,
            ),
            (
                row(&[1.0]),
                typed_complex(),
                row(&[1.0]),
                &INTERP1Q_INTEGER_DATA_EXTENSION,
            ),
            (
                row(&[1.0]),
                row(&[1.0]),
                typed_complex(),
                &INTERP1Q_INTEGER_QUERY_EXTENSION,
            ),
        ] {
            let err = block_on(interp1q_builtin(x.clone(), v.clone(), xq.clone()))
                .expect_err("compatible mode must reject the declared integer extension");
            assert_eq!(err.identifier(), extension.error_identifier);
            let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
            let err = block_on(interp1q_builtin(x, v, xq))
                .expect_err("typed complex integer input must reject");
            assert!(err.message().contains("complex numbers with integer types"));
        }
    }

    #[test]
    fn interp1q_out_of_range_returns_nan() {
        let out = block_on(interp1q_builtin(
            row(&[1.0, 2.0]),
            row(&[10.0, 20.0]),
            Value::Num(3.0),
        ))
        .expect("interp1q");
        let Value::Num(value) = out else {
            panic!("expected scalar");
        };
        assert!(value.is_nan());
    }

    #[test]
    fn interp1q_matrix_values_return_query_length_by_series_shape() {
        let x = row(&[1.0, 2.0, 3.0]);
        let v = Value::Tensor(
            Tensor::new(
                vec![
                    10.0, 20.0, 40.0, //
                    100.0, 200.0, 400.0,
                ],
                vec![3, 2],
            )
            .expect("matrix values"),
        );
        let xq = Value::Tensor(Tensor::new(vec![1.5, 2.5], vec![2, 1]).expect("query"));
        let out = block_on(interp1q_builtin(x, v, xq)).expect("interp1q");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.materialize_f64(), vec![15.0, 30.0, 150.0, 300.0]);
    }

    #[test]
    fn interp1q_interpolates_complex_vector_values() {
        let v = Value::ComplexTensor(
            ComplexTensor::new(vec![(10.0, 1.0), (20.0, 3.0), (40.0, 7.0)], vec![3, 1])
                .expect("complex values"),
        );
        let out = block_on(interp1q_builtin(row(&[1.0, 2.0, 3.0]), v, row(&[1.5, 2.5])))
            .expect("interp1q");
        let Value::ComplexTensor(tensor) = out else {
            panic!("expected complex tensor");
        };
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(tensor.materialize_f64(), vec![(15.0, 2.0), (30.0, 5.0)]);
    }

    #[test]
    fn interp1q_complex_out_of_range_returns_complex_nan() {
        let v = Value::ComplexTensor(
            ComplexTensor::new(vec![(10.0, 1.0), (20.0, 3.0)], vec![2, 1]).expect("complex values"),
        );
        let out =
            block_on(interp1q_builtin(row(&[1.0, 2.0]), v, Value::Num(3.0))).expect("interp1q");
        let Value::Complex(re, im) = out else {
            panic!("expected complex scalar");
        };
        assert!(re.is_nan());
        assert!(im.is_nan());
    }

    #[test]
    fn interp1q_complex_matrix_values_return_query_length_by_series_shape() {
        let v = Value::ComplexTensor(
            ComplexTensor::new(
                vec![
                    (10.0, 1.0),
                    (20.0, 3.0),
                    (40.0, 7.0),
                    (100.0, -1.0),
                    (200.0, -3.0),
                    (400.0, -7.0),
                ],
                vec![3, 2],
            )
            .expect("complex matrix values"),
        );
        let xq = Value::Tensor(Tensor::new(vec![1.5, 2.5], vec![2, 1]).expect("query"));
        let out = block_on(interp1q_builtin(row(&[1.0, 2.0, 3.0]), v, xq)).expect("interp1q");
        let Value::ComplexTensor(tensor) = out else {
            panic!("expected complex tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![(15.0, 2.0), (30.0, 5.0), (150.0, -2.0), (300.0, -5.0)]
        );
    }

    #[test]
    fn interp1q_accepts_zero_imaginary_complex_coordinates() {
        let x = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], vec![3, 1])
                .expect("complex x"),
        );
        let xq = Value::Complex(1.5, 0.0);
        let out = block_on(interp1q_builtin(x, row(&[10.0, 20.0, 40.0]), xq)).expect("interp1q");
        let Value::Num(value) = out else {
            panic!("expected scalar");
        };
        assert_eq!(value, 15.0);
    }

    #[test]
    fn interp1q_rejects_nonreal_complex_coordinates() {
        let err = block_on(interp1q_builtin(
            Value::ComplexTensor(
                ComplexTensor::new(vec![(1.0, 0.0), (2.0, 1.0)], vec![2, 1]).expect("complex x"),
            ),
            row(&[10.0, 20.0]),
            Value::Num(1.5),
        ))
        .expect_err("nonreal x should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn interp1q_rejects_unsorted_x() {
        let err = block_on(interp1q_builtin(
            row(&[2.0, 1.0]),
            row(&[20.0, 10.0]),
            Value::Num(1.5),
        ))
        .expect_err("unsorted x should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }
}
