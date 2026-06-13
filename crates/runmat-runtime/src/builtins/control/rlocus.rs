//! Root-locus analysis for SISO transfer-function models.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, ObjectInstance, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{
    control_error, poly_eval, polynomial_roots, scalar_f64, TfModel, EPS,
};
use crate::builtins::control::type_resolvers::rlocus_type;
use crate::{BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "rlocus";
const DEFAULT_GAIN_POINTS: usize = 121;
const DEFAULT_GAIN_DECADES: f64 = 4.0;

const RLOCUS_OUTPUT_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Closed-loop pole locations as a branches-by-gains matrix.",
};
const RLOCUS_OUTPUT_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Gain vector used to compute the root locus.",
};
const RLOCUS_INPUT_SYS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf or ss model.",
};
const RLOCUS_INPUT_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Finite nonnegative gain vector.",
};
const RLOCUS_OUTPUTS_R: [BuiltinParamDescriptor; 1] = [RLOCUS_OUTPUT_R];
const RLOCUS_OUTPUTS_R_K: [BuiltinParamDescriptor; 2] = [RLOCUS_OUTPUT_R, RLOCUS_OUTPUT_K];
const RLOCUS_INPUTS_SYS: [BuiltinParamDescriptor; 1] = [RLOCUS_INPUT_SYS];
const RLOCUS_INPUTS_SYS_K: [BuiltinParamDescriptor; 2] = [RLOCUS_INPUT_SYS, RLOCUS_INPUT_K];
const RLOCUS_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "r = rlocus(sys)",
        inputs: &RLOCUS_INPUTS_SYS,
        outputs: &RLOCUS_OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = rlocus(sys, k)",
        inputs: &RLOCUS_INPUTS_SYS_K,
        outputs: &RLOCUS_OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "[r,k] = rlocus(sys)",
        inputs: &RLOCUS_INPUTS_SYS,
        outputs: &RLOCUS_OUTPUTS_R_K,
    },
    BuiltinSignatureDescriptor {
        label: "[r,k] = rlocus(sys, k)",
        inputs: &RLOCUS_INPUTS_SYS_K,
        outputs: &RLOCUS_OUTPUTS_R_K,
    },
];
const RLOCUS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RLOCUS.INVALID_ARGUMENT",
    identifier: Some("RunMat:rlocus:InvalidArgument"),
    when: "Inputs do not match supported rlocus invocation forms.",
    message: "rlocus: invalid argument",
};
const RLOCUS_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RLOCUS.INVALID_MODEL",
    identifier: Some("RunMat:rlocus:InvalidModel"),
    when: "Input system is not a valid SISO tf or ss object.",
    message: "rlocus: invalid model",
};
const RLOCUS_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RLOCUS.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:rlocus:UnsupportedModel"),
    when: "Model form is not supported by the current implementation.",
    message: "rlocus: unsupported model",
};
const RLOCUS_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RLOCUS.PLOT_FAILED",
    identifier: Some("RunMat:rlocus:PlotFailed"),
    when: "Statement-form plotting failed for reasons other than known nonfatal setup conditions.",
    message: "rlocus: plotting failed",
};
const RLOCUS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RLOCUS.INTERNAL",
    identifier: Some("RunMat:rlocus:Internal"),
    when: "Root-locus computation or output construction failed.",
    message: "rlocus: internal error",
};
const RLOCUS_ERRORS: [BuiltinErrorDescriptor; 5] = [
    RLOCUS_ERROR_INVALID_ARGUMENT,
    RLOCUS_ERROR_INVALID_MODEL,
    RLOCUS_ERROR_UNSUPPORTED_MODEL,
    RLOCUS_ERROR_PLOT_FAILED,
    RLOCUS_ERROR_INTERNAL,
];
pub const RLOCUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RLOCUS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RLOCUS_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::rlocus")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rlocus",
    op_kind: GpuOpKind::Custom("control-root-locus"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "rlocus computes closed-loop polynomial roots on the host from transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::rlocus")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rlocus",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "rlocus is model analysis and plotting; it terminates numeric fusion chains.",
};

fn rlocus_error(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "rlocus",
    category = "control",
    summary = "Compute or plot root loci of SISO transfer-function models.",
    keywords = "rlocus,root locus,control system,transfer function,tf",
    sink = true,
    suppress_auto_output = true,
    type_resolver(rlocus_type),
    descriptor(crate::builtins::control::rlocus::RLOCUS_DESCRIPTOR),
    builtin_path = "crate::builtins::control::rlocus"
)]
async fn rlocus_builtin(sys: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if is_statement_form_call() {
        plot_root_locus_statement(sys, rest).await?;
        return Ok(Value::OutputList(Vec::new()));
    }

    if rest.len() > 1 {
        return Err(rlocus_error(
            "rlocus: expected rlocus(sys) or rlocus(sys, k)",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ));
    }

    let model = DynamicModel::from_value_async(sys).await?;
    let gains = match rest.first() {
        Some(value) => Some(parse_gain_arg(value).await?),
        None => None,
    };
    let eval = RootLocus::compute(&model.tf, gains)?;

    if crate::output_context::requested_output_count() == Some(0)
        && crate::output_count::current_output_count().is_none()
    {
        render_root_locus_plot(&eval, None).await?;
        return Ok(Value::OutputList(Vec::new()));
    }

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            render_root_locus_plot(&eval, None).await?;
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.roots_value()?]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs()?,
        ));
    }

    eval.roots_value()
}

fn is_statement_form_call() -> bool {
    matches!(crate::output_count::current_output_count(), Some(0))
        || (crate::output_context::requested_output_count() == Some(0)
            && crate::output_count::current_output_count().is_none())
}

async fn plot_root_locus_statement(first_sys: Value, rest: Vec<Value>) -> BuiltinResult<()> {
    let mut systems = vec![(first_sys, None)];
    let mut gains = None;

    for arg in rest {
        let gathered = crate::dispatcher::gather_if_needed_async(&arg).await?;
        if is_plot_style_arg(&gathered) {
            if let Some((_, style)) = systems.last_mut() {
                if style.is_some() {
                    return Err(rlocus_error(
                        "rlocus: only one style argument is supported per system",
                        &RLOCUS_ERROR_INVALID_ARGUMENT,
                    ));
                }
                *style = Some(gathered);
                continue;
            }
        }
        if is_dynamic_model_object(&gathered) {
            if gains.is_some() {
                return Err(rlocus_error(
                    "rlocus: gain vector must follow all systems in statement-form plots",
                    &RLOCUS_ERROR_INVALID_ARGUMENT,
                ));
            }
            systems.push((gathered, None));
            continue;
        }
        if is_numeric_vector_like(&gathered) && gains.is_none() {
            gains = Some(real_gain_vector(gathered)?);
            continue;
        }
        return Err(rlocus_error(
            "rlocus: unsupported statement-form plot argument",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ));
    }

    let mut first = true;
    for (system, style) in systems {
        let model = DynamicModel::from_value_async(system).await?;
        let eval = RootLocus::compute(&model.tf, gains.clone())?;
        render_root_locus_plot(&eval, style.as_ref()).await?;
        if first {
            first = false;
            let _ = crate::call_builtin_async("hold", &[Value::from("on")]).await;
        }
    }
    let _ = crate::call_builtin_async("hold", &[Value::from("off")]).await;
    Ok(())
}

fn is_dynamic_model_object(value: &Value) -> bool {
    matches!(value, Value::Object(object) if object.is_class("tf") || object.is_class("ss"))
}

fn is_plot_style_arg(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn is_numeric_vector_like(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => true,
        Value::Tensor(tensor) => is_vector_shape(&tensor.shape),
        Value::ComplexTensor(tensor) => is_vector_shape(&tensor.shape),
        Value::LogicalArray(logical) => is_vector_shape(&logical.shape),
        _ => false,
    }
}

async fn parse_gain_arg(value: &Value) -> BuiltinResult<Vec<f64>> {
    let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
    real_gain_vector(gathered)
}

fn real_gain_vector(value: Value) -> BuiltinResult<Vec<f64>> {
    let gains = match value {
        Value::Num(n) => vec![n],
        Value::Int(i) => vec![i.to_f64()],
        Value::Bool(b) => vec![if b { 1.0 } else { 0.0 }],
        Value::Complex(re, im) if im.abs() <= EPS => vec![re],
        Value::Tensor(tensor) => {
            ensure_vector_shape(&tensor.shape)?;
            tensor.data
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector_shape(&tensor.shape)?;
            tensor
                .data
                .into_iter()
                .map(|(re, im)| {
                    if im.abs() <= EPS {
                        Ok(re)
                    } else {
                        Err(rlocus_error(
                            "rlocus: gain vector must be real",
                            &RLOCUS_ERROR_INVALID_ARGUMENT,
                        ))
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()?
        }
        Value::LogicalArray(logical) => {
            ensure_vector_shape(&logical.shape)?;
            logical
                .data
                .into_iter()
                .map(|value| if value == 0 { 0.0 } else { 1.0 })
                .collect()
        }
        other => {
            return Err(rlocus_error(
                format!("rlocus: k must be a real numeric vector, got {other:?}"),
                &RLOCUS_ERROR_INVALID_ARGUMENT,
            ));
        }
    };
    validate_gains(gains)
}

fn ensure_vector_shape(shape: &[usize]) -> BuiltinResult<()> {
    if is_vector_shape(shape) {
        Ok(())
    } else {
        Err(rlocus_error(
            "rlocus: k must be a vector",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ))
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().copied().filter(|&dim| dim > 1).count() <= 1
}

fn validate_gains(gains: Vec<f64>) -> BuiltinResult<Vec<f64>> {
    if gains.is_empty() {
        return Err(rlocus_error(
            "rlocus: k must not be empty",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ));
    }
    if gains.iter().any(|gain| !gain.is_finite()) {
        return Err(rlocus_error(
            "rlocus: k values must be finite",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ));
    }
    if gains.iter().any(|gain| *gain < 0.0) {
        return Err(rlocus_error(
            "rlocus: k values must be nonnegative",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(gains)
}

#[derive(Clone, Debug)]
struct RootLocus {
    gains: Vec<f64>,
    roots: Vec<Complex64>,
    branches: usize,
    poles: Vec<Complex64>,
    zeros: Vec<Complex64>,
}

#[derive(Clone, Debug)]
struct DynamicModel {
    tf: TfModel,
}

impl DynamicModel {
    async fn from_value_async(value: Value) -> BuiltinResult<Self> {
        let gathered = crate::dispatcher::gather_if_needed_async(&value).await?;
        let Value::Object(object) = gathered else {
            return Err(rlocus_error(
                "rlocus: expected a SISO dynamic system model",
                &RLOCUS_ERROR_INVALID_MODEL,
            ));
        };
        if object.is_class("tf") {
            return Ok(Self {
                tf: TfModel::from_value(Value::Object(object), BUILTIN_NAME)?,
            });
        }
        if object.is_class("ss") {
            return Ok(Self {
                tf: ss_object_to_tf(&object)?,
            });
        }
        Err(rlocus_error(
            format!(
                "rlocus: unsupported model class '{}'; expected SISO tf or ss",
                object.class_name
            ),
            &RLOCUS_ERROR_UNSUPPORTED_MODEL,
        ))
    }
}

impl RootLocus {
    fn compute(model: &TfModel, gains: Option<Vec<f64>>) -> BuiltinResult<Self> {
        ensure_supported_model(model)?;
        let gains = gains.unwrap_or_else(|| default_gains(model));
        let branches = characteristic_branch_count(model);
        let poles = polynomial_roots(&model.denominator, BUILTIN_NAME)?;
        let zeros = if max_norm(&model.numerator) <= EPS {
            Vec::new()
        } else {
            polynomial_roots(&model.numerator, BUILTIN_NAME)?
        };

        let mut previous: Option<Vec<Complex64>> = None;
        let mut roots = Vec::with_capacity(branches.saturating_mul(gains.len()));
        for gain in &gains {
            let mut column = roots_for_gain(model, *gain, branches)?;
            column = match previous.as_ref() {
                Some(prev) => track_roots(prev, column),
                None => sort_roots(column),
            };
            previous = Some(column.clone());
            roots.extend(column);
        }

        Ok(Self {
            gains,
            roots,
            branches,
            poles,
            zeros,
        })
    }

    fn outputs(&self) -> BuiltinResult<Vec<Value>> {
        Ok(vec![self.roots_value()?, self.gains_value()?])
    }

    fn roots_value(&self) -> BuiltinResult<Value> {
        let shape = vec![self.branches, self.gains.len()];
        if self.roots.iter().all(|root| root.im.abs() <= EPS) {
            let data = self.roots.iter().map(|root| root.re).collect::<Vec<_>>();
            Tensor::new(data, shape).map(Value::Tensor).map_err(|err| {
                control_error(
                    BUILTIN_NAME,
                    "RunMat:rlocus:Internal",
                    format!("rlocus: failed to build root matrix: {err}"),
                )
            })
        } else {
            let data = self
                .roots
                .iter()
                .map(|root| (root.re, root.im))
                .collect::<Vec<_>>();
            ComplexTensor::new(data, shape)
                .map(Value::ComplexTensor)
                .map_err(|err| {
                    control_error(
                        BUILTIN_NAME,
                        "RunMat:rlocus:Internal",
                        format!("rlocus: failed to build complex root matrix: {err}"),
                    )
                })
        }
    }

    fn gains_value(&self) -> BuiltinResult<Value> {
        Tensor::new(self.gains.clone(), vec![1, self.gains.len()])
            .map(Value::Tensor)
            .map_err(|err| {
                control_error(
                    BUILTIN_NAME,
                    "RunMat:rlocus:Internal",
                    format!("rlocus: failed to build gain vector: {err}"),
                )
            })
    }

    fn root(&self, branch: usize, gain_index: usize) -> Complex64 {
        self.roots[branch + gain_index * self.branches]
    }
}

fn ensure_supported_model(model: &TfModel) -> BuiltinResult<()> {
    if model.input_delay.abs() > EPS || model.output_delay.abs() > EPS {
        return Err(rlocus_error(
            "rlocus: transfer functions with input or output delays are not supported",
            &RLOCUS_ERROR_UNSUPPORTED_MODEL,
        ));
    }
    Ok(())
}

fn ss_object_to_tf(object: &ObjectInstance) -> BuiltinResult<TfModel> {
    let a = matrix_property(object, "A")?;
    let b = matrix_property(object, "B")?;
    let c = matrix_property(object, "C")?;
    let d = matrix_property(object, "D")?;
    let sample_time = scalar_property(object, "Ts")?;
    if !sample_time.is_finite() || sample_time < 0.0 {
        return Err(rlocus_error(
            "rlocus: ss sample time must be a finite nonnegative scalar",
            &RLOCUS_ERROR_INVALID_MODEL,
        ));
    }
    ensure_zero_delay_property(object, "InputDelay")?;
    ensure_zero_delay_property(object, "OutputDelay")?;
    validate_ss_dimensions(&a, &b, &c, &d)?;

    let denominator = characteristic_polynomial_from_matrix(&a)?;
    let numerator = state_space_numerator(&a, &b, &c, d[(0, 0)], &denominator)?;
    let numerator = trim_leading_complex_zeros(clean_coefficients(numerator));
    let denominator = trim_leading_complex_zeros(clean_coefficients(denominator));
    if denominator.is_empty() || denominator[0].norm() <= EPS {
        return Err(rlocus_error(
            "rlocus: ss model produced an invalid denominator polynomial",
            &RLOCUS_ERROR_INTERNAL,
        ));
    }
    ensure_finite_coefficients("Numerator", &numerator)?;
    ensure_finite_coefficients("Denominator", &denominator)?;

    Ok(TfModel {
        numerator,
        denominator,
        variable: if sample_time > 0.0 { "z" } else { "s" }.to_string(),
        sample_time,
        input_delay: 0.0,
        output_delay: 0.0,
    })
}

fn property<'a>(object: &'a ObjectInstance, name: &str) -> BuiltinResult<&'a Value> {
    object.properties.get(name).ok_or_else(|| {
        rlocus_error(
            format!("rlocus: model object is missing {name} property"),
            &RLOCUS_ERROR_INVALID_MODEL,
        )
    })
}

fn matrix_property(object: &ObjectInstance, name: &str) -> BuiltinResult<DMatrix<Complex64>> {
    let value = property(object, name)?;
    match value {
        Value::Tensor(tensor) => {
            ensure_matrix_shape(name, &tensor.shape)?;
            let data = tensor
                .data
                .iter()
                .map(|&re| Complex64::new(re, 0.0))
                .collect::<Vec<_>>();
            ensure_finite_coefficients(name, &data)?;
            Ok(DMatrix::from_column_slice(tensor.rows, tensor.cols, &data))
        }
        Value::ComplexTensor(tensor) => {
            ensure_matrix_shape(name, &tensor.shape)?;
            let data = tensor
                .data
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect::<Vec<_>>();
            ensure_finite_coefficients(name, &data)?;
            Ok(DMatrix::from_column_slice(tensor.rows, tensor.cols, &data))
        }
        Value::LogicalArray(logical) => {
            ensure_matrix_shape(name, &logical.shape)?;
            let (rows, cols) = rows_cols(&logical.shape);
            let data = logical
                .data
                .iter()
                .map(|&value| Complex64::new(if value == 0 { 0.0 } else { 1.0 }, 0.0))
                .collect::<Vec<_>>();
            Ok(DMatrix::from_column_slice(rows, cols, &data))
        }
        Value::Num(n) => scalar_matrix(*n, 0.0, name),
        Value::Int(i) => scalar_matrix(i.to_f64(), 0.0, name),
        Value::Bool(b) => scalar_matrix(if *b { 1.0 } else { 0.0 }, 0.0, name),
        Value::Complex(re, im) => scalar_matrix(*re, *im, name),
        other => Err(rlocus_error(
            format!("rlocus: ss {name} must be a finite numeric matrix, got {other:?}"),
            &RLOCUS_ERROR_INVALID_MODEL,
        )),
    }
}

fn scalar_matrix(re: f64, im: f64, name: &str) -> BuiltinResult<DMatrix<Complex64>> {
    let value = Complex64::new(re, im);
    ensure_finite_coefficients(name, &[value])?;
    Ok(DMatrix::from_element(1, 1, value))
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> BuiltinResult<()> {
    if shape.len() <= 2 {
        Ok(())
    } else {
        Err(rlocus_error(
            format!("rlocus: ss {name} must be a 2-D matrix, got shape {shape:?}"),
            &RLOCUS_ERROR_INVALID_MODEL,
        ))
    }
}

fn rows_cols(shape: &[usize]) -> (usize, usize) {
    if shape.len() >= 2 {
        (shape[0], shape[1])
    } else if shape.len() == 1 {
        (1, shape[0])
    } else {
        (0, 0)
    }
}

fn scalar_property(object: &ObjectInstance, name: &str) -> BuiltinResult<f64> {
    let value = scalar_f64(property(object, name)?, name, BUILTIN_NAME)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(rlocus_error(
            format!("rlocus: ss {name} must be finite"),
            &RLOCUS_ERROR_INVALID_MODEL,
        ))
    }
}

fn ensure_zero_delay_property(object: &ObjectInstance, name: &str) -> BuiltinResult<()> {
    let values = numeric_values(property(object, name)?, name)?;
    if values.iter().any(|value| value.norm() > EPS) {
        return Err(rlocus_error(
            "rlocus: ss models with input or output delays are not supported",
            &RLOCUS_ERROR_UNSUPPORTED_MODEL,
        ));
    }
    Ok(())
}

fn numeric_values(value: &Value, name: &str) -> BuiltinResult<Vec<Complex64>> {
    let values = match value {
        Value::Num(n) => vec![Complex64::new(*n, 0.0)],
        Value::Int(i) => vec![Complex64::new(i.to_f64(), 0.0)],
        Value::Bool(b) => vec![Complex64::new(if *b { 1.0 } else { 0.0 }, 0.0)],
        Value::Complex(re, im) => vec![Complex64::new(*re, *im)],
        Value::Tensor(tensor) => tensor
            .data
            .iter()
            .map(|&re| Complex64::new(re, 0.0))
            .collect(),
        Value::ComplexTensor(tensor) => tensor
            .data
            .iter()
            .map(|&(re, im)| Complex64::new(re, im))
            .collect(),
        Value::LogicalArray(logical) => logical
            .data
            .iter()
            .map(|&value| Complex64::new(if value == 0 { 0.0 } else { 1.0 }, 0.0))
            .collect(),
        other => {
            return Err(rlocus_error(
                format!("rlocus: ss {name} must be numeric, got {other:?}"),
                &RLOCUS_ERROR_INVALID_MODEL,
            ));
        }
    };
    ensure_finite_coefficients(name, &values)?;
    Ok(values)
}

fn validate_ss_dimensions(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    c: &DMatrix<Complex64>,
    d: &DMatrix<Complex64>,
) -> BuiltinResult<()> {
    if a.nrows() != a.ncols() {
        return Err(rlocus_error(
            format!(
                "rlocus: ss A must be square, got {}x{}",
                a.nrows(),
                a.ncols()
            ),
            &RLOCUS_ERROR_INVALID_MODEL,
        ));
    }
    let states = a.nrows();
    if b.nrows() != states {
        return Err(rlocus_error(
            format!(
                "rlocus: ss B must have {states} rows to match A, got {}x{}",
                b.nrows(),
                b.ncols()
            ),
            &RLOCUS_ERROR_INVALID_MODEL,
        ));
    }
    if c.ncols() != states {
        return Err(rlocus_error(
            format!(
                "rlocus: ss C must have {states} columns to match A, got {}x{}",
                c.nrows(),
                c.ncols()
            ),
            &RLOCUS_ERROR_INVALID_MODEL,
        ));
    }
    if d.nrows() != c.nrows() || d.ncols() != b.ncols() {
        return Err(rlocus_error(
            format!(
                "rlocus: ss D must have shape {}x{} to match C outputs and B inputs, got {}x{}",
                c.nrows(),
                b.ncols(),
                d.nrows(),
                d.ncols()
            ),
            &RLOCUS_ERROR_INVALID_MODEL,
        ));
    }
    if b.ncols() != 1 || c.nrows() != 1 || d.nrows() != 1 || d.ncols() != 1 {
        return Err(rlocus_error(
            "rlocus: only SISO ss models are supported",
            &RLOCUS_ERROR_UNSUPPORTED_MODEL,
        ));
    }
    Ok(())
}

fn characteristic_polynomial_from_matrix(a: &DMatrix<Complex64>) -> BuiltinResult<Vec<Complex64>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(vec![Complex64::new(1.0, 0.0)]);
    }
    let mut coeffs = Vec::with_capacity(n + 1);
    coeffs.push(Complex64::new(1.0, 0.0));
    let mut b = DMatrix::<Complex64>::identity(n, n);
    for k in 1..=n {
        let ab = a * &b;
        let trace = (0..n).fold(Complex64::new(0.0, 0.0), |acc, idx| acc + ab[(idx, idx)]);
        let coeff = -trace / Complex64::new(k as f64, 0.0);
        coeffs.push(coeff);
        let mut next = ab;
        for idx in 0..n {
            next[(idx, idx)] += coeff;
        }
        b = next;
    }
    let coeffs = clean_coefficients(coeffs);
    ensure_finite_coefficients("ss characteristic polynomial", &coeffs)?;
    Ok(coeffs)
}

fn state_space_numerator(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    c: &DMatrix<Complex64>,
    d: Complex64,
    denominator: &[Complex64],
) -> BuiltinResult<Vec<Complex64>> {
    let degree = a.nrows();
    if degree == 0 {
        return Ok(vec![d]);
    }

    let mut points = Vec::with_capacity(degree + 1);
    let mut values = Vec::with_capacity(degree + 1);
    for point in interpolation_candidates(degree) {
        let Some(response) = state_space_response_at(a, b, c, d, point) else {
            continue;
        };
        points.push(point);
        values.push(response * poly_eval(denominator, point));
        if points.len() == degree + 1 {
            break;
        }
    }
    if points.len() != degree + 1 {
        return Err(rlocus_error(
            "rlocus: failed to find enough nonsingular interpolation points for ss model",
            &RLOCUS_ERROR_INTERNAL,
        ));
    }

    let mut vandermonde = DMatrix::<Complex64>::zeros(degree + 1, degree + 1);
    for (row, point) in points.iter().enumerate() {
        for col in 0..=degree {
            vandermonde[(row, col)] = point.powu((degree - col) as u32);
        }
    }
    let rhs = DVector::<Complex64>::from_vec(values);
    let coeffs = vandermonde.lu().solve(&rhs).ok_or_else(|| {
        rlocus_error(
            "rlocus: failed to solve ss numerator interpolation system",
            &RLOCUS_ERROR_INTERNAL,
        )
    })?;
    let coeffs = clean_coefficients(coeffs.iter().copied().collect());
    ensure_finite_coefficients("ss numerator polynomial", &coeffs)?;
    Ok(coeffs)
}

fn interpolation_candidates(degree: usize) -> Vec<Complex64> {
    let needed = degree + 1;
    let mut points = Vec::with_capacity(needed * 6);
    for radius_idx in 0..4 {
        let radius = 0.75 + radius_idx as f64;
        for idx in 0..needed {
            let angle = std::f64::consts::TAU * ((idx as f64 + 0.37) / needed as f64);
            points.push(Complex64::from_polar(radius, angle));
        }
    }
    for idx in 1..=(needed * 2) {
        let value = idx as f64;
        points.push(Complex64::new(value, 0.0));
        points.push(Complex64::new(-value, 0.0));
    }
    points
}

fn state_space_response_at(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    c: &DMatrix<Complex64>,
    d: Complex64,
    point: Complex64,
) -> Option<Complex64> {
    let n = a.nrows();
    if n == 0 {
        return Some(d);
    }
    let mut system = -a.clone();
    for idx in 0..n {
        system[(idx, idx)] += point;
    }
    let state = system.lu().solve(b)?;
    let output = c * state;
    Some(d + output[(0, 0)])
}

fn characteristic_branch_count(model: &TfModel) -> usize {
    model
        .denominator
        .len()
        .max(model.numerator.len())
        .saturating_sub(1)
}

fn roots_for_gain(model: &TfModel, gain: f64, branches: usize) -> BuiltinResult<Vec<Complex64>> {
    let polynomial = characteristic_polynomial(model, gain);
    let mut roots = polynomial_roots(&polynomial, BUILTIN_NAME)?;
    if roots.len() > branches {
        return Err(rlocus_error(
            "rlocus: root calculation returned more roots than characteristic branches",
            &RLOCUS_ERROR_INTERNAL,
        ));
    }
    roots.resize(branches, Complex64::new(f64::INFINITY, 0.0));
    Ok(roots)
}

fn characteristic_polynomial(model: &TfModel, gain: f64) -> Vec<Complex64> {
    let len = model.denominator.len().max(model.numerator.len()).max(1);
    let mut out = vec![Complex64::new(0.0, 0.0); len];
    let denominator_offset = len - model.denominator.len();
    for (idx, coeff) in model.denominator.iter().enumerate() {
        out[denominator_offset + idx] += *coeff;
    }
    let numerator_offset = len - model.numerator.len();
    let gain = Complex64::new(gain, 0.0);
    for (idx, coeff) in model.numerator.iter().enumerate() {
        out[numerator_offset + idx] += gain * *coeff;
    }
    out
}

fn default_gains(model: &TfModel) -> Vec<f64> {
    let numerator_scale = max_norm(&model.numerator);
    if numerator_scale <= EPS {
        return vec![0.0];
    }
    let denominator_scale = max_norm(&model.denominator).max(1.0);
    let center = (denominator_scale / numerator_scale).clamp(1.0e-9, 1.0e9);
    let start = center.log10() - DEFAULT_GAIN_DECADES;
    let stop = center.log10() + DEFAULT_GAIN_DECADES;

    let mut gains = Vec::with_capacity(DEFAULT_GAIN_POINTS + 16);
    gains.push(0.0);
    if DEFAULT_GAIN_POINTS == 1 {
        gains.push(center);
        return gains;
    }
    for idx in 0..DEFAULT_GAIN_POINTS {
        let fraction = idx as f64 / (DEFAULT_GAIN_POINTS - 1) as f64;
        let gain = 10.0_f64.powf(start + fraction * (stop - start));
        if gain.is_finite() && gain > 0.0 {
            gains.push(gain);
        }
    }
    for gain in critical_gains(model) {
        push_gain_window(&mut gains, gain);
    }
    sort_and_dedup_gains(gains)
}

fn max_norm(coeffs: &[Complex64]) -> f64 {
    coeffs
        .iter()
        .map(|coeff| coeff.norm())
        .fold(0.0_f64, f64::max)
}

fn critical_gains(model: &TfModel) -> Vec<f64> {
    let den_derivative = poly_derivative(&model.denominator);
    let num_derivative = poly_derivative(&model.numerator);
    let equation = poly_sub(
        &poly_mul(&den_derivative, &model.numerator),
        &poly_mul(&model.denominator, &num_derivative),
    );
    let Ok(points) = polynomial_roots(&equation, BUILTIN_NAME) else {
        return Vec::new();
    };
    let mut gains = Vec::new();
    for point in points {
        if point.im.abs() > 1.0e-7 {
            continue;
        }
        let numerator = poly_eval(&model.numerator, point);
        if numerator.norm() <= EPS {
            continue;
        }
        let gain = -poly_eval(&model.denominator, point) / numerator;
        if gain.im.abs() <= 1.0e-7 && gain.re.is_finite() && gain.re > 0.0 {
            gains.push(gain.re);
        }
    }
    gains
}

fn push_gain_window(gains: &mut Vec<f64>, gain: f64) {
    for factor in [0.9, 0.99, 1.0, 1.01, 1.1] {
        let candidate = gain * factor;
        if candidate.is_finite() && candidate > 0.0 {
            gains.push(candidate);
        }
    }
}

fn sort_and_dedup_gains(mut gains: Vec<f64>) -> Vec<f64> {
    gains.sort_by(f64::total_cmp);
    gains.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-10 * a.abs().max(b.abs()).max(1.0));
    gains
}

fn poly_derivative(coeffs: &[Complex64]) -> Vec<Complex64> {
    if coeffs.len() <= 1 {
        return vec![Complex64::new(0.0, 0.0)];
    }
    let degree = coeffs.len() - 1;
    coeffs
        .iter()
        .take(degree)
        .enumerate()
        .map(|(idx, coeff)| *coeff * Complex64::new((degree - idx) as f64, 0.0))
        .collect()
}

fn poly_mul(left: &[Complex64], right: &[Complex64]) -> Vec<Complex64> {
    if left.is_empty() || right.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Complex64::new(0.0, 0.0); left.len() + right.len() - 1];
    for (i, lhs) in left.iter().enumerate() {
        for (j, rhs) in right.iter().enumerate() {
            out[i + j] += *lhs * *rhs;
        }
    }
    clean_coefficients(out)
}

fn poly_sub(left: &[Complex64], right: &[Complex64]) -> Vec<Complex64> {
    let len = left.len().max(right.len());
    let mut out = vec![Complex64::new(0.0, 0.0); len];
    let left_offset = len - left.len();
    for (idx, coeff) in left.iter().enumerate() {
        out[left_offset + idx] += *coeff;
    }
    let right_offset = len - right.len();
    for (idx, coeff) in right.iter().enumerate() {
        out[right_offset + idx] -= *coeff;
    }
    clean_coefficients(out)
}

fn clean_coefficients(mut coeffs: Vec<Complex64>) -> Vec<Complex64> {
    let scale = max_norm(&coeffs).max(1.0);
    let tol = scale * 1.0e-10;
    for coeff in &mut coeffs {
        if coeff.re.abs() <= tol {
            coeff.re = 0.0;
        }
        if coeff.im.abs() <= tol {
            coeff.im = 0.0;
        }
    }
    coeffs
}

fn trim_leading_complex_zeros(coeffs: Vec<Complex64>) -> Vec<Complex64> {
    let first_nonzero = coeffs
        .iter()
        .position(|value| value.norm() > EPS)
        .unwrap_or(coeffs.len());
    coeffs[first_nonzero..].to_vec()
}

fn ensure_finite_coefficients(label: &str, coeffs: &[Complex64]) -> BuiltinResult<()> {
    if coeffs
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(rlocus_error(
            format!("rlocus: {label} values must be finite"),
            &RLOCUS_ERROR_INVALID_MODEL,
        ));
    }
    Ok(())
}

fn sort_roots(mut roots: Vec<Complex64>) -> Vec<Complex64> {
    roots.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    roots
}

fn track_roots(previous: &[Complex64], roots: Vec<Complex64>) -> Vec<Complex64> {
    if previous.len() != roots.len() {
        return sort_roots(roots);
    }
    let mut assigned = vec![false; roots.len()];
    let mut ordered = Vec::with_capacity(roots.len());
    for prev in previous {
        let mut best_idx = None;
        let mut best_distance = f64::INFINITY;
        for (idx, root) in roots.iter().enumerate() {
            if assigned[idx] {
                continue;
            }
            let Some(distance) = root_distance2(*prev, *root) else {
                continue;
            };
            if distance < best_distance {
                best_distance = distance;
                best_idx = Some(idx);
            }
        }
        let idx = best_idx
            .or_else(|| assigned.iter().position(|used| !*used))
            .unwrap_or(0);
        assigned[idx] = true;
        ordered.push(roots[idx]);
    }
    ordered
}

fn root_distance2(a: Complex64, b: Complex64) -> Option<f64> {
    if !a.re.is_finite() || !a.im.is_finite() || !b.re.is_finite() || !b.im.is_finite() {
        return None;
    }
    let dr = a.re - b.re;
    let di = a.im - b.im;
    Some(dr * dr + di * di)
}

async fn render_root_locus_plot(eval: &RootLocus, style: Option<&Value>) -> BuiltinResult<()> {
    let mut args = Vec::new();
    for branch in 0..eval.branches {
        let mut x = Vec::with_capacity(eval.gains.len());
        let mut y = Vec::with_capacity(eval.gains.len());
        for gain_idx in 0..eval.gains.len() {
            let root = eval.root(branch, gain_idx);
            if root.re.is_finite() && root.im.is_finite() {
                x.push(root.re);
                y.push(root.im);
            }
        }
        if x.is_empty() {
            continue;
        }
        args.push(column_tensor(x)?);
        args.push(column_tensor(y)?);
        if let Some(style) = style {
            args.push(style.clone());
        }
    }
    push_marker_series(&mut args, &eval.poles, "x")?;
    push_marker_series(&mut args, &eval.zeros, "o")?;

    if args.is_empty() {
        return Ok(());
    }
    if let Err(err) = crate::call_builtin_async("plot", &args).await {
        if super::is_nonfatal_plot_setup_error(&err) {
            return Ok(());
        }
        return Err(rlocus_error(
            format!("rlocus: plotting failed: {}", err.message()),
            &RLOCUS_ERROR_PLOT_FAILED,
        ));
    }
    let _ = crate::call_builtin_async("title", &[Value::from("Root Locus")]).await;
    let _ = crate::call_builtin_async("xlabel", &[Value::from("Real Axis")]).await;
    let _ = crate::call_builtin_async("ylabel", &[Value::from("Imaginary Axis")]).await;
    let _ = crate::call_builtin_async("grid", &[Value::from("on")]).await;
    Ok(())
}

fn push_marker_series(
    args: &mut Vec<Value>,
    values: &[Complex64],
    marker: &str,
) -> BuiltinResult<()> {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for value in values {
        if value.re.is_finite() && value.im.is_finite() {
            x.push(value.re);
            y.push(value.im);
        }
    }
    if !x.is_empty() {
        args.push(column_tensor(x)?);
        args.push(column_tensor(y)?);
        args.push(Value::from(marker));
    }
    Ok(())
}

fn column_tensor(data: Vec<f64>) -> BuiltinResult<Value> {
    let rows = data.len();
    Tensor::new(data, vec![rows, 1])
        .map(Value::Tensor)
        .map_err(|err| {
            control_error(
                BUILTIN_NAME,
                "RunMat:rlocus:Internal",
                format!("rlocus: failed to build plot vector: {err}"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn tf(num: Vec<f64>, den: Vec<f64>) -> Value {
        block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Tensor(Tensor::new(num.clone(), vec![1, num.len()]).unwrap()),
                Value::Tensor(Tensor::new(den.clone(), vec![1, den.len()]).unwrap()),
            ],
        ))
        .expect("tf")
    }

    fn discrete_tf(num: Vec<f64>, den: Vec<f64>, sample_time: f64) -> Value {
        block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Tensor(Tensor::new(num.clone(), vec![1, num.len()]).unwrap()),
                Value::Tensor(Tensor::new(den.clone(), vec![1, den.len()]).unwrap()),
                Value::Num(sample_time),
            ],
        ))
        .expect("tf")
    }

    fn ss(a: Value, b: Value, c: Value, d: Value) -> Value {
        block_on(crate::call_builtin_async("ss", &[a, b, c, d])).expect("ss")
    }

    fn run_rlocus(sys: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(rlocus_builtin(sys, rest))
    }

    fn tensor(value: &Value) -> &Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn descriptor_signatures_cover_output_forms() {
        let labels = RLOCUS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"r = rlocus(sys)"));
        assert!(labels.contains(&"r = rlocus(sys, k)"));
        assert!(labels.contains(&"[r,k] = rlocus(sys)"));
        assert!(labels.contains(&"[r,k] = rlocus(sys, k)"));
    }

    #[test]
    fn explicit_gain_returns_closed_loop_roots_and_gains() {
        let sys = tf(vec![1.0], vec![1.0, 1.0]);
        let gains = Value::Tensor(Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = run_rlocus(sys, vec![gains]).expect("rlocus");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);

        let roots = tensor(&outputs[0]);
        assert_eq!(roots.shape, vec![1, 3]);
        assert_eq!(roots.data, vec![-1.0, -2.0, -4.0]);

        let gains = tensor(&outputs[1]);
        assert_eq!(gains.shape, vec![1, 3]);
        assert_eq!(gains.data, vec![0.0, 1.0, 3.0]);
    }

    #[test]
    fn multi_branch_matrix_uses_branch_rows_and_gain_columns() {
        let sys = tf(vec![1.0, 0.0], vec![1.0, 3.0, 2.0]);
        let gains = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap());
        let result = run_rlocus(sys, vec![gains]).expect("rlocus");
        let roots = tensor(&result);
        assert_eq!(roots.shape, vec![2, 3]);

        for (gain_idx, gain) in [0.0_f64, 1.0, 2.0].iter().enumerate() {
            let column = &roots.data[gain_idx * 2..gain_idx * 2 + 2];
            for root in column {
                let residual = root * root + (3.0 + gain) * root + 2.0;
                assert!(
                    residual.abs() < 1.0e-8,
                    "gain={gain} root={root} residual={residual}"
                );
            }
        }
    }

    #[test]
    fn state_space_siso_matches_transfer_function_root_locus() {
        let sys = ss(
            Value::Num(-1.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(0.0),
        );
        let gains = Value::Tensor(Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap());
        let result = run_rlocus(sys, vec![gains]).expect("rlocus");
        let roots = tensor(&result);
        assert_eq!(roots.shape, vec![1, 3]);
        for (actual, expected) in roots.data.iter().zip([-1.0, -2.0, -4.0]) {
            assert!((actual - expected).abs() < 1.0e-8);
        }
    }

    #[test]
    fn state_space_mimo_is_rejected_with_rlocus_identifier() {
        let sys = ss(
            Value::Num(-1.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap()),
        );
        let err = run_rlocus(sys, Vec::new()).expect_err("MIMO ss should fail");
        assert!(err.message().contains("only SISO ss models"));
        assert_eq!(err.identifier(), RLOCUS_ERROR_UNSUPPORTED_MODEL.identifier);
    }

    #[test]
    fn non_model_input_uses_rlocus_invalid_model_identifier() {
        let err = run_rlocus(Value::Num(1.0), Vec::new()).expect_err("should fail");
        assert!(err.message().contains("dynamic system model"));
        assert_eq!(err.identifier(), RLOCUS_ERROR_INVALID_MODEL.identifier);
    }

    #[test]
    fn complex_root_matrix_is_returned_when_branches_are_complex() {
        let sys = tf(vec![1.0], vec![1.0, 2.0, 2.0]);
        let gains = Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
        let result = run_rlocus(sys, vec![gains]).expect("rlocus");
        let Value::ComplexTensor(roots) = result else {
            panic!("expected complex root matrix");
        };
        assert_eq!(roots.shape, vec![2, 1]);
        assert!(roots
            .data
            .iter()
            .any(|(re, im)| (*re + 1.0).abs() < 1.0e-8 && (*im - 1.0).abs() < 1.0e-8));
        assert!(roots
            .data
            .iter()
            .any(|(re, im)| (*re + 1.0).abs() < 1.0e-8 && (*im + 1.0).abs() < 1.0e-8));
    }

    #[test]
    fn discrete_system_uses_closed_loop_polynomial_in_z() {
        let sys = discrete_tf(vec![1.0], vec![1.0, -0.5], 0.1);
        let gains = Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap());
        let roots = run_rlocus(sys, vec![gains]).expect("rlocus");
        let roots = tensor(&roots);
        assert_eq!(roots.shape, vec![1, 1]);
        assert!(roots.data[0].abs() < 1.0e-12);
    }

    #[test]
    fn statement_form_plots_without_error() {
        let sys = tf(vec![1.0, 2.0], vec![1.0, 3.0, 4.0]);
        let _guard = crate::output_count::push_output_count(Some(0));
        let result = run_rlocus(sys, Vec::new()).expect("rlocus");
        assert!(matches!(result, Value::OutputList(outputs) if outputs.is_empty()));
    }

    #[test]
    fn rejects_negative_gain() {
        let sys = tf(vec![1.0], vec![1.0, 1.0]);
        let err = run_rlocus(sys, vec![Value::Num(-1.0)]).expect_err("should fail");
        assert!(err.message().contains("nonnegative"));
        assert_eq!(err.identifier(), RLOCUS_ERROR_INVALID_ARGUMENT.identifier);
    }
}
