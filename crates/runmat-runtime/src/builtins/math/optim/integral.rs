//! MATLAB-compatible `integral` builtin for finite scalar numerical integration.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    LogicalArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::common::call_function;
use crate::builtins::math::optim::type_resolvers::numerical_integral_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "integral";
const DEFAULT_ABS_TOL: f64 = 1.0e-10;
const DEFAULT_REL_TOL: f64 = 1.0e-6;
const DEFAULT_MAX_FUN_EVALS: usize = 10_000;
const MAX_DEPTH: usize = 30;

const INTEGRAL_OUTPUT_Q: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "q",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numerical integral estimate.",
}];

const INTEGRAL_INPUTS_CORE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar integrand callback.",
    },
    BuiltinParamDescriptor {
        name: "xmin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower integration bound.",
    },
    BuiltinParamDescriptor {
        name: "xmax",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper integration bound.",
    },
];

const INTEGRAL_INPUTS_OPTIONS_STRUCT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar integrand callback.",
    },
    BuiltinParamDescriptor {
        name: "xmin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower integration bound.",
    },
    BuiltinParamDescriptor {
        name: "xmax",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper integration bound.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Options struct for AbsTol/RelTol/MaxFunEvals.",
    },
];

const INTEGRAL_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar integrand callback.",
    },
    BuiltinParamDescriptor {
        name: "xmin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower integration bound.",
    },
    BuiltinParamDescriptor {
        name: "xmax",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper integration bound.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Option name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::PropertyValue,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value and additional name/value pairs.",
    },
];

const INTEGRAL_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "q = integral(fun, xmin, xmax)",
        inputs: &INTEGRAL_INPUTS_CORE,
        outputs: &INTEGRAL_OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "q = integral(fun, xmin, xmax, options)",
        inputs: &INTEGRAL_INPUTS_OPTIONS_STRUCT,
        outputs: &INTEGRAL_OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "q = integral(fun, xmin, xmax, name, value, ...)",
        inputs: &INTEGRAL_INPUTS_NAME_VALUE,
        outputs: &INTEGRAL_OUTPUT_Q,
    },
];

const INTEGRAL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTEGRAL.INVALID_ARGUMENT",
    identifier: Some("RunMat:integral:InvalidArgument"),
    when: "Option grammar/name-value parsing is invalid.",
    message: "integral: invalid argument",
};

const INTEGRAL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTEGRAL.INVALID_INPUT",
    identifier: Some("RunMat:integral:InvalidInput"),
    when: "Bounds/integrand/adaptive solver semantics are invalid.",
    message: "integral: invalid input",
};

const INTEGRAL_ERRORS: [BuiltinErrorDescriptor; 2] = [
    INTEGRAL_ERROR_INVALID_ARGUMENT,
    INTEGRAL_ERROR_INVALID_INPUT,
];

pub const INTEGRAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INTEGRAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INTEGRAL_ERRORS,
};

fn integral_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("integral:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn integral_map_error(
    err: RuntimeError,
    fallback: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        integral_error_with_detail(fallback, err.message())
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::integral")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "integral",
    op_kind: GpuOpKind::Custom("adaptive-quadrature"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host adaptive quadrature solver. Callback computations may use GPU-aware builtins, but the adaptive integration loop runs on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::integral")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "integral",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Adaptive integration repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "integral",
    category = "math/optim",
    summary = "Approximate finite scalar definite integrals using adaptive quadrature.",
    keywords = "integral,numerical integration,adaptive quadrature,quadrature,function handle",
    accel = "sink",
    type_resolver(numerical_integral_type),
    descriptor(crate::builtins::math::optim::integral::INTEGRAL_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::integral"
)]
async fn integral_builtin(
    function: Value,
    a: Value,
    b: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let options = IntegralOptions::parse(rest)
        .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_ARGUMENT))?;
    let a = scalar_bound("lower bound", a)
        .await
        .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_INPUT))?;
    let b = scalar_bound("upper bound", b)
        .await
        .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_INPUT))?;
    if a == b {
        return Ok(Value::Num(0.0));
    }

    let sign = if b < a { -1.0 } else { 1.0 };
    let lo = a.min(b);
    let hi = a.max(b);
    let result = integrate_finite_scalar(&function, lo, hi, &options)
        .await
        .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_INPUT))?;
    Ok(Value::Num(sign * result))
}

#[derive(Clone, Copy)]
struct IntegralOptions {
    abs_tol: f64,
    rel_tol: f64,
    max_fun_evals: usize,
}

impl IntegralOptions {
    fn parse(rest: Vec<Value>) -> BuiltinResult<Self> {
        let mut options = Self {
            abs_tol: DEFAULT_ABS_TOL,
            rel_tol: DEFAULT_REL_TOL,
            max_fun_evals: DEFAULT_MAX_FUN_EVALS,
        };
        if rest.is_empty() {
            return Ok(options);
        }
        if rest.len() == 1 {
            return match &rest[0] {
                Value::Struct(fields) => {
                    options.apply_struct(fields)?;
                    Ok(options)
                }
                other => Err(integral_error_with_detail(
                    &INTEGRAL_ERROR_INVALID_ARGUMENT,
                    format!("expected option name/value pairs, got {other:?}"),
                )),
            };
        }
        if !rest.len().is_multiple_of(2) {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "expected option name/value pairs",
            ));
        }
        for pair in rest.chunks(2) {
            let name = option_name(&pair[0])?;
            options.apply_option(&name, &pair[1])?;
        }
        options.validate()?;
        Ok(options)
    }

    fn apply_struct(&mut self, fields: &StructValue) -> BuiltinResult<()> {
        for (name, value) in &fields.fields {
            self.apply_option(name, value)?;
        }
        self.validate()
    }

    fn apply_option(&mut self, name: &str, value: &Value) -> BuiltinResult<()> {
        match name.to_ascii_lowercase().as_str() {
            "abstol" => self.abs_tol = numeric_option("AbsTol", value)?,
            "reltol" => self.rel_tol = numeric_option("RelTol", value)?,
            "maxfunevals" | "maxintervalcount" => {
                let parsed = integer_option(name, value)?;
                if parsed < 5 {
                    return Err(integral_error_with_detail(
                        &INTEGRAL_ERROR_INVALID_ARGUMENT,
                        "MaxFunEvals must be an integer scalar >= 5",
                    ));
                }
                self.max_fun_evals = parsed;
            }
            "arrayvalued" => {
                if bool_option("ArrayValued", value)? {
                    return Err(integral_error_with_detail(
                        &INTEGRAL_ERROR_INVALID_ARGUMENT,
                        "ArrayValued true is not supported yet",
                    ));
                }
            }
            other => {
                return Err(integral_error_with_detail(
                    &INTEGRAL_ERROR_INVALID_ARGUMENT,
                    format!("unsupported option {other}"),
                ))
            }
        }
        Ok(())
    }

    fn validate(&self) -> BuiltinResult<()> {
        if self.abs_tol < 0.0 {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "AbsTol must be nonnegative",
            ));
        }
        if self.rel_tol < 0.0 {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "RelTol must be nonnegative",
            ));
        }
        if self.abs_tol == 0.0 && self.rel_tol == 0.0 {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "AbsTol and RelTol cannot both be zero",
            ));
        }
        Ok(())
    }
}

fn option_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        other => Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option names must be strings, got {other:?}"),
        )),
    }
}

async fn scalar_bound(label: &str, value: Value) -> BuiltinResult<f64> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    let parsed = match value {
        Value::Num(n) => n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => {
            if b {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_INPUT,
                format!("{label} must be a finite real scalar, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            format!("{label} must be finite"),
        ))
    }
}

fn numeric_option(name: &str, value: &Value) -> BuiltinResult<f64> {
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(Tensor { data, .. }) if data.len() == 1 => data[0],
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                format!("option {name} must be numeric, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be finite"),
        ))
    }
}

fn integer_option(name: &str, value: &Value) -> BuiltinResult<usize> {
    let parsed = numeric_option(name, value)?;
    if parsed < 0.0 {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be nonnegative"),
        ));
    }
    if parsed.fract() != 0.0 {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be an integer scalar"),
        ));
    }
    Ok(parsed as usize)
}

fn bool_option(name: &str, value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw == 0 || raw == 1 {
                Ok(raw != 0)
            } else {
                Err(integral_error_with_detail(
                    &INTEGRAL_ERROR_INVALID_ARGUMENT,
                    format!("option {name} must be logical scalar"),
                ))
            }
        }
        other => Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be logical scalar, got {other:?}"),
        )),
    }
}

async fn integrate_finite_scalar(
    function: &Value,
    a: f64,
    b: f64,
    options: &IntegralOptions,
) -> BuiltinResult<f64> {
    let fa = call_integrand(function, a).await?;
    let m = 0.5 * (a + b);
    let fm = call_integrand(function, m).await?;
    let fb = call_integrand(function, b).await?;
    let mut evals = 3usize;
    let whole = simpson(a, b, fa, fm, fb);
    let tol = options.abs_tol.max(options.rel_tol * whole.abs());
    adaptive_simpson(
        function,
        SimpsonState {
            a,
            b,
            fa,
            fm,
            fb,
            whole,
            tol,
            depth: MAX_DEPTH,
        },
        &mut evals,
        options.max_fun_evals,
    )
    .await
}

#[derive(Clone, Copy)]
struct SimpsonState {
    a: f64,
    b: f64,
    fa: f64,
    fm: f64,
    fb: f64,
    whole: f64,
    tol: f64,
    depth: usize,
}

#[async_recursion::async_recursion(?Send)]
async fn adaptive_simpson(
    function: &Value,
    state: SimpsonState,
    evals: &mut usize,
    max_fun_evals: usize,
) -> BuiltinResult<f64> {
    if *evals + 2 > max_fun_evals {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "exceeded maximum function evaluations",
        ));
    }

    let c = 0.5 * (state.a + state.b);
    let d = 0.5 * (state.a + c);
    let e = 0.5 * (c + state.b);
    let fd = call_integrand(function, d).await?;
    let fe = call_integrand(function, e).await?;
    *evals += 2;

    let left = simpson(state.a, c, state.fa, fd, state.fm);
    let right = simpson(c, state.b, state.fm, fe, state.fb);
    let refined = left + right;
    let error = refined - state.whole;
    if error.abs() <= 15.0 * state.tol {
        return Ok(refined + error / 15.0);
    }
    if state.depth == 0 {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "adaptive quadrature did not converge",
        ));
    }

    let left_value = adaptive_simpson(
        function,
        SimpsonState {
            a: state.a,
            b: c,
            fa: state.fa,
            fm: fd,
            fb: state.fm,
            whole: left,
            tol: state.tol * 0.5,
            depth: state.depth - 1,
        },
        evals,
        max_fun_evals,
    )
    .await?;
    let right_value = adaptive_simpson(
        function,
        SimpsonState {
            a: c,
            b: state.b,
            fa: state.fm,
            fm: fe,
            fb: state.fb,
            whole: right,
            tol: state.tol * 0.5,
            depth: state.depth - 1,
        },
        evals,
        max_fun_evals,
    )
    .await?;
    Ok(left_value + right_value)
}

fn simpson(a: f64, b: f64, fa: f64, fm: f64, fb: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}

async fn call_integrand(function: &Value, x: f64) -> BuiltinResult<f64> {
    let value = call_function(function, vec![Value::Num(x)]).await?;
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    match value {
        Value::Num(n) if n.is_finite() => Ok(n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 && tensor.data[0].is_finite() => {
            Ok(tensor.data[0])
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        Value::Num(_) | Value::Tensor(_) => Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "function value must be a finite real scalar",
        )),
        other => Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            format!("function value must be real numeric scalar, got {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    const INTEGRAL_HELPER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "fx",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integrand scalar value.",
    }];

    const INTEGRAL_HELPER_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integrand sample location.",
    }];

    const INTEGRAL_HELPER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
        [BuiltinSignatureDescriptor {
            label: "fx = __integral_helper(x)",
            inputs: &INTEGRAL_HELPER_INPUTS,
            outputs: &INTEGRAL_HELPER_OUTPUT,
        }];

    const INTEGRAL_HELPER_ERRORS: [BuiltinErrorDescriptor; 0] = [];

    pub const INTEGRAL_TEST_HELPER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &INTEGRAL_HELPER_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::HiddenInternal,
        errors: &INTEGRAL_HELPER_ERRORS,
    };

    #[runtime_builtin(
        name = "__integral_square",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn square_helper(x: Value) -> crate::BuiltinResult<Value> {
        let x = scalar_bound("x", x).await?;
        Ok(Value::Num(x * x))
    }

    #[runtime_builtin(
        name = "__integral_vector",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn vector_helper(_x: Value) -> crate::BuiltinResult<Value> {
        Ok(Value::Tensor(
            Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
        ))
    }

    #[runtime_builtin(
        name = "__integral_nan",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn nan_helper(_x: Value) -> crate::BuiltinResult<Value> {
        Ok(Value::Num(f64::NAN))
    }

    fn run(function: Value, a: f64, b: f64) -> crate::BuiltinResult<Value> {
        block_on(integral_builtin(
            function,
            Value::Num(a),
            Value::Num(b),
            Vec::new(),
        ))
    }

    #[test]
    fn integral_test_helper_descriptor_is_attached_shape() {
        assert_eq!(
            INTEGRAL_TEST_HELPER_DESCRIPTOR.signatures[0].label,
            "fx = __integral_helper(x)"
        );
    }

    #[test]
    fn integrates_named_sine_function() {
        let result = run(
            Value::FunctionHandle("sin".into()),
            0.0,
            std::f64::consts::PI,
        )
        .expect("integral");
        match result {
            Value::Num(value) => assert!((value - 2.0).abs() < 1.0e-7),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn integrates_polynomial_helper() {
        let result =
            run(Value::FunctionHandle("__integral_square".into()), 0.0, 1.0).expect("integral");
        match result {
            Value::Num(value) => assert!((value - (1.0 / 3.0)).abs() < 1.0e-9),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn reversed_bounds_negate_result() {
        let result = run(
            Value::FunctionHandle("sin".into()),
            std::f64::consts::PI,
            0.0,
        )
        .expect("integral");
        match result {
            Value::Num(value) => assert!((value + 2.0).abs() < 1.0e-7),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn zero_width_interval_returns_zero_without_callback() {
        let result =
            run(Value::FunctionHandle("__integral_nan".into()), 1.0, 1.0).expect("integral");
        assert!(matches!(result, Value::Num(0.0)));
    }

    #[test]
    fn rejects_vector_valued_integrand_for_initial_scope() {
        let err = run(Value::FunctionHandle("__integral_vector".into()), 0.0, 1.0).unwrap_err();
        assert!(err.message().contains("finite real scalar"));
    }

    #[test]
    fn rejects_nonfinite_integrand_values() {
        let err = run(Value::FunctionHandle("__integral_nan".into()), 0.0, 1.0).unwrap_err();
        assert!(err.message().contains("finite real scalar"));
    }

    #[test]
    fn accepts_tolerance_name_value_options() {
        let result = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(std::f64::consts::PI),
            vec![
                Value::from("AbsTol"),
                Value::Num(1.0e-12),
                Value::from("RelTol"),
                Value::Num(1.0e-8),
            ],
        ))
        .expect("integral");
        match result {
            Value::Num(value) => assert!((value - 2.0).abs() < 1.0e-8),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn rejects_too_small_max_fun_evals() {
        let err = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("MaxFunEvals"), Value::Num(4.0)],
        ))
        .unwrap_err();
        assert!(err.message().contains("integer scalar >= 5"));
    }

    #[test]
    fn rejects_fractional_max_fun_evals() {
        let err = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("MaxFunEvals"), Value::Num(5.5)],
        ))
        .unwrap_err();
        assert!(err.message().contains("integer scalar"));
    }

    #[test]
    fn integral_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = INTEGRAL_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "q = integral(fun, xmin, xmax)",
                "q = integral(fun, xmin, xmax, options)",
                "q = integral(fun, xmin, xmax, name, value, ...)",
            ]
        );

        let codes: Vec<&str> = INTEGRAL_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec!["RM.INTEGRAL.INVALID_ARGUMENT", "RM.INTEGRAL.INVALID_INPUT"]
        );
    }

    #[test]
    fn integral_bad_name_value_pairs_use_stable_identifier() {
        let err = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("AbsTol")],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:integral:InvalidArgument"));
    }
}
