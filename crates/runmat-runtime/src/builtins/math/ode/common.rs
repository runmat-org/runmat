use nalgebra::{DMatrix, DVector};
use runmat_builtins::{BuiltinExtensionDescriptor, StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::math::optim::common::{call_function, lookup_option, value_to_real_vector};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const DEFAULT_REL_TOL: f64 = 1.0e-3;
const DEFAULT_ABS_TOL: f64 = 1.0e-6;
const DEFAULT_MAX_STEPS: usize = 100_000;
const ODE15S_NEWTON_MAX_ITERS: usize = 8;
const ODE15S_NEWTON_DAMPING_TRIES: usize = 6;

macro_rules! define_ode_integer_contract {
    ($builtin:literal, $identifier:literal) => {
        const INTEGER_TSPAN_EXTENSION: runmat_builtins::BuiltinExtensionDescriptor =
            runmat_builtins::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-integer-tspan"),
                mode: runmat_builtins::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " with native-class integer tspan is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "IntegerTspanExtension")),
            };
        const INTEGER_Y0_EXTENSION: runmat_builtins::BuiltinExtensionDescriptor =
            runmat_builtins::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-integer-y0"),
                mode: runmat_builtins::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " with native-class integer y0 is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "IntegerY0Extension")),
            };
        const INTEGER_OPTION_EXTENSION: runmat_builtins::BuiltinExtensionDescriptor =
            runmat_builtins::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-integer-option"),
                mode: runmat_builtins::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " with native-class integer numeric options is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "IntegerOptionExtension")),
            };
        const INTEGER_CALLBACK_EXTENSION: runmat_builtins::BuiltinExtensionDescriptor =
            runmat_builtins::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-integer-callback-result"),
                mode: runmat_builtins::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " with a native-class integer derivative result is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "IntegerCallbackExtension")),
            };
        const LOGICAL_NUMERIC_EXTENSION: runmat_builtins::BuiltinExtensionDescriptor =
            runmat_builtins::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-logical-numeric"),
                mode: runmat_builtins::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " with logical solver data is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "LogicalNumericExtension")),
            };
        const RESIDENT_INPUT_EXTENSION: runmat_builtins::BuiltinExtensionDescriptor =
            runmat_builtins::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-resident-input"),
                mode: runmat_builtins::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " host fallback for explicit gpuArray values is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "ResidentInputExtension")),
            };
        pub const EXTENSIONS: [runmat_builtins::BuiltinExtensionDescriptor; 6] = [
            INTEGER_TSPAN_EXTENSION,
            INTEGER_Y0_EXTENSION,
            INTEGER_OPTION_EXTENSION,
            INTEGER_CALLBACK_EXTENSION,
            LOGICAL_NUMERIC_EXTENSION,
            RESIDENT_INPUT_EXTENSION,
        ];
        const INTEGER_TSPAN_INPUT: [runmat_builtins::BuiltinIntegerInputCapability; 1] =
            [runmat_builtins::BuiltinIntegerInputCapability {
                name: "tspan",
                classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
                availability: runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: runmat_builtins::BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "R2026a documents single and double tspan; integer time points are gated and must cross the binary64 solver boundary exactly.",
            }];
        const INTEGER_Y0_INPUT: [runmat_builtins::BuiltinIntegerInputCapability; 1] =
            [runmat_builtins::BuiltinIntegerInputCapability {
                name: "y0",
                classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
                availability: runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: runmat_builtins::BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "R2026a documents single and double initial conditions; integer states are gated and must cross the binary64 solver boundary exactly.",
            }];
        const INTEGER_OPTION_INPUT: [runmat_builtins::BuiltinIntegerInputCapability; 1] =
            [runmat_builtins::BuiltinIntegerInputCapability {
                name: "numeric option field",
                classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
                availability: runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: runmat_builtins::BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "Typed numeric ODE option fields are gated before recursive gather; floating controls convert only when exact and MaxSteps remains structural.",
            }];
        const INTEGER_CALLBACK_INPUT: [runmat_builtins::BuiltinIntegerInputCapability; 1] =
            [runmat_builtins::BuiltinIntegerInputCapability {
                name: "odefun result",
                classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
                availability: runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: runmat_builtins::BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "R2026a requires single or double derivative output; RunMat gates integer derivatives and checks every floating conversion.",
            }];
        pub const INTEGER_CAPABILITIES: [runmat_builtins::BuiltinIntegerCapabilityDescriptor; 4] = [
            runmat_builtins::BuiltinIntegerCapabilityDescriptor { form: concat!("[t,y] = ", $builtin, "(odefun, integer_tspan, y0, ___)"), inputs: &INTEGER_TSPAN_INPUT, computation_domain: runmat_builtins::BuiltinIntegerComputationDomain::FloatingPoint, output_class: runmat_builtins::BuiltinIntegerOutputClassRule::Double, overflow: runmat_builtins::BuiltinIntegerOverflowRule::Error, backend: runmat_builtins::BuiltinIntegerBackendRule::GatherFallback, overload: runmat_builtins::BuiltinIntegerOverloadKind::Multiple, notes: "Integer time points are a checked RunMat-only input to the host double integration loop." },
            runmat_builtins::BuiltinIntegerCapabilityDescriptor { form: concat!("[t,y] = ", $builtin, "(odefun, tspan, integer_y0, ___)"), inputs: &INTEGER_Y0_INPUT, computation_domain: runmat_builtins::BuiltinIntegerComputationDomain::FloatingPoint, output_class: runmat_builtins::BuiltinIntegerOutputClassRule::Double, overflow: runmat_builtins::BuiltinIntegerOverflowRule::Error, backend: runmat_builtins::BuiltinIntegerBackendRule::GatherFallback, overload: runmat_builtins::BuiltinIntegerOverloadKind::Multiple, notes: "Integer initial states are a checked RunMat-only input and produce floating solver outputs." },
            runmat_builtins::BuiltinIntegerCapabilityDescriptor { form: concat!("[t,y] = ", $builtin, "(___, options_with_integer_field)"), inputs: &INTEGER_OPTION_INPUT, computation_domain: runmat_builtins::BuiltinIntegerComputationDomain::FunctionSpecific, output_class: runmat_builtins::BuiltinIntegerOutputClassRule::Double, overflow: runmat_builtins::BuiltinIntegerOverflowRule::Error, backend: runmat_builtins::BuiltinIntegerBackendRule::GatherFallback, overload: runmat_builtins::BuiltinIntegerOverloadKind::StructuralParameter, notes: "Integer option fields are independently gated; tolerances and step sizes cross exactly while the RunMat MaxSteps extension remains an exact count." },
            runmat_builtins::BuiltinIntegerCapabilityDescriptor { form: concat!("[t,y] = ", $builtin, "(integer_returning_odefun, ___)"), inputs: &INTEGER_CALLBACK_INPUT, computation_domain: runmat_builtins::BuiltinIntegerComputationDomain::FloatingPoint, output_class: runmat_builtins::BuiltinIntegerOutputClassRule::Double, overflow: runmat_builtins::BuiltinIntegerOverflowRule::Error, backend: runmat_builtins::BuiltinIntegerBackendRule::GatherFallback, overload: runmat_builtins::BuiltinIntegerOverloadKind::Multiple, notes: "Integer derivative values are a RunMat-only callback extension and enter the solver only after exact representability checks." },
        ];
        pub(crate) const ODE_COMPATIBILITY_EXTENSIONS: crate::builtins::math::ode::common::OdeCompatibilityExtensions =
            crate::builtins::math::ode::common::OdeCompatibilityExtensions {
                integer_tspan: &INTEGER_TSPAN_EXTENSION,
                integer_y0: &INTEGER_Y0_EXTENSION,
                integer_option: &INTEGER_OPTION_EXTENSION,
                integer_callback: &INTEGER_CALLBACK_EXTENSION,
                logical_numeric: &LOGICAL_NUMERIC_EXTENSION,
                resident_input: &RESIDENT_INPUT_EXTENSION,
            };
    };
}
pub(crate) use define_ode_integer_contract;

#[derive(Clone, Copy)]
pub(crate) struct OdeCompatibilityExtensions {
    pub(crate) integer_tspan: &'static BuiltinExtensionDescriptor,
    pub(crate) integer_y0: &'static BuiltinExtensionDescriptor,
    pub(crate) integer_option: &'static BuiltinExtensionDescriptor,
    pub(crate) integer_callback: &'static BuiltinExtensionDescriptor,
    pub(crate) logical_numeric: &'static BuiltinExtensionDescriptor,
    pub(crate) resident_input: &'static BuiltinExtensionDescriptor,
}

#[derive(Clone, Copy)]
pub(crate) enum OdeMethod {
    Ode45,
    Ode23,
    Ode15s,
}

impl OdeMethod {
    fn embedded_error_order(self) -> f64 {
        match self {
            // Embedded RK error estimate uses the lower-order formula.
            Self::Ode45 => 4.0,
            Self::Ode23 => 2.0,
            Self::Ode15s => 1.0,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct OdeOptions {
    pub rel_tol: f64,
    pub abs_tol: f64,
    pub initial_step: Option<f64>,
    pub max_step: Option<f64>,
    pub max_steps: usize,
}

#[derive(Clone)]
pub(crate) struct OdeInput {
    pub tspan: Vec<f64>,
    pub y0: Vec<f64>,
    pub y_shape: Vec<usize>,
    pub scalar_state: bool,
    extensions: OdeCompatibilityExtensions,
}

pub(crate) struct OdeResult {
    pub t: Vec<f64>,
    pub y_rows: Vec<Vec<f64>>,
}

pub(crate) fn ode_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

pub(crate) fn parse_options(
    name: &str,
    value: Option<&Value>,
) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options.clone())),
        Some(other) => Err(ode_error(
            name,
            format!("{name}: options must be a struct, got {other:?}"),
        )),
    }
}

pub(crate) fn ode_options_from_struct(
    name: &str,
    options: Option<&StructValue>,
) -> BuiltinResult<OdeOptions> {
    let rel_tol = option_f64(name, options, "RelTol", DEFAULT_REL_TOL)?;
    let abs_tol = option_f64(name, options, "AbsTol", DEFAULT_ABS_TOL)?;
    if rel_tol <= 0.0 || abs_tol <= 0.0 {
        return Err(ode_error(
            name,
            format!("{name}: RelTol and AbsTol must be positive"),
        ));
    }

    let initial_step = option_optional_f64(name, options, "InitialStep")?;
    if let Some(step) = initial_step {
        if step <= 0.0 {
            return Err(ode_error(
                name,
                format!("{name}: InitialStep must be positive"),
            ));
        }
    }

    let max_step = option_optional_f64(name, options, "MaxStep")?;
    if let Some(step) = max_step {
        if step <= 0.0 {
            return Err(ode_error(name, format!("{name}: MaxStep must be positive")));
        }
    }

    let max_steps = option_usize(name, options, "MaxSteps", DEFAULT_MAX_STEPS)?.max(1);

    Ok(OdeOptions {
        rel_tol,
        abs_tol,
        initial_step,
        max_step,
        max_steps,
    })
}

pub(crate) async fn prepare_ode_options(
    name: &str,
    options: Option<StructValue>,
    extensions: OdeCompatibilityExtensions,
) -> BuiltinResult<Option<StructValue>> {
    let Some(options) = options else {
        return Ok(None);
    };
    let value = Value::Struct(options);
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(extensions.resident_input, name)?;
    }
    if let Value::Struct(options) = &value {
        for field in ["RelTol", "AbsTol", "InitialStep", "MaxStep", "MaxSteps"] {
            let Some(field_value) = lookup_option(options, field) else {
                continue;
            };
            if crate::builtins::common::validation::value_contains_native_integer_class(field_value)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    extensions.integer_option,
                    name,
                )?;
                if field != "MaxSteps"
                    && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(
                        field_value,
                    )
                    .await?
                {
                    return Err(ode_error(
                        name,
                        format!(
                            "{name}: integer option {field} must be exactly representable as double"
                        ),
                    ));
                }
            }
        }
    }
    let gathered = crate::dispatcher::gather_if_needed_async(&value).await?;
    let Value::Struct(options) = gathered else {
        unreachable!("gather preserves struct shape")
    };
    Ok(Some(options))
}

pub(crate) async fn parse_ode_input(
    name: &str,
    tspan: Value,
    y0: Value,
    extensions: OdeCompatibilityExtensions,
) -> BuiltinResult<OdeInput> {
    let tspan_value =
        prepare_ode_numeric_value(name, "tspan", tspan, extensions.integer_tspan, extensions)
            .await?;
    let tspan = value_to_real_vector(name, tspan_value).await?;
    if tspan.len() < 2 {
        return Err(ode_error(
            name,
            format!("{name}: tspan must contain at least two time points"),
        ));
    }
    if tspan.windows(2).any(|w| w[0] == w[1]) {
        return Err(ode_error(
            name,
            format!("{name}: tspan values must be strictly monotonic"),
        ));
    }
    let forward = tspan[tspan.len() - 1] > tspan[0];
    if tspan.windows(2).any(|w| (w[1] > w[0]) != forward) {
        return Err(ode_error(
            name,
            format!("{name}: tspan values must be strictly monotonic"),
        ));
    }

    let y0_value =
        prepare_ode_numeric_value(name, "y0", y0, extensions.integer_y0, extensions).await?;
    let (y0, y_shape, scalar_state) = match y0_value {
        Value::Num(n) => (vec![n], vec![1, 1], true),
        Value::Int(i) => (vec![i.to_f64()], vec![1, 1], true),
        Value::Bool(b) => (vec![if b { 1.0 } else { 0.0 }], vec![1, 1], true),
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64(&tensor);
            if values.is_empty() {
                return Err(ode_error(name, format!("{name}: y0 cannot be empty")));
            }
            (values, tensor.shape, false)
        }
        Value::LogicalArray(logical) => {
            if logical.data.is_empty() {
                return Err(ode_error(name, format!("{name}: y0 cannot be empty")));
            }
            (
                logical
                    .data
                    .iter()
                    .map(|v| if *v == 0 { 0.0 } else { 1.0 })
                    .collect(),
                logical.shape,
                false,
            )
        }
        other => {
            return Err(ode_error(
                name,
                format!("{name}: y0 must be real numeric, got {other:?}"),
            ))
        }
    };

    if y0.iter().any(|value| !value.is_finite()) {
        return Err(ode_error(name, format!("{name}: y0 values must be finite")));
    }

    Ok(OdeInput {
        tspan,
        y0,
        y_shape,
        scalar_state,
        extensions,
    })
}

async fn prepare_ode_numeric_value(
    name: &str,
    role: &str,
    value: Value,
    integer_extension: &'static BuiltinExtensionDescriptor,
    extensions: OdeCompatibilityExtensions,
) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_contains_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(integer_extension, name)?;
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await?
        {
            return Err(ode_error(
                name,
                format!("{name}: integer {role} must be exactly representable as double"),
            ));
        }
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(&value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(extensions.logical_numeric, name)?;
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(extensions.resident_input, name)?;
    }
    crate::dispatcher::gather_if_needed_async(&value).await
}

pub(crate) async fn solve_ode(
    name: &str,
    method: OdeMethod,
    function: &Value,
    input: &OdeInput,
    options: &OdeOptions,
) -> BuiltinResult<OdeResult> {
    let start = input.tspan[0];
    let end = input.tspan[input.tspan.len() - 1];
    let direction = if end >= start { 1.0 } else { -1.0 };
    let dense_output = input.tspan.len() == 2;

    let mut t = start;
    let mut y = input.y0.clone();
    let mut history_t = vec![t];
    let mut history_y = vec![y.clone()];
    let mut steps = 0usize;
    let mut h = options
        .initial_step
        .unwrap_or_else(|| ((end - start).abs() * 1.0e-2).max(1.0e-6))
        .abs()
        * direction;

    if let Some(max_step) = options.max_step {
        h = h.abs().min(max_step) * direction;
    }
    let min_step = ((end - start).abs().max(1.0) * 1.0e-14).max(1.0e-12);

    if dense_output {
        while remaining_distance(t, end, direction) > 0.0 {
            if steps >= options.max_steps {
                return Err(ode_error(
                    name,
                    format!("{name}: exceeded maximum step count"),
                ));
            }
            let max_h = remaining_distance(t, end, direction);
            h = clamp_step(h, direction, max_h, options.max_step);
            if h.abs() < min_step {
                return Err(ode_error(
                    name,
                    format!("{name}: step size underflow before reaching final time"),
                ));
            }

            let step = attempt_step(name, method, function, t, &y, h, input, options).await?;
            if step.accepted {
                t += h;
                y = step.y_next;
                history_t.push(t);
                history_y.push(y.clone());
                steps += 1;
                h = scaled_next_step(
                    h,
                    step.error_norm,
                    method.embedded_error_order(),
                    direction,
                    options.max_step,
                );
            } else {
                h = scaled_next_step(
                    h,
                    step.error_norm,
                    method.embedded_error_order(),
                    direction,
                    options.max_step,
                );
                if h.abs() < min_step {
                    return Err(ode_error(
                        name,
                        format!("{name}: step size underflow during error control"),
                    ));
                }
            }
        }
    } else {
        for &target_t in input.tspan.iter().skip(1) {
            while remaining_distance(t, target_t, direction) > 0.0 {
                if steps >= options.max_steps {
                    return Err(ode_error(
                        name,
                        format!("{name}: exceeded maximum step count"),
                    ));
                }
                let max_h = remaining_distance(t, target_t, direction);
                h = clamp_step(h, direction, max_h, options.max_step);
                if h.abs() < min_step {
                    return Err(ode_error(
                        name,
                        format!(
                            "{name}: step size underflow before reaching requested tspan point"
                        ),
                    ));
                }

                let step = attempt_step(name, method, function, t, &y, h, input, options).await?;
                if step.accepted {
                    t += h;
                    y = step.y_next;
                    steps += 1;
                    h = scaled_next_step(
                        h,
                        step.error_norm,
                        method.embedded_error_order(),
                        direction,
                        options.max_step,
                    );
                } else {
                    h = scaled_next_step(
                        h,
                        step.error_norm,
                        method.embedded_error_order(),
                        direction,
                        options.max_step,
                    );
                    if h.abs() < min_step {
                        return Err(ode_error(
                            name,
                            format!("{name}: step size underflow during error control"),
                        ));
                    }
                }
            }

            history_t.push(t);
            history_y.push(y.clone());
        }
    }

    Ok(OdeResult {
        t: history_t,
        y_rows: history_y,
    })
}

struct StepAttempt {
    accepted: bool,
    error_norm: f64,
    y_next: Vec<f64>,
}

async fn attempt_step(
    name: &str,
    method: OdeMethod,
    function: &Value,
    t: f64,
    y: &[f64],
    h: f64,
    input: &OdeInput,
    options: &OdeOptions,
) -> BuiltinResult<StepAttempt> {
    let (y_next, err) = match method {
        OdeMethod::Ode45 => step_ode45(name, function, t, y, h, input).await?,
        OdeMethod::Ode23 => step_ode23(name, function, t, y, h, input).await?,
        OdeMethod::Ode15s => step_ode15s(name, function, t, y, h, input).await?,
    };
    let error_norm = scaled_error_norm(y, &y_next, &err, options.rel_tol, options.abs_tol);
    Ok(StepAttempt {
        accepted: error_norm <= 1.0 || error_norm == 0.0,
        error_norm,
        y_next,
    })
}

async fn step_ode45(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    h: f64,
    input: &OdeInput,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let k1 = eval_rhs(name, function, t, y, input).await?;
    let y2 = lincomb(y, h, &[(&k1, 1.0 / 5.0)]);
    let k2 = eval_rhs(name, function, t + h * (1.0 / 5.0), &y2, input).await?;

    let y3 = lincomb(y, h, &[(&k1, 3.0 / 40.0), (&k2, 9.0 / 40.0)]);
    let k3 = eval_rhs(name, function, t + h * (3.0 / 10.0), &y3, input).await?;

    let y4 = lincomb(
        y,
        h,
        &[(&k1, 44.0 / 45.0), (&k2, -56.0 / 15.0), (&k3, 32.0 / 9.0)],
    );
    let k4 = eval_rhs(name, function, t + h * (4.0 / 5.0), &y4, input).await?;

    let y5 = lincomb(
        y,
        h,
        &[
            (&k1, 19372.0 / 6561.0),
            (&k2, -25360.0 / 2187.0),
            (&k3, 64448.0 / 6561.0),
            (&k4, -212.0 / 729.0),
        ],
    );
    let k5 = eval_rhs(name, function, t + h * (8.0 / 9.0), &y5, input).await?;

    let y6 = lincomb(
        y,
        h,
        &[
            (&k1, 9017.0 / 3168.0),
            (&k2, -355.0 / 33.0),
            (&k3, 46732.0 / 5247.0),
            (&k4, 49.0 / 176.0),
            (&k5, -5103.0 / 18656.0),
        ],
    );
    let k6 = eval_rhs(name, function, t + h, &y6, input).await?;

    let y_high = lincomb(
        y,
        h,
        &[
            (&k1, 35.0 / 384.0),
            (&k3, 500.0 / 1113.0),
            (&k4, 125.0 / 192.0),
            (&k5, -2187.0 / 6784.0),
            (&k6, 11.0 / 84.0),
        ],
    );
    let k7 = eval_rhs(name, function, t + h, &y_high, input).await?;

    let y_low = lincomb(
        y,
        h,
        &[
            (&k1, 5179.0 / 57600.0),
            (&k3, 7571.0 / 16695.0),
            (&k4, 393.0 / 640.0),
            (&k5, -92097.0 / 339200.0),
            (&k6, 187.0 / 2100.0),
            (&k7, 1.0 / 40.0),
        ],
    );
    let err = y_high
        .iter()
        .zip(y_low.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    Ok((y_high, err))
}

async fn step_ode23(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    h: f64,
    input: &OdeInput,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let k1 = eval_rhs(name, function, t, y, input).await?;
    let y2 = lincomb(y, h, &[(&k1, 0.5)]);
    let k2 = eval_rhs(name, function, t + h * 0.5, &y2, input).await?;

    let y3 = lincomb(y, h, &[(&k2, 3.0 / 4.0)]);
    let k3 = eval_rhs(name, function, t + h * (3.0 / 4.0), &y3, input).await?;

    let y_high = lincomb(
        y,
        h,
        &[(&k1, 2.0 / 9.0), (&k2, 1.0 / 3.0), (&k3, 4.0 / 9.0)],
    );
    let k4 = eval_rhs(name, function, t + h, &y_high, input).await?;

    let y_low = lincomb(
        y,
        h,
        &[
            (&k1, 7.0 / 24.0),
            (&k2, 1.0 / 4.0),
            (&k3, 1.0 / 3.0),
            (&k4, 1.0 / 8.0),
        ],
    );
    let err = y_high
        .iter()
        .zip(y_low.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    Ok((y_high, err))
}

async fn step_ode15s(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    h: f64,
    input: &OdeInput,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let f_n = eval_rhs(name, function, t, y, input).await?;
    let predictor = lincomb(y, h, &[(&f_n, 1.0)]);
    let target_t = t + h;

    let Some(full_step) =
        implicit_euler_newton(name, function, target_t, y, h, &predictor, input).await?
    else {
        return Ok((predictor, vec![f64::INFINITY; y.len()]));
    };

    let half_h = h * 0.5;
    let midpoint_t = t + half_h;
    let half_predictor = lincomb(y, half_h, &[(&f_n, 1.0)]);
    let Some(midpoint) = implicit_euler_newton(
        name,
        function,
        midpoint_t,
        y,
        half_h,
        &half_predictor,
        input,
    )
    .await?
    else {
        return Ok((full_step, vec![f64::INFINITY; y.len()]));
    };

    let f_midpoint = eval_rhs(name, function, midpoint_t, &midpoint, input).await?;
    let second_half_predictor = lincomb(&midpoint, half_h, &[(&f_midpoint, 1.0)]);
    let Some(next) = implicit_euler_newton(
        name,
        function,
        target_t,
        &midpoint,
        half_h,
        &second_half_predictor,
        input,
    )
    .await?
    else {
        return Ok((full_step, vec![f64::INFINITY; y.len()]));
    };

    let err = next
        .iter()
        .zip(full_step.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    Ok((next, err))
}

async fn implicit_euler_newton(
    name: &str,
    function: &Value,
    target_t: f64,
    y_base: &[f64],
    h: f64,
    initial: &[f64],
    input: &OdeInput,
) -> BuiltinResult<Option<Vec<f64>>> {
    let mut next = initial.to_vec();
    let (mut residual, mut f_next) =
        implicit_euler_residual(name, function, target_t, y_base, h, &next, input).await?;

    if newton_converged(y_base, &next, &residual) {
        return Ok(Some(next));
    }

    for _ in 0..ODE15S_NEWTON_MAX_ITERS {
        let jacobian =
            implicit_euler_jacobian(name, function, target_t, &next, h, &f_next, input).await?;
        let matrix = DMatrix::from_row_slice(next.len(), next.len(), &jacobian);
        let rhs = -DVector::from_column_slice(&residual);
        let Some(delta) = matrix.lu().solve(&rhs) else {
            return Ok(None);
        };
        if delta.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }

        let residual_norm = max_abs(&residual);
        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..ODE15S_NEWTON_DAMPING_TRIES {
            let trial = next
                .iter()
                .zip(delta.iter())
                .map(|(value, step)| value + alpha * step)
                .collect::<Vec<_>>();
            if !all_finite(&trial) {
                alpha *= 0.5;
                continue;
            }

            let (trial_residual, trial_f_next) =
                implicit_euler_residual(name, function, target_t, y_base, h, &trial, input).await?;
            let trial_norm = max_abs(&trial_residual);
            if trial_norm < residual_norm || newton_converged(y_base, &trial, &trial_residual) {
                next = trial;
                residual = trial_residual;
                f_next = trial_f_next;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if !accepted {
            return Ok(None);
        }

        let step_norm = delta
            .iter()
            .fold(0.0_f64, |acc, value| acc.max((alpha * value).abs()));
        if newton_converged(y_base, &next, &residual)
            || step_norm <= 1.0e-9 * (1.0 + max_abs(&next))
        {
            return Ok(Some(next));
        }
    }

    Ok(None)
}

async fn implicit_euler_residual(
    name: &str,
    function: &Value,
    target_t: f64,
    y_base: &[f64],
    h: f64,
    y_trial: &[f64],
    input: &OdeInput,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let f_trial = eval_rhs(name, function, target_t, y_trial, input).await?;
    let residual = y_trial
        .iter()
        .zip(y_base.iter())
        .zip(f_trial.iter())
        .map(|((trial, base), f)| trial - base - h * f)
        .collect::<Vec<_>>();
    Ok((residual, f_trial))
}

async fn implicit_euler_jacobian(
    name: &str,
    function: &Value,
    target_t: f64,
    y: &[f64],
    h: f64,
    f_base: &[f64],
    input: &OdeInput,
) -> BuiltinResult<Vec<f64>> {
    let n = y.len();
    let mut jacobian = vec![0.0; n * n];

    for col in 0..n {
        let mut perturbed = y.to_vec();
        let step = f64::EPSILON.sqrt() * (y[col].abs() + 1.0);
        perturbed[col] += step;
        let f_perturbed = eval_rhs(name, function, target_t, &perturbed, input).await?;
        for row in 0..n {
            let df_dy = (f_perturbed[row] - f_base[row]) / step;
            jacobian[row * n + col] = if row == col { 1.0 } else { 0.0 } - h * df_dy;
        }
    }

    Ok(jacobian)
}

fn newton_converged(y_base: &[f64], y_next: &[f64], residual: &[f64]) -> bool {
    max_abs(residual) <= 1.0e-10 * (1.0 + max_abs(y_base).max(max_abs(y_next)))
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

async fn eval_rhs(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    input: &OdeInput,
) -> BuiltinResult<Vec<f64>> {
    let y_arg = if input.scalar_state {
        Value::Num(y[0])
    } else {
        Value::Tensor(
            Tensor::new(y.to_vec(), input.y_shape.clone())
                .map_err(|e| ode_error(name, format!("{name}: {e}")))?,
        )
    };
    let value = call_function(function, vec![Value::Num(t), y_arg]).await?;
    if crate::builtins::common::validation::value_contains_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            input.extensions.integer_callback,
            name,
        )?;
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await?
        {
            return Err(ode_error(
                name,
                format!(
                    "{name}: integer derivative values must be exactly representable as double"
                ),
            ));
        }
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(&value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            input.extensions.logical_numeric,
            name,
        )?;
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            input.extensions.resident_input,
            name,
        )?;
    }
    let rhs = value_to_real_vector(name, value).await?;
    if rhs.len() != y.len() {
        return Err(ode_error(
            name,
            format!(
                "{name}: derivative output length {} does not match state length {}",
                rhs.len(),
                y.len()
            ),
        ));
    }
    Ok(rhs)
}

fn scaled_error_norm(y: &[f64], y_next: &[f64], err: &[f64], rel_tol: f64, abs_tol: f64) -> f64 {
    let mut norm = 0.0_f64;
    for ((yn, yn1), e) in y.iter().zip(y_next.iter()).zip(err.iter()) {
        let scale = abs_tol + rel_tol * yn.abs().max(yn1.abs());
        let component = e.abs() / scale.max(1.0e-15);
        if !component.is_finite() {
            return f64::INFINITY;
        }
        norm = norm.max(component);
    }
    norm
}

fn lincomb(base: &[f64], h: f64, terms: &[(&[f64], f64)]) -> Vec<f64> {
    let mut out = base.to_vec();
    for (vec, coeff) in terms {
        for (dst, src) in out.iter_mut().zip(vec.iter()) {
            *dst += h * coeff * src;
        }
    }
    out
}

fn max_abs(values: &[f64]) -> f64 {
    let mut max = 0.0_f64;
    for value in values {
        if !value.is_finite() {
            return f64::INFINITY;
        }
        max = max.max(value.abs());
    }
    max
}

fn remaining_distance(current: f64, target: f64, direction: f64) -> f64 {
    ((target - current) * direction).max(0.0)
}

fn clamp_step(h: f64, direction: f64, remaining: f64, max_step: Option<f64>) -> f64 {
    let mut mag = h.abs().min(remaining);
    if let Some(max_step) = max_step {
        mag = mag.min(max_step);
    }
    mag * direction
}

fn scaled_next_step(
    h: f64,
    error_norm: f64,
    method_order: f64,
    direction: f64,
    max_step: Option<f64>,
) -> f64 {
    let safety = 0.9;
    let min_scale = 0.2;
    let max_scale = 5.0;
    let exponent = 1.0 / (method_order + 1.0);
    let scale = if error_norm <= 1.0e-12 {
        max_scale
    } else {
        (safety * error_norm.powf(-exponent)).clamp(min_scale, max_scale)
    };
    let mut next = h.abs() * scale;
    if let Some(max_step) = max_step {
        next = next.min(max_step);
    }
    next * direction
}

fn option_f64(
    name: &str,
    options: Option<&StructValue>,
    field: &str,
    default: f64,
) -> BuiltinResult<f64> {
    let Some(options) = options else {
        return Ok(default);
    };
    let Some(value) = lookup_option(options, field) else {
        return Ok(default);
    };
    match value {
        Value::Num(n) if n.is_finite() => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        other => Err(ode_error(
            name,
            format!("{name}: option {field} must be numeric, got {other:?}"),
        )),
    }
}

fn option_optional_f64(
    name: &str,
    options: Option<&StructValue>,
    field: &str,
) -> BuiltinResult<Option<f64>> {
    let Some(options) = options else {
        return Ok(None);
    };
    let Some(value) = lookup_option(options, field) else {
        return Ok(None);
    };
    match value {
        Value::Num(n) if n.is_finite() => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(i.to_f64())),
        other => Err(ode_error(
            name,
            format!("{name}: option {field} must be numeric, got {other:?}"),
        )),
    }
}

fn option_usize(
    name: &str,
    options: Option<&StructValue>,
    field: &str,
    default: usize,
) -> BuiltinResult<usize> {
    let Some(options) = options else {
        return Ok(default);
    };
    let Some(value) = lookup_option(options, field) else {
        return Ok(default);
    };
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|value| *value >= 1)
            .ok_or_else(|| ode_error(name, format!("{name}: option {field} must be at least 1")));
    }
    let parsed = match value {
        Value::Num(value) => *value,
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            tensor::tensor_value_f64(value, 0)
        }
        other => {
            return Err(ode_error(
                name,
                format!("{name}: option {field} must be numeric, got {other:?}"),
            ))
        }
    };
    if !parsed.is_finite()
        || parsed < 1.0
        || parsed.fract() != 0.0
        || parsed >= 2f64.powi(usize::BITS as i32)
    {
        return Err(ode_error(
            name,
            format!("{name}: option {field} must be a supported positive integer"),
        ));
    }
    Ok(parsed as usize)
}

pub(crate) fn build_ode_output(name: &str, result: OdeResult) -> BuiltinResult<Value> {
    let t = Tensor::new(result.t.clone(), vec![result.t.len(), 1])
        .map(Value::Tensor)
        .map_err(|e| ode_error(name, format!("{name}: {e}")))?;
    let y = rows_to_matrix_value(name, &result.y_rows)?;

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![y]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![t, y],
        ));
    }
    Ok(y)
}

fn rows_to_matrix_value(name: &str, rows: &[Vec<f64>]) -> BuiltinResult<Value> {
    if rows.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map(Value::Tensor)
            .map_err(|e| ode_error(name, format!("{name}: {e}")));
    }
    let row_count = rows.len();
    let col_count = rows[0].len();
    let mut data = vec![0.0; row_count * col_count];
    for (row_idx, row) in rows.iter().enumerate() {
        if row.len() != col_count {
            return Err(ode_error(
                name,
                format!("{name}: internal solver produced inconsistent row lengths"),
            ));
        }
        for (col_idx, value) in row.iter().enumerate() {
            data[row_idx + col_idx * row_count] = *value;
        }
    }
    Tensor::new(data, vec![row_count, col_count])
        .map(Value::Tensor)
        .map_err(|e| ode_error(name, format!("{name}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;
    use std::sync::Arc;

    #[test]
    fn parse_ode_input_reads_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tspan = Tensor::new_integer(IntegerStorage::U16(vec![0, 2]), vec![1, 2]).unwrap();
        let y0 = Tensor::new_integer(IntegerStorage::I16(vec![3, -4]), vec![2, 1]).unwrap();

        let input = block_on(parse_ode_input(
            "ode_test",
            Value::Tensor(tspan),
            Value::Tensor(y0),
            super::super::ode45::ODE_COMPATIBILITY_EXTENSIONS,
        ))
        .unwrap();

        assert_eq!(input.tspan, vec![0.0, 2.0]);
        assert_eq!(input.y0, vec![3.0, -4.0]);
        assert_eq!(input.y_shape, vec![2, 1]);
        assert!(!input.scalar_state);
    }

    #[test]
    fn ode_integer_metadata_covers_every_shared_role_for_each_solver() {
        let capability_sets = [
            &super::super::ode15s::INTEGER_CAPABILITIES,
            &super::super::ode23::INTEGER_CAPABILITIES,
            &super::super::ode45::INTEGER_CAPABILITIES,
        ];
        for capabilities in capability_sets {
            assert_eq!(capabilities.len(), 4);
            for input in capabilities.iter().flat_map(|capability| capability.inputs) {
                assert_eq!(
                    input.availability,
                    runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly
                );
            }
            assert!(capabilities.iter().all(|capability| {
                capability.backend == runmat_builtins::BuiltinIntegerBackendRule::GatherFallback
            }));
        }
    }

    #[test]
    fn step_scaling_uses_embedded_error_order_for_ode45() {
        let h = 0.1;
        let err = 1.0e-2;

        let next = scaled_next_step(h, err, OdeMethod::Ode45.embedded_error_order(), 1.0, None);
        let expected = h * 0.9 * err.powf(-1.0 / 5.0);
        assert!((next - expected).abs() < 1.0e-12);
    }

    #[test]
    fn step_scaling_uses_embedded_error_order_for_ode23() {
        let h = 0.1;
        let err = 1.0e-2;

        let next = scaled_next_step(h, err, OdeMethod::Ode23.embedded_error_order(), 1.0, None);
        let expected = h * 0.9 * err.powf(-1.0 / 3.0);
        assert!((next - expected).abs() < 1.0e-12);
    }

    #[test]
    fn scaled_error_norm_treats_nan_as_infinite_error() {
        let norm = scaled_error_norm(&[1.0], &[f64::NAN], &[f64::NAN], 1.0e-3, 1.0e-6);
        assert!(norm.is_infinite());
    }

    #[test]
    fn ode15s_newton_step_handles_picard_unstable_stiff_decay() {
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_name| {
                Some(0)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, args, _requested_outputs| {
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(-1000.0 * y)) })
            },
        )));
        let input = OdeInput {
            tspan: vec![0.0, 0.1],
            y0: vec![1.0],
            y_shape: vec![1, 1],
            scalar_state: true,
            extensions: super::super::ode15s::ODE_COMPATIBILITY_EXTENSIONS,
        };

        let (next, err) = block_on(step_ode15s(
            "ode15s",
            &Value::FunctionHandle("stiff_decay".into()),
            0.0,
            &[1.0],
            0.1,
            &input,
        ))
        .unwrap();

        assert_eq!(next.len(), 1);
        assert!(next[0].is_finite());
        assert!(next[0] > 0.0);
        assert!(next[0] < 0.02);
        assert!(err[0].is_finite());
    }
}
