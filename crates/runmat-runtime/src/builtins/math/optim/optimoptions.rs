//! MATLAB-compatible `optimoptions` options struct builder.

use std::collections::VecDeque;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, LogicalArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::common::canonical_option_name;
use crate::builtins::math::optim::type_resolvers::optim_options_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "optimoptions";

const OPTIMOPTIONS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Options struct for optimization solvers.",
}];

const OPTIMOPTIONS_INPUTS_SOLVER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "solver",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Solver name, such as fminbnd, fminunc, fzero, fsolve, or lsqcurvefit.",
}];

const OPTIMOPTIONS_INPUTS_SOLVER_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "solver",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Solver name, such as fminbnd, fminunc, fzero, fsolve, or lsqcurvefit.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value(s) and additional name/value pairs.",
    },
];

const OPTIMOPTIONS_INPUTS_EXISTING_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "oldopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing options struct to update.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value(s), additional name/value pairs, or another options struct.",
    },
];

const OPTIMOPTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "options = optimoptions(solver)",
        inputs: &OPTIMOPTIONS_INPUTS_SOLVER,
        outputs: &OPTIMOPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimoptions(solver, name, value, ...)",
        inputs: &OPTIMOPTIONS_INPUTS_SOLVER_PAIRS,
        outputs: &OPTIMOPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimoptions(oldopts, name, value, ...)",
        inputs: &OPTIMOPTIONS_INPUTS_EXISTING_PAIRS,
        outputs: &OPTIMOPTIONS_OUTPUT,
    },
];

const OPTIMOPTIONS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.INVALID_ARGUMENT",
    identifier: Some("RunMat:optimoptions:InvalidArgument"),
    when: "Argument grammar does not match supported optimoptions forms.",
    message: "optimoptions: invalid argument",
};
const OPTIMOPTIONS_ERROR_INVALID_SOLVER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.INVALID_SOLVER",
    identifier: Some("RunMat:optimoptions:InvalidSolver"),
    when: "The solver argument is not one of the supported optimization builtins.",
    message: "optimoptions: invalid solver",
};
const OPTIMOPTIONS_ERROR_INVALID_OPTION_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.INVALID_OPTION_NAME",
    identifier: Some("RunMat:optimoptions:InvalidOptionName"),
    when: "An option name is not a text scalar.",
    message: "optimoptions: invalid option name",
};
const OPTIMOPTIONS_ERROR_MISSING_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.MISSING_OPTION_VALUE",
    identifier: Some("RunMat:optimoptions:MissingOptionValue"),
    when: "A name-value option key is not followed by a value.",
    message: "optimoptions: missing option value",
};
const OPTIMOPTIONS_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.UNKNOWN_OPTION",
    identifier: Some("RunMat:optimoptions:UnknownOption"),
    when: "An option name is not supported by the selected solver.",
    message: "optimoptions: unknown option",
};
const OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.INVALID_OPTION_VALUE",
    identifier: Some("RunMat:optimoptions:InvalidOptionValue"),
    when: "An option value fails type or domain validation.",
    message: "optimoptions: invalid option value",
};
const OPTIMOPTIONS_ERROR_FLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMOPTIONS.FLOW",
    identifier: Some("RunMat:optimoptions:Flow"),
    when: "Nested flow fails while gathering input values.",
    message: "optimoptions: flow failure",
};

const OPTIMOPTIONS_ERRORS: [BuiltinErrorDescriptor; 7] = [
    OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
    OPTIMOPTIONS_ERROR_INVALID_SOLVER,
    OPTIMOPTIONS_ERROR_INVALID_OPTION_NAME,
    OPTIMOPTIONS_ERROR_MISSING_OPTION_VALUE,
    OPTIMOPTIONS_ERROR_UNKNOWN_OPTION,
    OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
    OPTIMOPTIONS_ERROR_FLOW,
];

pub const OPTIMOPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OPTIMOPTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &OPTIMOPTIONS_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::optimoptions")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "optimoptions",
    op_kind: GpuOpKind::Custom("optimization-options"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host metadata construction. gpuArray option values are gathered before validation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::optimoptions")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "optimoptions",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Option struct construction is host metadata work and terminates fusion planning.",
};

#[runtime_builtin(
    name = "optimoptions",
    category = "math/optim",
    summary = "Create or update a typed optimization options structure for fminbnd, fminunc, fzero, fsolve, and lsqcurvefit.",
    keywords = "optimoptions,options,TolX,TolFun,FunctionTolerance,StepTolerance,MaxIter,MaxFunEvals,Display,Algorithm,SpecifyObjectiveGradient",
    accel = "cpu",
    type_resolver(optim_options_type),
    descriptor(crate::builtins::math::optim::optimoptions::OPTIMOPTIONS_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::optimoptions"
)]
async fn optimoptions_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let mut gathered = Vec::with_capacity(rest.len());
    for value in rest {
        gathered.push(gather_if_needed_async(&value).await.map_err(|err| {
            remap_optimoptions_flow(&OPTIMOPTIONS_ERROR_FLOW, err, |source| {
                format!("optimoptions: {}", source.message())
            })
        })?);
    }

    let mut queue: VecDeque<Value> = gathered.into();
    let first = queue.pop_front().ok_or_else(|| {
        optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
            "optimoptions: expected a solver name or options struct",
        )
    })?;

    let mut solver;
    let explicit_solver;
    let mut options = match first {
        Value::Struct(existing) => {
            explicit_solver = false;
            solver = solver_from_options(&existing)?;
            canonicalize_existing_options(&existing, solver)?
        }
        other => {
            explicit_solver = true;
            solver = parse_solver(&other)?;
            default_options(solver)
        }
    };

    while let Some(arg) = queue.pop_front() {
        match arg {
            Value::Struct(existing) => {
                if explicit_solver {
                    let next_solver = solver_from_options(&existing)?;
                    let skip_defaults_from = match next_solver {
                        Solver::Generic => None,
                        other => Some(other),
                    };
                    apply_struct_fields(
                        &existing,
                        &mut options,
                        solver,
                        false,
                        skip_defaults_from,
                    )?;
                    options.insert("Solver", Value::from(solver.name()));
                    continue;
                } else {
                    let next_solver = solver_from_options(&existing)?;
                    let skip_defaults_from;
                    if next_solver != Solver::Generic && next_solver != solver {
                        options = if solver == Solver::Generic {
                            merge_generic_into_defaults(&options, next_solver)?
                        } else {
                            default_options(next_solver)
                        };
                        solver = next_solver;
                        skip_defaults_from = Some(next_solver);
                    } else if next_solver != Solver::Generic {
                        solver = next_solver;
                        skip_defaults_from = Some(next_solver);
                    } else {
                        skip_defaults_from = None;
                    }
                    apply_struct_fields(&existing, &mut options, solver, true, skip_defaults_from)?;
                    continue;
                }
            }
            name_value => {
                let name = expect_string_scalar(
                    &name_value,
                    "optimoptions: option names must be character vectors or string scalars",
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_NAME,
                )?;
                let value = queue.pop_front().ok_or_else(|| {
                    optimoptions_error_with(
                        &OPTIMOPTIONS_ERROR_MISSING_OPTION_VALUE,
                        format!("optimoptions: missing value for option '{name}'"),
                    )
                })?;
                set_option_field(&mut options, solver, &name, &value)?;
            }
        }
    }

    Ok(Value::Struct(options))
}

fn optimoptions_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_optimoptions_flow<F>(
    error: &'static BuiltinErrorDescriptor,
    err: RuntimeError,
    message: F,
) -> RuntimeError
where
    F: FnOnce(&RuntimeError) -> String,
{
    let mut builder = build_runtime_error(message(&err))
        .with_builtin(NAME)
        .with_source(err);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Solver {
    Fminbnd,
    Fminunc,
    Fzero,
    Fsolve,
    Lsqcurvefit,
    Generic,
}

impl Solver {
    fn name(self) -> &'static str {
        match self {
            Self::Fminbnd => "fminbnd",
            Self::Fminunc => "fminunc",
            Self::Fzero => "fzero",
            Self::Fsolve => "fsolve",
            Self::Lsqcurvefit => "lsqcurvefit",
            Self::Generic => "",
        }
    }

    fn default_display(self) -> &'static str {
        match self {
            Self::Fminbnd => "notify",
            Self::Fminunc | Self::Fzero | Self::Fsolve | Self::Lsqcurvefit | Self::Generic => "off",
        }
    }

    fn accepts_tol_fun(self) -> bool {
        matches!(
            self,
            Self::Fminunc | Self::Fsolve | Self::Lsqcurvefit | Self::Generic
        )
    }

    fn accepts_option(self, canonical: &str) -> bool {
        match canonical {
            "TolX" | "MaxIter" | "MaxFunEvals" | "Display" => true,
            "TolFun" => self.accepts_tol_fun(),
            "Algorithm" => matches!(self, Self::Fminunc | Self::Lsqcurvefit | Self::Generic),
            "SpecifyObjectiveGradient" => matches!(self, Self::Fminunc | Self::Generic),
            _ => false,
        }
    }

    fn accepts_display(self, display: &str) -> bool {
        match self {
            Self::Fminbnd | Self::Fminunc | Self::Generic => {
                matches!(display, "off" | "none" | "iter" | "notify" | "final")
            }
            Self::Fzero | Self::Fsolve | Self::Lsqcurvefit => {
                matches!(display, "off" | "none" | "iter" | "final")
            }
        }
    }

    fn accepts_algorithm(self, algorithm: &str) -> bool {
        match self {
            Self::Fminunc => matches!(algorithm, "quasi-newton" | "bfgs"),
            Self::Lsqcurvefit | Self::Generic => {
                matches!(
                    algorithm,
                    "quasi-newton" | "bfgs" | "levenberg-marquardt" | "trust-region-reflective"
                )
            }
            _ => false,
        }
    }
}

fn parse_solver(value: &Value) -> BuiltinResult<Solver> {
    let text = expect_string_scalar(
        value,
        "optimoptions: solver must be a character vector or string scalar",
        &OPTIMOPTIONS_ERROR_INVALID_SOLVER,
    )?;
    parse_solver_name(&text)
}

fn parse_solver_name(text: &str) -> BuiltinResult<Solver> {
    match text.trim().to_ascii_lowercase().as_str() {
        "fminbnd" => Ok(Solver::Fminbnd),
        "fminunc" => Ok(Solver::Fminunc),
        "fzero" => Ok(Solver::Fzero),
        "fsolve" => Ok(Solver::Fsolve),
        "lsqcurvefit" => Ok(Solver::Lsqcurvefit),
        other => Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_SOLVER,
            format!("optimoptions: unsupported solver '{other}'"),
        )),
    }
}

fn solver_from_options(options: &StructValue) -> BuiltinResult<Solver> {
    let Some(value) = lookup_case_insensitive(options, "Solver") else {
        return Ok(Solver::Generic);
    };
    parse_solver(value)
}

fn default_options(solver: Solver) -> StructValue {
    let mut out = StructValue::new();
    if solver != Solver::Generic {
        out.insert("Solver", Value::from(solver.name()));
    }
    match solver {
        Solver::Fminbnd => {
            out.insert("TolX", Value::Num(1.0e-4));
            out.insert("MaxIter", Value::Num(500.0));
            out.insert("MaxFunEvals", Value::Num(500.0));
            out.insert("Display", Value::from(solver.default_display()));
        }
        Solver::Fminunc => {
            out.insert("Algorithm", Value::from("quasi-newton"));
            out.insert("TolX", Value::Num(1.0e-6));
            out.insert("TolFun", Value::Num(1.0e-6));
            out.insert("MaxIter", Value::Num(400.0));
            out.insert("MaxFunEvals", Value::Num(40000.0));
            out.insert("Display", Value::from(solver.default_display()));
            out.insert("SpecifyObjectiveGradient", Value::Bool(false));
        }
        Solver::Fzero => {
            out.insert("TolX", Value::Num(1.0e-6));
            out.insert("MaxIter", Value::Num(400.0));
            out.insert("MaxFunEvals", Value::Num(500.0));
            out.insert("Display", Value::from(solver.default_display()));
        }
        Solver::Fsolve => {
            out.insert("TolX", Value::Num(1.0e-6));
            out.insert("TolFun", Value::Num(1.0e-6));
            out.insert("MaxIter", Value::Num(400.0));
            out.insert("MaxFunEvals", Value::Num(40000.0));
            out.insert("Display", Value::from(solver.default_display()));
        }
        Solver::Lsqcurvefit => {
            out.insert("Algorithm", Value::from("levenberg-marquardt"));
            out.insert("TolX", Value::Num(1.0e-6));
            out.insert("TolFun", Value::Num(1.0e-6));
            out.insert("MaxIter", Value::Num(400.0));
            out.insert("MaxFunEvals", Value::Num(40000.0));
            out.insert("Display", Value::from(solver.default_display()));
        }
        Solver::Generic => {}
    }
    out
}

fn canonicalize_existing_options(
    existing: &StructValue,
    solver: Solver,
) -> BuiltinResult<StructValue> {
    let mut out = if solver == Solver::Generic {
        StructValue::new()
    } else {
        default_options(solver)
    };
    apply_struct_fields(existing, &mut out, solver, true, None)?;
    Ok(out)
}

fn merge_generic_into_defaults(
    generic: &StructValue,
    solver: Solver,
) -> BuiltinResult<StructValue> {
    let mut out = default_options(solver);
    for (key, value) in &generic.fields {
        if key.eq_ignore_ascii_case("Solver") {
            continue;
        }
        let canonical = canonical_option_name(key);
        if !solver.accepts_option(&canonical) {
            continue;
        }
        if canonical == "Display" && display_value(solver, value).is_err() {
            continue;
        }
        set_option_field(&mut out, solver, key, value)?;
    }
    Ok(out)
}

fn apply_struct_fields(
    source: &StructValue,
    target: &mut StructValue,
    solver: Solver,
    copy_solver_field: bool,
    skip_defaults_from: Option<Solver>,
) -> BuiltinResult<()> {
    let source_defaults = skip_defaults_from.map(default_options);
    for (key, value) in &source.fields {
        if key.eq_ignore_ascii_case("Solver") {
            if !copy_solver_field {
                continue;
            }
            let parsed = parse_solver(value)?;
            target.insert("Solver", Value::from(parsed.name()));
            continue;
        }
        let canonical = canonical_option_name(key);
        if let Some(defaults) = &source_defaults {
            if solver.accepts_option(&canonical)
                && lookup_case_insensitive(defaults, &canonical).is_some_and(|default| {
                    normalized_option_value(solver, &canonical, value)
                        .is_ok_and(|normalized| default == &normalized)
                })
            {
                continue;
            }
        }
        set_option_field(target, solver, key, value)?;
    }
    Ok(())
}

fn set_option_field(
    options: &mut StructValue,
    solver: Solver,
    name: &str,
    value: &Value,
) -> BuiltinResult<()> {
    let canonical = canonical_option_name(name);
    if !solver.accepts_option(&canonical) {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_UNKNOWN_OPTION,
            format!(
                "optimoptions: option '{}' is not supported for {}",
                name,
                solver_label(solver)
            ),
        ));
    }

    let value = normalized_option_value(solver, &canonical, value)?;
    options.insert(canonical, value);
    Ok(())
}

fn normalized_option_value(solver: Solver, canonical: &str, value: &Value) -> BuiltinResult<Value> {
    match canonical {
        "TolX" | "TolFun" => Ok(Value::Num(positive_finite_scalar(canonical, value)?)),
        "MaxIter" | "MaxFunEvals" => {
            Ok(Value::Num(positive_integer_scalar(canonical, value)? as f64))
        }
        "Display" => Ok(Value::from(display_value(solver, value)?)),
        "Algorithm" => Ok(Value::from(algorithm_value(solver, value)?)),
        "SpecifyObjectiveGradient" => Ok(Value::Bool(logical_value(canonical, value)?)),
        _ => unreachable!("unsupported option passed accepts_option"),
    }
}

fn solver_label(solver: Solver) -> &'static str {
    match solver {
        Solver::Generic => "optimization solvers",
        _ => solver.name(),
    }
}

fn positive_finite_scalar(field: &str, value: &Value) -> BuiltinResult<f64> {
    let parsed = numeric_scalar(field, value)?;
    if parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be a finite positive scalar"),
        ))
    }
}

fn positive_integer_scalar(field: &str, value: &Value) -> BuiltinResult<usize> {
    let parsed = positive_finite_scalar(field, value)?;
    if parsed.fract() != 0.0 {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be an integer scalar"),
        ));
    }
    if parsed >= 2f64.powi(usize::BITS as i32) {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} is too large"),
        ));
    }
    Ok(parsed as usize)
}

fn numeric_scalar(field: &str, value: &Value) -> BuiltinResult<f64> {
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(Tensor { data, .. }) if data.len() == 1 => data[0],
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] == 0 {
                0.0
            } else {
                1.0
            }
        }
        other => {
            return Err(optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!("optimoptions: option {field} must be a numeric scalar, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be finite"),
        ))
    }
}

fn logical_value(field: &str, value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => Ok(data[0] != 0),
        Value::Num(n) => logical_from_number(field, *n),
        Value::Int(i) => logical_from_number(field, i.to_f64()),
        Value::Tensor(Tensor { data, .. }) if data.len() == 1 => {
            logical_from_number(field, data[0])
        }
        Value::String(s) => logical_from_text(field, s),
        Value::StringArray(sa) if sa.data.len() == 1 => logical_from_text(field, &sa.data[0]),
        Value::CharArray(CharArray { data, rows: 1, .. }) => {
            let text: String = data.iter().collect();
            logical_from_text(field, &text)
        }
        other => Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be logical, got {other:?}"),
        )),
    }
}

fn logical_from_number(field: &str, value: f64) -> BuiltinResult<bool> {
    if value == 0.0 {
        Ok(false)
    } else if value == 1.0 {
        Ok(true)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be logical 0 or 1"),
        ))
    }
}

fn logical_from_text(field: &str, value: &str) -> BuiltinResult<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "yes" => Ok(true),
        "off" | "false" | "no" => Ok(false),
        other => Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be 'on' or 'off', got '{other}'"),
        )),
    }
}

fn display_value(solver: Solver, value: &Value) -> BuiltinResult<String> {
    let display = expect_string_scalar(
        value,
        "optimoptions: Display must be a character vector or string scalar",
        &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
    )?
    .trim()
    .to_ascii_lowercase();
    if solver.accepts_display(&display) {
        Ok(display)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!(
                "optimoptions: unsupported Display '{}' for {}",
                display,
                solver_label(solver)
            ),
        ))
    }
}

fn algorithm_value(solver: Solver, value: &Value) -> BuiltinResult<String> {
    let algorithm = expect_string_scalar(
        value,
        "optimoptions: Algorithm must be a character vector or string scalar",
        &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
    )?
    .trim()
    .to_ascii_lowercase();
    if solver.accepts_algorithm(&algorithm) {
        Ok(algorithm)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!(
                "optimoptions: unsupported Algorithm '{}' for {}",
                algorithm,
                solver_label(solver)
            ),
        ))
    }
}

fn expect_string_scalar(
    value: &Value,
    context: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(CharArray { data, rows: 1, .. }) => Ok(data.iter().collect()),
        _ => Err(optimoptions_error_with(error, context)),
    }
}

fn lookup_case_insensitive<'a>(options: &'a StructValue, name: &str) -> Option<&'a Value> {
    options
        .fields
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call_builtin_async;
    use futures::executor::block_on;
    use runmat_builtins::IntValue;

    fn run_optimoptions(rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(optimoptions_builtin(rest))
    }

    fn run_call_builtin(name: &str, args: &[Value]) -> BuiltinResult<Value> {
        block_on(call_builtin_async(name, args))
    }

    fn struct_result(value: Value) -> StructValue {
        match value {
            Value::Struct(options) => options,
            other => panic!("expected struct, got {other:?}"),
        }
    }

    fn num_field(options: &StructValue, field: &str) -> f64 {
        match options.fields.get(field) {
            Some(Value::Num(value)) => *value,
            other => panic!("expected numeric field {field}, got {other:?}"),
        }
    }

    fn string_field<'a>(options: &'a StructValue, field: &str) -> &'a str {
        match options.fields.get(field) {
            Some(Value::String(value)) => value.as_str(),
            other => panic!("expected string field {field}, got {other:?}"),
        }
    }

    fn bool_field(options: &StructValue, field: &str) -> bool {
        match options.fields.get(field) {
            Some(Value::Bool(value)) => *value,
            other => panic!("expected bool field {field}, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_descriptor_signatures_and_errors_cover_core_forms() {
        let labels: Vec<&str> = OPTIMOPTIONS_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "options = optimoptions(solver)",
                "options = optimoptions(solver, name, value, ...)",
                "options = optimoptions(oldopts, name, value, ...)",
            ]
        );

        let codes: Vec<&str> = OPTIMOPTIONS_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec![
                "RM.OPTIMOPTIONS.INVALID_ARGUMENT",
                "RM.OPTIMOPTIONS.INVALID_SOLVER",
                "RM.OPTIMOPTIONS.INVALID_OPTION_NAME",
                "RM.OPTIMOPTIONS.MISSING_OPTION_VALUE",
                "RM.OPTIMOPTIONS.UNKNOWN_OPTION",
                "RM.OPTIMOPTIONS.INVALID_OPTION_VALUE",
                "RM.OPTIMOPTIONS.FLOW",
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_fminbnd_defaults_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![Value::from("fminbnd")]).expect("optimoptions fminbnd"),
        );
        assert_eq!(string_field(&options, "Solver"), "fminbnd");
        assert_eq!(num_field(&options, "TolX"), 1.0e-4);
        assert_eq!(num_field(&options, "MaxIter"), 500.0);
        assert_eq!(num_field(&options, "MaxFunEvals"), 500.0);
        assert_eq!(string_field(&options, "Display"), "notify");
        assert!(!options.fields.contains_key("TolFun"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_fzero_defaults_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![Value::from("fzero")]).expect("optimoptions fzero"),
        );
        assert_eq!(string_field(&options, "Solver"), "fzero");
        assert_eq!(num_field(&options, "TolX"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxIter"), 400.0);
        assert_eq!(num_field(&options, "MaxFunEvals"), 500.0);
        assert_eq!(string_field(&options, "Display"), "off");
        assert!(!options.fields.contains_key("TolFun"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_fsolve_defaults_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![Value::from("fsolve")]).expect("optimoptions fsolve"),
        );
        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-6);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxIter"), 400.0);
        assert_eq!(num_field(&options, "MaxFunEvals"), 40000.0);
        assert_eq!(string_field(&options, "Display"), "off");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_fminunc_defaults_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![Value::from("fminunc")]).expect("optimoptions fminunc"),
        );
        assert_eq!(string_field(&options, "Solver"), "fminunc");
        assert_eq!(string_field(&options, "Algorithm"), "quasi-newton");
        assert_eq!(num_field(&options, "TolX"), 1.0e-6);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxIter"), 400.0);
        assert_eq!(num_field(&options, "MaxFunEvals"), 40000.0);
        assert_eq!(string_field(&options, "Display"), "off");
        assert!(!bool_field(&options, "SpecifyObjectiveGradient"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_fminunc_accepts_gradient_and_algorithm_options() {
        let options = struct_result(
            run_optimoptions(vec![
                Value::from("fminunc"),
                Value::from("SpecifyObjectiveGradient"),
                Value::from("on"),
                Value::from("Algorithm"),
                Value::from("bfgs"),
                Value::from("Display"),
                Value::from("notify"),
            ])
            .expect("optimoptions fminunc"),
        );
        assert!(bool_field(&options, "SpecifyObjectiveGradient"));
        assert_eq!(string_field(&options, "Algorithm"), "bfgs");
        assert_eq!(string_field(&options, "Display"), "notify");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_lsqcurvefit_defaults_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![Value::from("lsqcurvefit")]).expect("optimoptions lsqcurvefit"),
        );
        assert_eq!(string_field(&options, "Solver"), "lsqcurvefit");
        assert_eq!(string_field(&options, "Algorithm"), "levenberg-marquardt");
        assert_eq!(num_field(&options, "TolX"), 1.0e-6);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxIter"), 400.0);
        assert_eq!(num_field(&options, "MaxFunEvals"), 40000.0);
        assert_eq!(string_field(&options, "Display"), "off");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_lsqcurvefit_accepts_modern_tolerance_aliases_and_algorithm() {
        let options = struct_result(
            run_optimoptions(vec![
                Value::from("lsqcurvefit"),
                Value::from("FunctionTolerance"),
                Value::Num(1.0e-9),
                Value::from("StepTolerance"),
                Value::Num(1.0e-8),
                Value::from("Algorithm"),
                Value::from("trust-region-reflective"),
            ])
            .expect("optimoptions lsqcurvefit aliases"),
        );
        assert_eq!(num_field(&options, "TolFun"), 1.0e-9);
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(
            string_field(&options, "Algorithm"),
            "trust-region-reflective"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_name_value_pairs_are_case_insensitive() {
        let options = struct_result(
            run_optimoptions(vec![
                Value::from("fsolve"),
                Value::from("tolx"),
                Value::Num(1.0e-8),
                Value::from("DISPLAY"),
                Value::from("Final"),
            ])
            .expect("optimoptions overrides"),
        );
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(string_field(&options, "Display"), "final");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_updates_existing_options_with_pairs() {
        let base = run_optimoptions(vec![
            Value::from("fzero"),
            Value::from("TolX"),
            Value::Num(1.0e-5),
        ])
        .expect("base options");
        let options = struct_result(
            run_optimoptions(vec![base, Value::from("MaxIter"), Value::Num(25.0)])
                .expect("updated options"),
        );
        assert_eq!(string_field(&options, "Solver"), "fzero");
        assert_eq!(num_field(&options, "TolX"), 1.0e-5);
        assert_eq!(num_field(&options, "MaxIter"), 25.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_merges_existing_options_structs() {
        let first = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-5),
        ])
        .expect("first");
        let second = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("MaxIter"),
            Value::Num(30.0),
        ])
        .expect("second");
        let options = struct_result(run_optimoptions(vec![first, second]).expect("merged options"));
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxIter"), 30.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_same_solver_struct_merge_preserves_prior_overrides() {
        let first = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("MaxFunEvals"),
            Value::Num(2000.0),
        ])
        .expect("first");
        let second = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
        ])
        .expect("second");

        let options = struct_result(run_optimoptions(vec![first, second]).expect("merged options"));

        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxFunEvals"), 2000.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_solver_form_same_solver_struct_preserves_prior_overrides() {
        let later = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
        ])
        .expect("later options");

        let options = struct_result(
            run_optimoptions(vec![
                Value::from("fsolve"),
                Value::from("MaxFunEvals"),
                Value::Num(2000.0),
                later,
            ])
            .expect("merged options"),
        );

        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxFunEvals"), 2000.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_default_skipping_compares_normalized_values() {
        let first = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("MaxFunEvals"),
            Value::Num(2000.0),
            Value::from("Display"),
            Value::from("final"),
        ])
        .expect("first");

        let mut later = StructValue::new();
        later.insert("Solver", Value::from("fsolve"));
        later.insert("TolX", Value::Num(1.0e-8));
        later.insert("MaxFunEvals", Value::Int(IntValue::I32(40000)));
        later.insert("Display", Value::CharArray(CharArray::new_row("off")));

        let options = struct_result(
            run_optimoptions(vec![first, Value::Struct(later)]).expect("merged options"),
        );

        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxFunEvals"), 2000.0);
        assert_eq!(string_field(&options, "Display"), "final");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_generic_to_concrete_solver_preserves_valid_generic_overrides() {
        let mut generic = StructValue::new();
        generic.insert("MaxFunEvals", Value::Num(2000.0));
        generic.insert("Display", Value::from("final"));

        let later = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
        ])
        .expect("later options");

        let options = struct_result(
            run_optimoptions(vec![Value::Struct(generic), later]).expect("merged options"),
        );

        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxFunEvals"), 2000.0);
        assert_eq!(string_field(&options, "Display"), "final");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_solver_form_keeps_requested_solver_when_struct_has_solver() {
        let fzero_options = run_optimoptions(vec![
            Value::from("fzero"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("MaxIter"),
            Value::Num(30.0),
        ])
        .expect("fzero options");

        let options = struct_result(
            run_optimoptions(vec![Value::from("fsolve"), fzero_options])
                .expect("merged into fsolve options"),
        );

        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxIter"), 30.0);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxFunEvals"), 40000.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_rejects_unknown_option_names() {
        let err = run_optimoptions(vec![
            Value::from("fzero"),
            Value::from("TolFun"),
            Value::Num(1.0e-8),
        ])
        .expect_err("TolFun is not accepted by fzero");
        assert_eq!(err.identifier(), Some("RunMat:optimoptions:UnknownOption"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_rejects_missing_option_values() {
        let err = run_optimoptions(vec![Value::from("fsolve"), Value::from("TolX")])
            .expect_err("missing option value");
        assert_eq!(
            err.identifier(),
            Some("RunMat:optimoptions:MissingOptionValue")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_rejects_invalid_option_values() {
        let err = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("MaxIter"),
            Value::Num(1.5),
        ])
        .expect_err("noninteger MaxIter should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:optimoptions:InvalidOptionValue")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_rejects_out_of_range_integer_options() {
        let err = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("MaxIter"),
            Value::Num(2f64.powi(usize::BITS as i32)),
        ])
        .expect_err("out-of-range MaxIter should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:optimoptions:InvalidOptionValue")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminbnd_accepts_optimoptions_output() {
        let options = run_optimoptions(vec![
            Value::from("fminbnd"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("Display"),
            Value::from("off"),
        ])
        .expect("optimoptions");
        let result = run_call_builtin(
            "fminbnd",
            &[
                Value::FunctionHandle("cos".into()),
                Value::Num(0.0),
                Value::Num(std::f64::consts::PI),
                options,
            ],
        )
        .expect("fminbnd");
        match result {
            Value::Num(value) => assert!((value - std::f64::consts::PI).abs() < 1.0e-4),
            other => panic!("unexpected fminbnd result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fzero_accepts_optimoptions_output() {
        let options = run_optimoptions(vec![
            Value::from("fzero"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
        ])
        .expect("optimoptions");
        let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result = run_call_builtin(
            "fzero",
            &[
                Value::FunctionHandle("sin".into()),
                Value::Tensor(bracket),
                options,
            ],
        )
        .expect("fzero");
        match result {
            Value::Num(value) => assert!((value - std::f64::consts::PI).abs() < 1.0e-6),
            other => panic!("unexpected fzero result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fsolve_accepts_optimoptions_output() {
        let options = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("TolFun"),
            Value::Num(1.0e-8),
        ])
        .expect("optimoptions");
        let result = run_call_builtin(
            "fsolve",
            &[
                Value::FunctionHandle("sin".into()),
                Value::Num(3.0),
                options,
            ],
        )
        .expect("fsolve");
        match result {
            Value::Num(value) => assert!((value - std::f64::consts::PI).abs() < 1.0e-6),
            other => panic!("unexpected fsolve result {other:?}"),
        }
    }
}
