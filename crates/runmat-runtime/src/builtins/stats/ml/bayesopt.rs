//! Deterministic MATLAB-compatible `bayesopt` surface for small in-memory workflows.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::time::Instant;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, LogicalArray, ObjectInstance, ResolveContext, StringArray, StructValue, Tensor,
    Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::optim::common::{call_function, value_to_scalar};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "bayesopt";
const RESULT_CLASS: &str = "BayesianOptimization";
const DEFAULT_MAX_EVALS: usize = 30;
const DEFAULT_SEED_POINTS: usize = 4;
const MAX_CANDIDATES: usize = 2048;
const MAX_EVALS_LIMIT: usize = MAX_CANDIDATES;
const GP_NOISE: f64 = 1.0e-8;

const OUTPUT_RESULT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "results",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "BayesianOptimization compatibility result object.",
}];

const PARAM_OBJECTIVE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Objective callback evaluated as fun(X), where X is a one-row table.",
};

const PARAM_VARIABLES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "vars",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "optimizableVariable object or cell/object collection.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Name-value options including MaxObjectiveEvaluations and AcquisitionFunctionName.",
};

const INPUTS: [BuiltinParamDescriptor; 3] = [PARAM_OBJECTIVE, PARAM_VARIABLES, PARAM_OPTIONS];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "results = bayesopt(fun, vars, Name, Value)",
    inputs: &INPUTS,
    outputs: &OUTPUT_RESULT,
}];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAYESOPT.INVALID_ARGUMENT",
    identifier: Some("RunMat:bayesopt:InvalidArgument"),
    when: "Variables, name-value options, callback outputs, or constraints are malformed.",
    message: "bayesopt: invalid argument",
};

const ERROR_OBJECTIVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAYESOPT.OBJECTIVE",
    identifier: Some("RunMat:bayesopt:Objective"),
    when: "The objective callback cannot be evaluated or returns an invalid objective value.",
    message: "bayesopt: objective evaluation failed",
};

const ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAYESOPT.UNSUPPORTED",
    identifier: Some("RunMat:bayesopt:Unsupported"),
    when: "A requested Bayesian optimization option requires infrastructure outside this slice.",
    message: "bayesopt: unsupported option",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_OBJECTIVE, ERROR_UNSUPPORTED];

pub const BAYESOPT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::ml::bayesopt")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("bayesian-optimization-loop"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "bayesopt is a host control-flow solver; variable metadata and option inputs are gathered before deterministic callback evaluation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::ml::bayesopt")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "bayesopt repeatedly invokes callbacks and terminates fusion planning.",
};

#[runtime_builtin(
    name = "bayesopt",
    category = "stats/ml",
    summary = "Run deterministic Bayesian optimization over optimizableVariable metadata.",
    keywords = "bayesopt,Bayesian optimization,optimizableVariable,expected improvement",
    accel = "cpu",
    type_resolver(bayesopt_type),
    descriptor(crate::builtins::stats::ml::bayesopt::BAYESOPT_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::bayesopt"
)]
async fn bayesopt_builtin(
    objective: Value,
    variables: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let variables = gather_if_needed_async(&variables)
        .await
        .map_err(|err| map_flow(err, "variables"))?;
    let mut gathered_rest = Vec::with_capacity(rest.len());
    for value in rest {
        gathered_rest.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| map_flow(err, "name-value option"))?,
        );
    }
    let vars = parse_variables(variables)?;
    let options = BayesoptOptions::parse(gathered_rest)?;
    let outcome = optimize(objective, vars, options).await?;
    finalize(outcome)
}

fn bayesopt_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn map_flow(err: RuntimeError, label: &str) -> RuntimeError {
    build_runtime_error(format!(
        "bayesopt: failed to gather {label}: {}",
        err.message()
    ))
    .with_builtin(NAME)
    .with_identifier("RunMat:bayesopt:Flow")
    .build()
}

fn bayesopt_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(NAME)
        .with_identifier(descriptor.identifier.unwrap_or("RunMat:bayesopt:Error"))
        .build()
}

#[derive(Clone, Debug)]
struct Variable {
    name: String,
    kind: VariableKind,
    transform: Transform,
    optimize: bool,
}

#[derive(Clone, Debug)]
enum VariableKind {
    Real { lower: f64, upper: f64 },
    Integer { lower: i64, upper: i64 },
    Categorical { categories: Vec<String> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transform {
    None,
    Log,
}

fn parse_variables(value: Value) -> BuiltinResult<Vec<Variable>> {
    let raw = match value {
        Value::Object(object) if object.class_name == "optimizableVariable" => {
            vec![Value::Object(object)]
        }
        Value::Cell(cell) => cell.data,
        Value::OutputList(values) => values,
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: variables must be optimizableVariable objects, got {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    if raw.is_empty() {
        return Err(bayesopt_error(
            "bayesopt: variables must not be empty",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let mut names = HashSet::with_capacity(raw.len());
    let mut vars = Vec::with_capacity(raw.len());
    for value in raw {
        let Value::Object(object) = value else {
            return Err(bayesopt_error(
                "bayesopt: variables must be optimizableVariable objects",
                &ERROR_INVALID_ARGUMENT,
            ));
        };
        if object.class_name != "optimizableVariable" {
            return Err(bayesopt_error(
                format!(
                    "bayesopt: expected optimizableVariable, got {}",
                    object.class_name
                ),
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        let variable = variable_from_object(&object)?;
        if !names.insert(variable.name.clone()) {
            return Err(bayesopt_error(
                format!("bayesopt: duplicate variable name '{}'", variable.name),
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        vars.push(variable);
    }
    if vars.iter().all(|var| !var.optimize) {
        return Err(bayesopt_error(
            "bayesopt: at least one variable must have Optimize=true",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(vars)
}

fn variable_from_object(object: &ObjectInstance) -> BuiltinResult<Variable> {
    let name = string_property(object, "Name")?;
    let var_type = string_property(object, "Type")?.to_ascii_lowercase();
    let transform = match string_property(object, "Transform")?
        .to_ascii_lowercase()
        .as_str()
    {
        "none" => Transform::None,
        "log" => Transform::Log,
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: unsupported Transform '{other}'"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    let optimize = match object.properties.get("Optimize") {
        Some(Value::Bool(value)) => *value,
        Some(other) => {
            return Err(bayesopt_error(
                format!("bayesopt: Optimize must be logical scalar, got {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
        None => true,
    };
    let range = object.properties.get("Range").ok_or_else(|| {
        bayesopt_error(
            format!("bayesopt: variable '{name}' is missing Range"),
            &ERROR_INVALID_ARGUMENT,
        )
    })?;
    let kind = match var_type.as_str() {
        "real" => {
            let (lower, upper) = numeric_range(range, &name)?;
            if transform == Transform::Log && lower <= 0.0 {
                return Err(bayesopt_error(
                    format!("bayesopt: log transform for '{name}' requires positive bounds"),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            VariableKind::Real { lower, upper }
        }
        "integer" => {
            let (lower, upper) = numeric_range(range, &name)?;
            if lower.fract() != 0.0 || upper.fract() != 0.0 {
                return Err(bayesopt_error(
                    format!("bayesopt: integer variable '{name}' must have whole-number bounds"),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            if transform == Transform::Log && lower < 0.0 {
                return Err(bayesopt_error(
                    format!("bayesopt: log transform for integer variable '{name}' requires nonnegative bounds"),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            VariableKind::Integer {
                lower: lower as i64,
                upper: upper as i64,
            }
        }
        "categorical" => {
            if transform != Transform::None {
                return Err(bayesopt_error(
                    format!(
                        "bayesopt: categorical variable '{name}' only supports Transform='none'"
                    ),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            VariableKind::Categorical {
                categories: categorical_range(range, &name)?,
            }
        }
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: unsupported variable Type '{other}'"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    Ok(Variable {
        name,
        kind,
        transform,
        optimize,
    })
}

fn string_property(object: &ObjectInstance, name: &str) -> BuiltinResult<String> {
    match object.properties.get(name) {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::StringArray(array)) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Some(Value::CharArray(chars)) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Some(other) => Err(bayesopt_error(
            format!("bayesopt: {name} must be a text scalar, got {other:?}"),
            &ERROR_INVALID_ARGUMENT,
        )),
        None => Err(bayesopt_error(
            format!("bayesopt: optimizableVariable is missing {name}"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn numeric_range(value: &Value, name: &str) -> BuiltinResult<(f64, f64)> {
    let values = match value {
        Value::Tensor(tensor) => tensor::tensor_values_f64(tensor),
        Value::Num(n) => vec![*n],
        Value::Int(i) => vec![i.to_f64()],
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: Range for '{name}' must be numeric, got {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    if values.len() != 2
        || !values[0].is_finite()
        || !values[1].is_finite()
        || values[0] >= values[1]
    {
        return Err(bayesopt_error(
            format!("bayesopt: Range for '{name}' must contain increasing finite bounds"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok((values[0], values[1]))
}

fn categorical_range(value: &Value, name: &str) -> BuiltinResult<Vec<String>> {
    let categories = match value {
        Value::String(text) => vec![text.clone()],
        Value::StringArray(array) => array.data.clone(),
        Value::CharArray(chars) if chars.rows == 1 => vec![chars.data.iter().collect()],
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for value in &cell.data {
                out.push(value_to_text(value, "Range")?);
            }
            out
        }
        other => {
            return Err(bayesopt_error(
                format!(
                "bayesopt: categorical Range for '{name}' must contain text values, got {other:?}"
            ),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    if categories.is_empty() || categories.iter().any(|text| text.trim().is_empty()) {
        return Err(bayesopt_error(
            format!("bayesopt: categorical Range for '{name}' must contain nonempty text"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let mut seen = HashSet::with_capacity(categories.len());
    for category in &categories {
        if !seen.insert(category.clone()) {
            return Err(bayesopt_error(
                format!("bayesopt: categorical Range for '{name}' contains duplicate '{category}'"),
                &ERROR_INVALID_ARGUMENT,
            ));
        }
    }
    Ok(categories)
}

#[derive(Clone)]
struct BayesoptOptions {
    max_evals: usize,
    num_seed_points: usize,
    acquisition: Acquisition,
    exploration_ratio: f64,
    deterministic: bool,
    verbose: bool,
    x_constraint: Option<Value>,
    raw_options: StructValue,
}

#[derive(Clone, Copy)]
enum Acquisition {
    ExpectedImprovement,
    ExpectedImprovementPlus,
    LowerConfidenceBound,
}

impl Acquisition {
    fn as_str(self) -> &'static str {
        match self {
            Self::ExpectedImprovement => "expected-improvement",
            Self::ExpectedImprovementPlus => "expected-improvement-plus",
            Self::LowerConfidenceBound => "lower-confidence-bound",
        }
    }
}

impl BayesoptOptions {
    fn parse(args: Vec<Value>) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(bayesopt_error(
                "bayesopt: name-value options must be paired",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        let mut raw_options = StructValue::new();
        let mut max_evals = DEFAULT_MAX_EVALS;
        let mut num_seed_points = DEFAULT_SEED_POINTS;
        let mut acquisition = Acquisition::ExpectedImprovementPlus;
        let mut exploration_ratio = 0.5;
        let mut deterministic = true;
        let mut verbose = false;
        let mut x_constraint = None;

        for pair in args.chunks_exact(2) {
            let name = value_to_text(&pair[0], "option name")?;
            let canonical = canonical_option(&name);
            let value = pair[1].clone();
            match canonical.as_str() {
                "MaxObjectiveEvaluations" => {
                    max_evals = option_usize(&value, &canonical, 1, MAX_EVALS_LIMIT)?
                }
                "NumSeedPoints" => {
                    num_seed_points = option_usize(&value, &canonical, 0, MAX_EVALS_LIMIT)?
                }
                "AcquisitionFunctionName" => {
                    acquisition = parse_acquisition(&value)?;
                }
                "ExplorationRatio" => {
                    exploration_ratio = option_unit_f64(&value, &canonical)?;
                }
                "IsObjectiveDeterministic" => {
                    deterministic = option_bool(&value, &canonical)?;
                }
                "Verbose" => {
                    verbose = option_bool(&value, &canonical)?;
                }
                "UseParallel" => {
                    if option_bool(&value, &canonical)? {
                        return Err(bayesopt_error(
                            "bayesopt: UseParallel=true requires the parallel execution infrastructure",
                            &ERROR_UNSUPPORTED,
                        ));
                    }
                }
                "XConstraintFcn" => {
                    if is_empty_option(&value) {
                        x_constraint = None;
                    } else if is_callable(&value) {
                        x_constraint = Some(value.clone());
                    } else {
                        return Err(bayesopt_error(
                            "bayesopt: XConstraintFcn must be empty or a function handle",
                            &ERROR_INVALID_ARGUMENT,
                        ));
                    }
                }
                "PlotFcn" | "OutputFcn" => {
                    if !is_empty_option(&value) {
                        return Err(bayesopt_error(
                            format!("bayesopt: {canonical} callbacks are not supported in this headless runtime slice"),
                            &ERROR_UNSUPPORTED,
                        ));
                    }
                }
                "ConditionalVariableFcn" => {
                    if !is_empty_option(&value) {
                        return Err(bayesopt_error(
                            "bayesopt: ConditionalVariableFcn requires dynamic variable activation support",
                            &ERROR_UNSUPPORTED,
                        ));
                    }
                }
                "InitialX" | "InitialObjective" | "InitialConstraintViolations" => {
                    return Err(bayesopt_error(
                        format!("bayesopt: {canonical} warm starts are not supported yet"),
                        &ERROR_UNSUPPORTED,
                    ));
                }
                other => {
                    return Err(bayesopt_error(
                        format!("bayesopt: unsupported option '{other}'"),
                        &ERROR_INVALID_ARGUMENT,
                    ));
                }
            }
            raw_options.insert(&canonical, value);
        }
        if num_seed_points > max_evals {
            num_seed_points = max_evals;
        }
        raw_options.insert("MaxObjectiveEvaluations", Value::Num(max_evals as f64));
        raw_options.insert("NumSeedPoints", Value::Num(num_seed_points as f64));
        raw_options.insert(
            "AcquisitionFunctionName",
            Value::String(acquisition.as_str().to_string()),
        );
        raw_options.insert("ExplorationRatio", Value::Num(exploration_ratio));
        raw_options.insert("IsObjectiveDeterministic", Value::Bool(deterministic));
        raw_options.insert("Verbose", Value::Bool(verbose));
        raw_options.insert("UseParallel", Value::Bool(false));

        Ok(Self {
            max_evals,
            num_seed_points,
            acquisition,
            exploration_ratio,
            deterministic,
            verbose,
            x_constraint,
            raw_options,
        })
    }
}

fn canonical_option(name: &str) -> String {
    match name.trim().to_ascii_lowercase().as_str() {
        "maxobjectiveevaluations" => "MaxObjectiveEvaluations",
        "numseedpoints" => "NumSeedPoints",
        "acquisitionfunctionname" => "AcquisitionFunctionName",
        "explorationratio" => "ExplorationRatio",
        "isobjectivedeterministic" => "IsObjectiveDeterministic",
        "verbose" => "Verbose",
        "useparallel" => "UseParallel",
        "xconstraintfcn" => "XConstraintFcn",
        "plotfcn" => "PlotFcn",
        "outputfcn" => "OutputFcn",
        "conditionalvariablefcn" => "ConditionalVariableFcn",
        "initialx" => "InitialX",
        "initialobjective" => "InitialObjective",
        "initialconstraintviolations" => "InitialConstraintViolations",
        other => other,
    }
    .to_string()
}

fn parse_acquisition(value: &Value) -> BuiltinResult<Acquisition> {
    let text = value_to_text(value, "AcquisitionFunctionName")?
        .to_ascii_lowercase()
        .replace('_', "-");
    match text.as_str() {
        "expected-improvement" | "expected-improvement-per-second" => {
            Ok(Acquisition::ExpectedImprovement)
        }
        "expected-improvement-plus" | "expected-improvement-per-second-plus" => {
            Ok(Acquisition::ExpectedImprovementPlus)
        }
        "lower-confidence-bound" => Ok(Acquisition::LowerConfidenceBound),
        other => Err(bayesopt_error(
            format!("bayesopt: unsupported AcquisitionFunctionName '{other}'"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

async fn optimize(
    objective: Value,
    variables: Vec<Variable>,
    options: BayesoptOptions,
) -> BuiltinResult<BayesoptOutcome> {
    if !is_callable(&objective) {
        return Err(bayesopt_error(
            "bayesopt: objective must be a function handle",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let start = Instant::now();
    let candidates = candidate_set(&variables, options.max_evals)?;
    let mut seen = HashSet::new();
    let mut observations = Vec::with_capacity(options.max_evals);
    let mut surrogate = SurrogateSummary::default();
    let mut exhausted_candidates = false;

    while observations.len() < options.max_evals {
        let Some(point) =
            choose_next_point(&options, &candidates, &observations, &seen, &mut surrogate)
        else {
            exhausted_candidates = true;
            break;
        };
        let key = point.key();
        if !seen.insert(key) {
            exhausted_candidates = true;
            break;
        }
        if let Some(constraint) = &options.x_constraint {
            let row = point.to_table(&variables)?;
            if !constraint_allows(constraint, row).await? {
                continue;
            }
        }
        let row = point.to_table(&variables)?;
        let eval_start = Instant::now();
        let callback = call_function(&objective, vec![row]).await.map_err(|err| {
            bayesopt_error(
                format!("bayesopt: objective evaluation failed: {}", err.message()),
                &ERROR_OBJECTIVE,
            )
        })?;
        let elapsed = eval_start.elapsed().as_secs_f64();
        let objective_value = parse_objective_value(callback).await?;
        observations.push(Observation {
            point,
            objective: objective_value.objective,
            constraint_violations: objective_value.constraint_violations,
            objective_time: elapsed,
        });
    }

    if observations.is_empty() {
        return Err(bayesopt_error(
            "bayesopt: no objective evaluations were completed",
            &ERROR_OBJECTIVE,
        ));
    }
    let total_time = start.elapsed().as_secs_f64();
    Ok(BayesoptOutcome {
        variables,
        options,
        observations,
        surrogate,
        total_time,
        exhausted_candidates,
    })
}

fn choose_next_point(
    options: &BayesoptOptions,
    candidates: &[Point],
    observations: &[Observation],
    seen: &HashSet<String>,
    surrogate: &mut SurrogateSummary,
) -> Option<Point> {
    if observations.len() < options.num_seed_points {
        return candidates
            .iter()
            .filter(|point| !seen.contains(&point.key()))
            .max_by(|a, b| {
                let da = min_distance_to_observations(a, observations);
                let db = min_distance_to_observations(b, observations);
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            })
            .cloned();
    }
    let model = GaussianProcess::fit(observations);
    surrogate.length_scale = model.length_scale;
    surrogate.noise = GP_NOISE;
    let best = best_feasible_objective(observations).unwrap_or_else(|| {
        observations
            .iter()
            .map(Observation::penalized_objective)
            .fold(f64::INFINITY, f64::min)
    });
    candidates
        .iter()
        .filter(|point| !seen.contains(&point.key()))
        .map(|point| {
            let (mean, variance) = model.predict(&point.encoded);
            let score = acquisition_score(
                options.acquisition,
                best,
                mean,
                variance,
                options.exploration_ratio,
            );
            (score, point)
        })
        .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(_, point)| point.clone())
}

fn min_distance_to_observations(point: &Point, observations: &[Observation]) -> f64 {
    if observations.is_empty() {
        return f64::INFINITY;
    }
    observations
        .iter()
        .map(|observation| squared_distance(&point.encoded, &observation.point.encoded).sqrt())
        .fold(f64::INFINITY, f64::min)
}

fn acquisition_score(
    acquisition: Acquisition,
    best: f64,
    mean: f64,
    variance: f64,
    exploration: f64,
) -> f64 {
    let std = variance.max(0.0).sqrt();
    match acquisition {
        Acquisition::LowerConfidenceBound => -(mean - (1.0 + 4.0 * exploration) * std),
        Acquisition::ExpectedImprovement | Acquisition::ExpectedImprovementPlus => {
            if std <= 1.0e-12 {
                return 0.0;
            }
            let improvement = best - mean;
            let z = improvement / std;
            let ei = improvement * normal_cdf(z) + std * normal_pdf(z);
            if matches!(acquisition, Acquisition::ExpectedImprovementPlus) {
                ei + exploration * std * 1.0e-3
            } else {
                ei
            }
        }
    }
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t)
            * (-x * x).exp();
    sign * y
}

#[derive(Clone)]
struct Point {
    values: Vec<VariableValue>,
    encoded: Vec<f64>,
}

impl Point {
    fn key(&self) -> String {
        self.values
            .iter()
            .map(VariableValue::key)
            .collect::<Vec<_>>()
            .join("|")
    }

    fn to_table(&self, variables: &[Variable]) -> BuiltinResult<Value> {
        let mut names = Vec::with_capacity(variables.len());
        let mut columns = Vec::with_capacity(variables.len());
        for (variable, value) in variables.iter().zip(&self.values) {
            names.push(variable.name.clone());
            columns.push(value.column_value()?);
        }
        crate::builtins::table::table_from_columns(names, columns)
            .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))
    }
}

#[derive(Clone)]
enum VariableValue {
    Numeric(f64),
    Categorical(String),
}

impl VariableValue {
    fn key(&self) -> String {
        match self {
            Self::Numeric(value) => format!("{value:.17}"),
            Self::Categorical(value) => value.clone(),
        }
    }

    fn column_value(&self) -> BuiltinResult<Value> {
        match self {
            Self::Numeric(value) => Tensor::new(vec![*value], vec![1, 1])
                .map(Value::Tensor)
                .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT)),
            Self::Categorical(value) => StringArray::new(vec![value.clone()], vec![1, 1])
                .map(Value::StringArray)
                .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT)),
        }
    }
}

fn candidate_set(variables: &[Variable], max_evals: usize) -> BuiltinResult<Vec<Point>> {
    let active_dims = variables.iter().filter(|var| var.optimize).count().max(1);
    let mut count = (max_evals * 24).clamp(128, MAX_CANDIDATES);
    count = count.max(max_evals);
    let mut candidates = Vec::with_capacity(count + 2);
    candidates.push(point_from_unit(variables, &vec![0.5; active_dims]));
    candidates.push(point_from_unit(variables, &vec![0.0; active_dims]));
    candidates.push(point_from_unit(variables, &vec![1.0; active_dims]));
    for idx in 0..count {
        let units = (0..active_dims)
            .map(|dim| halton(idx + 1, prime(dim)))
            .collect::<Vec<_>>();
        candidates.push(point_from_unit(variables, &units));
    }
    let mut seen = HashSet::new();
    candidates.retain(|point| seen.insert(point.key()));
    if candidates.is_empty() {
        return Err(bayesopt_error(
            "bayesopt: candidate generation produced no points",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(candidates)
}

fn point_from_unit(variables: &[Variable], units: &[f64]) -> Point {
    let mut active = 0usize;
    let mut values = Vec::with_capacity(variables.len());
    let mut encoded = Vec::new();
    for variable in variables {
        let unit = if variable.optimize {
            let value = units.get(active).copied().unwrap_or(0.5).clamp(0.0, 1.0);
            active += 1;
            value
        } else {
            0.0
        };
        match &variable.kind {
            VariableKind::Real { lower, upper } => {
                let value = decode_real(*lower, *upper, variable.transform, unit);
                values.push(VariableValue::Numeric(value));
                if variable.optimize {
                    encoded.push(encode_real(*lower, *upper, variable.transform, value));
                }
            }
            VariableKind::Integer { lower, upper } => {
                let value = decode_integer(*lower, *upper, variable.transform, unit);
                values.push(VariableValue::Numeric(value as f64));
                if variable.optimize {
                    let denom = (*upper - *lower).max(1) as f64;
                    encoded.push((value - *lower) as f64 / denom);
                }
            }
            VariableKind::Categorical { categories } => {
                let len = categories.len();
                let idx = if variable.optimize {
                    ((unit * len as f64).floor() as usize).min(len.saturating_sub(1))
                } else {
                    0
                };
                values.push(VariableValue::Categorical(categories[idx].clone()));
                if variable.optimize {
                    if len <= 1 {
                        encoded.push(0.0);
                    } else {
                        encoded.push(idx as f64 / (len - 1) as f64);
                    }
                }
            }
        }
    }
    Point { values, encoded }
}

fn decode_real(lower: f64, upper: f64, transform: Transform, unit: f64) -> f64 {
    match transform {
        Transform::None => lower + (upper - lower) * unit,
        Transform::Log => (lower.ln() + (upper.ln() - lower.ln()) * unit).exp(),
    }
}

fn encode_real(lower: f64, upper: f64, transform: Transform, value: f64) -> f64 {
    match transform {
        Transform::None => (value - lower) / (upper - lower),
        Transform::Log => (value.ln() - lower.ln()) / (upper.ln() - lower.ln()),
    }
    .clamp(0.0, 1.0)
}

fn decode_integer(lower: i64, upper: i64, transform: Transform, unit: f64) -> i64 {
    let value = match transform {
        Transform::None => lower as f64 + (upper - lower) as f64 * unit,
        Transform::Log => {
            let lo = (lower.max(1) as f64).ln();
            let hi = (upper.max(1) as f64).ln();
            (lo + (hi - lo) * unit).exp()
        }
    };
    value.round().clamp(lower as f64, upper as f64) as i64
}

fn halton(mut index: usize, base: usize) -> f64 {
    let mut f = 1.0;
    let mut r = 0.0;
    while index > 0 {
        f /= base as f64;
        r += f * (index % base) as f64;
        index /= base;
    }
    r
}

fn prime(index: usize) -> usize {
    const PRIMES: [usize; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
    PRIMES[index % PRIMES.len()]
}

#[derive(Clone)]
struct Observation {
    point: Point,
    objective: f64,
    constraint_violations: Vec<f64>,
    objective_time: f64,
}

impl Observation {
    fn feasible(&self) -> bool {
        self.constraint_violations.iter().all(|value| *value <= 0.0)
    }

    fn total_violation(&self) -> f64 {
        self.constraint_violations
            .iter()
            .map(|value| value.max(0.0))
            .sum()
    }

    fn penalized_objective(&self) -> f64 {
        self.objective + self.total_violation() * 1.0e6
    }
}

#[derive(Default)]
struct SurrogateSummary {
    length_scale: f64,
    noise: f64,
}

struct GaussianProcess {
    x: Vec<Vec<f64>>,
    y_mean: f64,
    y_scale: f64,
    alpha: Vec<f64>,
    length_scale: f64,
}

impl GaussianProcess {
    fn fit(observations: &[Observation]) -> Self {
        let x = observations
            .iter()
            .map(|obs| obs.point.encoded.clone())
            .collect::<Vec<_>>();
        let y_raw = observations
            .iter()
            .map(Observation::penalized_objective)
            .collect::<Vec<_>>();
        let y_mean = y_raw.iter().sum::<f64>() / y_raw.len() as f64;
        let variance = y_raw
            .iter()
            .map(|value| (value - y_mean).powi(2))
            .sum::<f64>()
            / y_raw.len().max(1) as f64;
        let y_scale = variance.sqrt().max(1.0);
        let y = y_raw
            .iter()
            .map(|value| (value - y_mean) / y_scale)
            .collect::<Vec<_>>();
        let dims = x.first().map(Vec::len).unwrap_or(1).max(1);
        let length_scale = (0.35 / (dims as f64).sqrt()).max(0.08);
        let mut kernel = vec![vec![0.0; x.len()]; x.len()];
        for row in 0..x.len() {
            for col in 0..x.len() {
                kernel[row][col] = rbf(&x[row], &x[col], length_scale);
            }
            kernel[row][row] += GP_NOISE;
        }
        let alpha = solve_spd(kernel, y).unwrap_or_else(|| vec![0.0; x.len()]);
        Self {
            x,
            y_mean,
            y_scale,
            alpha,
            length_scale,
        }
    }

    fn predict(&self, point: &[f64]) -> (f64, f64) {
        if self.x.is_empty() {
            return (0.0, 1.0);
        }
        let k = self
            .x
            .iter()
            .map(|x| rbf(x, point, self.length_scale))
            .collect::<Vec<_>>();
        let normalized_mean = k
            .iter()
            .zip(&self.alpha)
            .map(|(ki, alpha)| ki * alpha)
            .sum::<f64>();
        let nearest = self
            .x
            .iter()
            .map(|x| squared_distance(x, point))
            .fold(f64::INFINITY, f64::min);
        let variance = (1.0 - (-nearest / (2.0 * self.length_scale * self.length_scale)).exp())
            .clamp(1.0e-9, 1.0);
        (
            self.y_mean + normalized_mean * self.y_scale,
            variance * self.y_scale * self.y_scale,
        )
    }
}

fn rbf(a: &[f64], b: &[f64], length_scale: f64) -> f64 {
    (-squared_distance(a, b) / (2.0 * length_scale * length_scale)).exp()
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
}

fn solve_spd(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    if n == 0 {
        return Some(Vec::new());
    }
    for i in 0..n {
        let pivot = matrix[i][i];
        if !pivot.is_finite() || pivot.abs() < 1.0e-14 {
            return None;
        }
        for j in (i + 1)..n {
            let factor = matrix[j][i] / pivot;
            for k in i..n {
                matrix[j][k] -= factor * matrix[i][k];
            }
            rhs[j] -= factor * rhs[i];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = rhs[i];
        for j in (i + 1)..n {
            acc -= matrix[i][j] * x[j];
        }
        x[i] = acc / matrix[i][i];
    }
    Some(x)
}

struct ObjectiveValue {
    objective: f64,
    constraint_violations: Vec<f64>,
}

async fn parse_objective_value(value: Value) -> BuiltinResult<ObjectiveValue> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| bayesopt_error(format!("bayesopt: {}", err.message()), &ERROR_OBJECTIVE))?;
    match value {
        Value::Struct(st) => {
            let objective = st
                .fields
                .get("Objective")
                .or_else(|| st.fields.get("objective"))
                .cloned()
                .ok_or_else(|| {
                    bayesopt_error(
                        "bayesopt: objective struct must contain Objective",
                        &ERROR_OBJECTIVE,
                    )
                })?;
            let objective = value_to_scalar(NAME, objective).map_err(|err| {
                bayesopt_error(format!("bayesopt: {}", err.message()), &ERROR_OBJECTIVE)
            })?;
            let constraint_violations = match st
                .fields
                .get("ConstraintViolations")
                .or_else(|| st.fields.get("constraintViolations"))
                .or_else(|| st.fields.get("Constraints"))
            {
                Some(value) => finite_vector(value.clone(), "ConstraintViolations")?,
                None => Vec::new(),
            };
            Ok(ObjectiveValue {
                objective,
                constraint_violations,
            })
        }
        other => Ok(ObjectiveValue {
            objective: value_to_scalar(NAME, other).map_err(|err| {
                bayesopt_error(format!("bayesopt: {}", err.message()), &ERROR_OBJECTIVE)
            })?,
            constraint_violations: Vec::new(),
        }),
    }
}

async fn constraint_allows(handle: &Value, row: Value) -> BuiltinResult<bool> {
    let value = call_function(handle, vec![row]).await.map_err(|err| {
        bayesopt_error(
            format!(
                "bayesopt: XConstraintFcn evaluation failed: {}",
                err.message()
            ),
            &ERROR_OBJECTIVE,
        )
    })?;
    let value = gather_if_needed_async(&value).await.map_err(|err| {
        bayesopt_error(
            format!("bayesopt: XConstraintFcn gather failed: {}", err.message()),
            &ERROR_OBJECTIVE,
        )
    })?;
    match value {
        Value::Bool(value) => Ok(value),
        Value::LogicalArray(logical) if logical.data.len() == 1 => Ok(logical.data[0] != 0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
            Ok(tensor::tensor_value_f64(&tensor, 0) != 0.0)
        }
        Value::Num(value) => Ok(value != 0.0),
        other => Err(bayesopt_error(
            format!("bayesopt: XConstraintFcn must return a logical scalar, got {other:?}"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

struct BayesoptOutcome {
    variables: Vec<Variable>,
    options: BayesoptOptions,
    observations: Vec<Observation>,
    surrogate: SurrogateSummary,
    total_time: f64,
    exhausted_candidates: bool,
}

fn finalize(outcome: BayesoptOutcome) -> BuiltinResult<Value> {
    let result = result_object(outcome)?;
    match crate::output_count::current_output_count() {
        None => Ok(result),
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![result])),
        Some(requested) => Err(bayesopt_error(
            format!("bayesopt: requested {requested} outputs, but only one is supported"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn result_object(outcome: BayesoptOutcome) -> BuiltinResult<Value> {
    let best_index = best_observation_index(&outcome.observations).ok_or_else(|| {
        bayesopt_error(
            "bayesopt: no finite objective observations were recorded",
            &ERROR_OBJECTIVE,
        )
    })?;
    let best = &outcome.observations[best_index];
    let x_trace = trace_table(&outcome.variables, &outcome.observations)?;
    let x_at_min = best.point.to_table(&outcome.variables)?;
    let objective_trace = tensor_column(
        outcome
            .observations
            .iter()
            .map(|obs| obs.objective)
            .collect(),
    )?;
    let objective_minimum_trace = tensor_column(best_so_far_trace(&outcome.observations))?;
    let feasibility_trace = LogicalArray::new(
        outcome
            .observations
            .iter()
            .map(|obs| u8::from(obs.feasible()))
            .collect(),
        vec![outcome.observations.len(), 1],
    )
    .map(Value::LogicalArray)
    .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))?;
    let objective_time_trace = tensor_column(
        outcome
            .observations
            .iter()
            .map(|obs| obs.objective_time)
            .collect(),
    )?;
    let optimization_trace = optimization_trace_table(&outcome.observations)?;

    let mut output = StructValue::new();
    output.insert("iterations", Value::Num(outcome.observations.len() as f64));
    output.insert("funcCount", Value::Num(outcome.observations.len() as f64));
    output.insert(
        "algorithm",
        Value::String("deterministic Gaussian-process expected improvement".into()),
    );
    output.insert(
        "message",
        Value::String(
            if outcome.exhausted_candidates {
                "Optimization completed after exhausting the finite candidate set."
            } else {
                "Optimization completed MaxObjectiveEvaluations."
            }
            .into(),
        ),
    );

    let mut surrogate = StructValue::new();
    surrogate.insert("Kernel", Value::String("squared-exponential".into()));
    surrogate.insert("LengthScale", Value::Num(outcome.surrogate.length_scale));
    surrogate.insert("Noise", Value::Num(outcome.surrogate.noise));

    let mut object = ObjectInstance::new(RESULT_CLASS.to_string());
    object.properties.insert("XTrace".into(), x_trace);
    object
        .properties
        .insert("ObjectiveTrace".into(), objective_trace);
    object
        .properties
        .insert("ObjectiveMinimumTrace".into(), objective_minimum_trace);
    object
        .properties
        .insert("ObjectiveEvaluationTimeTrace".into(), objective_time_trace);
    object
        .properties
        .insert("FeasibilityTrace".into(), feasibility_trace);
    object
        .properties
        .insert("OptimizationTrace".into(), optimization_trace);
    object.properties.insert("XAtMinObjective".into(), x_at_min);
    object
        .properties
        .insert("MinObjective".into(), Value::Num(best.objective));
    object.properties.insert(
        "IndexOfMinimumTrace".into(),
        Value::Num((best_index + 1) as f64),
    );
    object.properties.insert(
        "NumObjectiveEvaluations".into(),
        Value::Num(outcome.observations.len() as f64),
    );
    object
        .properties
        .insert("TotalElapsedTime".into(), Value::Num(outcome.total_time));
    object
        .properties
        .insert("Options".into(), Value::Struct(outcome.options.raw_options));
    object
        .properties
        .insert("SurrogateModel".into(), Value::Struct(surrogate));
    object.properties.insert(
        "VariableDescriptions".into(),
        variable_descriptions(&outcome.variables)?,
    );
    object
        .properties
        .insert("Output".into(), Value::Struct(output));
    object.properties.insert("ExitFlag".into(), Value::Num(1.0));
    object
        .properties
        .insert("BestPoint".into(), best.point.to_table(&outcome.variables)?);
    object
        .properties
        .insert("BestObjective".into(), Value::Num(best.objective));
    object.properties.insert(
        "IsObjectiveDeterministic".into(),
        Value::Bool(outcome.options.deterministic),
    );
    object
        .properties
        .insert("Verbose".into(), Value::Bool(outcome.options.verbose));
    Ok(Value::Object(object))
}

fn best_observation_index(observations: &[Observation]) -> Option<usize> {
    observations
        .iter()
        .enumerate()
        .filter(|(_, obs)| obs.feasible() && obs.objective.is_finite())
        .min_by(|(_, a), (_, b)| {
            a.objective
                .partial_cmp(&b.objective)
                .unwrap_or(Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .or_else(|| {
            observations
                .iter()
                .enumerate()
                .filter(|(_, obs)| obs.objective.is_finite())
                .min_by(|(_, a), (_, b)| {
                    a.penalized_objective()
                        .partial_cmp(&b.penalized_objective())
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(idx, _)| idx)
        })
}

fn best_feasible_objective(observations: &[Observation]) -> Option<f64> {
    observations
        .iter()
        .filter(|obs| obs.feasible() && obs.objective.is_finite())
        .map(|obs| obs.objective)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
}

fn best_so_far_trace(observations: &[Observation]) -> Vec<f64> {
    let mut best = f64::INFINITY;
    let mut trace = Vec::with_capacity(observations.len());
    for obs in observations {
        if obs.feasible() {
            best = best.min(obs.objective);
        }
        trace.push(best);
    }
    trace
}

fn trace_table(variables: &[Variable], observations: &[Observation]) -> BuiltinResult<Value> {
    let mut names = Vec::with_capacity(variables.len());
    let mut columns = Vec::with_capacity(variables.len());
    for (idx, variable) in variables.iter().enumerate() {
        names.push(variable.name.clone());
        match &variable.kind {
            VariableKind::Categorical { .. } => {
                let data = observations
                    .iter()
                    .map(|obs| match &obs.point.values[idx] {
                        VariableValue::Categorical(value) => value.clone(),
                        VariableValue::Numeric(value) => value.to_string(),
                    })
                    .collect::<Vec<_>>();
                columns.push(Value::StringArray(
                    StringArray::new(data, vec![observations.len(), 1]).map_err(|err| {
                        bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT)
                    })?,
                ));
            }
            _ => {
                let data = observations
                    .iter()
                    .map(|obs| match &obs.point.values[idx] {
                        VariableValue::Numeric(value) => *value,
                        VariableValue::Categorical(_) => f64::NAN,
                    })
                    .collect::<Vec<_>>();
                columns.push(tensor_column(data)?);
            }
        }
    }
    crate::builtins::table::table_from_columns(names, columns)
        .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))
}

fn optimization_trace_table(observations: &[Observation]) -> BuiltinResult<Value> {
    let n = observations.len();
    let iterations = tensor_column((1..=n).map(|idx| idx as f64).collect())?;
    let objective = tensor_column(observations.iter().map(|obs| obs.objective).collect())?;
    let best = tensor_column(best_so_far_trace(observations))?;
    let feasible = LogicalArray::new(
        observations
            .iter()
            .map(|obs| u8::from(obs.feasible()))
            .collect(),
        vec![n, 1],
    )
    .map(Value::LogicalArray)
    .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))?;
    crate::builtins::table::table_from_columns(
        vec![
            "Iteration".into(),
            "Objective".into(),
            "BestSoFar".into(),
            "Feasible".into(),
        ],
        vec![iterations, objective, best, feasible],
    )
    .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))
}

fn variable_descriptions(variables: &[Variable]) -> BuiltinResult<Value> {
    let values = variables
        .iter()
        .map(|var| {
            let mut object = ObjectInstance::new("optimizableVariable".into());
            object
                .properties
                .insert("Name".into(), Value::String(var.name.clone()));
            object.properties.insert(
                "Type".into(),
                Value::String(
                    match &var.kind {
                        VariableKind::Real { .. } => "real",
                        VariableKind::Integer { .. } => "integer",
                        VariableKind::Categorical { .. } => "categorical",
                    }
                    .into(),
                ),
            );
            object.properties.insert(
                "Transform".into(),
                Value::String(
                    match var.transform {
                        Transform::None => "none",
                        Transform::Log => "log",
                    }
                    .into(),
                ),
            );
            object
                .properties
                .insert("Optimize".into(), Value::Bool(var.optimize));
            object
                .properties
                .insert("Range".into(), variable_range_value(var)?);
            Ok(Value::Object(object))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    CellArray::new(values, variables.len(), 1)
        .map(Value::Cell)
        .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))
}

fn variable_range_value(var: &Variable) -> BuiltinResult<Value> {
    match &var.kind {
        VariableKind::Real { lower, upper } => tensor_column(vec![*lower, *upper]),
        VariableKind::Integer { lower, upper } => tensor_column(vec![*lower as f64, *upper as f64]),
        VariableKind::Categorical { categories } => {
            StringArray::new(categories.clone(), vec![1, categories.len()])
                .map(Value::StringArray)
                .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))
        }
    }
}

fn tensor_column(values: Vec<f64>) -> BuiltinResult<Value> {
    Tensor::new(values.clone(), vec![values.len(), 1])
        .map(Value::Tensor)
        .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))
}

fn finite_vector(value: Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = match value {
        Value::Num(n) => vec![n],
        Value::Int(i) => vec![i.to_f64()],
        Value::Bool(b) => vec![if b { 1.0 } else { 0.0 }],
        Value::Tensor(tensor) => tensor::tensor_into_values_f64(tensor),
        Value::LogicalArray(logical) => logical
            .data
            .iter()
            .map(|value| if *value != 0 { 1.0 } else { 0.0 })
            .collect(),
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: {label} must be numeric or logical, got {other:?}"),
                &ERROR_OBJECTIVE,
            ))
        }
    };
    if data.iter().any(|value| !value.is_finite()) {
        return Err(bayesopt_error(
            format!("bayesopt: {label} must contain finite values"),
            &ERROR_OBJECTIVE,
        ));
    }
    Ok(data)
}

fn value_to_text(value: &Value, label: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        other => Err(bayesopt_error(
            format!("bayesopt: {label} must be a text scalar, got {other:?}"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn option_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(bayesopt_error(
                format!("bayesopt: option {name} must be logical scalar, got {value:?}"),
                &ERROR_INVALID_ARGUMENT,
            )),
        };
    }
    let parsed = match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(logical) if logical.data.len() == 1 => Ok(logical.data[0] != 0),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            let parsed = tensor::tensor_value_f64(tensor, 0);
            if parsed == 0.0 || parsed == 1.0 {
                Ok(parsed != 0.0)
            } else {
                Err(bayesopt_error(
                    format!("bayesopt: option {name} must be logical scalar, got {value:?}"),
                    &ERROR_INVALID_ARGUMENT,
                ))
            }
        }
        other => Err(bayesopt_error(
            format!("bayesopt: option {name} must be logical scalar, got {other:?}"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }?;
    Ok(parsed)
}

fn option_usize(value: &Value, name: &str, min: usize, max: usize) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        let parsed = integer.try_to_usize();
        return match parsed.filter(|parsed| *parsed >= min && *parsed <= max) {
            Some(parsed) => Ok(parsed),
            None => Err(bayesopt_error(
                format!("bayesopt: option {name} must be an integer in [{min}, {max}]"),
                &ERROR_INVALID_ARGUMENT,
            )),
        };
    }
    let parsed = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: option {name} must be numeric scalar, got {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    if !parsed.is_finite() || parsed.fract() != 0.0 || parsed < min as f64 || parsed > max as f64 {
        return Err(bayesopt_error(
            format!("bayesopt: option {name} must be an integer in [{min}, {max}]"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(parsed as usize)
}

fn option_unit_f64(value: &Value, name: &str) -> BuiltinResult<f64> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(0.0),
            Some(1) => Ok(1.0),
            _ => Err(bayesopt_error(
                format!("bayesopt: option {name} must be in [0, 1]"),
                &ERROR_INVALID_ARGUMENT,
            )),
        };
    }
    let parsed = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: option {name} must be numeric scalar, got {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err(bayesopt_error(
            format!("bayesopt: option {name} must be in [0, 1]"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(parsed)
}

fn is_empty_option(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::String(text) => text.is_empty() || text.eq_ignore_ascii_case("none"),
        Value::StringArray(array) => array.data.is_empty(),
        _ => false,
    }
}

fn is_callable(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;
    use std::sync::Arc;

    fn variable(name: &str, range: Vec<f64>, var_type: Option<&str>) -> Value {
        let mut object = ObjectInstance::new("optimizableVariable".into());
        object
            .properties
            .insert("Name".into(), Value::String(name.into()));
        object.properties.insert(
            "Range".into(),
            Value::Tensor(Tensor::new(range, vec![1, 2]).unwrap()),
        );
        object.properties.insert(
            "Type".into(),
            Value::String(var_type.unwrap_or("real").into()),
        );
        object
            .properties
            .insert("Transform".into(), Value::String("none".into()));
        object
            .properties
            .insert("Optimize".into(), Value::Bool(true));
        Value::Object(object)
    }

    fn variable_with_range(name: &str, range: Value, var_type: Option<&str>) -> Value {
        let mut object = ObjectInstance::new("optimizableVariable".into());
        object
            .properties
            .insert("Name".into(), Value::String(name.into()));
        object.properties.insert("Range".into(), range);
        object.properties.insert(
            "Type".into(),
            Value::String(var_type.unwrap_or("real").into()),
        );
        object
            .properties
            .insert("Transform".into(), Value::String("none".into()));
        object
            .properties
            .insert("Optimize".into(), Value::Bool(true));
        Value::Object(object)
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let mut tensor = Tensor::new_integer(storage, shape).unwrap();
        tensor.data.fill(f64::NAN);
        Value::Tensor(tensor)
    }

    #[test]
    fn option_parsers_read_all_integer_storage_variants() {
        let storages = [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for storage in storages {
            let value = poisoned_int_tensor(storage, vec![1, 1]);
            assert!(option_bool(&value, "Verbose").unwrap());
            assert_eq!(
                option_usize(&value, "MaxObjectiveEvaluations", 1, 2).unwrap(),
                1
            );
            assert_eq!(option_unit_f64(&value, "ExplorationRatio").unwrap(), 1.0);
        }
    }

    fn categorical_variable(name: &str, categories: Vec<&str>) -> Value {
        let mut object = ObjectInstance::new("optimizableVariable".into());
        object
            .properties
            .insert("Name".into(), Value::String(name.into()));
        object.properties.insert(
            "Range".into(),
            Value::StringArray(
                StringArray::new(
                    categories.into_iter().map(str::to_string).collect(),
                    vec![1, 2],
                )
                .unwrap(),
            ),
        );
        object
            .properties
            .insert("Type".into(), Value::String("categorical".into()));
        object
            .properties
            .insert("Transform".into(), Value::String("none".into()));
        object
            .properties
            .insert("Optimize".into(), Value::Bool(true));
        Value::Object(object)
    }

    fn variable_cell(values: Vec<Value>) -> Value {
        Value::Cell(CellArray::new(values, 1, 2).unwrap())
    }

    fn table_scalar_arg(args: &[Value], name: &str) -> f64 {
        let Value::Object(table) = &args[0] else {
            panic!("expected table");
        };
        let Value::Struct(vars) = table.properties.get("__table_variables").unwrap() else {
            panic!("expected table variables");
        };
        let Value::Tensor(tensor) = vars.fields.get(name).unwrap() else {
            panic!("expected numeric column");
        };
        tensor.data[0]
    }

    fn table_string_arg(args: &[Value], name: &str) -> String {
        let Value::Object(table) = &args[0] else {
            panic!("expected table");
        };
        let Value::Struct(vars) = table.properties.get("__table_variables").unwrap() else {
            panic!("expected table variables");
        };
        let Value::StringArray(array) = vars.fields.get(name).unwrap() else {
            panic!("expected string column");
        };
        array.data[0].clone()
    }

    #[test]
    fn bayesopt_minimizes_quadratic_over_optimizable_variables() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 1);
                let x = table_scalar_arg(args, "x");
                let y = table_scalar_arg(args, "y");
                Box::pin(async move { Ok(Value::Num((x - 0.25).powi(2) + (y + 0.5).powi(2))) })
            },
        )));
        let result = block_on(bayesopt_builtin(
            Value::BoundFunctionHandle {
                name: "objective".into(),
                function: 1,
            },
            variable_cell(vec![
                variable("x", vec![-1.0, 1.0], None),
                variable("y", vec![-1.0, 1.0], None),
            ]),
            vec![
                Value::String("MaxObjectiveEvaluations".into()),
                Value::Num(24.0),
                Value::String("NumSeedPoints".into()),
                Value::Num(6.0),
            ],
        ))
        .unwrap();
        let Value::Object(object) = result else {
            panic!("expected BayesianOptimization object");
        };
        assert_eq!(object.class_name, RESULT_CLASS);
        let Some(Value::Num(minimum)) = object.properties.get("MinObjective") else {
            panic!("expected minimum");
        };
        assert!(*minimum < 0.08, "minimum was {minimum}");
        assert!(matches!(
            object.properties.get("XTrace"),
            Some(Value::Object(table)) if table.class_name == "table"
        ));
    }

    #[test]
    fn bayesopt_rejects_parallel_option_explicitly() {
        let err = block_on(bayesopt_builtin(
            Value::FunctionHandle("objective".into()),
            variable("x", vec![0.0, 1.0], None),
            vec![Value::String("UseParallel".into()), Value::Bool(true)],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:bayesopt:Unsupported"));
    }

    #[test]
    fn bayesopt_honors_x_constraint_function() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, _requested_outputs| {
                let x = table_scalar_arg(args, "x");
                Box::pin(async move {
                    if function == 2 {
                        Ok(Value::Bool(x >= 0.0))
                    } else {
                        Ok(Value::Num((x - 0.4).powi(2)))
                    }
                })
            },
        )));
        let result = block_on(bayesopt_builtin(
            Value::BoundFunctionHandle {
                name: "objective".into(),
                function: 1,
            },
            variable("x", vec![-1.0, 1.0], None),
            vec![
                Value::String("MaxObjectiveEvaluations".into()),
                Value::Num(12.0),
                Value::String("XConstraintFcn".into()),
                Value::BoundFunctionHandle {
                    name: "constraint".into(),
                    function: 2,
                },
            ],
        ))
        .unwrap();
        let Value::Object(object) = result else {
            panic!("expected result object");
        };
        let Some(Value::Num(count)) = object.properties.get("NumObjectiveEvaluations") else {
            panic!("expected count");
        };
        assert_eq!(*count, 12.0);
    }

    #[test]
    fn bayesopt_supports_mixed_variables_and_objective_constraint_struct() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let n = table_scalar_arg(args, "n");
                let family = table_string_arg(args, "family");
                Box::pin(async move {
                    let mut out = StructValue::new();
                    let family_penalty = if family == "rbf" { 0.0 } else { 1.0 };
                    out.insert("Objective", Value::Num((n - 2.0).powi(2) + family_penalty));
                    out.insert("ConstraintViolations", Value::Num(1.5 - n));
                    Ok(Value::Struct(out))
                })
            },
        )));
        let result = block_on(bayesopt_builtin(
            Value::BoundFunctionHandle {
                name: "objective".into(),
                function: 3,
            },
            variable_cell(vec![
                variable("n", vec![1.0, 3.0], Some("integer")),
                categorical_variable("family", vec!["linear", "rbf"]),
            ]),
            vec![
                Value::String("MaxObjectiveEvaluations".into()),
                Value::Num(18.0),
                Value::String("AcquisitionFunctionName".into()),
                Value::String("expected-improvement".into()),
            ],
        ))
        .unwrap();
        let Value::Object(object) = result else {
            panic!("expected result object");
        };
        let Some(Value::Num(minimum)) = object.properties.get("MinObjective") else {
            panic!("expected minimum");
        };
        assert!(*minimum <= 0.25, "minimum was {minimum}");
        assert!(matches!(
            object.properties.get("FeasibilityTrace"),
            Some(Value::LogicalArray(_))
        ));
    }

    #[test]
    fn bayesopt_reads_typed_integer_storage_exactly() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, _requested_outputs| {
                let x = table_scalar_arg(args, "x");
                Box::pin(async move {
                    if function == 11 {
                        return Ok(poisoned_int_tensor(IntegerStorage::I8(vec![1]), vec![1, 1]));
                    }
                    let mut out = StructValue::new();
                    out.insert("Objective", Value::Num((x - 1.0).powi(2)));
                    out.insert(
                        "ConstraintViolations",
                        poisoned_int_tensor(IntegerStorage::I16(vec![0]), vec![1, 1]),
                    );
                    Ok(Value::Struct(out))
                })
            },
        )));
        let result = block_on(bayesopt_builtin(
            Value::BoundFunctionHandle {
                name: "objective".into(),
                function: 10,
            },
            variable_with_range(
                "x",
                poisoned_int_tensor(IntegerStorage::I16(vec![-1, 2]), vec![1, 2]),
                None,
            ),
            vec![
                Value::String("MaxObjectiveEvaluations".into()),
                poisoned_int_tensor(IntegerStorage::U8(vec![6]), vec![1, 1]),
                Value::String("NumSeedPoints".into()),
                poisoned_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
                Value::String("ExplorationRatio".into()),
                poisoned_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
                Value::String("IsObjectiveDeterministic".into()),
                poisoned_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
                Value::String("XConstraintFcn".into()),
                Value::BoundFunctionHandle {
                    name: "constraint".into(),
                    function: 11,
                },
            ],
        ))
        .unwrap();
        let Value::Object(object) = result else {
            panic!("expected BayesianOptimization object");
        };
        let Some(Value::Num(count)) = object.properties.get("NumObjectiveEvaluations") else {
            panic!("expected count");
        };
        assert_eq!(*count, 6.0);
        let Some(Value::Num(minimum)) = object.properties.get("MinObjective") else {
            panic!("expected minimum");
        };
        assert!(*minimum <= 0.25, "minimum was {minimum}");
    }
}
