//! Classification performance curve compatibility surface.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ObjectInstance, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::{
    build_runtime_error, gather_if_needed_async, output_count, BuiltinResult, RuntimeError,
};

const NAME: &str = "perfcurve";

const OUTPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X-coordinate criterion values for the performance curve.",
};

const OUTPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Y-coordinate criterion values for the performance curve.",
};

const OUTPUT_T: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Score thresholds used to compute the curve.",
};

const OUTPUT_AUC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "AUC",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Area under the X/Y performance curve.",
};

const OUTPUT_OPTROCPT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "OPTROCPT",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optimal operating point on the ROC curve.",
};

const OUTPUT_SUBY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "SUBY",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Y-criterion values computed for each negative subclass.",
};

const OUTPUT_SUBYNAMES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "SUBYNAMES",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Negative subclass labels corresponding to columns of SUBY.",
};

const PARAM_LABELS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "labels",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True class labels.",
};

const PARAM_SCORES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "scores",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Classifier scores for the positive class.",
};

const PARAM_POSCLASS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "posclass",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive class label.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as XCrit, YCrit, NegClass, Weights, Prior, Cost, TVals, XVals, UseNearest, and ProcessNaN.",
};

const INPUTS_BASIC: [BuiltinParamDescriptor; 3] = [PARAM_LABELS, PARAM_SCORES, PARAM_POSCLASS];
const INPUTS_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_LABELS, PARAM_SCORES, PARAM_POSCLASS, PARAM_OPTIONS];
const OUTPUTS_X: [BuiltinParamDescriptor; 1] = [OUTPUT_X];
const OUTPUTS_ALL: [BuiltinParamDescriptor; 7] = [
    OUTPUT_X,
    OUTPUT_Y,
    OUTPUT_T,
    OUTPUT_AUC,
    OUTPUT_OPTROCPT,
    OUTPUT_SUBY,
    OUTPUT_SUBYNAMES,
];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "X = perfcurve(labels, scores, posclass)",
        inputs: &INPUTS_BASIC,
        outputs: &OUTPUTS_X,
    },
    BuiltinSignatureDescriptor {
        label: "X = perfcurve(labels, scores, posclass, Name, Value)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUTS_X,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(___)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUTS_ALL,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERFCURVE.INVALID_ARGUMENT",
    identifier: Some("RunMat:perfcurve:InvalidArgument"),
    when: "Labels, scores, class selections, criteria, thresholds, weights, priors, costs, or name-value options are malformed.",
    message: "perfcurve: invalid argument",
};

const ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERFCURVE.UNSUPPORTED",
    identifier: Some("RunMat:perfcurve:Unsupported"),
    when: "A documented advanced mode requires confidence bounds, bootstrap/cross-validation outputs, or score matrix class-column inference that is not implemented yet.",
    message: "perfcurve: unsupported mode",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERFCURVE.INTERNAL",
    identifier: Some("RunMat:perfcurve:Internal"),
    when: "RunMat cannot allocate or construct perfcurve outputs.",
    message: "perfcurve: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_UNSUPPORTED, ERROR_INTERNAL];

pub const PERFCURVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn perfcurve_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![None, Some(1)]),
    }
}

fn error(message: impl Into<String>, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_INVALID_ARGUMENT)
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_UNSUPPORTED)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LabelKind {
    Numeric,
    String,
    Char,
    Cell,
    Logical,
    Categorical,
}

#[derive(Clone, Debug, PartialEq)]
enum Label {
    Numeric(f64),
    Text(String),
    Logical(bool),
}

#[derive(Clone, Debug)]
struct LabelVector {
    labels: Vec<Label>,
    kind: LabelKind,
    categories: Option<Vec<String>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Criterion {
    Fpr,
    Tpr,
    Tnr,
    Fnr,
    Precision,
    Npv,
    Accuracy,
    TruePositive,
    FalseNegative,
    FalsePositive,
    TrueNegative,
    PredictedPositive,
    RatePredictedPositive,
    RatePredictedNegative,
    ExpectedCost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProcessNaN {
    Ignore,
    AddToFalse,
}

#[derive(Clone, Debug)]
enum ThresholdSpec {
    AboveAll(f64),
    Value(f64),
}

#[derive(Clone, Debug)]
struct Options {
    xcrit: Criterion,
    ycrit: Criterion,
    neg_class: Option<Vec<Label>>,
    weights: Option<Vec<f64>>,
    prior: PriorOption,
    cost: [[f64; 2]; 2],
    tvals: Option<Vec<f64>>,
    xvals: Option<Vec<f64>>,
    use_nearest: bool,
    process_nan: ProcessNaN,
}

#[derive(Clone, Debug)]
enum PriorOption {
    Empirical,
    Uniform,
    Values(f64, f64),
}

#[derive(Clone, Debug)]
struct Observation {
    score: f64,
    weight: f64,
    is_positive: bool,
    neg_index: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct Counts {
    tp: f64,
    fp: f64,
    tn: f64,
    fn_: f64,
}

impl Counts {
    fn total(self) -> f64 {
        self.tp + self.fp + self.tn + self.fn_
    }
}

#[derive(Clone, Debug)]
struct Curve {
    x: Vec<f64>,
    y: Vec<f64>,
    thresholds: Vec<f64>,
    counts: Vec<Counts>,
    auc: f64,
}

#[runtime_builtin(
    name = "perfcurve",
    category = "stats/ml",
    summary = "Compute classifier performance curves and AUC values.",
    keywords = "perfcurve,roc,auc,classification,statistics,machine learning",
    type_resolver(perfcurve_type),
    descriptor(crate::builtins::stats::ml::perfcurve::PERFCURVE_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::perfcurve"
)]
async fn perfcurve_builtin(
    labels: Value,
    scores: Value,
    posclass: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let requested_outputs = output_count::current_output_count();
    if let Some(0) = requested_outputs {
        return Ok(Value::OutputList(Vec::new()));
    }
    if matches!(requested_outputs, Some(count) if count > 7) {
        return Err(invalid("perfcurve: too many output arguments"));
    }

    let labels = gather(labels).await?;
    let scores = gather(scores).await?;
    let posclass = gather(posclass).await?;
    let rest = gather_values(rest).await?;
    let output_limit = requested_outputs.unwrap_or(1).max(1);
    let outputs = perfcurve_compute(labels, scores, posclass, rest, output_limit)?;
    match requested_outputs {
        None => Ok(outputs[0].clone()),
        Some(count) => Ok(output_count::output_list_with_padding(count, outputs)),
    }
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("perfcurve: {err}")))
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(value).await?);
    }
    Ok(out)
}

fn perfcurve_compute(
    labels: Value,
    scores: Value,
    posclass: Value,
    rest: Vec<Value>,
    output_limit: usize,
) -> BuiltinResult<Vec<Value>> {
    let labels = labels_from_value(labels, "labels")?;
    let scores = score_vector(scores)?;
    if labels.labels.len() != scores.len() {
        return Err(invalid(
            "perfcurve: labels and scores must contain the same number of observations",
        ));
    }
    let posclass = coerce_label_to_kind(
        scalar_label_from_value(posclass, "posclass")?,
        labels.kind,
        "posclass",
    )?;
    if !labels
        .labels
        .iter()
        .any(|label| same_label(label, &posclass) && !is_missing_label(label))
    {
        return Err(invalid("perfcurve: posclass must appear in labels"));
    }
    let options = parse_options(rest, labels.kind)?;
    let neg_classes = negative_classes(&labels, &posclass, options.neg_class.as_ref())?;
    if neg_classes.is_empty() {
        return Err(invalid(
            "perfcurve: at least one negative class must be present",
        ));
    }

    let observations = prepare_observations(&labels, &scores, &posclass, &neg_classes, &options)?;
    if observations.iter().filter(|obs| obs.is_positive).count() == 0 {
        return Err(invalid(
            "perfcurve: at least one positive observation is required",
        ));
    }
    if observations.iter().filter(|obs| !obs.is_positive).count() == 0 {
        return Err(invalid(
            "perfcurve: at least one negative observation is required",
        ));
    }

    let thresholds = thresholds_for_scores(&observations, options.tvals.as_ref())?;
    let full_curve = compute_curve(
        &observations,
        &thresholds,
        options.xcrit,
        options.ycrit,
        &options.cost,
    )?;
    let auc = if let Some(xvals) = &options.xvals {
        auc_interval(&full_curve.x, &full_curve.y, xvals)
    } else {
        auc(&full_curve.x, &full_curve.y)
    };
    let mut curve = full_curve.clone();
    if let Some(xvals) = &options.xvals {
        curve = apply_xvals(&curve, xvals, options.use_nearest)?;
    }
    curve.auc = auc;
    let opt = optimal_point(&curve, &options);

    let mut outputs = vec![tensor_column(curve.x)?];
    if output_limit >= 2 {
        outputs.push(tensor_column(curve.y)?);
    }
    if output_limit >= 3 {
        outputs.push(tensor_column(curve.thresholds)?);
    }
    if output_limit >= 4 {
        outputs.push(Value::Num(curve.auc));
    }
    if output_limit >= 5 {
        outputs.push(tensor_row(vec![opt.0, opt.1])?);
    }
    if output_limit >= 6 {
        outputs.push(suby_output(
            &observations,
            &thresholds,
            &neg_classes,
            options.xcrit,
            options.ycrit,
            &options.cost,
            options.xvals.as_ref(),
            options.use_nearest,
        )?);
    }
    if output_limit >= 7 {
        outputs.push(labels_to_cell_value(&neg_classes)?);
    }
    Ok(outputs)
}

fn parse_options(rest: Vec<Value>, label_kind: LabelKind) -> BuiltinResult<Options> {
    if !rest.len().is_multiple_of(2) {
        return Err(invalid("perfcurve: name-value options must be paired"));
    }
    let mut options = Options {
        xcrit: Criterion::Fpr,
        ycrit: Criterion::Tpr,
        neg_class: None,
        weights: None,
        prior: PriorOption::Empirical,
        cost: [[0.0, 1.0], [1.0, 0.0]],
        tvals: None,
        xvals: None,
        use_nearest: true,
        process_nan: ProcessNaN::Ignore,
    };
    let mut index = 0usize;
    while index < rest.len() {
        let name = canonical(&scalar_text(&rest[index], "option name")?);
        let value = &rest[index + 1];
        match name.as_str() {
            "xcrit" => options.xcrit = parse_criterion(value, "XCrit")?,
            "ycrit" => options.ycrit = parse_criterion(value, "YCrit")?,
            "negclass" => {
                let labels = labels_from_value(value.clone(), "NegClass")?;
                let deduped = dedupe_labels(coerce_labels_to_kind(
                    labels.labels,
                    label_kind,
                    "NegClass",
                )?);
                if deduped.is_empty() {
                    return Err(invalid("perfcurve: NegClass cannot be empty"));
                }
                options.neg_class = Some(deduped);
            }
            "weights" => options.weights = Some(numeric_vector(value, "Weights")?),
            "prior" => options.prior = parse_prior(value)?,
            "cost" => options.cost = parse_cost(value)?,
            "tvals" => options.tvals = Some(numeric_vector(value, "TVals")?),
            "xvals" => options.xvals = Some(numeric_vector(value, "XVals")?),
            "usenearest" => options.use_nearest = scalar_bool(value, "UseNearest")?,
            "processnan" => options.process_nan = parse_process_nan(value)?,
            "nboot" | "alpha" | "bootstrap" | "bootstraps" | "nbootstraps" => {
                return Err(unsupported(
                    "perfcurve: confidence-bound and bootstrap modes are not implemented",
                ))
            }
            other => return Err(invalid(format!("perfcurve: unsupported option '{other}'"))),
        }
        index += 2;
    }
    if let Some(tvals) = &options.tvals {
        if tvals.is_empty() || tvals.iter().any(|value| value.is_nan()) {
            return Err(invalid("perfcurve: TVals must contain non-NaN thresholds"));
        }
    }
    if let Some(xvals) = &options.xvals {
        if xvals.is_empty() || xvals.iter().any(|value| !value.is_finite()) {
            return Err(invalid("perfcurve: XVals must contain finite values"));
        }
    }
    if options.tvals.is_some() && options.xvals.is_some() {
        return Err(invalid(
            "perfcurve: TVals and XVals cannot be specified together",
        ));
    }
    if !options.use_nearest {
        if let Some(xvals) = &mut options.xvals {
            xvals.sort_by(f64::total_cmp);
        }
    }
    Ok(options)
}

fn parse_criterion(value: &Value, name: &str) -> BuiltinResult<Criterion> {
    match canonical(&scalar_text(value, name)?).as_str() {
        "fpr" | "fallout" => Ok(Criterion::Fpr),
        "tpr" | "sens" | "sensitivity" | "recall" | "reca" => Ok(Criterion::Tpr),
        "tnr" | "spec" | "specificity" => Ok(Criterion::Tnr),
        "fnr" | "miss" => Ok(Criterion::Fnr),
        "ppv" | "precision" | "prec" => Ok(Criterion::Precision),
        "npv" => Ok(Criterion::Npv),
        "accu" | "accuracy" | "acc" => Ok(Criterion::Accuracy),
        "tp" => Ok(Criterion::TruePositive),
        "fn" => Ok(Criterion::FalseNegative),
        "fp" => Ok(Criterion::FalsePositive),
        "tn" => Ok(Criterion::TrueNegative),
        "tpfp" | "tp+fp" => Ok(Criterion::PredictedPositive),
        "rpp" => Ok(Criterion::RatePredictedPositive),
        "rnp" => Ok(Criterion::RatePredictedNegative),
        "ecost" => Ok(Criterion::ExpectedCost),
        other => Err(invalid(format!(
            "perfcurve: unsupported {name} criterion '{other}'"
        ))),
    }
}

fn parse_process_nan(value: &Value) -> BuiltinResult<ProcessNaN> {
    match canonical(&scalar_text(value, "ProcessNaN")?).as_str() {
        "ignore" => Ok(ProcessNaN::Ignore),
        "addtofalse" => Ok(ProcessNaN::AddToFalse),
        other => Err(invalid(format!(
            "perfcurve: unsupported ProcessNaN value '{other}'"
        ))),
    }
}

fn parse_prior(value: &Value) -> BuiltinResult<PriorOption> {
    if let Ok(text) = scalar_text(value, "Prior") {
        return match canonical(&text).as_str() {
            "empirical" => Ok(PriorOption::Empirical),
            "uniform" => Ok(PriorOption::Uniform),
            other => Err(invalid(format!(
                "perfcurve: unsupported Prior value '{other}'"
            ))),
        };
    }
    let values = numeric_vector(value, "Prior")?;
    if values.len() != 2 {
        return Err(invalid(
            "perfcurve: numeric Prior must contain positive and negative priors",
        ));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid(
            "perfcurve: Prior values must be finite and nonnegative",
        ));
    }
    let total = values[0] + values[1];
    if total <= 0.0 {
        return Err(invalid("perfcurve: Prior values must have positive sum"));
    }
    Ok(PriorOption::Values(values[0] / total, values[1] / total))
}

fn parse_cost(value: &Value) -> BuiltinResult<[[f64; 2]; 2]> {
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        _ => return Err(invalid("perfcurve: Cost must be a 2-by-2 numeric matrix")),
    };
    if tensor.shape.as_slice() != [2, 2] {
        return Err(invalid("perfcurve: Cost must be a 2-by-2 numeric matrix"));
    }
    if tensor
        .data
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid(
            "perfcurve: Cost entries must be finite and nonnegative",
        ));
    }
    Ok([
        [tensor.data[0], tensor.data[2]],
        [tensor.data[1], tensor.data[3]],
    ])
}

fn score_vector(value: Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            ensure_vector_shape(&tensor.shape, "scores")?;
            Ok(tensor.data)
        }
        Value::Num(value) => Ok(vec![value]),
        Value::Int(value) => Ok(vec![value.to_f64()]),
        other => Err(invalid(format!(
            "perfcurve: scores must be a numeric vector; got {other:?}"
        ))),
    }
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            ensure_vector_shape(&tensor.shape, name)?;
            Ok(tensor.data.clone())
        }
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => Ok(vec![value.to_f64()]),
        other => Err(invalid(format!(
            "perfcurve: {name} must be a numeric vector; got {other:?}"
        ))),
    }
}

fn scalar_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Int(value) => Ok(value.to_i64() != 0),
        _ => match canonical(&scalar_text(value, name)?).as_str() {
            "true" | "on" | "yes" => Ok(true),
            "false" | "off" | "no" => Ok(false),
            other => Err(invalid(format!(
                "perfcurve: {name} must be logical; got '{other}'"
            ))),
        },
    }
}

fn labels_from_value(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    match value {
        Value::Tensor(tensor) => Ok(LabelVector {
            labels: vector_values(&tensor, name)?
                .into_iter()
                .map(Label::Numeric)
                .collect(),
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Num(value) => Ok(LabelVector {
            labels: vec![Label::Numeric(value)],
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Int(value) => Ok(LabelVector {
            labels: vec![Label::Numeric(value.to_f64())],
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Bool(value) => Ok(LabelVector {
            labels: vec![Label::Logical(value)],
            kind: LabelKind::Logical,
            categories: None,
        }),
        Value::LogicalArray(array) => {
            ensure_vector_shape(&array.shape, name)?;
            Ok(LabelVector {
                labels: array
                    .data
                    .into_iter()
                    .map(|value| Label::Logical(value != 0))
                    .collect(),
                kind: LabelKind::Logical,
                categories: None,
            })
        }
        Value::String(text) => Ok(LabelVector {
            labels: vec![Label::Text(text)],
            kind: LabelKind::String,
            categories: None,
        }),
        Value::StringArray(array) => {
            ensure_vector_shape(&array.shape, name)?;
            Ok(LabelVector {
                labels: array.data.into_iter().map(Label::Text).collect(),
                kind: LabelKind::String,
                categories: None,
            })
        }
        Value::CharArray(array) => Ok(LabelVector {
            labels: char_rows(&array).into_iter().map(Label::Text).collect(),
            kind: LabelKind::Char,
            categories: None,
        }),
        Value::Cell(cell) => {
            ensure_vector_shape(&[cell.rows, cell.cols], name)?;
            let mut labels = Vec::with_capacity(cell.data.len());
            for item in cell.data {
                labels.push(Label::Text(scalar_text(&item, name)?));
            }
            Ok(LabelVector {
                labels,
                kind: LabelKind::Cell,
                categories: None,
            })
        }
        Value::Object(object) if object.is_class("categorical") => {
            ensure_categorical_vector_shape(&object, name)?;
            let labels = crate::builtins::table::categorical_labels(&Value::Object(object.clone()))?
                .into_iter()
                .map(Label::Text)
                .collect();
            Ok(LabelVector {
                labels,
                kind: LabelKind::Categorical,
                categories: categorical_categories(&object)?,
            })
        }
        other => Err(invalid(format!(
            "perfcurve: {name} must be numeric, logical, string, character, categorical, or cellstr labels; got {other:?}"
        ))),
    }
}

fn scalar_label_from_value(value: Value, name: &str) -> BuiltinResult<Label> {
    let vector = labels_from_value(value, name)?;
    if vector.labels.len() != 1 {
        return Err(invalid(format!("perfcurve: {name} must be a scalar label")));
    }
    Ok(vector.labels[0].clone())
}

fn coerce_labels_to_kind(
    labels: Vec<Label>,
    kind: LabelKind,
    name: &str,
) -> BuiltinResult<Vec<Label>> {
    labels
        .into_iter()
        .map(|label| coerce_label_to_kind(label, kind, name))
        .collect()
}

fn coerce_label_to_kind(label: Label, kind: LabelKind, name: &str) -> BuiltinResult<Label> {
    match (kind, label) {
        (LabelKind::Logical, Label::Logical(value)) => Ok(Label::Logical(value)),
        (LabelKind::Logical, Label::Text(value)) => match canonical(&value).as_str() {
            "true" | "logical1" | "1" => Ok(Label::Logical(true)),
            "false" | "logical0" | "0" => Ok(Label::Logical(false)),
            _ => Err(invalid(format!(
                "perfcurve: {name} must be true or false for logical labels"
            ))),
        },
        (LabelKind::Logical, Label::Numeric(value)) if value == 0.0 || value == 1.0 => {
            Ok(Label::Logical(value != 0.0))
        }
        (LabelKind::Numeric, Label::Numeric(value)) => Ok(Label::Numeric(value)),
        (LabelKind::Numeric, Label::Text(value)) => value
            .parse::<f64>()
            .map(Label::Numeric)
            .map_err(|_| invalid(format!("perfcurve: {name} must be numeric"))),
        (
            LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical,
            Label::Text(value),
        ) => Ok(Label::Text(value)),
        (
            LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical,
            Label::Numeric(value),
        ) => Ok(Label::Text(format_number(value))),
        (
            LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical,
            Label::Logical(value),
        ) => Ok(Label::Text(value.to_string())),
        _ => Err(invalid(format!(
            "perfcurve: {name} label type must match labels"
        ))),
    }
}

fn vector_values(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    ensure_vector_shape(&tensor.shape, name)?;
    Ok(tensor.data.clone())
}

fn ensure_vector_shape(shape: &[usize], name: &str) -> BuiltinResult<()> {
    if shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid(format!("perfcurve: {name} must be a vector")));
    }
    Ok(())
}

fn ensure_categorical_vector_shape(object: &ObjectInstance, name: &str) -> BuiltinResult<()> {
    match object.properties.get("Codes") {
        Some(Value::Tensor(tensor)) => ensure_vector_shape(&tensor.shape, name),
        Some(_) => Err(invalid("perfcurve: categorical Codes must be numeric")),
        None => Ok(()),
    }
}

fn categorical_categories(object: &ObjectInstance) -> BuiltinResult<Option<Vec<String>>> {
    match object.properties.get("Categories") {
        Some(Value::StringArray(array)) => Ok(Some(array.data.clone())),
        Some(_) => Err(invalid("perfcurve: categorical Categories must be strings")),
        None => Ok(None),
    }
}

fn negative_classes(
    labels: &LabelVector,
    posclass: &Label,
    explicit: Option<&Vec<Label>>,
) -> BuiltinResult<Vec<Label>> {
    let mut classes = if let Some(explicit) = explicit {
        explicit.clone()
    } else if labels.kind == LabelKind::Categorical {
        let mut out = Vec::new();
        for category in labels.categories.iter().flatten() {
            let label = Label::Text(category.clone());
            if !same_label(&label, posclass)
                && labels
                    .labels
                    .iter()
                    .any(|candidate| same_label(candidate, &label))
            {
                out.push(label);
            }
        }
        for label in &labels.labels {
            if !same_label(label, posclass)
                && !is_missing_label(label)
                && !out.iter().any(|existing| same_label(existing, label))
            {
                out.push(label.clone());
            }
        }
        out
    } else {
        let mut out = dedupe_labels(labels.labels.clone());
        sort_labels(&mut out, labels.kind);
        out.retain(|label| !same_label(label, posclass));
        out
    };
    if classes.iter().any(|label| same_label(label, posclass)) {
        return Err(invalid("perfcurve: NegClass cannot include posclass"));
    }
    classes.retain(|label| {
        !is_missing_label(label)
            && labels
                .labels
                .iter()
                .any(|candidate| same_label(candidate, label))
    });
    Ok(dedupe_labels(classes))
}

fn prepare_observations(
    labels: &LabelVector,
    scores: &[f64],
    posclass: &Label,
    neg_classes: &[Label],
    options: &Options,
) -> BuiltinResult<Vec<Observation>> {
    let weights = if let Some(weights) = &options.weights {
        if weights.len() != labels.labels.len() {
            return Err(invalid(
                "perfcurve: Weights must have one value per observation",
            ));
        }
        weights.clone()
    } else {
        vec![1.0; labels.labels.len()]
    };
    if weights
        .iter()
        .any(|value| value.is_nan() || *value < 0.0 || value.is_infinite())
    {
        return Err(invalid(
            "perfcurve: Weights must be finite nonnegative values",
        ));
    }
    let mut observations = Vec::with_capacity(labels.labels.len());
    for ((label, score), weight) in labels.labels.iter().zip(scores).zip(weights.iter()) {
        if is_missing_label(label) || *weight == 0.0 {
            continue;
        }
        if score.is_nan() && options.process_nan == ProcessNaN::Ignore {
            continue;
        }
        if score.is_infinite() {
            return Err(invalid("perfcurve: scores must be finite or NaN"));
        }
        let is_positive = same_label(label, posclass);
        let neg_index = if is_positive {
            None
        } else {
            neg_classes
                .iter()
                .position(|candidate| same_label(candidate, label))
        };
        if !is_positive && neg_index.is_none() {
            continue;
        }
        observations.push(Observation {
            score: *score,
            weight: *weight,
            is_positive,
            neg_index,
        });
    }
    apply_prior_weights(observations, &options.prior)
}

fn apply_prior_weights(
    mut observations: Vec<Observation>,
    prior: &PriorOption,
) -> BuiltinResult<Vec<Observation>> {
    let pos_total: f64 = observations
        .iter()
        .filter(|obs| obs.is_positive)
        .map(|obs| obs.weight)
        .sum();
    let neg_total: f64 = observations
        .iter()
        .filter(|obs| !obs.is_positive)
        .map(|obs| obs.weight)
        .sum();
    if pos_total <= 0.0 || neg_total <= 0.0 {
        return Ok(observations);
    }
    let (pos_prior, neg_prior) = match prior {
        PriorOption::Empirical => return Ok(observations),
        PriorOption::Uniform => (0.5, 0.5),
        PriorOption::Values(pos, neg) => (*pos, *neg),
    };
    if pos_prior <= 0.0 || neg_prior <= 0.0 {
        return Err(invalid(
            "perfcurve: positive and negative prior probabilities must be nonzero",
        ));
    }
    let pos_scale = pos_prior / pos_total;
    let neg_scale = neg_prior / neg_total;
    for obs in &mut observations {
        obs.weight *= if obs.is_positive {
            pos_scale
        } else {
            neg_scale
        };
    }
    Ok(observations)
}

fn thresholds_for_scores(
    observations: &[Observation],
    explicit: Option<&Vec<f64>>,
) -> BuiltinResult<Vec<ThresholdSpec>> {
    if let Some(values) = explicit {
        let mut out: Vec<ThresholdSpec> =
            values.iter().copied().map(ThresholdSpec::Value).collect();
        out.sort_by(|left, right| {
            threshold_output_value(right).total_cmp(&threshold_output_value(left))
        });
        return Ok(out);
    }
    let mut values: Vec<f64> = observations
        .iter()
        .filter_map(|obs| obs.score.is_finite().then_some(obs.score))
        .collect();
    values.sort_by(|left, right| right.total_cmp(left));
    values.dedup_by(|left, right| left.total_cmp(right) == Ordering::Equal);
    if values.is_empty() {
        return Err(invalid(
            "perfcurve: at least one finite score is required to form thresholds",
        ));
    }
    let mut out = Vec::with_capacity(values.len() + 1);
    out.push(ThresholdSpec::AboveAll(values[0]));
    out.extend(values.into_iter().map(ThresholdSpec::Value));
    Ok(out)
}

fn compute_curve(
    observations: &[Observation],
    thresholds: &[ThresholdSpec],
    xcrit: Criterion,
    ycrit: Criterion,
    cost: &[[f64; 2]; 2],
) -> BuiltinResult<Curve> {
    let mut x = Vec::with_capacity(thresholds.len());
    let mut y = Vec::with_capacity(thresholds.len());
    let mut t = Vec::with_capacity(thresholds.len());
    let counts = counts_for_thresholds(observations, thresholds, None);
    for (threshold, count) in thresholds.iter().zip(&counts) {
        x.push(criterion_value(*count, xcrit, cost));
        y.push(criterion_value(*count, ycrit, cost));
        t.push(threshold_output_value(threshold));
    }
    Ok(Curve {
        x,
        y,
        thresholds: t,
        counts,
        auc: 0.0,
    })
}

fn counts_for_thresholds(
    observations: &[Observation],
    thresholds: &[ThresholdSpec],
    neg_index: Option<usize>,
) -> Vec<Counts> {
    let mut current = Counts {
        tp: 0.0,
        fp: 0.0,
        tn: 0.0,
        fn_: 0.0,
    };
    let mut finite = Vec::new();
    for obs in observations {
        if !observation_matches_subclass(obs, neg_index) {
            continue;
        }
        if obs.score.is_nan() {
            if obs.is_positive {
                current.fn_ += obs.weight;
            } else {
                current.fp += obs.weight;
            }
        } else {
            if obs.is_positive {
                current.fn_ += obs.weight;
            } else {
                current.tn += obs.weight;
            }
            finite.push(obs);
        }
    }
    finite.sort_by(|left, right| right.score.total_cmp(&left.score));

    let mut out = Vec::with_capacity(thresholds.len());
    let mut cursor = 0usize;
    for threshold in thresholds {
        match threshold {
            ThresholdSpec::AboveAll(_) => {}
            ThresholdSpec::Value(value) => {
                while cursor < finite.len() && finite[cursor].score >= *value {
                    let obs = finite[cursor];
                    if obs.is_positive {
                        current.fn_ -= obs.weight;
                        current.tp += obs.weight;
                    } else {
                        current.tn -= obs.weight;
                        current.fp += obs.weight;
                    }
                    cursor += 1;
                }
            }
        }
        out.push(current);
    }
    out
}

fn observation_matches_subclass(obs: &Observation, neg_index: Option<usize>) -> bool {
    obs.is_positive || neg_index.is_none() || obs.neg_index == neg_index
}

fn criterion_value(counts: Counts, criterion: Criterion, cost: &[[f64; 2]; 2]) -> f64 {
    let pos = counts.tp + counts.fn_;
    let neg = counts.fp + counts.tn;
    match criterion {
        Criterion::Fpr => ratio(counts.fp, neg),
        Criterion::Tpr => ratio(counts.tp, pos),
        Criterion::Tnr => ratio(counts.tn, neg),
        Criterion::Fnr => ratio(counts.fn_, pos),
        Criterion::Precision => ratio(counts.tp, counts.tp + counts.fp),
        Criterion::Npv => ratio(counts.tn, counts.tn + counts.fn_),
        Criterion::Accuracy => ratio(counts.tp + counts.tn, counts.total()),
        Criterion::TruePositive => counts.tp,
        Criterion::FalseNegative => counts.fn_,
        Criterion::FalsePositive => counts.fp,
        Criterion::TrueNegative => counts.tn,
        Criterion::PredictedPositive => counts.tp + counts.fp,
        Criterion::RatePredictedPositive => ratio(counts.tp + counts.fp, counts.total()),
        Criterion::RatePredictedNegative => ratio(counts.tn + counts.fn_, counts.total()),
        Criterion::ExpectedCost => ratio(
            (cost[0][0] * counts.tn)
                + (cost[0][1] * counts.fp)
                + (cost[1][0] * counts.fn_)
                + (cost[1][1] * counts.tp),
            counts.total(),
        ),
    }
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::NAN
    }
}

fn threshold_output_value(threshold: &ThresholdSpec) -> f64 {
    match threshold {
        ThresholdSpec::AboveAll(value) => *value,
        ThresholdSpec::Value(value) => *value,
    }
}

fn auc(x: &[f64], y: &[f64]) -> f64 {
    let mut points: Vec<(f64, f64)> = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    points.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.total_cmp(&right.1)));
    let mut area = 0.0;
    for pair in points.windows(2) {
        let dx = pair[1].0 - pair[0].0;
        if dx >= 0.0 {
            area += dx * (pair[0].1 + pair[1].1) / 2.0;
        }
    }
    area
}

fn auc_interval(x: &[f64], y: &[f64], xvals: &[f64]) -> f64 {
    let lo = xvals.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = xvals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut points: Vec<(f64, f64)> = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite() && *x >= lo && *x <= hi)
        .collect();
    if !points.iter().any(|(x, _)| (*x - lo).abs() <= f64::EPSILON) {
        let (_, yi, _, _) = interpolate_from_xy(x, y, lo);
        if yi.is_finite() {
            points.push((lo, yi));
        }
    }
    if !points.iter().any(|(x, _)| (*x - hi).abs() <= f64::EPSILON) {
        let (_, yi, _, _) = interpolate_from_xy(x, y, hi);
        if yi.is_finite() {
            points.push((hi, yi));
        }
    }
    points.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.total_cmp(&right.1)));
    let (xs, ys): (Vec<_>, Vec<_>) = points.into_iter().unzip();
    auc(&xs, &ys)
}

fn optimal_point(curve: &Curve, options: &Options) -> (f64, f64) {
    if options.xcrit != Criterion::Fpr || options.ycrit != Criterion::Tpr {
        return (f64::NAN, f64::NAN);
    }
    let slope = roc_slope(options);
    let mut best = (f64::NAN, f64::NAN);
    let mut best_score = f64::NEG_INFINITY;
    for ((x, y), counts) in curve.x.iter().zip(&curve.y).zip(&curve.counts) {
        let score = if slope.is_finite() {
            *y - slope * *x
        } else {
            -((criterion_value(*counts, Criterion::Fpr, &options.cost).powi(2))
                + (1.0 - criterion_value(*counts, Criterion::Tpr, &options.cost)).powi(2))
        };
        if score.is_finite() && score > best_score {
            best_score = score;
            best = (*x, *y);
        }
    }
    best
}

fn roc_slope(options: &Options) -> f64 {
    let (pos_prior, neg_prior) = match options.prior {
        PriorOption::Empirical => (0.5, 0.5),
        PriorOption::Uniform => (0.5, 0.5),
        PriorOption::Values(pos, neg) => (pos, neg),
    };
    let false_positive_cost = options.cost[0][1] - options.cost[0][0];
    let false_negative_cost = options.cost[1][0] - options.cost[1][1];
    if false_positive_cost <= 0.0 || false_negative_cost <= 0.0 || pos_prior <= 0.0 {
        f64::NAN
    } else {
        (false_positive_cost * neg_prior) / (false_negative_cost * pos_prior)
    }
}

fn apply_xvals(curve: &Curve, xvals: &[f64], use_nearest: bool) -> BuiltinResult<Curve> {
    let mut x = Vec::with_capacity(xvals.len());
    let mut y = Vec::with_capacity(xvals.len());
    let mut t = Vec::with_capacity(xvals.len());
    let mut counts = Vec::with_capacity(xvals.len());
    for target in xvals {
        let (xi, yi, ti, ci) = if use_nearest {
            nearest_at_x(curve, *target)
        } else {
            interpolate_at_x(curve, *target)
        };
        x.push(xi);
        y.push(yi);
        t.push(ti);
        counts.push(ci);
    }
    Ok(Curve {
        x,
        y,
        thresholds: t,
        counts,
        auc: 0.0,
    })
}

fn nearest_at_x(curve: &Curve, target: f64) -> (f64, f64, f64, Counts) {
    let mut best = 0usize;
    let mut best_dist = f64::INFINITY;
    for (idx, x) in curve.x.iter().enumerate() {
        let dist = (*x - target).abs();
        if dist < best_dist {
            best_dist = dist;
            best = idx;
        }
    }
    (
        curve.x[best],
        curve.y[best],
        curve.thresholds[best],
        curve.counts[best],
    )
}

fn interpolate_at_x(curve: &Curve, target: f64) -> (f64, f64, f64, Counts) {
    interpolate_from_points(
        curve
            .x
            .iter()
            .copied()
            .zip(curve.y.iter().copied())
            .zip(curve.thresholds.iter().copied())
            .zip(curve.counts.iter().copied())
            .map(|(((x, y), t), c)| (x, y, t, c))
            .collect(),
        target,
        curve.counts[0],
    )
}

fn interpolate_from_xy(x: &[f64], y: &[f64], target: f64) -> (f64, f64, f64, Counts) {
    interpolate_from_points(
        x.iter()
            .copied()
            .zip(y.iter().copied())
            .map(|(x, y)| {
                (
                    x,
                    y,
                    f64::NAN,
                    Counts {
                        tp: 0.0,
                        fp: 0.0,
                        tn: 0.0,
                        fn_: 0.0,
                    },
                )
            })
            .collect(),
        target,
        Counts {
            tp: 0.0,
            fp: 0.0,
            tn: 0.0,
            fn_: 0.0,
        },
    )
}

fn interpolate_from_points(
    raw_points: Vec<(f64, f64, f64, Counts)>,
    target: f64,
    fallback_counts: Counts,
) -> (f64, f64, f64, Counts) {
    let mut points: Vec<(f64, f64, f64, Counts)> = raw_points
        .into_iter()
        .filter(|(x, y, _, _)| x.is_finite() && y.is_finite())
        .collect();
    points.sort_by(|left, right| left.0.total_cmp(&right.0));
    if points.is_empty() {
        return (target, f64::NAN, f64::NAN, fallback_counts);
    }
    if target <= points[0].0 {
        return (target, points[0].1, points[0].2, points[0].3);
    }
    for pair in points.windows(2) {
        let (x0, y0, t0, c0) = pair[0];
        let (x1, y1, t1, c1) = pair[1];
        if target <= x1 {
            if (x1 - x0).abs() <= f64::EPSILON {
                return (target, y1, t1, c1);
            }
            let alpha = (target - x0) / (x1 - x0);
            return (target, y0 + alpha * (y1 - y0), t0 + alpha * (t1 - t0), c0);
        }
    }
    let last = points[points.len() - 1];
    (target, last.1, last.2, last.3)
}

fn suby_output(
    observations: &[Observation],
    thresholds: &[ThresholdSpec],
    neg_classes: &[Label],
    xcrit: Criterion,
    ycrit: Criterion,
    cost: &[[f64; 2]; 2],
    xvals: Option<&Vec<f64>>,
    use_nearest: bool,
) -> BuiltinResult<Value> {
    let rows = xvals.map_or(thresholds.len(), Vec::len);
    let cols = neg_classes.len();
    let mut data = vec![0.0; rows * cols];
    for col in 0..cols {
        let mut x = Vec::with_capacity(thresholds.len());
        let mut y = Vec::with_capacity(thresholds.len());
        let counts = counts_for_thresholds(observations, thresholds, Some(col));
        for count in &counts {
            x.push(criterion_value(*count, xcrit, cost));
            y.push(criterion_value(*count, ycrit, cost));
        }
        let col_values = if let Some(xvals) = xvals {
            let curve = Curve {
                x,
                y,
                thresholds: thresholds.iter().map(threshold_output_value).collect(),
                counts,
                auc: 0.0,
            };
            xvals
                .iter()
                .map(|target| {
                    if use_nearest {
                        nearest_at_x(&curve, *target).1
                    } else {
                        interpolate_at_x(&curve, *target).1
                    }
                })
                .collect::<Vec<_>>()
        } else {
            y
        };
        for (row, value) in col_values.into_iter().enumerate() {
            data[row + col * rows] = value;
        }
    }
    Ok(Value::Tensor(
        Tensor::new(data, vec![rows, cols]).map_err(|err| internal(format!("perfcurve: {err}")))?,
    ))
}

fn dedupe_labels<I>(labels: I) -> Vec<Label>
where
    I: IntoIterator<Item = Label>,
{
    let mut out = Vec::new();
    for label in labels {
        if is_missing_label(&label) {
            continue;
        }
        if !out.iter().any(|existing| same_label(existing, &label)) {
            out.push(label);
        }
    }
    out
}

fn sort_labels(labels: &mut [Label], kind: LabelKind) {
    match kind {
        LabelKind::Numeric => labels.sort_by(|left, right| match (left, right) {
            (Label::Numeric(a), Label::Numeric(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            _ => Ordering::Equal,
        }),
        LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical => labels
            .sort_by(|left, right| match (left, right) {
                (Label::Text(a), Label::Text(b)) => a.cmp(b),
                _ => Ordering::Equal,
            }),
        LabelKind::Logical => labels.sort_by_key(|label| match label {
            Label::Logical(value) => *value,
            _ => false,
        }),
    }
}

fn label_key(label: &Label) -> String {
    match label {
        Label::Numeric(value) if *value == 0.0 => "n:0".to_string(),
        Label::Numeric(value) => format!("n:{:016x}", value.to_bits()),
        Label::Text(value) => format!("t:{value}"),
        Label::Logical(value) => format!("l:{}", usize::from(*value)),
    }
}

fn same_label(left: &Label, right: &Label) -> bool {
    label_key(left) == label_key(right)
        || matches!(
            (left, right),
            (Label::Numeric(a), Label::Numeric(b)) if a.is_nan() && b.is_nan()
        )
}

fn is_missing_label(label: &Label) -> bool {
    match label {
        Label::Numeric(value) => value.is_nan(),
        Label::Text(value) => value.is_empty() || value == "<missing>",
        Label::Logical(_) => false,
    }
}

fn labels_to_cell_value(labels: &[Label]) -> BuiltinResult<Value> {
    Ok(Value::Cell(
        CellArray::new(
            labels
                .iter()
                .map(|label| Value::CharArray(CharArray::new_row(&label_display(label))))
                .collect(),
            labels.len(),
            1,
        )
        .map_err(|err| internal(format!("perfcurve: {err}")))?,
    ))
}

fn label_display(label: &Label) -> String {
    match label {
        Label::Numeric(value) => format_number(*value),
        Label::Text(value) => value.clone(),
        Label::Logical(value) => value.to_string(),
    }
}

fn char_rows(array: &CharArray) -> Vec<String> {
    (0..array.rows)
        .map(|row| {
            let mut text = String::with_capacity(array.cols);
            for col in 0..array.cols {
                text.push(array.data[row + col * array.rows]);
            }
            text.trim_end().to_string()
        })
        .collect()
}

fn scalar_text(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect::<String>()),
        Value::Num(value) if value.is_finite() => Ok(format_number(*value)),
        Value::Int(value) => Ok(value.decimal_string()),
        Value::Bool(value) => Ok(value.to_string()),
        other => Err(invalid(format!(
            "perfcurve: {name} must be text scalar; got {other:?}"
        ))),
    }
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn canonical(value: &str) -> String {
    value
        .chars()
        .filter(|ch| *ch != '_' && !ch.is_ascii_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn tensor_column(data: Vec<f64>) -> BuiltinResult<Value> {
    let rows = data.len();
    Ok(Value::Tensor(
        Tensor::new(data, vec![rows, 1]).map_err(|err| internal(format!("perfcurve: {err}")))?,
    ))
}

fn tensor_row(data: Vec<f64>) -> BuiltinResult<Value> {
    let cols = data.len();
    Ok(Value::Tensor(
        Tensor::new(data, vec![1, cols]).map_err(|err| internal(format!("perfcurve: {err}")))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::StringArray;

    #[test]
    fn perfcurve_scalar_text_preserves_exact_uint64() {
        assert_eq!(
            scalar_text(
                &Value::Int(runmat_builtins::IntValue::U64(u64::MAX)),
                "option"
            )
            .expect("scalar text"),
            "18446744073709551615"
        );
    }

    fn tensor(data: &[f64], shape: &[usize]) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), shape.to_vec()).unwrap())
    }

    fn output_tensor(value: &Value, index: usize) -> &Tensor {
        match value {
            Value::OutputList(values) => match &values[index] {
                Value::Tensor(tensor) => tensor,
                other => panic!("expected tensor at {index}, got {other:?}"),
            },
            other => panic!("expected output list, got {other:?}"),
        }
    }

    fn output_num(value: &Value, index: usize) -> f64 {
        match value {
            Value::OutputList(values) => match &values[index] {
                Value::Num(value) => *value,
                other => panic!("expected num at {index}, got {other:?}"),
            },
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn default_roc_returns_thresholds_auc_and_optimal_point() {
        let _guard = output_count::push_output_count(Some(5));
        let result = perfcurve_builtin(
            tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1]),
            tensor(&[0.9, 0.8, 0.4, 0.1], &[4, 1]),
            Value::Num(1.0),
            vec![],
        )
        .await
        .unwrap();
        let x = output_tensor(&result, 0);
        let y = output_tensor(&result, 1);
        let thresholds = output_tensor(&result, 2);
        assert_eq!(x.shape, vec![5, 1]);
        assert_eq!(y.shape, vec![5, 1]);
        assert_eq!(thresholds.shape, vec![5, 1]);
        assert!((output_num(&result, 3) - 0.75).abs() < 1.0e-12);
        assert_eq!(x.data[0], 0.0);
        assert_eq!(y.data[0], 0.0);
        assert_eq!(thresholds.data[0], thresholds.data[1]);
        assert_eq!(x.data[4], 1.0);
        assert_eq!(y.data[4], 1.0);
        let opt = output_tensor(&result, 4);
        assert_eq!(opt.shape, vec![1, 2]);
    }

    #[tokio::test]
    async fn supports_text_labels_negclass_weights_and_precision_recall() {
        let _guard = output_count::push_output_count(Some(7));
        let labels = Value::StringArray(
            StringArray::new(
                vec![
                    "hit".into(),
                    "miss".into(),
                    "hit".into(),
                    "other".into(),
                    "miss".into(),
                ],
                vec![5, 1],
            )
            .unwrap(),
        );
        let result = perfcurve_builtin(
            labels,
            tensor(&[0.9, 0.8, 0.4, 0.2, f64::NAN], &[5, 1]),
            Value::String("hit".into()),
            vec![
                Value::String("NegClass".into()),
                Value::String("miss".into()),
                Value::String("Weights".into()),
                tensor(&[1.0, 2.0, 1.0, 10.0, 2.0], &[5, 1]),
                Value::String("XCrit".into()),
                Value::String("reca".into()),
                Value::String("YCrit".into()),
                Value::String("prec".into()),
            ],
        )
        .await
        .unwrap();
        let x = output_tensor(&result, 0);
        let y = output_tensor(&result, 1);
        let suby = output_tensor(&result, 5);
        assert_eq!(x.shape, vec![4, 1]);
        assert_eq!(suby.shape, vec![4, 1]);
        assert!(y.data.iter().any(|value| (*value - 0.5).abs() < 1.0e-12));
        match &result {
            Value::OutputList(values) => match &values[6] {
                Value::Cell(names) => assert_eq!(
                    names.data,
                    vec![Value::CharArray(CharArray::new_row("miss"))]
                ),
                other => panic!("expected cell names, got {other:?}"),
            },
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn xvals_nearest_returns_actual_points_and_interpolation_sorts_targets() {
        let _guard = output_count::push_output_count(Some(4));
        let result = perfcurve_builtin(
            tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1]),
            tensor(&[0.9, 0.8, 0.4, 0.1], &[4, 1]),
            Value::Num(1.0),
            vec![Value::String("XVals".into()), tensor(&[0.6, 0.0], &[1, 2])],
        )
        .await
        .unwrap();
        let x = output_tensor(&result, 0);
        assert_eq!(x.shape, vec![2, 1]);
        assert_eq!(x.data, vec![0.5, 0.0]);
        assert!(output_num(&result, 3).is_finite());

        let result = perfcurve_builtin(
            tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1]),
            tensor(&[0.9, 0.8, 0.4, 0.1], &[4, 1]),
            Value::Num(1.0),
            vec![
                Value::String("XVals".into()),
                tensor(&[0.5, 0.0], &[1, 2]),
                Value::String("UseNearest".into()),
                Value::Bool(false),
            ],
        )
        .await
        .unwrap();
        assert_eq!(output_tensor(&result, 0).data, vec![0.0, 0.5]);
    }

    #[tokio::test]
    async fn rejects_missing_classes_and_bad_option_shapes() {
        let err = perfcurve_builtin(
            tensor(&[1.0, 1.0], &[2, 1]),
            tensor(&[0.2, 0.3], &[2, 1]),
            Value::Num(2.0),
            vec![],
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("posclass"));

        let err = perfcurve_builtin(
            tensor(&[1.0, 0.0], &[2, 1]),
            tensor(&[0.2, 0.3], &[2, 1]),
            Value::Num(1.0),
            vec![Value::String("Weights".into()), tensor(&[1.0], &[1, 1])],
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("Weights"));

        let err = perfcurve_builtin(
            tensor(&[1.0, 0.0], &[2, 1]),
            tensor(&[0.2, 0.3], &[2, 1]),
            Value::Num(1.0),
            vec![
                Value::String("TVals".into()),
                tensor(&[0.2], &[1, 1]),
                Value::String("XVals".into()),
                tensor(&[0.0], &[1, 1]),
            ],
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("TVals and XVals"));
    }

    #[tokio::test]
    async fn handles_nan_addtofalse_logical_aliases_and_non_roc_optimal_point() {
        let _guard = output_count::push_output_count(Some(5));
        let result = perfcurve_builtin(
            Value::LogicalArray(
                runmat_builtins::LogicalArray::new(vec![1, 0, 1, 0], vec![4, 1]).unwrap(),
            ),
            tensor(&[0.9, f64::NAN, f64::NAN, 0.1], &[4, 1]),
            Value::String("true".into()),
            vec![
                Value::String("ProcessNaN".into()),
                Value::String("addtofalse".into()),
                Value::String("YCrit".into()),
                Value::String("ecost".into()),
            ],
        )
        .await
        .unwrap();
        let y = output_tensor(&result, 1);
        assert!(y.data.iter().any(|value| (*value - 0.5).abs() < 1.0e-12));
        let opt = output_tensor(&result, 4);
        assert!(opt.data.iter().all(|value| value.is_nan()));
    }
}
