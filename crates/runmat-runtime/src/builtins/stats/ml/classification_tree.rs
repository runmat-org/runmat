//! Numeric classification tree compatibility surface.

use std::cmp::Ordering;

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
use runmat_value::{
    CharArray, IntValue, IntegerStorage, LogicalArray, ObjectInstance, StringArray, StructValue,
    Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::builtins::table::{
    is_tabular_object, table_variable_names_from_object, table_variables,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

pub(crate) const CLASSIFICATION_TREE_CLASS: &str = "ClassificationTree";

const FITCTREE_NAME: &str = "fitctree";
const PREDICT_NAME: &str = "predict";
const EPS: f64 = 1.0e-12;
const MAX_TREE_DEPTH: usize = 4096;

const INTEGER_PREDICTOR_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fitctree-integer-predictors",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fitctree with typed-integer predictors is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FitctreeIntegerPredictorExtension"),
};
const INTEGER_NUMERIC_ARGUMENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fitctree-integer-numeric-arguments",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fitctree with typed-integer weights or numeric controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FitctreeIntegerNumericArgumentExtension"),
};
const RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fitctree-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fitctree host fitting from provider-resident input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FitctreeResidentInputExtension"),
};
pub const FITCTREE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    INTEGER_PREDICTOR_EXTENSION,
    INTEGER_NUMERIC_ARGUMENT_EXTENSION,
    RESIDENT_INPUT_EXTENSION,
];

const INTEGER_PREDICTOR_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "X", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Integer predictors must be exactly representable in binary64 before host fitting; values that would round are rejected." }];
const INTEGER_LABEL_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "Y or ClassNames", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Grouping labels retain their exact integer class and identity through model storage and prediction, including above flintmax." }];
const INTEGER_NUMERIC_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "Weights or numeric option", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Typed-integer statistical inputs are independently gated and must cross the binary64 boundary exactly." }];
pub const FITCTREE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "Mdl = fitctree(integer_X, Y, ___)", inputs: &INTEGER_PREDICTOR_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Matrix and table predictor roles share the same exact conversion boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "Mdl = fitctree(X, integer_Y, ClassNames=integer_names)", inputs: &INTEGER_LABEL_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Class grouping never passes through floating point." },
    BuiltinIntegerCapabilityDescriptor { form: "Mdl = fitctree(X, Y, Weights=integer_W, numericOption=integer_value)", inputs: &INTEGER_NUMERIC_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer weights and numeric controls are RunMat-only extensions and reject rather than round." },
];

const OUTPUT_TREE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Mdl",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "ClassificationTree object containing class names, split rules, and posterior probabilities.",
}];

const PARAM_TBL_OR_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tblOrX",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input table or numeric predictor matrix.",
};

const PARAM_Y_OR_RESPONSE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "yOrResponse",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Response vector, response variable name, or table formula.",
};

const PARAM_REST: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as ClassNames, PredictorNames, ResponseName, MaxNumSplits, MinLeafSize, MinParentSize, SplitCriterion, and Weights.",
};

const FITCTREE_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_TBL_OR_X, PARAM_Y_OR_RESPONSE];
const FITCTREE_INPUTS_FULL: [BuiltinParamDescriptor; 3] =
    [PARAM_TBL_OR_X, PARAM_Y_OR_RESPONSE, PARAM_REST];

const FITCTREE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Mdl = fitctree(Tbl, ResponseVarName)",
        inputs: &FITCTREE_INPUTS_X_Y,
        outputs: &OUTPUT_TREE,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitctree(Tbl, formula)",
        inputs: &FITCTREE_INPUTS_X_Y,
        outputs: &OUTPUT_TREE,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitctree(X, Y)",
        inputs: &FITCTREE_INPUTS_X_Y,
        outputs: &OUTPUT_TREE,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitctree(___, Name, Value)",
        inputs: &FITCTREE_INPUTS_FULL,
        outputs: &OUTPUT_TREE,
    },
];

const ERROR_FITCTREE_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITCTREE.INVALID_ARGUMENT",
    identifier: Some("RunMat:fitctree:InvalidArgument"),
    when:
        "Inputs, response labels, dimensions, or name-value options are malformed or unsupported.",
    message: "fitctree: invalid argument",
};

const ERROR_FITCTREE_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITCTREE.INTERNAL",
    identifier: Some("RunMat:fitctree:Internal"),
    when: "RunMat cannot construct the ClassificationTree result.",
    message: "fitctree: internal error",
};

const ERROR_PREDICT_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PREDICT.INVALID_ARGUMENT",
    identifier: Some("RunMat:predict:InvalidArgument"),
    when: "Classification-tree prediction inputs are malformed or incompatible with the model.",
    message: "predict: invalid argument",
};

const ERROR_PREDICT_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PREDICT.INTERNAL",
    identifier: Some("RunMat:predict:Internal"),
    when: "Classification-tree prediction cannot construct its output.",
    message: "predict: internal error",
};

const FITCTREE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ERROR_FITCTREE_INVALID_ARGUMENT, ERROR_FITCTREE_INTERNAL];

pub const FITCTREE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FITCTREE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FITCTREE_ERRORS,
};

fn fitctree_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn fitctree_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(FITCTREE_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fitctree_invalid(message: impl Into<String>) -> RuntimeError {
    fitctree_error(message, &ERROR_FITCTREE_INVALID_ARGUMENT)
}

fn fitctree_internal(message: impl Into<String>) -> RuntimeError {
    fitctree_error(message, &ERROR_FITCTREE_INTERNAL)
}

fn predict_invalid(message: impl Into<String>) -> RuntimeError {
    predict_error(message, &ERROR_PREDICT_INVALID_ARGUMENT)
}

fn predict_internal(message: impl Into<String>) -> RuntimeError {
    predict_error(message, &ERROR_PREDICT_INTERNAL)
}

fn predict_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(PREDICT_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone, Debug)]
struct FitOptions {
    class_names: Option<Vec<ClassLabel>>,
    predictor_names: Option<Vec<String>>,
    response_name: Option<String>,
    max_num_splits: Option<usize>,
    min_leaf_size: usize,
    min_parent_size: usize,
    split_criterion: SplitCriterion,
    weights: Option<WeightSpec>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            class_names: None,
            predictor_names: None,
            response_name: None,
            max_num_splits: None,
            min_leaf_size: 1,
            min_parent_size: 10,
            split_criterion: SplitCriterion::Gdi,
            weights: None,
        }
    }
}

#[derive(Clone, Debug)]
enum WeightSpec {
    Values(Vec<f64>),
    Variable(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SplitCriterion {
    Gdi,
    Deviance,
}

#[derive(Clone, Debug, PartialEq)]
enum ClassLabel {
    Numeric(f64),
    Integer(IntValue),
    Text(String),
    Logical(bool),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LabelKind {
    Numeric,
    Integer(IntegerClass),
    Text,
    Logical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IntegerClass {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

#[derive(Clone, Debug)]
struct LabelVector {
    labels: Vec<ClassLabel>,
    kind: LabelKind,
}

#[derive(Clone, Debug)]
struct FitSpec {
    x: Tensor,
    class_indices: Vec<usize>,
    class_names: Vec<ClassLabel>,
    class_kind: LabelKind,
    predictor_names: Vec<String>,
    response_name: String,
    weights: Vec<f64>,
    options: FitOptions,
}

#[derive(Clone, Debug)]
struct Node {
    left: Option<usize>,
    right: Option<usize>,
    predictor: Option<usize>,
    threshold: f64,
    class_index: usize,
    probabilities: Vec<f64>,
}

#[derive(Clone, Debug)]
struct TreeFit {
    nodes: Vec<Node>,
    num_splits: usize,
}

#[derive(Clone, Debug)]
struct SplitCandidate {
    predictor: usize,
    threshold: f64,
    gain: f64,
    left_rows: Vec<usize>,
    right_rows: Vec<usize>,
}

#[runtime_builtin(
    name = "fitctree",
    category = "stats/ml",
    summary = "Fit a binary decision tree for multiclass classification.",
    keywords = "fitctree,classification tree,decision tree,machine learning,statistics",
    type_resolver(fitctree_type),
    descriptor(crate::builtins::stats::ml::classification_tree::FITCTREE_DESCRIPTOR),
    extensions(crate::builtins::stats::ml::classification_tree::FITCTREE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::stats::ml::classification_tree::FITCTREE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::stats::ml::classification_tree"
)]
async fn fitctree_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_fitctree_extensions(&first, &rest)?;
    let first = gather(first)
        .await
        .map_err(|err| fitctree_invalid(err.message))?;
    let rest = gather_all(rest)
        .await
        .map_err(|err| fitctree_invalid(err.message))?;
    let spec = build_fit_spec(first, rest)?;
    let fit = fit_tree(&spec)?;
    model_object(spec, fit).map(Value::Object)
}

fn enable_extension(extension: &BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, FITCTREE_NAME)
}

fn typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn ensure_fitctree_extensions(first: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if crate::dispatcher::value_contains_gpu(first)
        || rest.iter().any(crate::dispatcher::value_contains_gpu)
    {
        enable_extension(&RESIDENT_INPUT_EXTENSION)?;
    }
    let option_start = if is_table_value(first) {
        ensure_table_integer_extensions(first, rest)?
    } else {
        if typed_integer_value(first) {
            enable_extension(&INTEGER_PREDICTOR_EXTENSION)?;
        }
        1
    };
    for pair in rest[option_start.min(rest.len())..].chunks_exact(2) {
        let Some(name) = scalar_text(&pair[0]) else {
            continue;
        };
        if typed_integer_value(&pair[1]) && !name.eq_ignore_ascii_case("ClassNames") {
            enable_extension(&INTEGER_NUMERIC_ARGUMENT_EXTENSION)?;
        }
    }
    Ok(())
}

fn ensure_table_integer_extensions(first: &Value, rest: &[Value]) -> BuiltinResult<usize> {
    let Value::Object(object) = first else {
        return Ok(0);
    };
    let names = table_variable_names_from_object(object)
        .map_err(|err| fitctree_invalid(format!("fitctree: {err}")))?;
    let variables =
        table_variables(object).map_err(|err| fitctree_invalid(format!("fitctree: {err}")))?;
    let mut response = names.last().cloned().unwrap_or_default();
    let mut predictors: Option<Vec<String>> = None;
    let mut external_y = false;
    let mut option_start = 0;
    if let Some(arg) = rest.first().filter(|arg| !is_name_value_option(arg)) {
        option_start = 1;
        if let Some(text) = scalar_text(arg) {
            if let Some((lhs, rhs)) = text.split_once('~') {
                response = lhs.trim().to_string();
                predictors = Some(
                    rhs.split('+')
                        .map(str::trim)
                        .filter(|term| !term.is_empty() && *term != "1")
                        .map(str::to_string)
                        .collect(),
                );
            } else {
                response = text;
            }
        } else {
            external_y = true;
        }
    }
    let mut weight_variable = None;
    for pair in rest[option_start..].chunks_exact(2) {
        let Some(name) = scalar_text(&pair[0]) else {
            continue;
        };
        match name.to_ascii_lowercase().as_str() {
            "responsename" if !external_y => {
                if let Some(value) = scalar_text(&pair[1]) {
                    response = value;
                }
            }
            "predictornames" => predictors = text_list(&pair[1], "PredictorNames").ok(),
            "weights" => weight_variable = scalar_text(&pair[1]),
            _ => {}
        }
    }
    let predictors = predictors.unwrap_or_else(|| {
        names
            .iter()
            .filter(|name| {
                (external_y || **name != response)
                    && weight_variable.as_deref() != Some(name.as_str())
            })
            .cloned()
            .collect()
    });
    if predictors
        .iter()
        .any(|name| variables.fields.get(name).is_some_and(typed_integer_value))
    {
        enable_extension(&INTEGER_PREDICTOR_EXTENSION)?;
    }
    if weight_variable
        .as_ref()
        .is_some_and(|name| variables.fields.get(name).is_some_and(typed_integer_value))
    {
        enable_extension(&INTEGER_NUMERIC_ARGUMENT_EXTENSION)?;
    }
    Ok(option_start)
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| fitctree_invalid(format!("fitctree: {err}")))
}

async fn gather_all(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(value).await?);
    }
    Ok(out)
}

fn build_fit_spec(first: Value, mut rest: Vec<Value>) -> BuiltinResult<FitSpec> {
    if is_table_value(&first) {
        build_table_fit_spec(first, rest)
    } else {
        if rest.is_empty() {
            return Err(fitctree_invalid(
                "fitctree: matrix input requires response vector Y",
            ));
        }
        let y = rest.remove(0);
        build_matrix_fit_spec(first, y, rest)
    }
}

fn is_table_value(value: &Value) -> bool {
    matches!(value, Value::Object(object) if is_tabular_object(object))
}

fn build_matrix_fit_spec(
    x_value: Value,
    y_value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<FitSpec> {
    let x = tensor::value_into_tensor_for(FITCTREE_NAME, x_value)
        .map_err(|err| fitctree_invalid(format!("fitctree: {err}")))?;
    ensure_integer_tensor_exact(&x, "predictors")?;
    if x.shape.len() > 2 {
        return Err(fitctree_invalid("fitctree: X must be a 2-D numeric matrix"));
    }
    let labels = label_vector(y_value, "Y")?;
    if labels.labels.len() != x.rows {
        return Err(fitctree_invalid(
            "fitctree: length(Y) must match the number of rows in X",
        ));
    }
    let mut options = FitOptions::default();
    parse_fit_options(&rest, &mut options)?;
    if let Some(WeightSpec::Variable(_)) = options.weights {
        return Err(fitctree_invalid(
            "fitctree: weights variable names require table input",
        ));
    }
    let predictor_names = if let Some(names) = options.predictor_names.clone() {
        if names.len() != x.cols {
            return Err(fitctree_invalid(format!(
                "fitctree: PredictorNames must contain {} names",
                x.cols
            )));
        }
        names
    } else {
        (1..=x.cols).map(|idx| format!("x{idx}")).collect()
    };
    let response_name = options
        .response_name
        .clone()
        .unwrap_or_else(|| "Y".to_string());
    let weights = match &options.weights {
        Some(WeightSpec::Values(values)) => values.clone(),
        Some(WeightSpec::Variable(_)) => unreachable!("checked above"),
        None => vec![1.0; x.rows],
    };
    finalize_fit_spec(x, labels, predictor_names, response_name, weights, options)
}

fn build_table_fit_spec(first: Value, rest: Vec<Value>) -> BuiltinResult<FitSpec> {
    let object = match first {
        Value::Object(object) => object,
        _ => unreachable!("table checked"),
    };
    let variable_names = table_variable_names_from_object(&object)
        .map_err(|err| fitctree_invalid(format!("fitctree: {err}")))?;
    if variable_names.len() < 2 {
        return Err(fitctree_invalid(
            "fitctree: table input must contain at least one predictor and one response",
        ));
    }
    let variables =
        table_variables(&object).map_err(|err| fitctree_invalid(format!("fitctree: {err}")))?;
    let mut options = FitOptions::default();
    let mut rest_start = 0usize;
    let mut response_name = variable_names
        .last()
        .cloned()
        .ok_or_else(|| fitctree_invalid("fitctree: response variable missing"))?;
    let mut external_y = None;
    let mut formula_predictors = None;
    if let Some(first_arg) = rest.first() {
        if !is_name_value_option(first_arg) {
            if let Some(text) = scalar_text(first_arg) {
                if text.contains('~') {
                    let parsed = parse_formula(&text, &variable_names)?;
                    response_name = parsed.0;
                    formula_predictors = Some(parsed.1);
                } else {
                    response_name = text;
                }
            } else {
                external_y = Some(label_vector(first_arg.clone(), "Y")?);
                response_name = options
                    .response_name
                    .clone()
                    .unwrap_or_else(|| "Y".to_string());
            }
            rest_start = 1;
        }
    }
    parse_fit_options(&rest[rest_start..], &mut options)?;
    if external_y.is_none() {
        response_name = options.response_name.clone().unwrap_or(response_name);
    }
    let has_external_y = external_y.is_some();
    let labels = if let Some(labels) = external_y {
        labels
    } else {
        let value = variables.fields.get(&response_name).ok_or_else(|| {
            fitctree_invalid(format!(
                "fitctree: response variable '{response_name}' is not in the table"
            ))
        })?;
        label_vector(value.clone(), &response_name)?
    };
    let predictor_names = if let Some(names) = options.predictor_names.clone() {
        names
    } else if let Some(names) = formula_predictors {
        names
    } else {
        let weight_variable = match &options.weights {
            Some(WeightSpec::Variable(name)) => Some(name.as_str()),
            _ => None,
        };
        variable_names
            .iter()
            .filter(|name| {
                (has_external_y || **name != response_name)
                    && weight_variable != Some(name.as_str())
            })
            .cloned()
            .collect()
    };
    if predictor_names.is_empty() {
        return Err(fitctree_invalid(
            "fitctree: at least one predictor variable is required",
        ));
    }
    let mut columns = Vec::with_capacity(predictor_names.len());
    for name in &predictor_names {
        let value = variables.fields.get(name).ok_or_else(|| {
            fitctree_invalid(format!(
                "fitctree: predictor variable '{name}' is not in the table"
            ))
        })?;
        let tensor = tensor::value_into_tensor_for(FITCTREE_NAME, value.clone()).map_err(|_| {
            fitctree_invalid(format!(
                "fitctree: predictor variable '{name}' must be numeric"
            ))
        })?;
        ensure_integer_tensor_exact(&tensor, name)?;
        columns.push(vector_values(&tensor, name)?);
    }
    let x = columns_to_tensor(columns)?;
    let weights = match &options.weights {
        Some(WeightSpec::Values(values)) => values.clone(),
        Some(WeightSpec::Variable(name)) => {
            let value = variables.fields.get(name).ok_or_else(|| {
                fitctree_invalid(format!(
                    "fitctree: weights variable '{name}' is not in the table"
                ))
            })?;
            numeric_vector(value, "Weights")?
        }
        None => vec![1.0; x.rows],
    };
    finalize_fit_spec(x, labels, predictor_names, response_name, weights, options)
}

fn finalize_fit_spec(
    x: Tensor,
    labels: LabelVector,
    predictor_names: Vec<String>,
    response_name: String,
    weights: Vec<f64>,
    mut options: FitOptions,
) -> BuiltinResult<FitSpec> {
    if labels.labels.len() != x.rows {
        return Err(fitctree_invalid(
            "fitctree: response length must match the number of observations",
        ));
    }
    if weights.len() != x.rows {
        return Err(fitctree_invalid(
            "fitctree: Weights length must match the number of observations",
        ));
    }
    for weight in &weights {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(fitctree_invalid(
                "fitctree: weights must be nonnegative finite values",
            ));
        }
    }
    let class_names = if let Some(names) = options.class_names.clone() {
        validate_class_names(names, labels.kind)?
    } else {
        unique_labels(&labels.labels, labels.kind)
    };
    if class_names.len() < 2 {
        return Err(fitctree_invalid(
            "fitctree: at least two classes are required",
        ));
    }
    let mut clean_rows = Vec::new();
    let mut clean_classes = Vec::new();
    let mut clean_weights = Vec::new();
    for row in 0..x.rows {
        let weight = weights[row];
        if weight == 0.0 {
            continue;
        }
        if is_missing_label(&labels.labels[row]) {
            continue;
        }
        let class_index = class_names
            .iter()
            .position(|label| same_label(label, &labels.labels[row]))
            .ok_or_else(|| {
                fitctree_invalid("fitctree: response contains a class outside ClassNames")
            })?;
        let mut row_values = Vec::with_capacity(x.cols);
        let mut has_observed_predictor = false;
        for col in 0..x.cols {
            let value = x_value(&x, row, col);
            if value.is_infinite() {
                return Err(fitctree_invalid(
                    "fitctree: predictor values must not contain Inf",
                ));
            }
            if !value.is_nan() {
                has_observed_predictor = true;
            }
            row_values.push(value);
        }
        if has_observed_predictor {
            clean_rows.push(row_values);
            clean_classes.push(class_index);
            clean_weights.push(weight);
        }
    }
    if clean_classes.is_empty() {
        return Err(fitctree_invalid(
            "fitctree: at least one weighted observation with an observed predictor is required",
        ));
    }
    let mut observed_classes = vec![false; class_names.len()];
    for class in &clean_classes {
        observed_classes[*class] = true;
    }
    if observed_classes
        .iter()
        .filter(|observed| **observed)
        .count()
        < 2
    {
        return Err(fitctree_invalid(
            "fitctree: at least two observed classes are required",
        ));
    }
    let mut clean_data = Vec::with_capacity(clean_rows.len() * x.cols);
    for col in 0..x.cols {
        for row in &clean_rows {
            clean_data.push(row[col]);
        }
    }
    let clean_x = Tensor::new(clean_data, vec![clean_rows.len(), x.cols])
        .map_err(|err| fitctree_internal(format!("fitctree: {err}")))?;
    if options.max_num_splits.is_none() {
        options.max_num_splits = Some(clean_classes.len().saturating_sub(1));
    }
    Ok(FitSpec {
        x: clean_x,
        class_indices: clean_classes,
        class_names,
        class_kind: labels.kind,
        predictor_names,
        response_name,
        weights: clean_weights,
        options,
    })
}

fn parse_fit_options(values: &[Value], options: &mut FitOptions) -> BuiltinResult<()> {
    if !values.len().is_multiple_of(2) {
        return Err(fitctree_invalid(
            "fitctree: name-value options must be paired",
        ));
    }
    let mut index = 0usize;
    while index < values.len() {
        let name = scalar_text(&values[index])
            .ok_or_else(|| fitctree_invalid("fitctree: option name must be text"))?;
        let value = &values[index + 1];
        match name.to_ascii_lowercase().as_str() {
            "classnames" => {
                let labels = label_vector(value.clone(), "ClassNames")?;
                options.class_names = Some(labels.labels);
            }
            "predictornames" => {
                options.predictor_names = Some(text_list(value, "PredictorNames")?);
            }
            "responsename" => {
                options.response_name = Some(
                    scalar_text(value)
                        .ok_or_else(|| fitctree_invalid("fitctree: ResponseName must be text"))?,
                );
            }
            "maxnumsplits" => {
                options.max_num_splits = Some(nonnegative_integer(value, "MaxNumSplits")?);
            }
            "minleafsize" => {
                options.min_leaf_size = positive_integer(value, "MinLeafSize")?;
            }
            "minparentsize" => {
                options.min_parent_size = positive_integer(value, "MinParentSize")?;
            }
            "splitcriterion" => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitctree_invalid("fitctree: SplitCriterion must be text"))?;
                options.split_criterion = match text.to_ascii_lowercase().as_str() {
                    "gdi" => SplitCriterion::Gdi,
                    "deviance" => SplitCriterion::Deviance,
                    other => {
                        return Err(fitctree_invalid(format!(
                            "fitctree: unsupported SplitCriterion '{other}'"
                        )));
                    }
                };
            }
            "weights" => {
                options.weights = Some(if let Some(name) = scalar_text(value) {
                    WeightSpec::Variable(name)
                } else {
                    WeightSpec::Values(numeric_vector(value, "Weights")?)
                });
            }
            "scoretransform" => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitctree_invalid("fitctree: ScoreTransform must be text"))?;
                if !text.eq_ignore_ascii_case("none") {
                    return Err(fitctree_invalid(
                        "fitctree: only ScoreTransform='none' is supported",
                    ));
                }
            }
            "algorithmforcategorical"
            | "categoricalpredictors"
            | "cost"
            | "crossval"
            | "cvpartition"
            | "holdout"
            | "kfold"
            | "leaveout"
            | "maxnumcategories"
            | "mergeleaves"
            | "numbins"
            | "numvariablestosample"
            | "optimizehyperparameters"
            | "hyperparameteroptimizationoptions"
            | "predictorselection"
            | "prior"
            | "prune"
            | "prunecriterion"
            | "reproducible"
            | "surrogate" => {
                return Err(fitctree_invalid(format!(
                    "fitctree: option '{name}' is not supported yet"
                )));
            }
            other => {
                return Err(fitctree_invalid(format!(
                    "fitctree: unknown option '{other}'"
                )))
            }
        }
        index += 2;
    }
    Ok(())
}

fn parse_formula(text: &str, variables: &[String]) -> BuiltinResult<(String, Vec<String>)> {
    let Some((lhs, rhs)) = text.split_once('~') else {
        return Err(fitctree_invalid("fitctree: formula must contain '~'"));
    };
    let response = lhs.trim().to_string();
    if response.is_empty() {
        return Err(fitctree_invalid("fitctree: formula response is empty"));
    }
    let mut predictors = Vec::new();
    for term in rhs.split('+') {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }
        if term == "1" {
            continue;
        }
        if term.contains('*') || term.contains(':') || term.contains('^') {
            return Err(fitctree_invalid(
                "fitctree: formula interactions are not supported yet",
            ));
        }
        if !variables.iter().any(|name| name == term) {
            return Err(fitctree_invalid(format!(
                "fitctree: formula predictor '{term}' is not in the table"
            )));
        }
        predictors.push(term.to_string());
    }
    if predictors.is_empty() {
        return Err(fitctree_invalid(
            "fitctree: formula must contain at least one predictor",
        ));
    }
    Ok((response, predictors))
}

fn fit_tree(spec: &FitSpec) -> BuiltinResult<TreeFit> {
    let mut nodes = Vec::new();
    let mut splits = 0usize;
    let rows = (0..spec.x.rows).collect::<Vec<_>>();
    build_node(spec, &rows, &mut nodes, &mut splits, 0)?;
    Ok(TreeFit {
        nodes,
        num_splits: splits,
    })
}

fn build_node(
    spec: &FitSpec,
    rows: &[usize],
    nodes: &mut Vec<Node>,
    split_count: &mut usize,
    depth: usize,
) -> BuiltinResult<usize> {
    if depth > MAX_TREE_DEPTH {
        return Err(fitctree_invalid(
            "fitctree: tree depth exceeds RunMat safety limit",
        ));
    }
    let counts = class_counts(
        rows,
        &spec.class_indices,
        &spec.weights,
        spec.class_names.len(),
    );
    let total_weight = counts.iter().sum::<f64>();
    let probabilities = if total_weight > 0.0 {
        counts.iter().map(|value| value / total_weight).collect()
    } else {
        vec![0.0; counts.len()]
    };
    let class_index = probabilities
        .iter()
        .enumerate()
        .max_by(|left, right| {
            left.1
                .partial_cmp(right.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let node_index = nodes.len();
    nodes.push(Node {
        left: None,
        right: None,
        predictor: None,
        threshold: f64::NAN,
        class_index,
        probabilities,
    });
    let max_splits = spec.options.max_num_splits.unwrap_or(0);
    if *split_count >= max_splits
        || rows.len() < spec.options.min_parent_size
        || rows.len() < spec.options.min_leaf_size.saturating_mul(2)
        || counts.iter().filter(|value| **value > 0.0).count() <= 1
    {
        return Ok(node_index);
    }
    if let Some(best) = best_split(spec, rows, &counts)? {
        if best.gain <= EPS {
            return Ok(node_index);
        }
        *split_count += 1;
        let left = build_node(spec, &best.left_rows, nodes, split_count, depth + 1)?;
        let right = build_node(spec, &best.right_rows, nodes, split_count, depth + 1)?;
        nodes[node_index].left = Some(left);
        nodes[node_index].right = Some(right);
        nodes[node_index].predictor = Some(best.predictor);
        nodes[node_index].threshold = best.threshold;
    }
    Ok(node_index)
}

fn best_split(
    spec: &FitSpec,
    rows: &[usize],
    _parent_counts: &[f64],
) -> BuiltinResult<Option<SplitCandidate>> {
    let mut best = None;
    for col in 0..spec.x.cols {
        let mut entries = rows
            .iter()
            .filter(|row| !x_value(&spec.x, **row, col).is_nan())
            .map(|row| {
                (
                    x_value(&spec.x, *row, col),
                    spec.class_indices[*row],
                    spec.weights[*row],
                    *row,
                )
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
        if entries.len() < 2 {
            continue;
        }
        let mut right_counts = vec![0.0; spec.class_names.len()];
        for (_, class_idx, weight, _) in &entries {
            right_counts[*class_idx] += *weight;
        }
        let parent_weight = right_counts.iter().sum::<f64>();
        let parent_impurity = impurity(&right_counts, spec.options.split_criterion);
        let mut left_counts = vec![0.0; right_counts.len()];
        let mut left_weight = 0.0;
        let mut right_weight = parent_weight;
        let mut left_len = 0usize;
        let mut right_len = entries.len();
        for idx in 0..entries.len() - 1 {
            let (_, class_idx, weight, _) = entries[idx];
            left_counts[class_idx] += weight;
            right_counts[class_idx] -= weight;
            left_weight += weight;
            right_weight -= weight;
            left_len += 1;
            right_len -= 1;
            let current = entries[idx].0;
            let next = entries[idx + 1].0;
            if (current - next).abs() <= EPS {
                continue;
            }
            if left_len < spec.options.min_leaf_size || right_len < spec.options.min_leaf_size {
                continue;
            }
            if left_weight <= 0.0 || right_weight <= 0.0 {
                continue;
            }
            let left_impurity = impurity(&left_counts, spec.options.split_criterion);
            let right_impurity = impurity(&right_counts, spec.options.split_criterion);
            let gain = parent_impurity
                - (left_weight / parent_weight) * left_impurity
                - (right_weight / parent_weight) * right_impurity;
            if gain
                > best
                    .as_ref()
                    .map(|candidate: &SplitCandidate| candidate.gain)
                    .unwrap_or(-1.0)
            {
                let threshold = current + (next - current) / 2.0;
                let mut left_rows = Vec::with_capacity(left_len);
                let mut right_rows = Vec::with_capacity(right_len);
                for (value, _, _, row) in &entries {
                    if *value <= threshold {
                        left_rows.push(*row);
                    } else {
                        right_rows.push(*row);
                    }
                }
                best = Some(SplitCandidate {
                    predictor: col,
                    threshold,
                    gain,
                    left_rows,
                    right_rows,
                });
            }
        }
    }
    Ok(best)
}

fn impurity(counts: &[f64], criterion: SplitCriterion) -> f64 {
    let total = counts.iter().sum::<f64>();
    if total <= 0.0 {
        return 0.0;
    }
    match criterion {
        SplitCriterion::Gdi => {
            1.0 - counts
                .iter()
                .map(|count| {
                    let p = count / total;
                    p * p
                })
                .sum::<f64>()
        }
        SplitCriterion::Deviance => counts
            .iter()
            .filter(|count| **count > 0.0)
            .map(|count| {
                let p = count / total;
                -p * p.ln()
            })
            .sum(),
    }
}

fn class_counts(
    rows: &[usize],
    classes: &[usize],
    weights: &[f64],
    class_count: usize,
) -> Vec<f64> {
    let mut counts = vec![0.0; class_count];
    for row in rows {
        counts[classes[*row]] += weights[*row];
    }
    counts
}

fn model_object(spec: FitSpec, fit: TreeFit) -> BuiltinResult<ObjectInstance> {
    let mut object = ObjectInstance::new(CLASSIFICATION_TREE_CLASS.to_string());
    object.properties.insert(
        "ResponseName".to_string(),
        Value::String(spec.response_name),
    );
    object.properties.insert(
        "PredictorNames".to_string(),
        string_column(&spec.predictor_names)?,
    );
    object.properties.insert(
        "ClassNames".to_string(),
        labels_value(&spec.class_names, spec.class_kind)?,
    );
    object.properties.insert(
        "CategoricalPredictors".to_string(),
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).map_err(|err| {
            fitctree_internal(format!(
                "fitctree: categorical predictor shape error: {err}"
            ))
        })?),
    );
    object.properties.insert(
        "ScoreTransform".to_string(),
        Value::String("none".to_string()),
    );
    object.properties.insert(
        "NumObservations".to_string(),
        Value::Num(spec.x.rows as f64),
    );
    object
        .properties
        .insert("NumPredictors".to_string(), Value::Num(spec.x.cols as f64));
    object
        .properties
        .insert("NumNodes".to_string(), Value::Num(fit.nodes.len() as f64));
    object
        .properties
        .insert("NumSplits".to_string(), Value::Num(fit.num_splits as f64));
    let mut params = StructValue::new();
    params.insert(
        "MaxNumSplits",
        Value::Num(spec.options.max_num_splits.unwrap_or(0) as f64),
    );
    params.insert("MinLeaf", Value::Num(spec.options.min_leaf_size as f64));
    params.insert("MinParent", Value::Num(spec.options.min_parent_size as f64));
    params.insert(
        "SplitCriterion",
        Value::String(
            match spec.options.split_criterion {
                SplitCriterion::Gdi => "gdi",
                SplitCriterion::Deviance => "deviance",
            }
            .to_string(),
        ),
    );
    object
        .properties
        .insert("ModelParameters".to_string(), Value::Struct(params));
    object.properties.insert(
        "__RunMatTreeClassKind".to_string(),
        Value::String(
            match spec.class_kind {
                LabelKind::Numeric => "numeric",
                LabelKind::Integer(class) => integer_class_name(class),
                LabelKind::Text => "text",
                LabelKind::Logical => "logical",
            }
            .to_string(),
        ),
    );
    object.properties.insert(
        "__RunMatTreeClassNames".to_string(),
        labels_value(&spec.class_names, spec.class_kind)?,
    );
    object.properties.insert(
        "__RunMatTreeChildren".to_string(),
        tree_children_tensor(&fit.nodes)?,
    );
    object.properties.insert(
        "__RunMatTreePredictor".to_string(),
        tree_predictor_tensor(&fit.nodes)?,
    );
    object.properties.insert(
        "__RunMatTreeThreshold".to_string(),
        tree_threshold_tensor(&fit.nodes)?,
    );
    object.properties.insert(
        "__RunMatTreeNodeClass".to_string(),
        tree_class_tensor(&fit.nodes)?,
    );
    object.properties.insert(
        "__RunMatTreeNodeProb".to_string(),
        tree_prob_tensor(&fit.nodes, spec.class_names.len())?,
    );
    Ok(object)
}

pub(crate) fn predict_classification_tree_object(
    object: ObjectInstance,
    xnew: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    if !rest.is_empty() {
        return Err(predict_invalid(
            "predict: ClassificationTree Subtrees and name-value options are not supported yet",
        ));
    }
    let predictor_names = string_property(&object, "PredictorNames")?;
    let x = predictors_for_prediction(xnew, &predictor_names)?;
    let children = matrix_property(&object, "__RunMatTreeChildren")?;
    let predictors = numeric_property(&object, "__RunMatTreePredictor")?;
    let thresholds = numeric_property(&object, "__RunMatTreeThreshold")?;
    let node_classes = numeric_property(&object, "__RunMatTreeNodeClass")?;
    let node_probs = matrix_property(&object, "__RunMatTreeNodeProb")?;
    let class_names_value = object
        .properties
        .get("__RunMatTreeClassNames")
        .cloned()
        .ok_or_else(|| predict_invalid("predict: tree is missing class names"))?;
    let class_kind = class_kind_property(&object)?;
    let class_names = labels_from_value(class_names_value, "ClassNames")?;
    let class_count = class_names.labels.len();
    let node_count = predictors.len();
    if children.len() != node_count * 2
        || thresholds.len() != node_count
        || node_classes.len() != node_count
        || node_probs.len() != node_count * class_count
    {
        return Err(predict_invalid("predict: tree metadata is inconsistent"));
    }
    let mut label_indices = Vec::with_capacity(x.rows);
    let mut score_data = vec![0.0; x.rows * class_count];
    let mut node_data = Vec::with_capacity(x.rows);
    let mut class_number_data = Vec::with_capacity(x.rows);
    for row in 0..x.rows {
        let node = traverse_tree(&x, row, &children, &predictors, &thresholds, node_count)?;
        let class_index = checked_one_based_index(node_classes[node], class_count, "class number")?;
        label_indices.push(class_index);
        node_data.push(node as f64 + 1.0);
        class_number_data.push(class_index as f64 + 1.0);
        for class in 0..class_count {
            score_data[class * x.rows + row] = node_probs[class * node_count + node];
        }
    }
    let labels = predicted_labels_value(&class_names.labels, class_kind, &label_indices)?;
    let scores = Value::Tensor(
        Tensor::new(score_data, vec![x.rows, class_count])
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    let node = Value::Tensor(
        Tensor::new(node_data, vec![x.rows, 1])
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    let cnum = Value::Tensor(
        Tensor::new(class_number_data, vec![x.rows, 1])
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    Ok(vec![labels, scores, node, cnum])
}

fn traverse_tree(
    x: &Tensor,
    row: usize,
    children: &[f64],
    predictors: &[f64],
    thresholds: &[f64],
    node_count: usize,
) -> BuiltinResult<usize> {
    let mut node = 0usize;
    for _ in 0..node_count {
        let predictor = predictors[node];
        if predictor <= 0.0 {
            return Ok(node);
        }
        let predictor_idx = checked_one_based_index(predictor, x.cols, "predictor")?;
        let value = x_value(x, row, predictor_idx);
        if value.is_nan() {
            return Ok(node);
        }
        let child_col = if value <= thresholds[node] { 0 } else { 1 };
        let child = children[child_col * node_count + node];
        if child <= 0.0 {
            return Ok(node);
        }
        node = checked_one_based_index(child, node_count, "child node")?;
    }
    Err(predict_invalid("predict: tree traversal did not terminate"))
}

fn checked_one_based_index(value: f64, len: usize, label: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(predict_invalid(format!("predict: invalid {label}")));
    }
    let index = value as usize - 1;
    if index >= len {
        return Err(predict_invalid(format!("predict: {label} out of range")));
    }
    Ok(index)
}

fn predictors_for_prediction(value: Value, predictor_names: &[String]) -> BuiltinResult<Tensor> {
    if let Value::Object(object) = &value {
        if is_tabular_object(object) {
            let variables = table_variables(object)
                .map_err(|err| predict_invalid(format!("predict: {err}")))?;
            let mut columns = Vec::with_capacity(predictor_names.len());
            for name in predictor_names {
                let raw = variables.fields.get(name).ok_or_else(|| {
                    predict_invalid(format!("predict: missing predictor '{name}'"))
                })?;
                let tensor =
                    tensor::value_into_tensor_for(PREDICT_NAME, raw.clone()).map_err(|_| {
                        predict_invalid(format!("predict: predictor '{name}' must be numeric"))
                    })?;
                ensure_integer_tensor_exact_predict(&tensor, name)?;
                columns.push(vector_values_predict(&tensor, name)?);
            }
            return columns_to_tensor_predict(columns);
        }
    }
    let tensor = tensor::value_into_tensor_for(PREDICT_NAME, value)
        .map_err(|err| predict_invalid(format!("predict: {err}")))?;
    ensure_integer_tensor_exact_predict(&tensor, "predictors")?;
    if tensor.shape.len() > 2 || tensor.cols != predictor_names.len() {
        return Err(predict_invalid(format!(
            "predict: X must have {} predictor columns",
            predictor_names.len()
        )));
    }
    Ok(tensor)
}

fn label_vector(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    labels_from_value(value, name).map_err(|err| fitctree_invalid(err.message))
}

fn labels_from_value(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    match value {
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            if tensor.shape.len() > 2 || (tensor.rows != 1 && tensor.cols != 1) {
                return Err(predict_invalid(format!("predict: {name} must be a vector")));
            }
            let storage = tensor.integer_storage().expect("checked above");
            Ok(LabelVector {
                labels: storage
                    .exact_values()
                    .into_iter()
                    .map(ClassLabel::Integer)
                    .collect(),
                kind: LabelKind::Integer(integer_class(storage)),
            })
        }
        Value::Tensor(tensor) => Ok(LabelVector {
            labels: vector_values_predict(&tensor, name)?
                .into_iter()
                .map(ClassLabel::Numeric)
                .collect(),
            kind: LabelKind::Numeric,
        }),
        Value::Num(value) => Ok(LabelVector {
            labels: vec![ClassLabel::Numeric(value)],
            kind: LabelKind::Numeric,
        }),
        Value::Int(value) => Ok(LabelVector {
            kind: LabelKind::Integer(integer_class_for_value(&value)),
            labels: vec![ClassLabel::Integer(value)],
        }),
        Value::String(text) => Ok(LabelVector {
            labels: vec![ClassLabel::Text(text)],
            kind: LabelKind::Text,
        }),
        Value::CharArray(array) => Ok(LabelVector {
            labels: char_rows(&array).into_iter().map(ClassLabel::Text).collect(),
            kind: LabelKind::Text,
        }),
        Value::StringArray(array) => Ok(LabelVector {
            labels: array.data.into_iter().map(ClassLabel::Text).collect(),
            kind: LabelKind::Text,
        }),
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data.len());
            for item in cell.data {
                labels.push(ClassLabel::Text(
                    scalar_text(&item).ok_or_else(|| {
                        predict_invalid(format!("predict: {name} cell entries must be text"))
                    })?,
                ));
            }
            Ok(LabelVector {
                labels,
                kind: LabelKind::Text,
            })
        }
        Value::Bool(value) => Ok(LabelVector {
            labels: vec![ClassLabel::Logical(value)],
            kind: LabelKind::Logical,
        }),
        Value::LogicalArray(array) => Ok(LabelVector {
            labels: array
                .data
                .into_iter()
                .map(|value| ClassLabel::Logical(value != 0))
                .collect(),
            kind: LabelKind::Logical,
        }),
        other => Err(predict_invalid(format!(
            "predict: {name} must be numeric, logical, string, character, or cellstr labels; got {other:?}"
        ))),
    }
}

fn validate_class_names(names: Vec<ClassLabel>, kind: LabelKind) -> BuiltinResult<Vec<ClassLabel>> {
    if names.is_empty() {
        return Err(fitctree_invalid("fitctree: ClassNames must not be empty"));
    }
    for name in &names {
        if is_missing_label(name) {
            return Err(fitctree_invalid(
                "fitctree: ClassNames must not contain missing values",
            ));
        }
        if label_kind(name) != kind {
            return Err(fitctree_invalid(
                "fitctree: ClassNames must have the same type as the response",
            ));
        }
    }
    let mut out = Vec::new();
    for name in names {
        if out.iter().any(|existing| same_label(existing, &name)) {
            return Err(fitctree_invalid(
                "fitctree: ClassNames must contain unique values",
            ));
        }
        out.push(name);
    }
    Ok(out)
}

fn unique_labels(labels: &[ClassLabel], kind: LabelKind) -> Vec<ClassLabel> {
    let mut out = Vec::new();
    for label in labels {
        if is_missing_label(label) {
            continue;
        }
        if !out.iter().any(|existing| same_label(existing, label)) {
            out.push(label.clone());
        }
    }
    match kind {
        LabelKind::Numeric => out.sort_by(|left, right| match (left, right) {
            (ClassLabel::Numeric(a), ClassLabel::Numeric(b)) => {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            _ => Ordering::Equal,
        }),
        LabelKind::Integer(_) => out.sort_by(|left, right| match (left, right) {
            (ClassLabel::Integer(a), ClassLabel::Integer(b)) => integer_cmp(a, b),
            _ => Ordering::Equal,
        }),
        LabelKind::Text => out.sort_by(|left, right| match (left, right) {
            (ClassLabel::Text(a), ClassLabel::Text(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }),
        LabelKind::Logical => out.sort_by_key(|label| match label {
            ClassLabel::Logical(value) => *value,
            _ => false,
        }),
    }
    out
}

fn label_kind(label: &ClassLabel) -> LabelKind {
    match label {
        ClassLabel::Numeric(_) => LabelKind::Numeric,
        ClassLabel::Integer(value) => LabelKind::Integer(integer_class_for_value(value)),
        ClassLabel::Text(_) => LabelKind::Text,
        ClassLabel::Logical(_) => LabelKind::Logical,
    }
}

fn same_label(left: &ClassLabel, right: &ClassLabel) -> bool {
    match (left, right) {
        (ClassLabel::Numeric(a), ClassLabel::Numeric(b)) => (a == b) || (a.is_nan() && b.is_nan()),
        (ClassLabel::Integer(a), ClassLabel::Integer(b)) => a == b,
        (ClassLabel::Text(a), ClassLabel::Text(b)) => a == b,
        (ClassLabel::Logical(a), ClassLabel::Logical(b)) => a == b,
        _ => false,
    }
}

fn is_missing_label(label: &ClassLabel) -> bool {
    matches!(label, ClassLabel::Numeric(value) if value.is_nan())
}

fn integer_class(storage: &IntegerStorage) -> IntegerClass {
    match storage {
        IntegerStorage::I8(_) => IntegerClass::I8,
        IntegerStorage::I16(_) => IntegerClass::I16,
        IntegerStorage::I32(_) => IntegerClass::I32,
        IntegerStorage::I64(_) => IntegerClass::I64,
        IntegerStorage::U8(_) => IntegerClass::U8,
        IntegerStorage::U16(_) => IntegerClass::U16,
        IntegerStorage::U32(_) => IntegerClass::U32,
        IntegerStorage::U64(_) => IntegerClass::U64,
    }
}

fn integer_class_for_value(value: &IntValue) -> IntegerClass {
    match value {
        IntValue::I8(_) => IntegerClass::I8,
        IntValue::I16(_) => IntegerClass::I16,
        IntValue::I32(_) => IntegerClass::I32,
        IntValue::I64(_) => IntegerClass::I64,
        IntValue::U8(_) => IntegerClass::U8,
        IntValue::U16(_) => IntegerClass::U16,
        IntValue::U32(_) => IntegerClass::U32,
        IntValue::U64(_) => IntegerClass::U64,
    }
}

fn integer_class_name(class: IntegerClass) -> &'static str {
    match class {
        IntegerClass::I8 => "int8",
        IntegerClass::I16 => "int16",
        IntegerClass::I32 => "int32",
        IntegerClass::I64 => "int64",
        IntegerClass::U8 => "uint8",
        IntegerClass::U16 => "uint16",
        IntegerClass::U32 => "uint32",
        IntegerClass::U64 => "uint64",
    }
}

fn integer_class_from_name(name: &str) -> Option<IntegerClass> {
    match name {
        "int8" => Some(IntegerClass::I8),
        "int16" => Some(IntegerClass::I16),
        "int32" => Some(IntegerClass::I32),
        "int64" => Some(IntegerClass::I64),
        "uint8" => Some(IntegerClass::U8),
        "uint16" => Some(IntegerClass::U16),
        "uint32" => Some(IntegerClass::U32),
        "uint64" => Some(IntegerClass::U64),
        _ => None,
    }
}

fn integer_cmp(left: &IntValue, right: &IntValue) -> Ordering {
    match (left, right) {
        (IntValue::I8(a), IntValue::I8(b)) => a.cmp(b),
        (IntValue::I16(a), IntValue::I16(b)) => a.cmp(b),
        (IntValue::I32(a), IntValue::I32(b)) => a.cmp(b),
        (IntValue::I64(a), IntValue::I64(b)) => a.cmp(b),
        (IntValue::U8(a), IntValue::U8(b)) => a.cmp(b),
        (IntValue::U16(a), IntValue::U16(b)) => a.cmp(b),
        (IntValue::U32(a), IntValue::U32(b)) => a.cmp(b),
        (IntValue::U64(a), IntValue::U64(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

fn integer_storage(class: IntegerClass, values: Vec<IntValue>) -> Result<IntegerStorage, String> {
    match class {
        IntegerClass::I8 => collect_integer_values(values, "int8", |v| match v {
            IntValue::I8(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::I8),
        IntegerClass::I16 => collect_integer_values(values, "int16", |v| match v {
            IntValue::I16(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::I16),
        IntegerClass::I32 => collect_integer_values(values, "int32", |v| match v {
            IntValue::I32(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::I32),
        IntegerClass::I64 => collect_integer_values(values, "int64", |v| match v {
            IntValue::I64(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::I64),
        IntegerClass::U8 => collect_integer_values(values, "uint8", |v| match v {
            IntValue::U8(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::U8),
        IntegerClass::U16 => collect_integer_values(values, "uint16", |v| match v {
            IntValue::U16(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::U16),
        IntegerClass::U32 => collect_integer_values(values, "uint32", |v| match v {
            IntValue::U32(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::U32),
        IntegerClass::U64 => collect_integer_values(values, "uint64", |v| match v {
            IntValue::U64(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::U64),
    }
}

fn collect_integer_values<T>(
    values: Vec<IntValue>,
    expected: &str,
    convert: impl Fn(IntValue) -> Option<T>,
) -> Result<Vec<T>, String> {
    values
        .into_iter()
        .map(|value| {
            let actual = value.class_name();
            convert(value).ok_or_else(|| {
                format!("integer label invariant expected {expected}, found {actual}")
            })
        })
        .collect()
}

fn labels_value(labels: &[ClassLabel], kind: LabelKind) -> BuiltinResult<Value> {
    predicted_labels_value(labels, kind, &(0..labels.len()).collect::<Vec<_>>())
}

fn predicted_labels_value(
    labels: &[ClassLabel],
    kind: LabelKind,
    indices: &[usize],
) -> BuiltinResult<Value> {
    match kind {
        LabelKind::Numeric => {
            let data = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Numeric(value)) => *value,
                    _ => f64::NAN,
                })
                .collect::<Vec<_>>();
            Ok(Value::Tensor(
                Tensor::new(data, vec![indices.len(), 1])
                    .map_err(|err| predict_internal(format!("predict: {err}")))?,
            ))
        }
        LabelKind::Integer(class) => {
            let values = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Integer(value)) => Ok(value.clone()),
                    Some(_) => Err(predict_internal(
                        "predict: integer class metadata does not match stored labels",
                    )),
                    None => Err(predict_internal(
                        "predict: predicted class index is out of range",
                    )),
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(Value::Tensor(
                Tensor::new_integer(
                    integer_storage(class, values).map_err(predict_internal)?,
                    vec![indices.len(), 1],
                )
                .map_err(|err| predict_internal(format!("predict: {err}")))?,
            ))
        }
        LabelKind::Text => {
            let data = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Text(value)) => value.clone(),
                    _ => String::new(),
                })
                .collect::<Vec<_>>();
            Ok(Value::StringArray(StringArray {
                data,
                shape: vec![indices.len(), 1],
                rows: indices.len(),
                cols: 1,
            }))
        }
        LabelKind::Logical => {
            let data = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Logical(value)) if *value => 1,
                    _ => 0,
                })
                .collect::<Vec<_>>();
            Ok(Value::LogicalArray(
                LogicalArray::new(data, vec![indices.len(), 1])
                    .map_err(|err| predict_internal(format!("predict: {err}")))?,
            ))
        }
    }
}

fn class_kind_property(object: &ObjectInstance) -> BuiltinResult<LabelKind> {
    match object.properties.get("__RunMatTreeClassKind") {
        Some(Value::String(text)) if text == "numeric" => Ok(LabelKind::Numeric),
        Some(Value::String(text)) if integer_class_from_name(text).is_some() => Ok(
            LabelKind::Integer(integer_class_from_name(text).expect("checked above")),
        ),
        Some(Value::String(text)) if text == "text" => Ok(LabelKind::Text),
        Some(Value::String(text)) if text == "logical" => Ok(LabelKind::Logical),
        _ => Err(predict_invalid(
            "predict: tree is missing class type metadata",
        )),
    }
}

fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

fn text_list(value: &Value, name: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::CharArray(array) => Ok(char_rows(array)),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|value| {
                scalar_text(value).ok_or_else(|| {
                    fitctree_invalid(format!("fitctree: {name} entries must be text"))
                })
            })
            .collect(),
        _ => Err(fitctree_invalid(format!(
            "fitctree: {name} must be text or a text collection"
        ))),
    }
}

fn char_rows(array: &CharArray) -> Vec<String> {
    let mut out = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        out.push(array.data[start..start + array.cols].iter().collect());
    }
    out
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => {
            ensure_int_exact_f64(value, name)?;
            Ok(vec![value.to_f64()])
        }
        Value::Tensor(tensor) => {
            ensure_integer_tensor_exact(tensor, name)?;
            vector_values(tensor, name)
        }
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|value| if *value == 0 { 0.0 } else { 1.0 })
            .collect()),
        Value::Bool(value) => Ok(vec![if *value { 1.0 } else { 0.0 }]),
        _ => Err(fitctree_invalid(format!(
            "fitctree: {name} must be numeric"
        ))),
    }
}

fn vector_values(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    vector_values_predict(tensor, name).map_err(|err| fitctree_invalid(err.message))
}

fn vector_values_predict(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    if tensor.shape.len() > 2 || (tensor.rows != 1 && tensor.cols != 1) {
        return Err(predict_invalid(format!("predict: {name} must be a vector")));
    }
    Ok(tensor::tensor_values_f64(tensor))
}

fn nonnegative_integer(value: &Value, name: &str) -> BuiltinResult<usize> {
    let value = numeric_scalar(value, name)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(fitctree_invalid(format!(
            "fitctree: {name} must be a nonnegative integer"
        )));
    }
    Ok(value as usize)
}

fn positive_integer(value: &Value, name: &str) -> BuiltinResult<usize> {
    let value = numeric_scalar(value, name)?;
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(fitctree_invalid(format!(
            "fitctree: {name} must be a positive integer"
        )));
    }
    Ok(value as usize)
}

fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => {
            ensure_int_exact_f64(value, name)?;
            Ok(value.to_f64())
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            ensure_integer_tensor_exact(tensor, name)?;
            Ok(tensor::tensor_values_f64(tensor)[0])
        }
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        _ => Err(fitctree_invalid(format!(
            "fitctree: {name} must be a numeric scalar"
        ))),
    }
}

fn ensure_integer_tensor_exact(tensor: &Tensor, name: &str) -> BuiltinResult<()> {
    if let Some(storage) = tensor.integer_storage() {
        for value in storage.exact_values() {
            ensure_int_exact_f64(&value, name)?;
        }
    }
    Ok(())
}

fn ensure_integer_tensor_exact_predict(tensor: &Tensor, name: &str) -> BuiltinResult<()> {
    if let Some(storage) = tensor.integer_storage() {
        for value in storage.exact_values() {
            if !crate::builtins::math::trigonometry::cos::integer_is_exact_f64(&value) {
                return Err(predict_invalid(format!(
                    "predict: integer {name} must be exactly representable as double"
                )));
            }
        }
    }
    Ok(())
}

fn ensure_int_exact_f64(value: &IntValue, name: &str) -> BuiltinResult<()> {
    if crate::builtins::math::trigonometry::cos::integer_is_exact_f64(value) {
        Ok(())
    } else {
        Err(fitctree_invalid(format!(
            "fitctree: integer {name} must be exactly representable as double"
        )))
    }
}

fn is_name_value_option(value: &Value) -> bool {
    scalar_text(value).is_some_and(|text| {
        matches!(
            text.to_ascii_lowercase().as_str(),
            "algorithmforcategorical"
                | "categoricalpredictors"
                | "classnames"
                | "cost"
                | "crossval"
                | "cvpartition"
                | "holdout"
                | "hyperparameteroptimizationoptions"
                | "kfold"
                | "leaveout"
                | "maxnumcategories"
                | "maxnumsplits"
                | "mergeleaves"
                | "minleafsize"
                | "minparentsize"
                | "numbins"
                | "numvariablestosample"
                | "optimizehyperparameters"
                | "predictornames"
                | "predictorselection"
                | "prior"
                | "prune"
                | "prunecriterion"
                | "reproducible"
                | "responsename"
                | "scoretransform"
                | "splitcriterion"
                | "surrogate"
                | "weights"
        )
    })
}

fn columns_to_tensor(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if columns.is_empty() {
        return Err(fitctree_invalid("fitctree: no predictor columns"));
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        if column.len() != rows {
            return Err(fitctree_invalid(
                "fitctree: predictor variables must have the same height",
            ));
        }
        data.extend(column);
    }
    Tensor::new(data, vec![rows, cols]).map_err(|err| fitctree_internal(format!("fitctree: {err}")))
}

fn columns_to_tensor_predict(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if columns.is_empty() {
        return Err(predict_invalid("predict: no predictor columns"));
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        if column.len() != rows {
            return Err(predict_invalid(
                "predict: predictor table variables must have the same height",
            ));
        }
        data.extend(column);
    }
    Tensor::new(data, vec![rows, cols]).map_err(|err| predict_internal(format!("predict: {err}")))
}

fn x_value(tensor: &Tensor, row: usize, col: usize) -> f64 {
    tensor::tensor_value_f64(tensor, col * tensor.rows + row)
}

fn string_column(values: &[String]) -> BuiltinResult<Value> {
    Ok(Value::StringArray(StringArray {
        data: values.to_vec(),
        shape: vec![values.len(), 1],
        rows: values.len(),
        cols: 1,
    }))
}

fn numeric_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    match object.properties.get(name) {
        Some(Value::Tensor(tensor)) => Ok(tensor::tensor_values_f64(tensor)),
        _ => Err(predict_invalid(format!("predict: tree is missing {name}"))),
    }
}

fn matrix_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    numeric_property(object, name)
}

fn string_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<String>> {
    match object.properties.get(name) {
        Some(Value::StringArray(array)) => Ok(array.data.clone()),
        Some(Value::String(text)) => Ok(vec![text.clone()]),
        _ => Err(predict_invalid(format!("predict: tree is missing {name}"))),
    }
}

fn tree_children_tensor(nodes: &[Node]) -> BuiltinResult<Value> {
    let rows = nodes.len();
    let mut data = Vec::with_capacity(rows * 2);
    for node in nodes {
        data.push(node.left.map(|idx| idx as f64 + 1.0).unwrap_or(0.0));
    }
    for node in nodes {
        data.push(node.right.map(|idx| idx as f64 + 1.0).unwrap_or(0.0));
    }
    Ok(Value::Tensor(Tensor::new(data, vec![rows, 2]).map_err(
        |err| fitctree_internal(format!("fitctree: {err}")),
    )?))
}

fn tree_predictor_tensor(nodes: &[Node]) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(
            nodes
                .iter()
                .map(|node| node.predictor.map(|idx| idx as f64 + 1.0).unwrap_or(0.0))
                .collect(),
            vec![nodes.len(), 1],
        )
        .map_err(|err| fitctree_internal(format!("fitctree: {err}")))?,
    ))
}

fn tree_threshold_tensor(nodes: &[Node]) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(
            nodes.iter().map(|node| node.threshold).collect(),
            vec![nodes.len(), 1],
        )
        .map_err(|err| fitctree_internal(format!("fitctree: {err}")))?,
    ))
}

fn tree_class_tensor(nodes: &[Node]) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(
            nodes
                .iter()
                .map(|node| node.class_index as f64 + 1.0)
                .collect(),
            vec![nodes.len(), 1],
        )
        .map_err(|err| fitctree_internal(format!("fitctree: {err}")))?,
    ))
}

fn tree_prob_tensor(nodes: &[Node], class_count: usize) -> BuiltinResult<Value> {
    let rows = nodes.len();
    let mut data = Vec::with_capacity(rows * class_count);
    for class in 0..class_count {
        for node in nodes {
            data.push(node.probabilities.get(class).copied().unwrap_or(0.0));
        }
    }
    Ok(Value::Tensor(
        Tensor::new(data, vec![rows, class_count])
            .map_err(|err| fitctree_internal(format!("fitctree: {err}")))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn tensor_value(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
        Value::Tensor(tensor)
    }

    fn object(value: Value) -> ObjectInstance {
        match value {
            Value::Object(object) => object,
            other => panic!("expected object, got {other:?}"),
        }
    }

    fn tensor(value: &Value) -> &Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn fitctree_matrix_predicts_numeric_labels_and_scores() {
        let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let y = tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1);
        let model = block_on(fitctree_builtin(
            x,
            vec![
                y,
                Value::from("MaxNumSplits"),
                Value::Num(1.0),
                Value::from("MinParentSize"),
                Value::Num(2.0),
            ],
        ))
        .unwrap();
        let output_count = crate::output_count::push_output_count(Some(4));
        let predicted = crate::builtins::stats::ml::linear_model::predict_builtin(
            model,
            tensor_value(vec![0.5, 2.5], 2, 1),
            Vec::new(),
        );
        let predicted = block_on(predicted).unwrap();
        drop(output_count);
        let Value::OutputList(values) = predicted else {
            panic!("expected output list");
        };
        assert_eq!(tensor(&values[0]).materialize_f64(), vec![0.0, 1.0]);
        assert_eq!(tensor(&values[1]).shape, vec![2, 2]);
        assert_eq!(tensor(&values[2]).shape, vec![2, 1]);
        assert_eq!(tensor(&values[3]).materialize_f64(), vec![1.0, 2.0]);
    }

    #[test]
    fn fitctree_and_predict_read_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let model = block_on(fitctree_builtin(
            poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1),
            vec![
                poisoned_int_tensor(IntegerStorage::U8(vec![0, 0, 1, 1]), 4, 1),
                Value::from("ClassNames"),
                poisoned_int_tensor(IntegerStorage::U8(vec![0, 1]), 2, 1),
                Value::from("Weights"),
                poisoned_int_tensor(IntegerStorage::U8(vec![1, 1, 1, 1]), 4, 1),
                Value::from("MaxNumSplits"),
                poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
                Value::from("MinLeafSize"),
                poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
                Value::from("MinParentSize"),
                poisoned_int_tensor(IntegerStorage::U8(vec![2]), 1, 1),
            ],
        ))
        .unwrap();
        let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
            model,
            poisoned_int_tensor(IntegerStorage::I16(vec![0, 3]), 2, 1),
            Vec::new(),
        ))
        .unwrap();
        assert_eq!(tensor(&predicted).materialize_f64(), vec![0.0, 1.0]);
    }

    #[test]
    fn integer_grouping_labels_round_trip_all_classes() {
        let storages = [
            IntegerStorage::I8(vec![-1, 1]),
            IntegerStorage::I16(vec![-1, 1]),
            IntegerStorage::I32(vec![-1, 1]),
            IntegerStorage::I64(vec![-1, 1]),
            IntegerStorage::U8(vec![0, 1]),
            IntegerStorage::U16(vec![0, 1]),
            IntegerStorage::U32(vec![0, 1]),
            IntegerStorage::U64(vec![0, 1]),
        ];
        for storage in storages {
            let expected = storage.clone();
            let labels = labels_from_value(
                Value::Tensor(Tensor::new_integer(storage, vec![2, 1]).unwrap()),
                "Y",
            )
            .unwrap();
            let Value::Tensor(round_trip) = labels_value(&labels.labels, labels.kind).unwrap()
            else {
                panic!("expected integer tensor");
            };
            assert_eq!(round_trip.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn fitctree_preserves_adjacent_wide_uint64_classes() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let base = 9_007_199_254_740_992_u64;
        let model = block_on(fitctree_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            vec![
                poisoned_int_tensor(
                    IntegerStorage::U64(vec![base, base, base + 1, base + 1]),
                    4,
                    1,
                ),
                Value::from("MaxNumSplits"),
                Value::Num(1.0),
                Value::from("MinParentSize"),
                Value::Num(2.0),
            ],
        ))
        .unwrap();
        let Value::Object(model_object) = &model else {
            panic!("expected model");
        };
        assert_eq!(
            model_object
                .properties
                .get("ClassNames")
                .and_then(|value| match value {
                    Value::Tensor(tensor) => tensor.integer_storage(),
                    _ => None,
                }),
            Some(&IntegerStorage::U64(vec![base, base + 1]))
        );
        let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
            model,
            tensor_value(vec![0.0, 3.0], 2, 1),
            Vec::new(),
        ))
        .unwrap();
        assert_eq!(
            match &predicted {
                Value::Tensor(tensor) => tensor.integer_storage(),
                _ => None,
            },
            Some(&IntegerStorage::U64(vec![base, base + 1]))
        );
    }

    #[test]
    fn fitctree_strict_mode_rejects_integer_numeric_extensions() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(fitctree_builtin(
            poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1),
            vec![tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1)],
        ))
        .expect_err("integer predictor extension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FitctreeIntegerPredictorExtension")
        );

        let error = block_on(fitctree_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            vec![
                tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
                Value::from("MaxNumSplits"),
                poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
            ],
        ))
        .expect_err("integer control extension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FitctreeIntegerNumericArgumentExtension")
        );

        let poison = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![4, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let error = block_on(fitctree_builtin(
            poison,
            vec![tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1)],
        ))
        .expect_err("resident gate before gather");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FitctreeResidentInputExtension")
        );
    }

    #[test]
    fn fitctree_rejects_inexact_integer_predictors_and_weights() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let inexact = 9_007_199_254_740_993_u64;
        let error = block_on(fitctree_builtin(
            poisoned_int_tensor(IntegerStorage::U64(vec![0, 1, 2, inexact]), 4, 1),
            vec![tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1)],
        ))
        .expect_err("inexact predictor");
        assert!(error.message.contains("exactly representable"));

        let error = block_on(fitctree_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            vec![
                tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
                Value::from("Weights"),
                poisoned_int_tensor(IntegerStorage::U64(vec![1, 1, 1, inexact]), 4, 1),
            ],
        ))
        .expect_err("inexact weights");
        assert!(error.message.contains("exactly representable"));
    }

    #[test]
    fn fitctree_retains_rows_with_partially_missing_predictors() {
        let model = block_on(fitctree_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0, f64::NAN, 10.0, 11.0, 12.0], 4, 2),
            vec![
                tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
                Value::from("MaxNumSplits"),
                Value::Num(1.0),
                Value::from("MinParentSize"),
                Value::Num(2.0),
            ],
        ))
        .unwrap();
        let Value::Object(object) = model else {
            panic!("expected model");
        };
        assert_eq!(
            object.properties.get("NumObservations"),
            Some(&Value::Num(4.0))
        );
    }

    #[test]
    fn fitctree_table_response_name_predicts_string_labels() {
        let table = crate::builtins::table::table_from_columns(
            vec!["A".into(), "Y".into()],
            vec![
                tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
                Value::StringArray(StringArray {
                    data: vec!["low".into(), "low".into(), "high".into(), "high".into()],
                    shape: vec![4, 1],
                    rows: 4,
                    cols: 1,
                }),
            ],
        )
        .unwrap();
        let model = block_on(fitctree_builtin(
            table.clone(),
            vec![
                Value::String("Y".into()),
                Value::from("MaxNumSplits"),
                Value::Num(1.0),
                Value::from("MinParentSize"),
                Value::Num(2.0),
            ],
        ))
        .unwrap();
        let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
            model,
            table,
            Vec::new(),
        ))
        .unwrap();
        let Value::StringArray(labels) = predicted else {
            panic!("expected string labels");
        };
        assert_eq!(labels.shape, vec![4, 1]);
        assert_eq!(labels.data[0], "low");
        assert_eq!(labels.data[3], "high");
    }

    #[test]
    fn fitctree_respects_class_names_and_depth_controls() {
        let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let y = tensor_value(vec![2.0, 2.0, 5.0, 5.0], 4, 1);
        let model = object(
            block_on(fitctree_builtin(
                x,
                vec![
                    y,
                    Value::from("ClassNames"),
                    tensor_value(vec![5.0, 2.0], 2, 1),
                    Value::from("MaxNumSplits"),
                    Value::Num(0.0),
                ],
            ))
            .unwrap(),
        );
        assert_eq!(model.properties.get("NumSplits"), Some(&Value::Num(0.0)));
        let classes = tensor(model.properties.get("ClassNames").unwrap());
        assert_eq!(classes.materialize_f64(), vec![5.0, 2.0]);
    }

    #[test]
    fn fitctree_omits_nan_responses_and_requires_two_observed_classes() {
        let model = object(
            block_on(fitctree_builtin(
                tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
                vec![
                    tensor_value(vec![0.0, f64::NAN, 1.0, 1.0], 4, 1),
                    Value::from("MinParentSize"),
                    Value::Num(2.0),
                ],
            ))
            .unwrap(),
        );
        assert_eq!(
            model.properties.get("NumObservations"),
            Some(&Value::Num(3.0))
        );
        assert_eq!(
            tensor(model.properties.get("ClassNames").unwrap()).materialize_f64(),
            vec![0.0, 1.0]
        );

        let err = block_on(fitctree_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            vec![
                tensor_value(vec![0.0, f64::NAN, f64::NAN, 0.0], 4, 1),
                Value::from("ClassNames"),
                tensor_value(vec![0.0, 1.0], 2, 1),
            ],
        ))
        .unwrap_err();
        assert!(err.message().contains("at least two observed classes"));
    }

    #[test]
    fn fitctree_excludes_table_weight_variable_from_default_predictors() {
        let table = crate::builtins::table::table_from_columns(
            vec!["A".into(), "W".into(), "Y".into()],
            vec![
                tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
                tensor_value(vec![1.0, 2.0, 1.0, 2.0], 4, 1),
                tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            ],
        )
        .unwrap();
        let model = object(
            block_on(fitctree_builtin(
                table,
                vec![Value::from("Y"), Value::from("Weights"), Value::from("W")],
            ))
            .unwrap(),
        );
        assert_eq!(
            model.properties.get("NumPredictors"),
            Some(&Value::Num(1.0))
        );
        let Value::StringArray(names) = model.properties.get("PredictorNames").unwrap() else {
            panic!("expected predictor names");
        };
        assert_eq!(names.data, vec!["A"]);
    }

    #[test]
    fn fitctree_predict_rejects_too_many_outputs() {
        let model = block_on(fitctree_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            vec![
                tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
                Value::from("MaxNumSplits"),
                Value::Num(1.0),
                Value::from("MinParentSize"),
                Value::Num(2.0),
            ],
        ))
        .unwrap();
        let output_count = crate::output_count::push_output_count(Some(5));
        let err = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
            model,
            tensor_value(vec![0.5, 2.5], 2, 1),
            Vec::new(),
        ))
        .unwrap_err();
        drop(output_count);
        assert!(err.message().contains("too many output"));
    }

    #[test]
    fn fitctree_rejects_unsupported_cross_validation_options() {
        let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let y = tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1);
        let err = block_on(fitctree_builtin(
            x,
            vec![y, Value::from("CrossVal"), Value::String("on".into())],
        ))
        .unwrap_err();
        assert!(err.message.contains("not supported"));
    }
}
