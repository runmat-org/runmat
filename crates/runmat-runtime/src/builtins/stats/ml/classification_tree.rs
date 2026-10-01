//! Numeric classification tree compatibility surface.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, FitctreeOption,
    FITCTREE_ERROR_INTERNAL as ERROR_FITCTREE_INTERNAL,
    FITCTREE_ERROR_INVALID_ARGUMENT as ERROR_FITCTREE_INVALID_ARGUMENT,
    FITCTREE_INTEGER_NUMERIC_ARGUMENT_EXTENSION as INTEGER_NUMERIC_ARGUMENT_EXTENSION,
    FITCTREE_INTEGER_PREDICTOR_EXTENSION as INTEGER_PREDICTOR_EXTENSION,
    FITCTREE_RESIDENT_INPUT_EXTENSION as RESIDENT_INPUT_EXTENSION,
    PREDICT_ERROR_INTERNAL as ERROR_PREDICT_INTERNAL,
    PREDICT_ERROR_INVALID_ARGUMENT as ERROR_PREDICT_INVALID_ARGUMENT,
};
use runmat_macros::runtime_builtin;
use runmat_types::IntegerClass;
use runmat_value::{
    CharArray, IntValue, IntegerStorage, LogicalArray, ObjectInstance, StringArray, StructValue,
    Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::builtins::table::{
    is_tabular_object, table_variable_names_from_object, table_variables,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const FITCTREE_NAME: &str = "fitctree";
const PREDICT_NAME: &str = "predict";
const EPS: f64 = 1.0e-12;
const MAX_TREE_DEPTH: usize = 4096;

mod training;
use training::fit_tree;
mod admission;
mod options;
mod prediction;
mod projection;
use admission::{build_fit_spec, is_table_value};
use options::{parse_fit_options, parse_formula};
pub(crate) use prediction::predict_classification_tree_object;
use projection::model_object;
mod labels;
mod values;
use labels::{
    class_kind_property, integer_class_name, integer_storage, is_missing_label, label_vector,
    labels_from_value, labels_value, predicted_labels_value, same_label, unique_labels,
    validate_class_names,
};
use values::{
    char_rows, columns_to_tensor, columns_to_tensor_predict, ensure_integer_tensor_exact,
    ensure_integer_tensor_exact_predict, is_name_value_option, matrix_property,
    nonnegative_integer, numeric_property, numeric_vector, positive_integer, scalar_text,
    string_column, string_property, text_list, tree_children_tensor, tree_class_tensor,
    tree_predictor_tensor, tree_prob_tensor, tree_threshold_tensor, vector_values,
    vector_values_predict, x_value,
};

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
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::classification_tree"
)]
async fn fitctree_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_fitctree_extensions(&first, &rest)?;
    let first = gather(first).await?;
    let rest = gather_all(rest).await?;
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
        if typed_integer_value(&pair[1])
            && FitctreeOption::from_public_name(&name) != Some(FitctreeOption::ClassNames)
        {
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
        match FitctreeOption::from_public_name(&name) {
            Some(FitctreeOption::ResponseName) if !external_y => {
                if let Some(value) = scalar_text(&pair[1]) {
                    response = value;
                }
            }
            Some(FitctreeOption::PredictorNames) => {
                predictors = text_list(&pair[1], "PredictorNames").ok()
            }
            Some(FitctreeOption::Weights) => weight_variable = scalar_text(&pair[1]),
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
    gather_if_needed_async(&value).await
}

async fn gather_all(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(value).await?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
