//! Confusion matrix compatibility surface.

use std::{cmp::Ordering, collections::HashMap};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, IntValue, IntegerStorage, LogicalArray, NumericDType, ObjectInstance,
    ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "confusionmat";
const MAX_CONFUSIONMAT_CELLS: usize = 25_000_000;

pub const CONFUSIONMAT_INTEGER_GROUP_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "confusionmat-integer-group",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "confusionmat with typed-integer group labels is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ConfusionmatIntegerGroupExtension"),
    };
pub const CONFUSIONMAT_INTEGER_GROUPHAT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "confusionmat-integer-grouphat",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "confusionmat with typed-integer grouphat labels is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ConfusionmatIntegerGrouphatExtension"),
    };
pub const CONFUSIONMAT_INTEGER_ORDER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "confusionmat-integer-order",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "confusionmat with typed-integer Order labels is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ConfusionmatIntegerOrderExtension"),
    };
pub const CONFUSIONMAT_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "confusionmat-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "confusionmat with resident inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ConfusionmatResidentInputExtension"),
    };
pub const CONFUSIONMAT_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    CONFUSIONMAT_INTEGER_GROUP_EXTENSION,
    CONFUSIONMAT_INTEGER_GROUPHAT_EXTENSION,
    CONFUSIONMAT_INTEGER_ORDER_EXTENSION,
    CONFUSIONMAT_RESIDENT_INPUT_EXTENSION,
];

const INTEGER_GROUP_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "group",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
    notes: "Integer labels are exact and must use the same class as grouphat and Order.",
}];
const INTEGER_GROUPHAT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "grouphat",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Integer predicted labels are exact and must use the same class as group and Order.",
    }];
const INTEGER_ORDER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Order",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
    notes: "Integer Order labels are deduplicated and matched without floating conversion.",
}];
pub const CONFUSIONMAT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "C = confusionmat(integer_group, grouphat)", inputs: &INTEGER_GROUP_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat mode groups and sorts all eight integer classes exactly; C contains double counts." },
    BuiltinIntegerCapabilityDescriptor { form: "C = confusionmat(group, integer_grouphat)", inputs: &INTEGER_GROUPHAT_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "This role is independently gated before gather; integer label classes must match." },
    BuiltinIntegerCapabilityDescriptor { form: "[C, order] = confusionmat(integer_group, integer_grouphat, 'Order', integer_order)", inputs: &INTEGER_ORDER_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "C contains double counts; the optional order output retains the exact integer label class and values, including values above flintmax." },
];

const OUTPUT_C: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Confusion matrix with true labels in rows and predicted labels in columns.",
};

const OUTPUT_ORDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "order",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Class labels corresponding to rows and columns of C.",
};

const PARAM_GROUP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "group",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True class labels.",
};

const PARAM_GROUPHAT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "grouphat",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predicted class labels.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as Order.",
};

const INPUTS_BASIC: [BuiltinParamDescriptor; 2] = [PARAM_GROUP, PARAM_GROUPHAT];
const INPUTS_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_GROUP, PARAM_GROUPHAT, PARAM_OPTIONS];
const OUTPUTS_C: [BuiltinParamDescriptor; 1] = [OUTPUT_C];
const OUTPUTS_ALL: [BuiltinParamDescriptor; 2] = [OUTPUT_C, OUTPUT_ORDER];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "C = confusionmat(group, grouphat)",
        inputs: &INPUTS_BASIC,
        outputs: &OUTPUTS_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = confusionmat(group, grouphat, Name, Value)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUTS_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C,order] = confusionmat(___)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUTS_ALL,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONFUSIONMAT.INVALID_ARGUMENT",
    identifier: Some("RunMat:confusionmat:InvalidArgument"),
    when: "Label vectors, dimensions, order labels, or name-value options are malformed.",
    message: "confusionmat: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONFUSIONMAT.INTERNAL",
    identifier: Some("RunMat:confusionmat:Internal"),
    when: "RunMat cannot allocate or construct confusion matrix outputs.",
    message: "confusionmat: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const CONFUSIONMAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn confusionmat_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![None, None]),
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

fn internal(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LabelKind {
    Numeric,
    Integer(NumericDType),
    String,
    Char,
    Cell,
    Logical,
    Categorical,
}

#[derive(Clone, Debug, PartialEq)]
enum Label {
    Numeric(f64),
    Integer(IntValue),
    Text(String),
    Logical(bool),
}

#[derive(Clone, Debug)]
struct LabelVector {
    labels: Vec<Label>,
    kind: LabelKind,
    categories: Option<Vec<String>>,
}

#[runtime_builtin(
    name = "confusionmat",
    category = "stats/ml",
    summary = "Compute a confusion matrix from true and predicted class labels.",
    keywords = "confusionmat,confusion matrix,classification,statistics,machine learning",
    type_resolver(confusionmat_type),
    extensions(CONFUSIONMAT_EXTENSIONS),
    integer_capabilities(CONFUSIONMAT_INTEGER_CAPABILITIES),
    descriptor(crate::builtins::stats::ml::confusionmat::CONFUSIONMAT_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::confusionmat"
)]
async fn confusionmat_builtin(
    group: Value,
    grouphat: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    if let Some(0) = requested_outputs {
        return Ok(Value::OutputList(Vec::new()));
    }
    if matches!(requested_outputs, Some(count) if count > 2) {
        return Err(invalid("confusionmat: too many output arguments"));
    }
    ensure_confusionmat_extensions(&group, &grouphat, &rest)?;

    let group = gather(group).await?;
    let grouphat = gather(grouphat).await?;
    let rest = gather_values(rest).await?;
    let outputs =
        confusionmat_compute(group, grouphat, rest, requested_outputs.unwrap_or(1).max(1))?;
    match requested_outputs {
        Some(1) => Ok(Value::OutputList(vec![outputs[0].clone()])),
        Some(count) => Ok(crate::output_count::output_list_with_padding(
            count, outputs,
        )),
        None => Ok(outputs[0].clone()),
    }
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("confusionmat: {err}")))
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(value).await?);
    }
    Ok(out)
}

fn ensure_confusionmat_extensions(
    group: &Value,
    grouphat: &Value,
    rest: &[Value],
) -> BuiltinResult<()> {
    if is_typed_integer_value(group) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONFUSIONMAT_INTEGER_GROUP_EXTENSION,
            NAME,
        )?;
    }
    if is_typed_integer_value(grouphat) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONFUSIONMAT_INTEGER_GROUPHAT_EXTENSION,
            NAME,
        )?;
    }
    for pair in rest.chunks_exact(2) {
        if scalar_text(&pair[0], "option name").is_ok_and(|name| canonical(&name) == "order")
            && is_typed_integer_value(&pair[1])
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CONFUSIONMAT_INTEGER_ORDER_EXTENSION,
                NAME,
            )?;
        }
    }
    if matches!(group, Value::GpuTensor(_))
        || matches!(grouphat, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONFUSIONMAT_RESIDENT_INPUT_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn confusionmat_compute(
    group: Value,
    grouphat: Value,
    rest: Vec<Value>,
    output_limit: usize,
) -> BuiltinResult<Vec<Value>> {
    let group = labels_from_value(group, "group")?;
    let grouphat = labels_from_value(grouphat, "grouphat")?;
    if group.labels.len() != grouphat.labels.len() {
        return Err(invalid(
            "confusionmat: group and grouphat must contain the same number of labels",
        ));
    }
    if !label_kinds_compatible(group.kind, grouphat.kind) {
        return Err(invalid(
            "confusionmat: group and grouphat label types must be compatible",
        ));
    }
    let order = parse_options(rest)?;
    let (order, order_kind) = if let Some(order) = order {
        if !label_kinds_compatible(group.kind, order.kind) {
            return Err(invalid(
                "confusionmat: Order labels must match group label type",
            ));
        }
        let order_kind = order.kind;
        let order = dedupe_labels(order.labels);
        ensure_order_covers(&order, &group.labels, &grouphat.labels)?;
        (order, order_kind)
    } else {
        (inferred_order(&group, &grouphat), group.kind)
    };
    let class_count = order.len();
    let cell_count = class_count
        .checked_mul(class_count)
        .ok_or_else(|| invalid("confusionmat: too many classes"))?;
    if cell_count > MAX_CONFUSIONMAT_CELLS {
        return Err(invalid("confusionmat: too many classes"));
    }
    let mut index = HashMap::with_capacity(order.len());
    for (idx, label) in order.iter().enumerate() {
        index.insert(label_key(label), idx);
    }
    let mut counts = vec![0.0; cell_count];
    for (actual, predicted) in group.labels.iter().zip(grouphat.labels.iter()) {
        if is_missing_label(actual) || is_missing_label(predicted) {
            continue;
        }
        let row = *index
            .get(&label_key(actual))
            .ok_or_else(|| internal("confusionmat: missing actual label in order"))?;
        let col = *index
            .get(&label_key(predicted))
            .ok_or_else(|| internal("confusionmat: missing predicted label in order"))?;
        counts[row + col * class_count] += 1.0;
    }
    let matrix = Tensor::new(counts, vec![class_count, class_count])
        .map_err(|err| internal(format!("confusionmat: {err}")))?;
    let mut outputs = vec![Value::Tensor(matrix)];
    if output_limit >= 2 {
        outputs.push(labels_to_value(&order, order_kind)?);
    }
    Ok(outputs)
}

fn parse_options(rest: Vec<Value>) -> BuiltinResult<Option<LabelVector>> {
    if rest.is_empty() {
        return Ok(None);
    }
    if !rest.len().is_multiple_of(2) {
        return Err(invalid("confusionmat: name-value options must be paired"));
    }
    let mut order = None;
    let mut index = 0usize;
    while index < rest.len() {
        let name = scalar_text(&rest[index], "option name")?;
        match canonical(&name).as_str() {
            "order" => {
                order = Some(labels_from_value(rest[index + 1].clone(), "Order")?);
            }
            other => {
                return Err(invalid(format!(
                    "confusionmat: unsupported option '{other}'"
                )))
            }
        }
        index += 2;
    }
    if let Some(order) = &order {
        if order.labels.iter().any(is_missing_label) {
            return Err(invalid("confusionmat: Order cannot contain missing labels"));
        }
        if order.labels.is_empty() {
            return Err(invalid(
                "confusionmat: Order must contain at least one label",
            ));
        }
    }
    Ok(order)
}

fn labels_from_value(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    match value {
        Value::Tensor(tensor) => {
            ensure_vector_shape(&tensor.shape, name)?;
            if let Some(storage) = tensor.integer_storage() {
                Ok(LabelVector { labels: storage.exact_values().into_iter().map(Label::Integer).collect(), kind: LabelKind::Integer(storage.numeric_dtype()), categories: None })
            } else {
                Ok(LabelVector { labels: vector_values(&tensor, name)?.into_iter().map(Label::Numeric).collect(), kind: LabelKind::Numeric, categories: None })
            }
        }
        Value::Num(value) => Ok(LabelVector {
            labels: vec![Label::Numeric(value)],
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Int(value) => Ok(LabelVector {
            kind: LabelKind::Integer(int_value_dtype(&value)),
            labels: vec![Label::Integer(value)],
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
            "confusionmat: {name} must be numeric, logical, string, character, categorical, or cellstr labels; got {other:?}"
        ))),
    }
}

fn vector_values(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    ensure_vector_shape(&tensor.shape, name)?;
    Ok(tensor::tensor_values_f64(tensor))
}

fn ensure_vector_shape(shape: &[usize], name: &str) -> BuiltinResult<()> {
    if shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid(format!("confusionmat: {name} must be a vector")));
    }
    Ok(())
}

fn inferred_order(group: &LabelVector, grouphat: &LabelVector) -> Vec<Label> {
    match group.kind {
        LabelKind::Logical => vec![Label::Logical(false), Label::Logical(true)],
        LabelKind::Categorical => categorical_order(group, grouphat),
        kind => {
            let mut combined = Vec::with_capacity(group.labels.len() + grouphat.labels.len());
            combined.extend(group.labels.iter().cloned());
            combined.extend(grouphat.labels.iter().cloned());
            let mut out = dedupe_labels(combined);
            if matches!(kind, LabelKind::Numeric | LabelKind::Integer(_)) {
                sort_labels(&mut out, kind);
            }
            out
        }
    }
}

fn categorical_order(group: &LabelVector, grouphat: &LabelVector) -> Vec<Label> {
    let mut out = Vec::new();
    for category in group
        .categories
        .iter()
        .chain(grouphat.categories.iter())
        .flatten()
    {
        let label = Label::Text(category.clone());
        if !is_missing_label(&label)
            && group_or_grouphat_contains(&label, group, grouphat)
            && !out.iter().any(|existing| same_label(existing, &label))
        {
            out.push(label);
        }
    }
    for label in group.labels.iter().chain(grouphat.labels.iter()) {
        if !is_missing_label(label) && !out.iter().any(|existing| same_label(existing, label)) {
            out.push(label.clone());
        }
    }
    out
}

fn group_or_grouphat_contains(label: &Label, group: &LabelVector, grouphat: &LabelVector) -> bool {
    group
        .labels
        .iter()
        .chain(grouphat.labels.iter())
        .any(|candidate| same_label(candidate, label))
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
        LabelKind::Integer(_) => labels.sort_by(|left, right| match (left, right) {
            (Label::Integer(a), Label::Integer(b)) => integer_sort_key(a).cmp(&integer_sort_key(b)),
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

fn ensure_order_covers(order: &[Label], group: &[Label], grouphat: &[Label]) -> BuiltinResult<()> {
    for label in group.iter().chain(grouphat.iter()) {
        if is_missing_label(label) {
            continue;
        }
        if !order
            .iter()
            .any(|order_label| same_label(order_label, label))
        {
            return Err(invalid(
                "confusionmat: Order must contain all nonmissing labels in group and grouphat",
            ));
        }
    }
    Ok(())
}

fn label_key(label: &Label) -> String {
    match label {
        Label::Numeric(value) if *value == 0.0 => "n:0".to_string(),
        Label::Numeric(value) => format!("n:{:016x}", value.to_bits()),
        Label::Integer(value) => format!("i:{value:?}"),
        Label::Text(value) => format!("t:{value}"),
        Label::Logical(value) => format!("l:{}", usize::from(*value)),
    }
}

fn same_label(left: &Label, right: &Label) -> bool {
    match (left, right) {
        (Label::Numeric(a), Label::Numeric(b)) => (a == b) || (a.is_nan() && b.is_nan()),
        (Label::Integer(a), Label::Integer(b)) => a == b,
        (Label::Text(a), Label::Text(b)) => a == b,
        (Label::Logical(a), Label::Logical(b)) => a == b,
        _ => false,
    }
}

fn is_missing_label(label: &Label) -> bool {
    match label {
        Label::Numeric(value) => value.is_nan(),
        Label::Integer(_) => false,
        Label::Text(value) => value.is_empty() || value == "<missing>",
        Label::Logical(_) => false,
    }
}

fn label_kinds_compatible(left: LabelKind, right: LabelKind) -> bool {
    left == right || (is_text_kind(left) && is_text_kind(right))
}

fn is_text_kind(kind: LabelKind) -> bool {
    matches!(
        kind,
        LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical
    )
}

fn labels_to_value(labels: &[Label], kind: LabelKind) -> BuiltinResult<Value> {
    match kind {
        LabelKind::Numeric => Ok(Value::Tensor(
            Tensor::new(
                labels
                    .iter()
                    .map(|label| match label {
                        Label::Numeric(value) => *value,
                        _ => f64::NAN,
                    })
                    .collect(),
                vec![labels.len(), 1],
            )
            .map_err(|err| internal(format!("confusionmat: {err}")))?,
        )),
        LabelKind::Integer(dtype) => {
            let values = labels
                .iter()
                .map(|label| match label {
                    Label::Integer(value) => Ok(value.clone()),
                    _ => Err(internal("confusionmat: noninteger label in integer order")),
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            let storage = integer_storage_from_values(dtype, values)?;
            Ok(Value::Tensor(
                Tensor::new_integer(storage, vec![labels.len(), 1])
                    .map_err(|err| internal(format!("confusionmat: {err}")))?,
            ))
        }
        LabelKind::Logical => Ok(Value::LogicalArray(
            LogicalArray::new(
                labels
                    .iter()
                    .map(|label| match label {
                        Label::Logical(true) => 1,
                        _ => 0,
                    })
                    .collect(),
                vec![labels.len(), 1],
            )
            .map_err(|err| internal(format!("confusionmat: {err}")))?,
        )),
        LabelKind::String => Ok(Value::StringArray(
            StringArray::new(text_labels(labels), vec![labels.len(), 1])
                .map_err(|err| internal(format!("confusionmat: {err}")))?,
        )),
        LabelKind::Char => Ok(Value::CharArray(char_array_from_rows(&text_labels(
            labels,
        ))?)),
        LabelKind::Cell => Ok(Value::Cell(
            CellArray::new(
                text_labels(labels)
                    .into_iter()
                    .map(|label| Value::CharArray(CharArray::new_row(&label)))
                    .collect(),
                labels.len(),
                1,
            )
            .map_err(|err| internal(format!("confusionmat: {err}")))?,
        )),
        LabelKind::Categorical => {
            let values = text_labels(labels);
            let value_array = Value::StringArray(
                StringArray::new(values.clone(), vec![labels.len(), 1])
                    .map_err(|err| internal(format!("confusionmat: {err}")))?,
            );
            let categories = Value::StringArray(
                StringArray::new(values, vec![1, labels.len()])
                    .map_err(|err| internal(format!("confusionmat: {err}")))?,
            );
            crate::builtins::table::categorical_from_args(vec![value_array, categories])
        }
    }
}

fn int_value_dtype(value: &IntValue) -> NumericDType {
    IntegerStorage::from_scalar(value.clone()).numeric_dtype()
}

fn integer_sort_key(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(v) => *v as i128,
        IntValue::I16(v) => *v as i128,
        IntValue::I32(v) => *v as i128,
        IntValue::I64(v) => *v as i128,
        IntValue::U8(v) => *v as i128,
        IntValue::U16(v) => *v as i128,
        IntValue::U32(v) => *v as i128,
        IntValue::U64(v) => *v as i128,
    }
}

fn integer_storage_from_values(
    dtype: NumericDType,
    values: Vec<IntValue>,
) -> BuiltinResult<IntegerStorage> {
    macro_rules! collect_variant {
        ($variant:ident, $ty:ty) => {{
            values
                .into_iter()
                .map(|value| match value {
                    IntValue::$variant(value) => Ok(value),
                    _ => Err(internal("confusionmat: integer order class mismatch")),
                })
                .collect::<BuiltinResult<Vec<$ty>>>()
                .map(IntegerStorage::$variant)
        }};
    }
    match dtype {
        NumericDType::I8 => collect_variant!(I8, i8),
        NumericDType::I16 => collect_variant!(I16, i16),
        NumericDType::I32 => collect_variant!(I32, i32),
        NumericDType::I64 => collect_variant!(I64, i64),
        NumericDType::U8 => collect_variant!(U8, u8),
        NumericDType::U16 => collect_variant!(U16, u16),
        NumericDType::U32 => collect_variant!(U32, u32),
        NumericDType::U64 => collect_variant!(U64, u64),
        NumericDType::F32 | NumericDType::F64 => {
            Err(internal("confusionmat: floating dtype in integer order"))
        }
    }
}

fn categorical_categories(object: &ObjectInstance) -> BuiltinResult<Option<Vec<String>>> {
    match object.properties.get("Categories") {
        Some(Value::StringArray(array)) => Ok(Some(array.data.clone())),
        Some(_) => Err(invalid(
            "confusionmat: categorical Categories must be strings",
        )),
        None => Ok(None),
    }
}

fn ensure_categorical_vector_shape(object: &ObjectInstance, name: &str) -> BuiltinResult<()> {
    match object.properties.get("Codes") {
        Some(Value::Tensor(tensor)) => ensure_vector_shape(&tensor.shape, name),
        Some(_) => Err(invalid("confusionmat: categorical Codes must be numeric")),
        None => Ok(()),
    }
}

fn text_labels(labels: &[Label]) -> Vec<String> {
    labels
        .iter()
        .map(|label| match label {
            Label::Text(value) => value.clone(),
            _ => String::new(),
        })
        .collect()
}

fn char_rows(array: &CharArray) -> Vec<String> {
    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        rows.push(array.data[start..start + array.cols].iter().collect());
    }
    rows
}

fn char_array_from_rows(rows: &[String]) -> BuiltinResult<CharArray> {
    let width = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(rows.len() * width);
    for row in rows {
        let mut chars = row.chars().collect::<Vec<_>>();
        chars.resize(width, ' ');
        data.extend(chars);
    }
    CharArray::new(data, rows.len(), width).map_err(|err| internal(format!("confusionmat: {err}")))
}

fn scalar_text(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(invalid(format!(
            "confusionmat: {name} must be a string scalar, got {other:?}"
        ))),
    }
}

fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).unwrap())
    }

    fn with_outputs<T>(count: usize, f: impl FnOnce() -> T) -> T {
        let _guard = crate::output_count::push_output_count(Some(count));
        f()
    }

    fn confusionmat(group: Value, grouphat: Value, rest: Vec<Value>, outputs: usize) -> Vec<Value> {
        with_outputs(outputs, || {
            let Value::OutputList(values) =
                block_on(confusionmat_builtin(group, grouphat, rest)).expect("confusionmat")
            else {
                panic!("expected output list");
            };
            values
        })
    }

    #[test]
    fn counts_numeric_labels_in_sorted_order() {
        let values = confusionmat(
            tensor(vec![2.0, 1.0, 2.0, 3.0], vec![4, 1]),
            tensor(vec![2.0, 2.0, 1.0, 3.0], vec![4, 1]),
            Vec::new(),
            2,
        );
        let Value::Tensor(c) = &values[0] else {
            panic!("matrix");
        };
        assert_eq!(c.shape, vec![3, 3]);
        assert_eq!(
            c.materialize_f64(),
            vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        let Value::Tensor(order) = &values[1] else {
            panic!("order");
        };
        assert_eq!(order.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn typed_integer_labels_cover_all_classes_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cases = [
            IntegerStorage::I8(vec![2, 1, 2]),
            IntegerStorage::I16(vec![2, 1, 2]),
            IntegerStorage::I32(vec![2, 1, 2]),
            IntegerStorage::I64(vec![2, 1, 2]),
            IntegerStorage::U8(vec![2, 1, 2]),
            IntegerStorage::U16(vec![2, 1, 2]),
            IntegerStorage::U32(vec![2, 1, 2]),
            IntegerStorage::U64(vec![2, 1, 2]),
        ];
        for storage in cases {
            let values = confusionmat(
                int_tensor(storage.clone(), vec![3, 1]),
                int_tensor(storage.clone(), vec![3, 1]),
                Vec::new(),
                2,
            );
            let Value::Tensor(c) = &values[0] else {
                panic!("matrix");
            };
            assert_eq!(c.materialize_f64(), vec![1.0, 0.0, 0.0, 2.0]);
            let Value::Tensor(order) = &values[1] else {
                panic!("order");
            };
            assert_eq!(
                order.integer_storage().map(IntegerStorage::numeric_dtype),
                Some(storage.numeric_dtype())
            );
            assert_eq!(order.integer_storage().unwrap().exact_values().len(), 2);
        }
    }

    #[test]
    fn typed_integer_labels_reject_mixed_classes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = block_on(confusionmat_builtin(
            int_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]),
            int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message.contains("label types must be compatible"));
    }

    #[test]
    fn uint64_labels_above_flintmax_remain_distinct() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let a = 9_007_199_254_740_993_u64;
        let b = u64::MAX;
        let values = confusionmat(
            int_tensor(IntegerStorage::U64(vec![b, a, b]), vec![3, 1]),
            int_tensor(IntegerStorage::U64(vec![a, a, b]), vec![3, 1]),
            Vec::new(),
            2,
        );
        let Value::Tensor(c) = &values[0] else {
            panic!("matrix");
        };
        assert_eq!(c.materialize_f64(), vec![1.0, 1.0, 0.0, 1.0]);
        let Value::Tensor(order) = &values[1] else {
            panic!("order");
        };
        assert_eq!(
            order.integer_storage(),
            Some(&IntegerStorage::U64(vec![a, b]))
        );
    }

    #[test]
    fn typed_integer_roles_are_independently_compatibility_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let group_err = block_on(confusionmat_builtin(
            Value::Int(IntValue::I8(1)),
            Value::Int(IntValue::I8(1)),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            group_err.identifier(),
            CONFUSIONMAT_INTEGER_GROUP_EXTENSION.error_identifier
        );
        let grouphat_err = block_on(confusionmat_builtin(
            Value::Num(1.0),
            Value::Int(IntValue::I8(1)),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            grouphat_err.identifier(),
            CONFUSIONMAT_INTEGER_GROUPHAT_EXTENSION.error_identifier
        );
        let order_err = block_on(confusionmat_builtin(
            Value::Num(1.0),
            Value::Num(1.0),
            vec![Value::from("Order"), Value::Int(IntValue::I8(1))],
        ))
        .unwrap_err();
        assert_eq!(
            order_err.identifier(),
            CONFUSIONMAT_INTEGER_ORDER_EXTENSION.error_identifier
        );
    }

    #[test]
    fn resident_integer_labels_gate_before_gather_and_remain_exact_in_runmat_mode() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                vec![2, 1],
            )
            .unwrap();
            let group = gpu_helpers::upload_tensor(provider, &tensor).expect("group upload");
            let grouphat = gpu_helpers::upload_tensor(provider, &tensor).expect("grouphat upload");
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let err = block_on(confusionmat_builtin(
                    Value::GpuTensor(group.clone()),
                    Value::GpuTensor(grouphat.clone()),
                    Vec::new(),
                ))
                .unwrap_err();
                assert_eq!(
                    err.identifier(),
                    CONFUSIONMAT_INTEGER_GROUP_EXTENSION.error_identifier
                );
            }
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let values = confusionmat(
                Value::GpuTensor(group),
                Value::GpuTensor(grouphat),
                Vec::new(),
                2,
            );
            let Value::Tensor(order) = &values[1] else {
                panic!("order");
            };
            assert_eq!(order.integer_storage(), tensor.integer_storage());
        });
    }

    #[test]
    fn resident_floating_labels_use_the_resident_extension_gate() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let data = tensor.materialize_f64();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &data,
                    shape: &tensor.shape,
                })
                .expect("upload");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = block_on(confusionmat_builtin(
                Value::GpuTensor(handle.clone()),
                Value::GpuTensor(handle),
                Vec::new(),
            ))
            .unwrap_err();
            assert_eq!(
                err.identifier(),
                CONFUSIONMAT_RESIDENT_INPUT_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn honors_explicit_order_and_ignores_missing_labels() {
        let values = confusionmat(
            tensor(vec![1.0, f64::NAN, 2.0, 3.0], vec![4, 1]),
            tensor(vec![2.0, 1.0, 2.0, 1.0], vec![4, 1]),
            vec![
                Value::from("Order"),
                tensor(vec![3.0, 2.0, 1.0], vec![3, 1]),
            ],
            2,
        );
        let Value::Tensor(c) = &values[0] else {
            panic!("matrix");
        };
        assert_eq!(c.shape, vec![3, 3]);
        assert_eq!(
            c.materialize_f64(),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        );
        let Value::Tensor(order) = &values[1] else {
            panic!("order");
        };
        assert_eq!(order.materialize_f64(), vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn supports_string_and_cell_text_labels() {
        let group = Value::StringArray(
            StringArray::new(
                vec!["dog".into(), "cat".into(), "dog".into(), "<missing>".into()],
                vec![4, 1],
            )
            .unwrap(),
        );
        let predicted = Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(CharArray::new_row("dog")),
                    Value::CharArray(CharArray::new_row("dog")),
                    Value::CharArray(CharArray::new_row("cat")),
                    Value::CharArray(CharArray::new_row("cat")),
                ],
                4,
                1,
            )
            .unwrap(),
        );
        let values = confusionmat(group, predicted, Vec::new(), 2);
        let Value::Tensor(c) = &values[0] else {
            panic!("matrix");
        };
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.materialize_f64(), vec![1.0, 1.0, 1.0, 0.0]);
        let Value::StringArray(order) = &values[1] else {
            panic!("order");
        };
        assert_eq!(order.data, vec!["dog", "cat"]);
    }

    #[test]
    fn inferred_logical_order_contains_false_and_true() {
        let values = confusionmat(Value::Bool(true), Value::Bool(true), Vec::new(), 2);
        let Value::Tensor(c) = &values[0] else {
            panic!("matrix");
        };
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.materialize_f64(), vec![0.0, 0.0, 0.0, 1.0]);
        let Value::LogicalArray(order) = &values[1] else {
            panic!("order");
        };
        assert_eq!(order.data, vec![0, 1]);
    }

    #[test]
    fn explicit_order_must_cover_all_observed_labels() {
        let err = block_on(confusionmat_builtin(
            tensor(vec![1.0, 2.0], vec![2, 1]),
            tensor(vec![1.0, 3.0], vec![2, 1]),
            vec![Value::from("Order"), tensor(vec![1.0, 2.0], vec![2, 1])],
        ))
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:confusionmat:InvalidArgument")
        );
        assert!(err
            .message
            .contains("Order must contain all nonmissing labels"));
    }

    #[test]
    fn supports_categorical_labels_in_category_order() {
        let categories = Value::StringArray(
            StringArray::new(
                vec!["low".into(), "medium".into(), "high".into()],
                vec![1, 3],
            )
            .unwrap(),
        );
        let group = crate::builtins::table::categorical_from_args(vec![
            Value::StringArray(
                StringArray::new(vec!["high".into(), "low".into(), "high".into()], vec![3, 1])
                    .unwrap(),
            ),
            categories.clone(),
        ])
        .unwrap();
        let predicted = crate::builtins::table::categorical_from_args(vec![
            Value::StringArray(
                StringArray::new(
                    vec!["low".into(), "medium".into(), "high".into()],
                    vec![3, 1],
                )
                .unwrap(),
            ),
            categories,
        ])
        .unwrap();

        let values = confusionmat(group, predicted, Vec::new(), 2);
        let Value::Tensor(c) = &values[0] else {
            panic!("matrix");
        };
        assert_eq!(c.shape, vec![3, 3]);
        assert_eq!(
            c.materialize_f64(),
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        let Value::Object(order) = &values[1] else {
            panic!("categorical order");
        };
        assert!(order.is_class("categorical"));
        let order_labels =
            crate::builtins::table::categorical_labels(&Value::Object(order.clone())).unwrap();
        assert_eq!(order_labels, vec!["low", "medium", "high"]);
    }

    #[test]
    fn rejects_bad_shapes_and_options() {
        let err = block_on(confusionmat_builtin(
            tensor(vec![1.0, 2.0], vec![2, 1]),
            tensor(vec![1.0], vec![1, 1]),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:confusionmat:InvalidArgument")
        );

        let err = block_on(confusionmat_builtin(
            tensor(vec![1.0], vec![1, 1]),
            tensor(vec![1.0], vec![1, 1]),
            vec![Value::from("Rows"), Value::from("all")],
        ))
        .unwrap_err();
        assert!(err.message.contains("unsupported option"));

        let err = block_on(confusionmat_builtin(
            Value::StringArray(
                StringArray::new(
                    vec!["a".into(), "b".into(), "c".into(), "d".into()],
                    vec![2, 2],
                )
                .unwrap(),
            ),
            Value::StringArray(
                StringArray::new(
                    vec!["a".into(), "b".into(), "c".into(), "d".into()],
                    vec![2, 2],
                )
                .unwrap(),
            ),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message.contains("must be a vector"));
    }
}
