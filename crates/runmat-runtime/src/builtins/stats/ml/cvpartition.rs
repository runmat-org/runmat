//! Cross-validation partition helpers for Statistics and Machine Learning workflows.

use std::collections::BTreeMap;

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
use runmat_value::{LogicalArray, ObjectInstance, StringArray, Tensor, Value};

use crate::builtins::common::{random, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const CVPARTITION_NAME: &str = "cvpartition";
const CVPARTITION_CLASS: &str = "cvpartition";
const TEST_SETS_PROPERTY: &str = "__RunMatCvPartitionTestSets";
const ACTIVE_ROWS_PROPERTY: &str = "__RunMatCvPartitionActiveRows";
const MAX_MATERIALIZED_MASK_CELLS: usize = 10_000_000;
const EPS: f64 = 1.0e-12;

pub const CVPARTITION_INTEGER_N_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-integer-observation-count",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cvpartition with a typed-integer observation count is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionIntegerObservationCountExtension"),
    };
pub const CVPARTITION_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-integer-partition-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "cvpartition with typed-integer KFold or Holdout control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionIntegerPartitionControlExtension"),
    };
pub const CVPARTITION_INTEGER_STRATIFICATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-integer-stratification",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cvpartition with typed-integer stratification labels is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionIntegerStratificationExtension"),
    };
pub const CVPARTITION_INTEGER_CUSTOM_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-integer-custom-testsets",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cvpartition with typed-integer custom test sets is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionIntegerCustomTestsetsExtension"),
    };
pub const CVPARTITION_BOOLEAN_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-nonlogical-stratify-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cvpartition with a nonlogical Stratify option is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionNonlogicalStratifyOptionExtension"),
    };
pub const CVPARTITION_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    CVPARTITION_INTEGER_N_EXTENSION,
    CVPARTITION_INTEGER_CONTROL_EXTENSION,
    CVPARTITION_INTEGER_STRATIFICATION_EXTENSION,
    CVPARTITION_INTEGER_CUSTOM_EXTENSION,
    CVPARTITION_BOOLEAN_ALIAS_EXTENSION,
];

const INTEGER_N_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "n", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The public observation count lists single and double; all eight typed integer scalar classes are independently gated and read exactly." }];
const INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "kOrP", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Typed KFold and integer-count Holdout controls are independently gated; documented single/double integer-valued controls remain accepted." }];
const INTEGER_STRAT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "stratvar", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "The public numeric stratification class list is single/double; RunMat mode additionally preserves all eight integer label classes through structural grouping." }];
const INTEGER_CUSTOM_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "testSets", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "The public custom-test-set class list is single/double/logical; typed integer assignment vectors or 0/1 masks are independently gated." }];
pub const CVPARTITION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "c = cvpartition(integer_n, ___)", inputs: &INTEGER_N_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Typed n determines object and mask sizes only; public count properties remain double and mask properties remain logical." },
    BuiltinIntegerCapabilityDescriptor { form: "c = cvpartition(___, integer_kOrP)", inputs: &INTEGER_CONTROL_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Typed partition controls are validated exactly before random assignment and do not determine property classes." },
    BuiltinIntegerCapabilityDescriptor { form: "c = cvpartition(integer_stratvar, ___)", inputs: &INTEGER_STRAT_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer labels are compared structurally from authoritative storage and never treated as floating observations." },
    BuiltinIntegerCapabilityDescriptor { form: "c = cvpartition(CustomPartition=integer_testSets)", inputs: &INTEGER_CUSTOM_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer test-set identifiers and 0/1 masks are interpreted exactly before logical mask construction." },
];

const OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cross-validation partition object.",
}];

const OUTPUT_MASK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mask",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical observation mask for one or more test sets.",
}];

const PARAM_N_OR_STRATVAR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nOrStratvar",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation count or stratification variable.",
};

const PARAM_KIND: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "kind",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Partition kind such as KFold, Holdout, Leaveout, Resubstitution, or CustomPartition.",
};

const PARAM_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Partition parameter such as fold count, holdout fraction, holdout count, or custom test sets.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options including Stratify.",
};

const PARAM_C: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "cvpartition object.",
};

const PARAM_INDEX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "i",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "One-based test-set index, vector of indices, or 'all'.",
};

const CVPARTITION_INPUTS: [BuiltinParamDescriptor; 3] =
    [PARAM_N_OR_STRATVAR, PARAM_KIND, PARAM_VALUE];
const CVPARTITION_INPUTS_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_N_OR_STRATVAR, PARAM_KIND, PARAM_VALUE, PARAM_OPTIONS];
const MASK_INPUTS_C: [BuiltinParamDescriptor; 1] = [PARAM_C];
const MASK_INPUTS_C_I: [BuiltinParamDescriptor; 2] = [PARAM_C, PARAM_INDEX];

const CVPARTITION_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "c = cvpartition(n, 'KFold', k)",
        inputs: &CVPARTITION_INPUTS,
        outputs: &OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "c = cvpartition(stratvar, 'KFold', k, 'Stratify', tf)",
        inputs: &CVPARTITION_INPUTS_OPTIONS,
        outputs: &OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "c = cvpartition(n, 'Holdout', p)",
        inputs: &CVPARTITION_INPUTS,
        outputs: &OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "c = cvpartition(n, 'Leaveout')",
        inputs: &[PARAM_N_OR_STRATVAR, PARAM_KIND],
        outputs: &OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "c = cvpartition('CustomPartition', testSets)",
        inputs: &[PARAM_KIND, PARAM_VALUE],
        outputs: &OUTPUT_C,
    },
];

const MASK_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "mask = training(c)",
        inputs: &MASK_INPUTS_C,
        outputs: &OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "mask = training(c, i)",
        inputs: &MASK_INPUTS_C_I,
        outputs: &OUTPUT_MASK,
    },
];

const TEST_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "mask = test(c)",
        inputs: &MASK_INPUTS_C,
        outputs: &OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "mask = test(c, i)",
        inputs: &MASK_INPUTS_C_I,
        outputs: &OUTPUT_MASK,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CVPARTITION.INVALID_ARGUMENT",
    identifier: Some("RunMat:cvpartition:InvalidArgument"),
    when: "Inputs, partition kinds, dimensions, indices, or name-value options are malformed.",
    message: "cvpartition: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CVPARTITION.INTERNAL",
    identifier: Some("RunMat:cvpartition:Internal"),
    when: "RunMat cannot allocate or construct a partition output.",
    message: "cvpartition: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const CVPARTITION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CVPARTITION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TRAINING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MASK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TEST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEST_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn object_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn logical_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Logical { shape: None }
}

fn cvpartition_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(CVPARTITION_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    cvpartition_error(message, &ERROR_INVALID_ARGUMENT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    cvpartition_error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PartitionKind {
    KFold,
    Holdout,
    Leaveout,
    Resubstitution,
    Custom,
}

impl PartitionKind {
    fn property_name(self) -> &'static str {
        match self {
            Self::KFold => "kfold",
            Self::Holdout => "holdout",
            Self::Leaveout => "leaveout",
            Self::Resubstitution => "resubstitution",
            Self::Custom => "custom",
        }
    }
}

#[derive(Clone, Debug)]
struct CvOptions {
    stratify: Option<bool>,
}

#[derive(Clone, Debug)]
struct PartitionSpec {
    kind: PartitionKind,
    n: usize,
    test_sets: Vec<u8>,
    test_set_count: usize,
    active_rows: Vec<u8>,
    is_custom: bool,
    is_stratified: bool,
}

#[runtime_builtin(
    name = "cvpartition",
    category = "stats/ml",
    summary = "Create cross-validation partition objects.",
    keywords = "cvpartition,cross validation,kfold,holdout,leaveout,resubstitution,statistics,machine learning",
    type_resolver(object_type),
    descriptor(crate::builtins::stats::ml::cvpartition::CVPARTITION_DESCRIPTOR),
    extensions(crate::builtins::stats::ml::cvpartition::CVPARTITION_EXTENSIONS),
    integer_capabilities(
        crate::builtins::stats::ml::cvpartition::CVPARTITION_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::stats::ml::cvpartition"
)]
async fn cvpartition_builtin(
    first: Value,
    second: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_cvpartition_extensions(&first, &second, &rest)?;
    let first = gather_value(first).await?;
    let second = gather_value(second).await?;
    let rest = gather_values(rest).await?;
    let spec = cvpartition_compute(first, second, rest)?;
    partition_object(spec).map(Value::Object)
}

fn enable_extension(extension: &BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, CVPARTITION_NAME)
}

fn typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn scalar_value(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_) | Value::Bool(_))
        || matches!(value, Value::Tensor(value_tensor) if tensor::is_scalar_tensor(value_tensor))
        || matches!(value, Value::LogicalArray(array) if array.data.len() == 1)
        || matches!(value, Value::GpuTensor(handle) if tensor::element_count(&handle.shape) == 1)
}

fn documented_logical_scalar(value: &Value) -> bool {
    matches!(value, Value::Bool(_))
        || matches!(value, Value::LogicalArray(array) if array.data.len() == 1)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle) && tensor::element_count(&handle.shape) == 1)
}

fn ensure_cvpartition_extensions(
    first: &Value,
    second: &Value,
    rest: &[Value],
) -> BuiltinResult<()> {
    if string_matches(first, "CustomPartition") {
        if typed_integer_value(second) {
            enable_extension(&CVPARTITION_INTEGER_CUSTOM_EXTENSION)?;
        }
        return Ok(());
    }

    if typed_integer_value(first) {
        if scalar_value(first) {
            enable_extension(&CVPARTITION_INTEGER_N_EXTENSION)?;
        } else {
            enable_extension(&CVPARTITION_INTEGER_STRATIFICATION_EXTENSION)?;
        }
    }

    let kind = scalar_text(second, "partition kind")
        .map(|text| canonical(&text))
        .unwrap_or_default();
    let option_start = if matches!(kind.as_str(), "kfold" | "holdout") {
        if rest.first().is_some_and(typed_integer_value) {
            enable_extension(&CVPARTITION_INTEGER_CONTROL_EXTENSION)?;
        }
        1
    } else {
        0
    };
    let options = &rest[option_start.min(rest.len())..];
    for pair in options.chunks_exact(2) {
        if string_matches(&pair[0], "Stratify") && !documented_logical_scalar(&pair[1]) {
            enable_extension(&CVPARTITION_BOOLEAN_ALIAS_EXTENSION)?;
        }
    }
    Ok(())
}

#[runtime_builtin(
    name = "training",
    category = "stats/ml",
    summary = "Return training-set masks from a cvpartition object.",
    keywords = "training,cvpartition,cross validation,mask,statistics,machine learning",
    type_resolver(logical_type),
    descriptor(crate::builtins::stats::ml::cvpartition::TRAINING_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::cvpartition"
)]
async fn training_builtin(c: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let c = gather_value(c).await?;
    let rest = gather_values(rest).await?;
    mask_from_partition(c, rest, true)
}

#[runtime_builtin(
    name = "test",
    category = "stats/ml",
    summary = "Return test-set masks from a cvpartition object.",
    keywords = "test,cvpartition,cross validation,mask,statistics,machine learning",
    type_resolver(logical_type),
    descriptor(crate::builtins::stats::ml::cvpartition::TEST_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::cvpartition"
)]
async fn test_builtin(c: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let c = gather_value(c).await?;
    let rest = gather_values(rest).await?;
    mask_from_partition(c, rest, false)
}

async fn gather_value(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("cvpartition: {err}")))
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather_value(value).await?);
    }
    Ok(out)
}

fn cvpartition_compute(
    first: Value,
    second: Value,
    rest: Vec<Value>,
) -> BuiltinResult<PartitionSpec> {
    if string_matches(&first, "CustomPartition") {
        if !rest.is_empty() {
            return Err(invalid_argument(
                "cvpartition: CustomPartition accepts exactly one test-set argument",
            ));
        }
        return custom_partition(second);
    }

    let kind_text = scalar_text(&second, "partition kind")?;
    let kind = parse_kind(&kind_text)?;
    let (value, option_start) = match kind {
        PartitionKind::Leaveout | PartitionKind::Resubstitution => (None, 0usize),
        PartitionKind::KFold | PartitionKind::Holdout => {
            let Some(value) = rest.first() else {
                return Err(invalid_argument(format!(
                    "cvpartition: {} requires a partition value",
                    kind.property_name()
                )));
            };
            (Some(value.clone()), 1usize)
        }
        PartitionKind::Custom => {
            if !rest.is_empty() {
                return Err(invalid_argument(
                    "cvpartition: CustomPartition accepts exactly one test-set argument",
                ));
            }
            return custom_partition(first);
        }
    };
    let options = parse_options(&rest[option_start..])?;
    let input = PartitionInput::from_value(first)?;
    let stratify = options.stratify.unwrap_or(input.default_stratify());
    if stratify && input.labels.is_none() {
        return Err(invalid_argument(
            "cvpartition: Stratify true requires a grouping variable input",
        ));
    }
    let labels = if stratify {
        input.labels.as_deref()
    } else {
        None
    };
    let n = input.n;
    match kind {
        PartitionKind::KFold => {
            let k = positive_integer(value.as_ref().expect("kfold value"), "KFold")?;
            kfold_partition(n, k, labels)
        }
        PartitionKind::Holdout => {
            let test_count = holdout_count(value.as_ref().expect("holdout value"), n)?;
            holdout_partition(n, test_count, labels)
        }
        PartitionKind::Leaveout => leaveout_partition(n),
        PartitionKind::Resubstitution => resubstitution_partition(n),
        PartitionKind::Custom => unreachable!(),
    }
}

#[derive(Clone, Debug)]
struct PartitionInput {
    n: usize,
    labels: Option<Vec<Option<String>>>,
}

impl PartitionInput {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(number) => Ok(Self {
                n: positive_integer_number(number, "n")?,
                labels: None,
            }),
            Value::Int(integer) => Ok(Self {
                n: integer.try_to_usize().filter(|n| *n > 0).ok_or_else(|| {
                    invalid_argument("cvpartition: n must be a positive integer scalar")
                })?,
                labels: None,
            }),
            Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
                let n = if let Some(integer) = tensor.integer_storage().and_then(|s| s.value_at(0)) {
                    integer.try_to_usize().filter(|n| *n > 0).ok_or_else(|| {
                        invalid_argument("cvpartition: n must be a positive integer scalar")
                    })?
                } else {
                    positive_integer_number(tensor::tensor_value_f64(&tensor, 0), "n")?
                };
                Ok(Self { n, labels: None })
            }
            Value::Tensor(tensor) => numeric_labels(tensor),
            Value::LogicalArray(array) => logical_labels(array),
            Value::Bool(flag) => Ok(Self {
                n: 1,
                labels: Some(vec![Some(if flag { "true" } else { "false" }.to_string())]),
            }),
            Value::String(text) => Ok(Self {
                n: 1,
                labels: Some(vec![Some(text)]),
            }),
            Value::StringArray(array) => string_array_labels(array),
            Value::CharArray(chars) => Ok(Self {
                n: chars.rows,
                labels: Some(char_row_labels(chars)),
            }),
            other => Err(invalid_argument(format!(
                "cvpartition: first input must be an observation count or grouping variable, got {other:?}"
            ))),
        }
    }

    fn default_stratify(&self) -> bool {
        self.labels.is_some()
    }
}

fn numeric_labels(tensor: Tensor) -> BuiltinResult<PartitionInput> {
    let len = tensor::tensor_element_len(&tensor);
    if len == 0 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be nonempty",
        ));
    }
    if tensor.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be a vector",
        ));
    }
    let labels = if let Some(storage) = tensor.integer_storage() {
        storage
            .exact_values()
            .iter()
            .map(|value| Some(integer_label_key(value)))
            .collect()
    } else {
        tensor::tensor_values_f64_cow(&tensor)
            .iter()
            .map(|value| {
                if value.is_nan() {
                    None
                } else {
                    Some(label_number(*value))
                }
            })
            .collect()
    };
    Ok(PartitionInput {
        n: len,
        labels: Some(labels),
    })
}

fn integer_label_key(value: &runmat_value::IntValue) -> String {
    match value {
        runmat_value::IntValue::I8(value) => format!("i:{value}"),
        runmat_value::IntValue::I16(value) => format!("i:{value}"),
        runmat_value::IntValue::I32(value) => format!("i:{value}"),
        runmat_value::IntValue::I64(value) => format!("i:{value}"),
        runmat_value::IntValue::U8(value) => format!("u:{value}"),
        runmat_value::IntValue::U16(value) => format!("u:{value}"),
        runmat_value::IntValue::U32(value) => format!("u:{value}"),
        runmat_value::IntValue::U64(value) => format!("u:{value}"),
    }
}

fn logical_labels(array: LogicalArray) -> BuiltinResult<PartitionInput> {
    if array.data.is_empty() {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be nonempty",
        ));
    }
    if array.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be a vector",
        ));
    }
    Ok(PartitionInput {
        n: array.data.len(),
        labels: Some(
            array
                .data
                .into_iter()
                .map(|flag| {
                    Some(if flag == 0 {
                        "false".to_string()
                    } else {
                        "true".to_string()
                    })
                })
                .collect(),
        ),
    })
}

fn string_array_labels(array: StringArray) -> BuiltinResult<PartitionInput> {
    if array.data.is_empty() {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be nonempty",
        ));
    }
    if array.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be a vector",
        ));
    }
    Ok(PartitionInput {
        n: array.data.len(),
        labels: Some(array.data.into_iter().map(Some).collect()),
    })
}

fn char_row_labels(chars: runmat_value::CharArray) -> Vec<Option<String>> {
    let mut labels = Vec::with_capacity(chars.rows);
    for row in 0..chars.rows {
        let mut label = String::with_capacity(chars.cols);
        for col in 0..chars.cols {
            label.push(chars.data[row + col * chars.rows]);
        }
        labels.push(Some(label));
    }
    labels
}

fn label_number(value: f64) -> String {
    if value.is_infinite() {
        value.to_string()
    } else if value.fract().abs() <= EPS {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn parse_kind(text: &str) -> BuiltinResult<PartitionKind> {
    match canonical(text).as_str() {
        "kfold" => Ok(PartitionKind::KFold),
        "holdout" => Ok(PartitionKind::Holdout),
        "leaveout" => Ok(PartitionKind::Leaveout),
        "resubstitution" => Ok(PartitionKind::Resubstitution),
        "custompartition" => Ok(PartitionKind::Custom),
        other => Err(invalid_argument(format!(
            "cvpartition: unsupported partition kind '{other}'"
        ))),
    }
}

fn parse_options(rest: &[Value]) -> BuiltinResult<CvOptions> {
    if !rest.len().is_multiple_of(2) {
        return Err(invalid_argument(
            "cvpartition: name-value options must be supplied in pairs",
        ));
    }
    let mut options = CvOptions { stratify: None };
    let mut idx = 0usize;
    while idx < rest.len() {
        let name = scalar_text(&rest[idx], "option name")?;
        let value = &rest[idx + 1];
        match canonical(&name).as_str() {
            "stratify" => options.stratify = Some(scalar_bool(value, "Stratify")?),
            other => {
                return Err(invalid_argument(format!(
                    "cvpartition: unsupported option '{other}'"
                )))
            }
        }
        idx += 2;
    }
    Ok(options)
}

fn kfold_partition(
    n: usize,
    k: usize,
    labels: Option<&[Option<String>]>,
) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    if k < 2 || k >= n {
        return Err(invalid_argument(
            "cvpartition: KFold must be an integer in the interval [2, n)",
        ));
    }
    let len = materialized_mask_len(n, k, "KFold")?;
    let mut test_sets = vec![0u8; len];
    let active_rows = active_rows_for(n, labels);
    if let Some(labels) = labels {
        if labels.len() != n {
            return Err(invalid_argument(
                "cvpartition: stratification labels must match n",
            ));
        }
        for indices in grouped_indices(labels).values() {
            for (offset, row) in shuffled_indices(indices)?.iter().enumerate() {
                test_sets[row + n * (offset % k)] = 1;
            }
        }
    } else {
        for (offset, row) in shuffled_indices(&(0..n).collect::<Vec<_>>())?
            .iter()
            .enumerate()
        {
            test_sets[row + n * (offset % k)] = 1;
        }
    }
    Ok(PartitionSpec {
        kind: PartitionKind::KFold,
        n,
        test_sets,
        test_set_count: k,
        active_rows,
        is_custom: false,
        is_stratified: labels.is_some(),
    })
}

fn holdout_partition(
    n: usize,
    test_count: usize,
    labels: Option<&[Option<String>]>,
) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    if test_count == 0 || test_count >= n {
        return Err(invalid_argument(
            "cvpartition: Holdout must select between 1 and n-1 observations",
        ));
    }
    let mut test_sets = vec![0u8; n];
    let active_rows = active_rows_for(n, labels);
    if let Some(labels) = labels {
        let groups = grouped_indices(labels);
        let mut selected = 0usize;
        for indices in groups.values() {
            let group_target = ((indices.len() * test_count) + (n / 2)) / n;
            let group_target = group_target.min(indices.len());
            for row in shuffled_indices(indices)?.iter().take(group_target) {
                if selected < test_count {
                    test_sets[*row] = 1;
                    selected += 1;
                }
            }
        }
        for row in shuffled_indices(&(0..n).collect::<Vec<_>>())? {
            if selected >= test_count {
                break;
            }
            if active_rows[row] != 0 && test_sets[row] == 0 {
                test_sets[row] = 1;
                selected += 1;
            }
        }
    } else {
        for row in shuffled_indices(&(0..n).collect::<Vec<_>>())?
            .into_iter()
            .take(test_count)
        {
            test_sets[row] = 1;
        }
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Holdout,
        n,
        test_sets,
        test_set_count: 1,
        active_rows,
        is_custom: false,
        is_stratified: labels.is_some(),
    })
}

fn shuffled_indices(indices: &[usize]) -> BuiltinResult<Vec<usize>> {
    let mut output = indices.to_vec();
    if output.len() <= 1 {
        return Ok(output);
    }
    let uniforms = random::generate_uniform(output.len() - 1, CVPARTITION_NAME)?;
    for (index, uniform) in uniforms.into_iter().enumerate() {
        let span = output.len() - index;
        let offset = ((uniform * span as f64).floor() as usize).min(span - 1);
        output.swap(index, index + offset);
    }
    Ok(output)
}

fn leaveout_partition(n: usize) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    let len = materialized_mask_len(n, n, "Leaveout")?;
    let mut test_sets = vec![0u8; len];
    for row in 0..n {
        test_sets[row + n * row] = 1;
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Leaveout,
        n,
        test_sets,
        test_set_count: n,
        active_rows: vec![1; n],
        is_custom: false,
        is_stratified: false,
    })
}

fn resubstitution_partition(n: usize) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Resubstitution,
        n,
        test_sets: vec![1u8; n],
        test_set_count: 1,
        active_rows: vec![1; n],
        is_custom: false,
        is_stratified: false,
    })
}

fn custom_partition(value: Value) -> BuiltinResult<PartitionSpec> {
    let array = match value {
        Value::LogicalArray(array) => array,
        Value::Bool(flag) => LogicalArray::new(vec![u8::from(flag)], vec![1, 1])
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?,
        Value::Tensor(tensor) => {
            return custom_numeric_partition(tensor);
        }
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: CustomPartition test sets must be logical, got {other:?}"
            )))
        }
    };
    let shape = logical_shape(&array)?;
    let n = shape.0;
    let test_set_count = shape.1;
    if n == 0 || test_set_count == 0 {
        return Err(invalid_argument(
            "cvpartition: CustomPartition test sets must be nonempty",
        ));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Custom,
        n,
        test_sets: array.data,
        test_set_count,
        active_rows: vec![1; n],
        is_custom: true,
        is_stratified: false,
    })
}

fn custom_numeric_partition(tensor: Tensor) -> BuiltinResult<PartitionSpec> {
    if tensor.integer_storage().is_some() {
        return custom_integer_partition(tensor);
    }
    let values = tensor::tensor_values_f64_cow(&tensor);
    if values.is_empty() {
        return Err(invalid_argument(
            "cvpartition: CustomPartition test sets must be nonempty",
        ));
    }
    let is_vector = tensor.shape.iter().filter(|dim| **dim > 1).count() <= 1;
    if !is_vector {
        let data = values
            .iter()
            .map(|value| {
                if (*value - 0.0).abs() <= EPS {
                    Ok(0)
                } else if (*value - 1.0).abs() <= EPS {
                    Ok(1)
                } else {
                    Err(invalid_argument(
                        "cvpartition: numeric custom matrices must contain only 0 or 1",
                    ))
                }
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        let array = LogicalArray::new(data, tensor.shape)
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?;
        return custom_partition(Value::LogicalArray(array));
    }

    if values.iter().any(|value| (*value - 0.0).abs() <= EPS) {
        let data = values
            .iter()
            .map(|value| {
                if (*value - 0.0).abs() <= EPS {
                    Ok(0)
                } else if (*value - 1.0).abs() <= EPS {
                    Ok(1)
                } else {
                    Err(invalid_argument(
                        "cvpartition: numeric custom logical vectors must contain only 0 or 1",
                    ))
                }
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        let array = LogicalArray::new(data, vec![values.len(), 1])
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?;
        return custom_partition(Value::LogicalArray(array));
    }

    let mut ids = Vec::with_capacity(values.len());
    let mut max_id = 0usize;
    for value in values.iter() {
        if !value.is_finite()
            || value.fract() != 0.0
            || *value < 1.0
            || *value > usize::MAX as f64
            || (usize::BITS == 64 && *value == usize::MAX as f64)
        {
            return Err(invalid_argument(
                "cvpartition: custom partition ids must be positive integers",
            ));
        }
        let id = *value as usize;
        max_id = max_id.max(id);
        ids.push(id);
    }
    let n = ids.len();
    let len = materialized_mask_len(n, max_id, "CustomPartition")?;
    let mut test_sets = vec![0u8; len];
    let mut nonempty = vec![0usize; max_id];
    for (row, id) in ids.into_iter().enumerate() {
        test_sets[row + n * (id - 1)] = 1;
        nonempty[id - 1] += 1;
    }
    if nonempty.contains(&0) {
        return Err(invalid_argument(
            "cvpartition: custom partition ids must form nonempty test sets",
        ));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Custom,
        n,
        test_sets,
        test_set_count: max_id,
        active_rows: vec![1; n],
        is_custom: true,
        is_stratified: false,
    })
}

fn custom_integer_partition(tensor: Tensor) -> BuiltinResult<PartitionSpec> {
    let storage = tensor
        .integer_storage()
        .expect("custom_integer_partition requires integer storage");
    let values = storage.exact_values();
    if values.is_empty() {
        return Err(invalid_argument(
            "cvpartition: CustomPartition test sets must be nonempty",
        ));
    }
    let is_vector = tensor.shape.iter().filter(|dim| **dim > 1).count() <= 1;
    let logical_byte = |value: &runmat_value::IntValue| match value.try_to_usize() {
        Some(0) => Ok(0),
        Some(1) => Ok(1),
        _ => Err(invalid_argument(
            "cvpartition: numeric custom logical values must contain only 0 or 1",
        )),
    };
    if !is_vector || values.iter().any(|value| value.is_zero()) {
        let data = values
            .iter()
            .map(logical_byte)
            .collect::<BuiltinResult<Vec<_>>>()?;
        let shape = if is_vector {
            vec![values.len(), 1]
        } else {
            tensor.shape.clone()
        };
        let array = LogicalArray::new(data, shape)
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?;
        return custom_partition(Value::LogicalArray(array));
    }

    let mut ids = Vec::with_capacity(values.len());
    let mut max_id = 0usize;
    for value in values {
        let id = value.try_to_usize().filter(|id| *id > 0).ok_or_else(|| {
            invalid_argument("cvpartition: custom partition ids must be positive integers")
        })?;
        max_id = max_id.max(id);
        ids.push(id);
    }
    custom_partition_from_ids(ids, max_id)
}

fn custom_partition_from_ids(ids: Vec<usize>, max_id: usize) -> BuiltinResult<PartitionSpec> {
    let n = ids.len();
    let len = materialized_mask_len(n, max_id, "CustomPartition")?;
    let mut test_sets = vec![0u8; len];
    let mut nonempty = vec![0usize; max_id];
    for (row, id) in ids.into_iter().enumerate() {
        test_sets[row + n * (id - 1)] = 1;
        nonempty[id - 1] += 1;
    }
    if nonempty.contains(&0) {
        return Err(invalid_argument(
            "cvpartition: custom partition ids must form nonempty test sets",
        ));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Custom,
        n,
        test_sets,
        test_set_count: max_id,
        active_rows: vec![1; n],
        is_custom: true,
        is_stratified: false,
    })
}

fn partition_object(spec: PartitionSpec) -> BuiltinResult<ObjectInstance> {
    let mut object = ObjectInstance::new(CVPARTITION_CLASS.to_string());
    object
        .properties
        .insert("NumObservations".to_string(), Value::Num(spec.n as f64));
    object.properties.insert(
        "NumTestSets".to_string(),
        Value::Num(spec.test_set_count as f64),
    );
    object.properties.insert(
        "TestSize".to_string(),
        size_vector(&spec.test_sets, spec.n, spec.test_set_count, true)?,
    );
    object.properties.insert(
        "TrainSize".to_string(),
        size_vector(&spec.test_sets, spec.n, spec.test_set_count, false)?,
    );
    object.properties.insert(
        "Type".to_string(),
        Value::String(spec.kind.property_name().to_string()),
    );
    object
        .properties
        .insert("IsCustom".to_string(), Value::Bool(spec.is_custom));
    object
        .properties
        .insert("IsGrouped".to_string(), Value::Bool(false));
    object
        .properties
        .insert("IsStratified".to_string(), Value::Bool(spec.is_stratified));
    object.properties.insert(
        TEST_SETS_PROPERTY.to_string(),
        Value::LogicalArray(
            LogicalArray::new(spec.test_sets, vec![spec.n, spec.test_set_count])
                .map_err(|err| internal_error(format!("cvpartition: {err}")))?,
        ),
    );
    object.properties.insert(
        ACTIVE_ROWS_PROPERTY.to_string(),
        Value::LogicalArray(
            LogicalArray::new(spec.active_rows, vec![spec.n, 1])
                .map_err(|err| internal_error(format!("cvpartition: {err}")))?,
        ),
    );
    Ok(object)
}

fn size_vector(
    test_sets: &[u8],
    rows: usize,
    cols: usize,
    test_size: bool,
) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(cols);
    for col in 0..cols {
        let count = test_sets[col * rows..(col + 1) * rows]
            .iter()
            .filter(|flag| **flag != 0)
            .count();
        let size = if test_size { count } else { rows - count };
        data.push(size as f64);
    }
    if cols == 1 {
        Ok(Value::Num(data[0]))
    } else {
        Tensor::new(data, vec![1, cols])
            .map(Value::Tensor)
            .map_err(|err| internal_error(format!("cvpartition: {err}")))
    }
}

fn mask_from_partition(c: Value, rest: Vec<Value>, training: bool) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(invalid_argument(
            "cvpartition: training/test accept at most one index argument",
        ));
    }
    let object = match c {
        Value::Object(object) if object.is_class(CVPARTITION_CLASS) => object,
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: expected cvpartition object, got {other:?}"
            )))
        }
    };
    let test_sets = test_sets_property(&object)?;
    let active_rows = active_rows_property(&object)?;
    let resubstitution = matches!(
        object.properties.get("Type"),
        Some(Value::String(kind)) if kind.eq_ignore_ascii_case("resubstitution")
    );
    let (rows, cols) = logical_shape(&test_sets)?;
    if active_rows.data.len() != rows {
        return Err(invalid_argument(
            "cvpartition: object active-row metadata is inconsistent",
        ));
    }
    let selected = if let Some(index) = rest.first() {
        selected_indices(index, cols)?
    } else if cols == 1 {
        vec![0usize]
    } else {
        return Err(invalid_argument(
            "cvpartition: an index or 'all' is required for partitions with multiple test sets",
        ));
    };
    let _ = materialized_mask_len(rows, selected.len(), "training/test output")?;
    let mut data = Vec::with_capacity(rows * selected.len());
    for col in &selected {
        let base = *col * rows;
        for row in 0..rows {
            let active = active_rows.data[row] != 0;
            let is_test = test_sets.data[base + row] != 0;
            data.push(u8::from(if training {
                active && (resubstitution || !is_test)
            } else {
                active && is_test
            }));
        }
    }
    LogicalArray::new(
        data,
        vec![rows, if rest.is_empty() { 1 } else { selected.len() }],
    )
    .map(Value::LogicalArray)
    .map_err(|err| internal_error(format!("cvpartition: {err}")))
}

fn selected_indices(value: &Value, cols: usize) -> BuiltinResult<Vec<usize>> {
    if string_matches(value, "all") {
        return Ok((0..cols).collect());
    }
    if let Value::Int(integer) = value {
        return integer
            .try_to_usize()
            .filter(|index| *index >= 1 && *index <= cols)
            .map(|index| vec![index - 1])
            .ok_or_else(|| {
                invalid_argument(format!(
                    "cvpartition: index must be an integer between 1 and {cols}"
                ))
            });
    }
    let raw = match value {
        Value::Num(number) => vec![*number],
        Value::Tensor(tensor) => tensor::tensor_values_f64(tensor),
        Value::LogicalArray(array) => array
            .data
            .iter()
            .map(|flag| if *flag == 0 { 0.0 } else { 1.0 })
            .collect(),
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: index must be numeric, logical, or 'all', got {other:?}"
            )))
        }
    };
    let mut indices = Vec::with_capacity(raw.len());
    for value in raw {
        if !value.is_finite() || value.fract() != 0.0 || value < 1.0 || value > cols as f64 {
            return Err(invalid_argument(format!(
                "cvpartition: index must be an integer between 1 and {cols}"
            )));
        }
        indices.push(value as usize - 1);
    }
    if indices.is_empty() {
        return Err(invalid_argument("cvpartition: index must be nonempty"));
    }
    Ok(indices)
}

fn test_sets_property(object: &ObjectInstance) -> BuiltinResult<LogicalArray> {
    match object.properties.get(TEST_SETS_PROPERTY) {
        Some(Value::LogicalArray(array)) => Ok(array.clone()),
        _ => Err(invalid_argument(
            "cvpartition: object is missing internal test-set data",
        )),
    }
}

fn active_rows_property(object: &ObjectInstance) -> BuiltinResult<LogicalArray> {
    match object.properties.get(ACTIVE_ROWS_PROPERTY) {
        Some(Value::LogicalArray(array)) => Ok(array.clone()),
        _ => Err(invalid_argument(
            "cvpartition: object is missing internal active-row data",
        )),
    }
}

fn logical_shape(array: &LogicalArray) -> BuiltinResult<(usize, usize)> {
    match array.shape.as_slice() {
        [] => Ok((array.data.len(), 1)),
        [rows] => Ok((*rows, 1)),
        [1, cols] => Ok((*cols, 1)),
        [rows, 1] => Ok((*rows, 1)),
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(invalid_argument(
            "cvpartition: logical test sets must be a vector or 2-D matrix",
        )),
    }
}

fn grouped_indices(labels: &[Option<String>]) -> BTreeMap<String, Vec<usize>> {
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for (idx, label) in labels.iter().enumerate() {
        if let Some(label) = label {
            groups.entry(label.clone()).or_default().push(idx);
        }
    }
    groups
}

fn active_rows_for(n: usize, labels: Option<&[Option<String>]>) -> Vec<u8> {
    match labels {
        Some(labels) => labels
            .iter()
            .map(|label| u8::from(label.is_some()))
            .collect(),
        None => vec![1; n],
    }
}

fn materialized_mask_len(rows: usize, cols: usize, label: &str) -> BuiltinResult<usize> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| internal_error(format!("cvpartition: {label} size overflow")))?;
    if len > MAX_MATERIALIZED_MASK_CELLS {
        return Err(internal_error(format!(
            "cvpartition: {label} would materialize {len} logical entries; limit is {MAX_MATERIALIZED_MASK_CELLS}"
        )));
    }
    Ok(len)
}

fn holdout_count(value: &Value, n: usize) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|count| *count > 0 && *count < n)
            .ok_or_else(|| {
                invalid_argument(
                    "cvpartition: Holdout must be a fraction in (0,1) or an integer in [1,n)",
                )
            });
    }
    let raw = scalar_number(value, "Holdout")?;
    if raw > 0.0 && raw < 1.0 {
        let count = (raw * n as f64).round() as usize;
        return Ok(count.clamp(1, n.saturating_sub(1)));
    }
    if raw.fract() == 0.0 && raw >= 1.0 && raw < n as f64 {
        return Ok(raw as usize);
    }
    Err(invalid_argument(
        "cvpartition: Holdout must be a fraction in (0,1) or an integer in [1,n)",
    ))
}

fn positive_integer(value: &Value, label: &str) -> BuiltinResult<usize> {
    if let Value::Int(integer) = value {
        return integer
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                invalid_argument(format!(
                    "cvpartition: {label} must be a positive integer scalar"
                ))
            });
    }
    positive_integer_number(scalar_number(value, label)?, label)
}

fn positive_integer_number(raw: f64, label: &str) -> BuiltinResult<usize> {
    if !raw.is_finite()
        || raw.fract() != 0.0
        || raw < 1.0
        || raw > usize::MAX as f64
        || (usize::BITS == 64 && raw == usize::MAX as f64)
    {
        return Err(invalid_argument(format!(
            "cvpartition: {label} must be a positive integer scalar"
        )));
    }
    Ok(raw as usize)
}

fn scalar_number(value: &Value, label: &str) -> BuiltinResult<f64> {
    let number = match value {
        Value::Num(number) => *number,
        Value::Int(integer) => integer.to_f64(),
        Value::Bool(flag) => {
            if *flag {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            if array.data[0] == 0 {
                0.0
            } else {
                1.0
            }
        }
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: {label} must be a numeric scalar, got {other:?}"
            )))
        }
    };
    if !number.is_finite() {
        return Err(invalid_argument(format!(
            "cvpartition: {label} must be finite"
        )));
    }
    Ok(number)
}

fn scalar_bool(value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(number) if (*number - 0.0).abs() <= EPS || (*number - 1.0).abs() <= EPS => {
            Ok((*number - 1.0).abs() <= EPS)
        }
        Value::Int(integer) => match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(invalid_argument(format!(
                "cvpartition: {label} must be logical scalar"
            ))),
        },
        Value::String(text)
            if text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("false") =>
        {
            Ok(text.eq_ignore_ascii_case("true"))
        }
        Value::CharArray(chars) => {
            let text = chars.data.iter().collect::<String>();
            if text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("false") {
                Ok(text.eq_ignore_ascii_case("true"))
            } else {
                Err(invalid_argument(format!(
                    "cvpartition: {label} must be logical scalar"
                )))
            }
        }
        _ => Err(invalid_argument(format!(
            "cvpartition: {label} must be logical scalar"
        ))),
    }
}

fn scalar_text(value: &Value, label: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(invalid_argument(format!(
            "cvpartition: {label} must be a string scalar, got {other:?}"
        ))),
    }
}

fn string_matches(value: &Value, expected: &str) -> bool {
    scalar_text(value, "text")
        .map(|text| text.eq_ignore_ascii_case(expected))
        .unwrap_or(false)
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
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn logical(data: Vec<u8>, shape: Vec<usize>) -> Value {
        Value::LogicalArray(LogicalArray::new(data, shape).unwrap())
    }

    fn cleared_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).unwrap();
        Value::Tensor(tensor)
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).unwrap();
        Value::Tensor(tensor)
    }

    #[test]
    fn scalar_counts_read_every_integer_storage_variant_not_the_float_mirror() {
        for storage in [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ] {
            let value = poisoned_int_tensor(storage, vec![1, 1]);
            assert_eq!(PartitionInput::from_value(value.clone()).unwrap().n, 2);
            assert_eq!(holdout_count(&value, 6).unwrap(), 2);
        }
    }

    fn cv(first: Value, second: Value, rest: Vec<Value>) -> Value {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(cvpartition_builtin(first, second, rest)).expect("cvpartition")
    }

    fn logical_output(value: Value) -> LogicalArray {
        let Value::LogicalArray(array) = value else {
            panic!("expected logical array");
        };
        array
    }

    #[test]
    fn kfold_partition_exposes_balanced_masks() {
        let _lock = random::test_guard();
        random::set_seed(2026).unwrap();
        let partition = cv(
            Value::Num(6.0),
            Value::String("KFold".into()),
            vec![Value::Num(3.0)],
        );
        let Value::Object(object) = &partition else {
            panic!("object");
        };
        assert_eq!(object.properties.get("NumTestSets"), Some(&Value::Num(3.0)));
        assert_eq!(
            object.properties.get("TestSize"),
            Some(&tensor(vec![2.0, 2.0, 2.0], vec![1, 3]))
        );

        let all_test = logical_output(
            block_on(test_builtin(
                partition.clone(),
                vec![Value::String("all".into())],
            ))
            .expect("test all"),
        );
        assert_eq!(all_test.shape, vec![6, 3]);
        for fold in 0..3 {
            assert_eq!(
                all_test.data[fold * 6..(fold + 1) * 6]
                    .iter()
                    .filter(|flag| **flag != 0)
                    .count(),
                2
            );
        }
        for row in 0..6 {
            assert_eq!(
                (0..3)
                    .filter(|fold| all_test.data[row + fold * 6] != 0)
                    .count(),
                1
            );
        }

        let train1 = logical_output(
            block_on(training_builtin(partition, vec![Value::Num(1.0)])).expect("training 1"),
        );
        assert_eq!(train1.shape, vec![6, 1]);
        assert_eq!(train1.data.iter().filter(|flag| **flag != 0).count(), 4);
    }

    #[test]
    fn holdout_fraction_and_stratification_work() {
        let _lock = random::test_guard();
        random::set_seed(2027).unwrap();
        let labels = Value::StringArray(
            StringArray::new(
                vec!["a".into(), "a".into(), "b".into(), "b".into()],
                vec![4, 1],
            )
            .unwrap(),
        );
        let partition = cv(
            labels,
            Value::String("Holdout".into()),
            vec![
                Value::Num(0.5),
                Value::String("Stratify".into()),
                Value::Bool(true),
            ],
        );
        let test_mask =
            logical_output(block_on(test_builtin(partition, Vec::new())).expect("test"));
        assert_eq!(test_mask.shape, vec![4, 1]);
        assert_eq!(test_mask.data.iter().filter(|flag| **flag != 0).count(), 2);
        assert_eq!(
            test_mask.data[..2]
                .iter()
                .filter(|flag| **flag != 0)
                .count(),
            1
        );
        assert_eq!(
            test_mask.data[2..]
                .iter()
                .filter(|flag| **flag != 0)
                .count(),
            1
        );
    }

    #[test]
    fn leaveout_and_resubstitution_have_expected_shapes() {
        let leaveout = cv(
            Value::Num(3.0),
            Value::String("Leaveout".into()),
            Vec::new(),
        );
        let all = logical_output(
            block_on(test_builtin(leaveout, vec![Value::String("all".into())]))
                .expect("leaveout all"),
        );
        assert_eq!(all.shape, vec![3, 3]);
        assert_eq!(all.data, vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);

        let resub = cv(
            Value::Num(3.0),
            Value::String("Resubstitution".into()),
            Vec::new(),
        );
        let train = logical_output(block_on(training_builtin(resub, Vec::new())).expect("train"));
        assert_eq!(train.shape, vec![3, 1]);
        assert_eq!(train.data, vec![1, 1, 1]);

        let resub = cv(
            Value::Num(3.0),
            Value::String("Resubstitution".into()),
            Vec::new(),
        );
        let test = logical_output(block_on(test_builtin(resub, Vec::new())).expect("test"));
        assert_eq!(test.shape, vec![3, 1]);
        assert_eq!(test.data, vec![1, 1, 1]);
    }

    #[test]
    fn custom_partition_preserves_columns() {
        let partition = cv(
            Value::String("CustomPartition".into()),
            logical(vec![1, 0, 0, 1, 0, 1], vec![3, 2]),
            Vec::new(),
        );
        let all = logical_output(
            block_on(test_builtin(partition, vec![Value::String("all".into())]))
                .expect("custom all"),
        );
        assert_eq!(all.shape, vec![3, 2]);
        assert_eq!(all.data, vec![1, 0, 0, 1, 0, 1]);
    }

    #[test]
    fn custom_integer_vector_expands_to_test_set_columns() {
        let partition = cv(
            Value::String("CustomPartition".into()),
            tensor(vec![1.0, 2.0, 2.0, 1.0], vec![1, 4]),
            Vec::new(),
        );
        let all = logical_output(
            block_on(test_builtin(partition, vec![Value::String("all".into())]))
                .expect("custom integer all"),
        );
        assert_eq!(all.shape, vec![4, 2]);
        assert_eq!(all.data, vec![1, 0, 0, 1, 0, 1, 1, 0]);
    }

    #[test]
    fn custom_integer_vector_reads_typed_integer_storage_exactly() {
        let partition = cv(
            Value::String("CustomPartition".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![1, 2, 2, 1]), vec![1, 4]),
            Vec::new(),
        );
        let all = logical_output(
            block_on(test_builtin(partition, vec![Value::String("all".into())]))
                .expect("custom integer all"),
        );
        assert_eq!(all.shape, vec![4, 2]);
        assert_eq!(all.data, vec![1, 0, 0, 1, 0, 1, 1, 0]);

        let partition = cv(
            Value::String("CustomPartition".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![1, 0, 1, 0]), vec![1, 4]),
            Vec::new(),
        );
        let mask = logical_output(block_on(test_builtin(partition, Vec::new())).expect("test"));
        assert_eq!(mask.shape, vec![4, 1]);
        assert_eq!(mask.data, vec![1, 0, 1, 0]);

        let err = block_on(cvpartition_builtin(
            Value::String("CustomPartition".into()),
            tensor(vec![usize::MAX as f64], vec![1, 1]),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cvpartition:InvalidArgument"));
    }

    #[test]
    fn custom_logical_row_vector_is_single_test_set() {
        let partition = cv(
            Value::String("CustomPartition".into()),
            logical(vec![1, 0, 1, 0], vec![1, 4]),
            Vec::new(),
        );
        let Value::Object(object) = &partition else {
            panic!("object");
        };
        assert_eq!(
            object.properties.get("NumObservations"),
            Some(&Value::Num(4.0))
        );
        assert_eq!(object.properties.get("NumTestSets"), Some(&Value::Num(1.0)));
        let mask = logical_output(block_on(test_builtin(partition, Vec::new())).expect("test"));
        assert_eq!(mask.shape, vec![4, 1]);
        assert_eq!(mask.data, vec![1, 0, 1, 0]);
    }

    #[test]
    fn invalid_kfold_rejects_large_and_equal_fold_count() {
        let err = block_on(cvpartition_builtin(
            Value::Num(3.0),
            Value::String("KFold".into()),
            vec![Value::Num(4.0)],
        ))
        .unwrap_err();
        assert!(err.message.contains("KFold"));

        let err = block_on(cvpartition_builtin(
            Value::Num(3.0),
            Value::String("KFold".into()),
            vec![Value::Num(3.0)],
        ))
        .unwrap_err();
        assert!(err.message.contains("KFold"));
    }

    #[test]
    fn selected_fold_vector_returns_one_column_per_index() {
        let _lock = random::test_guard();
        random::set_seed(2028).unwrap();
        let partition = cv(
            Value::Num(6.0),
            Value::String("KFold".into()),
            vec![Value::Num(3.0)],
        );
        let selected = logical_output(
            block_on(test_builtin(
                partition,
                vec![tensor(vec![1.0, 3.0], vec![1, 2])],
            ))
            .expect("selected folds"),
        );
        assert_eq!(selected.shape, vec![6, 2]);
        assert_eq!(
            selected.data[..6].iter().filter(|flag| **flag != 0).count(),
            2
        );
        assert_eq!(
            selected.data[6..].iter().filter(|flag| **flag != 0).count(),
            2
        );
    }

    #[test]
    fn missing_numeric_stratification_rows_are_discarded_from_masks() {
        let _lock = random::test_guard();
        random::set_seed(2029).unwrap();
        let partition = cv(
            tensor(vec![1.0, f64::NAN, 1.0, 2.0, 2.0], vec![5, 1]),
            Value::String("KFold".into()),
            vec![Value::Num(2.0)],
        );
        let all_test = logical_output(
            block_on(test_builtin(
                partition.clone(),
                vec![Value::String("all".into())],
            ))
            .expect("test all"),
        );
        assert_eq!(all_test.shape, vec![5, 2]);
        assert_eq!(all_test.data[1], 0);
        assert_eq!(all_test.data[6], 0);

        let all_train = logical_output(
            block_on(training_builtin(
                partition,
                vec![Value::String("all".into())],
            ))
            .expect("training all"),
        );
        assert_eq!(all_train.data[1], 0);
        assert_eq!(all_train.data[6], 0);
    }

    #[test]
    fn typed_integer_partition_counts_and_indices_are_exact() {
        let input = PartitionInput::from_value(Value::Int(runmat_value::IntValue::U16(6))).unwrap();
        assert_eq!(input.n, 6);
        let input = PartitionInput::from_value(cleared_int_tensor(
            IntegerStorage::U16(vec![6]),
            vec![1, 1],
        ))
        .unwrap();
        assert_eq!(input.n, 6);
        assert_eq!(
            positive_integer(&Value::Int(runmat_value::IntValue::U8(3)), "KFold").unwrap(),
            3
        );
        assert_eq!(
            positive_integer(
                &cleared_int_tensor(IntegerStorage::U8(vec![3]), vec![1, 1]),
                "KFold"
            )
            .unwrap(),
            3
        );
        assert_eq!(
            holdout_count(&Value::Int(runmat_value::IntValue::U8(2)), 6).unwrap(),
            2
        );
        assert_eq!(
            holdout_count(
                &cleared_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
                6
            )
            .unwrap(),
            2
        );
        assert_eq!(
            selected_indices(&Value::Int(runmat_value::IntValue::U8(3)), 3).unwrap(),
            vec![2]
        );
        assert_eq!(
            selected_indices(
                &cleared_int_tensor(IntegerStorage::U8(vec![3]), vec![1, 1]),
                3
            )
            .unwrap(),
            vec![2]
        );

        let partition = cv(
            cleared_int_tensor(IntegerStorage::U16(vec![6]), vec![1, 1]),
            Value::from("KFold"),
            vec![cleared_int_tensor(IntegerStorage::U8(vec![3]), vec![1, 1])],
        );
        let selected = logical_output(
            block_on(test_builtin(
                partition,
                vec![cleared_int_tensor(IntegerStorage::U8(vec![3]), vec![1, 1])],
            ))
            .unwrap(),
        );
        assert_eq!(selected.shape, vec![6, 1]);

        for value in [
            Value::Int(runmat_value::IntValue::I8(-1)),
            Value::Num(1.5),
            Value::Num(usize::MAX as f64),
            Value::Num(usize::MAX as f64 + 1.0),
        ] {
            assert!(positive_integer(&value, "KFold").is_err());
        }
        assert!(selected_indices(&Value::Int(runmat_value::IntValue::I8(-1)), 3).is_err());
    }

    #[test]
    fn cvpartition_extensions_are_independently_mode_gated() {
        let integer_scalar = || Value::Int(runmat_value::IntValue::U16(6));
        let integer_labels =
            || cleared_int_tensor(IntegerStorage::U8(vec![1, 1, 2, 2]), vec![4, 1]);
        let cases = [
            (
                integer_scalar(),
                Value::from("KFold"),
                vec![Value::Num(2.0)],
                CVPARTITION_INTEGER_N_EXTENSION.error_identifier,
            ),
            (
                Value::Num(6.0),
                Value::from("KFold"),
                vec![Value::Int(runmat_value::IntValue::U8(2))],
                CVPARTITION_INTEGER_CONTROL_EXTENSION.error_identifier,
            ),
            (
                integer_labels(),
                Value::from("KFold"),
                vec![Value::Num(2.0)],
                CVPARTITION_INTEGER_STRATIFICATION_EXTENSION.error_identifier,
            ),
            (
                Value::from("CustomPartition"),
                integer_labels(),
                Vec::new(),
                CVPARTITION_INTEGER_CUSTOM_EXTENSION.error_identifier,
            ),
            (
                Value::Num(6.0),
                Value::from("KFold"),
                vec![Value::Num(2.0), Value::from("Stratify"), Value::Num(1.0)],
                CVPARTITION_BOOLEAN_ALIAS_EXTENSION.error_identifier,
            ),
        ];
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for (first, second, rest, identifier) in cases {
            let error =
                block_on(cvpartition_builtin(first, second, rest)).expect_err("strict rejection");
            assert_eq!(error.identifier(), identifier);
        }
    }

    #[test]
    fn scalar_string_array_kind_cannot_bypass_typed_control_gate() {
        let kind = Value::StringArray(
            StringArray::new(vec!["KFold".into()], vec![1, 1]).expect("scalar string"),
        );
        let control = cleared_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]);
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(cvpartition_builtin(Value::Num(6.0), kind, vec![control]))
            .expect_err("typed control gate");
        assert_eq!(
            error.identifier(),
            CVPARTITION_INTEGER_CONTROL_EXTENSION.error_identifier
        );
    }

    #[test]
    fn wide_integer_stratification_keys_remain_distinct_and_custom_validation_is_exact() {
        let wide = 9_007_199_254_740_992_i64;
        for value in [
            cleared_int_tensor(
                IntegerStorage::I64(vec![wide, wide + 1, wide, wide + 1]),
                vec![4, 1],
            ),
            cleared_int_tensor(
                IntegerStorage::U64(vec![
                    wide as u64,
                    wide as u64 + 1,
                    wide as u64,
                    wide as u64 + 1,
                ]),
                vec![4, 1],
            ),
        ] {
            let input = PartitionInput::from_value(value).expect("exact integer labels");
            let labels = input.labels.expect("stratification labels");
            assert_eq!(labels[0], labels[2]);
            assert_eq!(labels[1], labels[3]);
            assert_ne!(labels[0], labels[1]);
        }

        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in [
            IntegerStorage::I64(vec![wide, wide + 1]),
            IntegerStorage::U64(vec![wide as u64, wide as u64 + 1]),
        ] {
            let error = block_on(cvpartition_builtin(
                Value::from("CustomPartition"),
                cleared_int_tensor(storage, vec![2, 1]),
                Vec::new(),
            ))
            .expect_err("wide custom identifiers reject structurally");
            assert!(error.message().contains("CustomPartition"));
        }
    }

    #[test]
    fn resident_integer_roles_gate_before_gather() {
        fn resident(shape: Vec<usize>, buffer_id: u64) -> Value {
            let handle = runmat_accelerate_api::GpuTensorHandle {
                shape,
                device_id: u32::MAX,
                buffer_id,
            };
            runmat_accelerate_api::set_handle_integer_type(
                &handle,
                runmat_accelerate_api::IntegerElementType::U64,
            );
            Value::GpuTensor(handle)
        }
        let cases = [
            (
                resident(vec![1, 1], u64::MAX - 510),
                Value::from("KFold"),
                vec![Value::Num(2.0)],
                CVPARTITION_INTEGER_N_EXTENSION.error_identifier,
            ),
            (
                Value::Num(6.0),
                Value::from("KFold"),
                vec![resident(vec![1, 1], u64::MAX - 511)],
                CVPARTITION_INTEGER_CONTROL_EXTENSION.error_identifier,
            ),
            (
                resident(vec![4, 1], u64::MAX - 512),
                Value::from("KFold"),
                vec![Value::Num(2.0)],
                CVPARTITION_INTEGER_STRATIFICATION_EXTENSION.error_identifier,
            ),
            (
                Value::from("CustomPartition"),
                resident(vec![4, 1], u64::MAX - 513),
                Vec::new(),
                CVPARTITION_INTEGER_CUSTOM_EXTENSION.error_identifier,
            ),
        ];
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for (first, second, rest, identifier) in cases {
            let error = block_on(cvpartition_builtin(first, second, rest))
                .expect_err("resident role rejects at extension gate");
            assert_eq!(error.identifier(), identifier);
        }
    }

    #[test]
    fn cvpartition_logical_stratify_is_documented_and_rng_respects_seed() {
        let _lock = random::test_guard();
        let call = |seed| {
            random::set_seed(seed).unwrap();
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let partition = block_on(cvpartition_builtin(
                Value::Num(12.0),
                Value::from("KFold"),
                vec![Value::Num(3.0), Value::from("Stratify"), Value::Bool(false)],
            ))
            .unwrap();
            logical_output(block_on(test_builtin(partition, vec![Value::from("all")])).unwrap())
                .data
        };
        let first = call(101);
        let same = call(101);
        let different = call(102);
        assert_eq!(first, same);
        assert_ne!(first, different);
    }

    #[test]
    fn cvpartition_integer_capabilities_cover_all_classes() {
        assert_eq!(CVPARTITION_INTEGER_CAPABILITIES.len(), 4);
        assert!(CVPARTITION_INTEGER_CAPABILITIES
            .iter()
            .all(|capability| capability.inputs[0].classes.len() == 8));
    }
}
