//! Deep Learning Toolbox compatibility builtins.

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ClassDef, MethodDef, ObjectInstance, ResolveContext, StringArray, Tensor, Type, Value,
};
use std::cell::Cell;
use std::collections::HashMap;

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

pub(super) const MAX_COMBVEC_COLUMNS: usize = 1_000_000;
pub(super) const MAX_PAD_ELEMENTS: usize = 10_000_000;

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when:
        "Inputs or name-value options do not match the supported Deep Learning compatibility forms.",
    message: "deep learning builtin received invalid input",
};

const ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.UNSUPPORTED",
    identifier: Some("RunMat:deepLearning:Unsupported"),
    when: "The requested operation requires training, autodiff, export, or UI infrastructure outside this compatibility slice.",
    message: "deep learning operation is not supported in this slice",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_UNSUPPORTED];

thread_local! {
    static DLARRAY_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

const OUT_OBJECT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Deep Learning Toolbox compatibility object.",
}];

const OUT_ARRAY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric or cell array output.",
}];

const IN_REST: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Builtin-specific positional and name-value arguments.",
}];

const OBJECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = deepLearningBuiltin(args...)",
    inputs: &IN_REST,
    outputs: &OUT_OBJECT,
}];

const ARRAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = deepLearningUtility(args...)",
    inputs: &IN_REST,
    outputs: &OUT_ARRAY,
}];

const OUT_ADAMUPDATE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "params",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Updated numeric parameters.",
    },
    BuiltinParamDescriptor {
        name: "averageGrad",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Updated first-moment moving average.",
    },
    BuiltinParamDescriptor {
        name: "averageSqGrad",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Updated second-moment moving average.",
    },
];

const ADAMUPDATE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[params, averageGrad, averageSqGrad] = adamupdate(params, grad, averageGrad, averageSqGrad, iteration)",
        inputs: &IN_REST,
        outputs: &OUT_ADAMUPDATE,
    },
    BuiltinSignatureDescriptor {
        label: "[params, averageGrad, averageSqGrad] = adamupdate(params, grad, averageGrad, averageSqGrad, iteration, learnRate, gradDecay, sqGradDecay, epsilon)",
        inputs: &IN_REST,
        outputs: &OUT_ADAMUPDATE,
    },
];

const OUT_VARARG: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Outputs returned by the invoked function.",
}];

const DLFEVAL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle to evaluate.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Arguments forwarded to the function handle.",
    },
];

const DLFEVAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "varargout = dlfeval(fun, args...)",
    inputs: &DLFEVAL_INPUTS,
    outputs: &OUT_VARARG,
}];

const DLUPDATE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle applied to matching leaves in the parameter trees.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "One or more compatible parameter trees.",
    },
];

const DLUPDATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "varargout = dlupdate(fun, args...)",
    inputs: &DLUPDATE_INPUTS,
    outputs: &OUT_VARARG,
}];

const DLGRADIENT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "loss",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar traced dlarray loss.",
    },
    BuiltinParamDescriptor {
        name: "targets",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Traced dlarray values, learnables, or dlnetworks to differentiate.",
    },
];

const DLGRADIENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "varargout = dlgradient(loss, targets...)",
    inputs: &DLGRADIENT_INPUTS,
    outputs: &OUT_VARARG,
}];

pub const OBJECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OBJECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ARRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARRAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ADAMUPDATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADAMUPDATE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DLFEVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DLFEVAL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DLUPDATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DLUPDATE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DLGRADIENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DLGRADIENT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub(super) fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

pub(super) async fn gather_args(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(gather_if_needed_async(&value).await?);
    }
    Ok(gathered)
}

pub(super) fn deep_learning_error(
    function: &'static str,
    message: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(function)
        .with_identifier("RunMat:deepLearning:InvalidInput")
        .build()
}

pub(super) fn unsupported_error(
    function: &'static str,
    message: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(function)
        .with_identifier("RunMat:deepLearning:Unsupported")
        .build()
}

pub(super) fn ensure_dlarray_class_registered() {
    DLARRAY_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }
        let methods = ["plus", "minus", "times", "rdivide", "mtimes", "sum"]
            .into_iter()
            .map(|name| {
                (
                    name.to_string(),
                    MethodDef {
                        name: name.to_string(),
                        is_static: false,
                        is_abstract: false,
                        is_sealed: false,
                        access: Access::Public,
                        function_name: format!("dlarray.{name}"),
                        implicit_class_argument: None,
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        runmat_builtins::register_class(ClassDef {
            name: "dlarray".to_string(),
            parent: None,
            properties: HashMap::new(),
            methods,
        });
        registered.set(true);
    });
}

pub(super) fn scalar_text(value: &Value, function: &'static str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(deep_learning_error(
            function,
            format!("{function}: expected text scalar, got {other:?}"),
        )),
    }
}

pub(super) fn numeric_scalar(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) if n.is_finite() => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t)
            if crate::builtins::common::tensor::is_scalar_tensor(t)
                && crate::builtins::common::tensor::tensor_value_f64(t, 0).is_finite() =>
        {
            Ok(crate::builtins::common::tensor::tensor_value_f64(t, 0))
        }
        other => Err(deep_learning_error(
            function,
            format!("{function}: {label} must be a finite numeric scalar, got {other:?}"),
        )),
    }
}

/// Parse a scalar flag without consulting an integer tensor's compatibility
/// `f64` mirror.  Structural options use this rather than treating an integer
/// value as ordinary numeric data.
pub(super) fn logical_scalar(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<bool> {
    if let Value::Bool(flag) = value {
        return Ok(*flag);
    }
    if let Some(integer) = crate::builtins::common::tensor::scalar_integer_value(value) {
        return match integer.try_to_i64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(deep_learning_error(
                function,
                format!("{function}: {label} must be logical scalar true or false"),
            )),
        };
    }
    let number = match value {
        Value::Num(number) => *number,
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            crate::builtins::common::tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(deep_learning_error(
                function,
                format!("{function}: {label} must be logical scalar true or false, got {other:?}"),
            ));
        }
    };
    match number {
        0.0 => Ok(false),
        1.0 => Ok(true),
        _ => Err(deep_learning_error(
            function,
            format!("{function}: {label} must be logical scalar true or false"),
        )),
    }
}

pub(super) fn positive_i64(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<i64> {
    if let Some(integer) = crate::builtins::common::tensor::scalar_integer_value(value) {
        return integer
            .try_to_i64()
            .filter(|value| *value >= 1)
            .ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: {label} must be a positive integer scalar"),
                )
            });
    }
    let number = numeric_scalar(value, function, label)?;
    if number.fract().abs() > f64::EPSILON || number < 1.0 || number >= i64::MAX as f64 {
        return Err(deep_learning_error(
            function,
            format!("{function}: {label} must be a positive integer scalar"),
        ));
    }
    Ok(number as i64)
}

pub(super) fn positive_usize(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<usize> {
    match value {
        Value::Int(value) => value
            .try_to_usize()
            .filter(|value| *value >= 1)
            .ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: {label} must be a positive integer"),
                )
            }),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return value
                    .try_to_usize()
                    .filter(|value| *value >= 1)
                    .ok_or_else(|| {
                        deep_learning_error(
                            function,
                            format!("{function}: {label} must be a positive integer"),
                        )
                    });
            }
            let n = crate::builtins::common::tensor::tensor_value_f64(tensor, 0);
            positive_usize_from_f64(n, function, label)
        }
        _ => {
            let n = numeric_scalar(value, function, label)?;
            positive_usize_from_f64(n, function, label)
        }
    }
}

pub(super) fn nonnegative_usize(
    value: &Value,
    function: &'static str,
    label: &str,
) -> Option<usize> {
    match value {
        Value::Int(value) => value.try_to_usize(),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return value.try_to_usize();
            }
            nonnegative_usize_from_f64(crate::builtins::common::tensor::tensor_value_f64(tensor, 0))
        }
        Value::Num(n) => nonnegative_usize_from_f64(*n),
        _ => {
            let _ = (function, label);
            None
        }
    }
}

fn positive_usize_from_f64(n: f64, function: &'static str, label: &str) -> BuiltinResult<usize> {
    if !n.is_finite()
        || n.fract().abs() > f64::EPSILON
        || n < 1.0
        || n > usize::MAX as f64
        || (usize::BITS == 64 && n == usize::MAX as f64)
    {
        return Err(deep_learning_error(
            function,
            format!("{function}: {label} must be a positive integer"),
        ));
    }
    Ok(n as usize)
}

fn nonnegative_usize_from_f64(n: f64) -> Option<usize> {
    if n.is_finite()
        && n >= 0.0
        && n.fract() == 0.0
        && (n < usize::MAX as f64 || (usize::BITS < 64 && n == usize::MAX as f64))
    {
        Some(n as usize)
    } else {
        None
    }
}

pub(super) fn numeric_vector(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(value) => value
            .try_to_usize()
            .filter(|value| *value >= 1)
            .map(|value| vec![value])
            .ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: {label} must contain positive integers"),
                )
            }),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let storage = tensor.integer_storage().expect("checked integer storage");
            let mut out = Vec::with_capacity(storage.len());
            for index in 0..storage.len() {
                let Some(value) = storage
                    .value_at(index)
                    .and_then(|value| value.try_to_usize())
                    .filter(|value| *value >= 1)
                else {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: {label} must contain positive integers"),
                    ));
                };
                out.push(value);
            }
            Ok(out)
        }
        Value::Num(_) | Value::Tensor(_) => {
            let values = numeric_values(value, function, label)?;
            let mut out = Vec::with_capacity(values.len());
            for item in values {
                if !item.is_finite()
                    || item.fract().abs() > f64::EPSILON
                    || item < 1.0
                    || item > usize::MAX as f64
                    || (usize::BITS == 64 && item == usize::MAX as f64)
                {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: {label} must contain positive integers"),
                    ));
                }
                out.push(item as usize);
            }
            Ok(out)
        }
        other => Err(deep_learning_error(
            function,
            format!("{function}: {label} must be numeric, got {other:?}"),
        )),
    }
}

pub(super) fn numeric_values(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Tensor(t) => Ok(crate::builtins::common::tensor::tensor_values_f64(t)),
        other => Err(deep_learning_error(
            function,
            format!("{function}: {label} must be numeric, got {other:?}"),
        )),
    }
}

pub(super) fn text_or_missing(
    value: Option<&Value>,
    default: &str,
    function: &'static str,
) -> BuiltinResult<String> {
    match value {
        Some(v) => scalar_text(v, function),
        None => Ok(default.to_string()),
    }
}

pub(super) fn string_array(
    values: Vec<String>,
    shape: Vec<usize>,
    function: &'static str,
) -> BuiltinResult<Value> {
    StringArray::new(values, shape)
        .map(Value::StringArray)
        .map_err(|err| deep_learning_error(function, err))
}

pub(super) fn tensor_value(
    data: Vec<f64>,
    shape: Vec<usize>,
    function: &'static str,
) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(|err| deep_learning_error(function, err))
}

pub(super) fn object<K, I>(class_name: &str, properties: I) -> Value
where
    K: Into<String>,
    I: IntoIterator<Item = (K, Value)>,
{
    let mut object = ObjectInstance::new(class_name.to_string());
    for (name, value) in properties {
        object.properties.insert(name.into(), value);
    }
    Value::Object(object)
}

pub(super) fn layer_object(
    class_name: &str,
    type_name: &str,
    mut properties: Vec<(&str, Value)>,
    rest: Vec<Value>,
    function: &'static str,
) -> BuiltinResult<Value> {
    let mut owned_properties = properties
        .drain(..)
        .map(|(name, value)| (name.to_string(), value))
        .collect::<Vec<_>>();
    let mut name = String::new();
    let mut description = String::new();
    let mut extra = parse_name_values(rest, function)?;
    if let Some(value) = extra.remove("name") {
        name = scalar_text(&value, function)?;
    }
    if let Some(value) = extra.remove("description") {
        description = scalar_text(&value, function)?;
    }
    owned_properties.push(("Type".to_string(), Value::String(type_name.to_string())));
    owned_properties.push(("Name".to_string(), Value::String(name)));
    owned_properties.push(("Description".to_string(), Value::String(description)));
    for (key, value) in extra {
        owned_properties.push((canonical_property_name(&key), value));
    }
    Ok(object(class_name, owned_properties))
}

fn canonical_property_name(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "biaslearnratefactor" => "BiasLearnRateFactor",
        "biasl2factor" => "BiasL2Factor",
        "biasinitializer" => "BiasInitializer",
        "weightslearnratefactor" => "WeightsLearnRateFactor",
        "weightsl2factor" => "WeightsL2Factor",
        "weightsinitializer" => "WeightsInitializer",
        "inputnames" => "InputNames",
        "outputnames" => "OutputNames",
        "padding" => "Padding",
        "stride" => "Stride",
        "dilationfactor" => "DilationFactor",
        "numchannels" => "NumChannels",
        "hasstateinputs" => "HasStateInputs",
        "hasstateoutputs" => "HasStateOutputs",
        "outputmode" => "OutputMode",
        "stateactivationfunction" => "StateActivationFunction",
        "gateactivationfunction" => "GateActivationFunction",
        "normalization" => "Normalization",
        "splitcomplexinputs" => "SplitComplexInputs",
        "weights" => "Weights",
        "bias" => "Bias",
        "classes" => "Classes",
        "epsilon" => "Epsilon",
        "alphalearnratefactor" => "AlphaLearnRateFactor",
        "betalearnratefactor" => "BetaLearnRateFactor",
        "offset" => "Offset",
        "scale" => "Scale",
        other => other,
    }
    .to_string()
}

pub(super) fn parse_name_values(
    args: Vec<Value>,
    function: &'static str,
) -> BuiltinResult<std::collections::BTreeMap<String, Value>> {
    if !args.len().is_multiple_of(2) {
        return Err(deep_learning_error(
            function,
            format!("{function}: name-value options must be paired"),
        ));
    }
    let mut map = std::collections::BTreeMap::new();
    let mut idx = 0;
    while idx < args.len() {
        let name = scalar_text(&args[idx], function)?.to_ascii_lowercase();
        map.insert(name, args[idx + 1].clone());
        idx += 2;
    }
    Ok(map)
}

pub(super) fn layers_from_value(value: Value, function: &'static str) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Object(_) => Ok(vec![value]),
        Value::Cell(cell) => Ok(cell.data),
        Value::OutputList(values) => Ok(values),
        other => Err(deep_learning_error(
            function,
            format!("{function}: layers must be a layer object, cell array, or object array, got {other:?}"),
        )),
    }
}

pub(super) fn layer_names(layers: &[Value], function: &'static str) -> BuiltinResult<Vec<String>> {
    let mut names = Vec::with_capacity(layers.len());
    for (idx, layer) in layers.iter().enumerate() {
        match layer {
            Value::Object(object) => {
                let name = object
                    .properties
                    .get("Name")
                    .and_then(|value| match value {
                        Value::String(s) if !s.is_empty() => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| format!("layer_{}", idx + 1));
                names.push(name);
            }
            other => {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: layer list contains non-object value {other:?}"),
                ));
            }
        }
    }
    Ok(names)
}

pub(crate) mod autodiff;
pub(crate) mod graph;
pub(crate) mod layers;
pub(crate) mod losses;
pub(crate) mod model;
pub(crate) mod onnx;
pub(crate) mod sequences;
pub(crate) mod supervised;
pub(crate) mod training;

#[cfg(test)]
mod tests;
