//! MATLAB-compatible `structfun` builtin.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::structs::type_resolvers::structfun_type;
use crate::{
    build_runtime_error, call_feval_async_with_outputs, current_requested_outputs,
    gather_if_needed_async, BuiltinResult, RuntimeError,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, LogicalArray, StringArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::structfun")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "structfun",
    op_kind: GpuOpKind::Custom("host-struct-map"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes callbacks on the host through the cellfun mapping engine; GPU-resident field values are gathered before callback invocation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::structfun")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "structfun",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Callback execution happens on the host; fusion planners should treat structfun as a fusion barrier.",
};

const BUILTIN_NAME: &str = "structfun";

const STRUCTFUN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mapped callback outputs.",
}];

const STRUCTFUN_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or builtin/function name.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar struct input.",
    },
];

const STRUCTFUN_UNIFORM_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or builtin/function name.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar struct input.",
    },
    BuiltinParamDescriptor {
        name: "UniformOutput",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"UniformOutput\""),
        description: "Name-value key for uniform output mode.",
    },
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Required,
        default: Some("true"),
        description: "Whether callback outputs must be scalar-uniform.",
    },
];

const STRUCTFUN_ERROR_HANDLER_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or builtin/function name.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar struct input.",
    },
    BuiltinParamDescriptor {
        name: "ErrorHandler",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ErrorHandler\""),
        description: "Name-value key for callback error handler.",
    },
    BuiltinParamDescriptor {
        name: "errfunc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Error handler function handle or name.",
    },
];

const STRUCTFUN_BOTH_OPTIONS_INPUTS: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or builtin/function name.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar struct input.",
    },
    BuiltinParamDescriptor {
        name: "UniformOutput",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"UniformOutput\""),
        description: "Name-value key for uniform output mode.",
    },
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Required,
        default: Some("true"),
        description: "Whether callback outputs must be scalar-uniform.",
    },
    BuiltinParamDescriptor {
        name: "ErrorHandler",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ErrorHandler\""),
        description: "Name-value key for callback error handler.",
    },
    BuiltinParamDescriptor {
        name: "errfunc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Error handler function handle or name.",
    },
];

const STRUCTFUN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "A = structfun(func, S)",
        inputs: &STRUCTFUN_INPUTS,
        outputs: &STRUCTFUN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = structfun(func, S, \"UniformOutput\", tf)",
        inputs: &STRUCTFUN_UNIFORM_INPUTS,
        outputs: &STRUCTFUN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = structfun(func, S, \"ErrorHandler\", errfunc)",
        inputs: &STRUCTFUN_ERROR_HANDLER_INPUTS,
        outputs: &STRUCTFUN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = structfun(func, S, \"UniformOutput\", tf, \"ErrorHandler\", errfunc)",
        inputs: &STRUCTFUN_BOTH_OPTIONS_INPUTS,
        outputs: &STRUCTFUN_OUTPUT,
    },
];

const STRUCTFUN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRUCTFUN.INVALID_INPUT",
    identifier: Some("RunMat:structfun:InvalidInput"),
    when: "Input callback, struct argument, or name-value forms are invalid.",
    message: "structfun: invalid input arguments",
};

const STRUCTFUN_ERROR_NOT_SCALAR_STRUCT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRUCTFUN.NOT_SCALAR_STRUCT",
    identifier: Some("RunMat:structfun:NotScalarStruct"),
    when: "Second argument is not a scalar struct.",
    message: "structfun: second input must be a scalar struct",
};

const STRUCTFUN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRUCTFUN.INTERNAL",
    identifier: Some("RunMat:structfun:Internal"),
    when: "Internal field vector materialisation fails.",
    message: "structfun: internal error",
};

const STRUCTFUN_ERROR_UNIFORM_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRUCTFUN.UNIFORM_OUTPUT",
    identifier: Some("RunMat:structfun:UniformOutput"),
    when: "UniformOutput mode requirements are violated.",
    message: "structfun: uniform output contract violated",
};

const STRUCTFUN_ERROR_FUNCTION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRUCTFUN.FUNCTION_ERROR",
    identifier: Some("RunMat:structfun:FunctionError"),
    when: "Callback execution or callback output arity fails.",
    message: "structfun: callback execution error",
};

const STRUCTFUN_ERRORS: [BuiltinErrorDescriptor; 5] = [
    STRUCTFUN_ERROR_INVALID_INPUT,
    STRUCTFUN_ERROR_NOT_SCALAR_STRUCT,
    STRUCTFUN_ERROR_INTERNAL,
    STRUCTFUN_ERROR_UNIFORM_OUTPUT,
    STRUCTFUN_ERROR_FUNCTION_ERROR,
];

pub const STRUCTFUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRUCTFUN_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRUCTFUN_ERRORS,
};

#[runtime_builtin(
    name = "structfun",
    category = "structs/core",
    summary = "Apply a function to each field of a scalar struct.",
    keywords = "structfun,struct,functional,uniformoutput,errorhandler",
    type_resolver(structfun_type),
    descriptor(crate::builtins::structs::core::structfun::STRUCTFUN_DESCRIPTOR),
    builtin_path = "crate::builtins::structs::core::structfun"
)]
async fn structfun_builtin(
    func: Value,
    struct_value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let options = parse_options(&rest)?;
    let fields = match struct_value {
        Value::Struct(st) => FieldValues::from_struct(st),
        Value::Cell(_) => {
            return Err(structfun_error(
                STRUCTFUN_ERROR_NOT_SCALAR_STRUCT.message,
                &STRUCTFUN_ERROR_NOT_SCALAR_STRUCT,
            ))
        }
        other => {
            return Err(structfun_error(
                format!(
                    "{} (got {other:?})",
                    STRUCTFUN_ERROR_NOT_SCALAR_STRUCT.message
                ),
                &STRUCTFUN_ERROR_NOT_SCALAR_STRUCT,
            ))
        }
    };

    execute_structfun(func, fields, options).await
}

struct FieldValues {
    names: Vec<String>,
    values: Vec<Value>,
}

impl FieldValues {
    fn from_struct(struct_value: StructValue) -> Self {
        let mut names = Vec::with_capacity(struct_value.fields.len());
        let mut values = Vec::with_capacity(struct_value.fields.len());
        for (name, value) in struct_value.fields {
            names.push(name);
            values.push(value);
        }
        Self { names, values }
    }
}

#[derive(Clone)]
struct StructfunOptions {
    uniform_output: bool,
    error_handler: Option<Value>,
}

fn parse_options(rest: &[Value]) -> BuiltinResult<StructfunOptions> {
    if !rest.len().is_multiple_of(2) {
        return Err(structfun_error(
            "structfun: optional arguments must be name-value pairs",
            &STRUCTFUN_ERROR_INVALID_INPUT,
        ));
    }

    let mut options = StructfunOptions {
        uniform_output: true,
        error_handler: None,
    };

    for pair in rest.chunks_exact(2) {
        let Some(name) = extract_option_name(&pair[0]) else {
            return Err(structfun_error(
                "structfun: option names must be character vectors or string scalars",
                &STRUCTFUN_ERROR_INVALID_INPUT,
            ));
        };
        match name.trim().to_ascii_lowercase().as_str() {
            "uniformoutput" => {
                options.uniform_output = parse_bool_option(&pair[1])?;
            }
            "errorhandler" => {
                options.error_handler = Some(pair[1].clone());
            }
            other => {
                return Err(structfun_error(
                    format!("structfun: unknown name-value argument '{other}'"),
                    &STRUCTFUN_ERROR_INVALID_INPUT,
                ))
            }
        }
    }

    Ok(options)
}

fn extract_option_name(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(StringArray { data, .. }) if data.len() == 1 => Some(data[0].clone()),
        _ => None,
    }
}

fn parse_bool_option(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(n) => Ok(*n != 0.0),
        Value::Int(int) => Ok(int.to_f64() != 0.0),
        Value::String(text) => parse_bool_text(text).ok_or_else(uniform_option_error),
        Value::CharArray(chars) if chars.rows == 1 => {
            let text: String = chars.data.iter().collect();
            parse_bool_text(&text).ok_or_else(uniform_option_error)
        }
        Value::StringArray(StringArray { data, .. }) if data.len() == 1 => {
            parse_bool_text(&data[0]).ok_or_else(uniform_option_error)
        }
        _ => Err(uniform_option_error()),
    }
}

fn parse_bool_text(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "on" => Some(true),
        "false" | "off" => Some(false),
        _ => None,
    }
}

fn uniform_option_error() -> RuntimeError {
    structfun_error(
        "structfun: UniformOutput must be logical true or false",
        &STRUCTFUN_ERROR_UNIFORM_OUTPUT,
    )
}

async fn execute_structfun(
    func: Value,
    fields: FieldValues,
    options: StructfunOptions,
) -> BuiltinResult<Value> {
    let requested_outputs = current_requested_outputs();
    if requested_outputs == 0 {
        return Ok(Value::OutputList(Vec::new()));
    }

    let mut collectors = (0..requested_outputs)
        .map(|_| OutputCollector::new(options.uniform_output, &fields.names))
        .collect::<Vec<_>>();

    for (linear_idx, value) in fields.values.iter().enumerate() {
        let field_value = gather_if_needed_async(value).await?;
        let result = match call_feval_async_with_outputs(
            func.clone(),
            std::slice::from_ref(&field_value),
            requested_outputs,
        )
        .await
        {
            Ok(value) => value,
            Err(err) => {
                let Some(handler) = options.error_handler.as_ref() else {
                    return Err(wrap_callback_error(err));
                };
                let err_struct = make_error_struct(&err, linear_idx, &fields.names[linear_idx]);
                call_feval_async_with_outputs(
                    handler.clone(),
                    &[err_struct, field_value.clone()],
                    requested_outputs,
                )
                .await
                .map_err(wrap_callback_error)?
            }
        };

        let outputs = normalize_callback_outputs(result, requested_outputs)?;
        for (collector, output) in collectors.iter_mut().zip(outputs.into_iter()) {
            let host_output = gather_if_needed_async(&output).await?;
            collector.push(&fields.names[linear_idx], host_output)?;
        }
    }

    let outputs = collectors
        .into_iter()
        .map(OutputCollector::finish)
        .collect::<BuiltinResult<Vec<_>>>()?;
    if requested_outputs == 1 {
        Ok(outputs.into_iter().next().unwrap())
    } else {
        Ok(Value::OutputList(outputs))
    }
}

fn normalize_callback_outputs(value: Value, requested_outputs: usize) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::OutputList(values) if values.len() == requested_outputs => Ok(values),
        Value::OutputList(values) => Err(structfun_error(
            format!(
                "structfun: callback returned {} outputs but {} were requested",
                values.len(),
                requested_outputs
            ),
            &STRUCTFUN_ERROR_FUNCTION_ERROR,
        )),
        other if requested_outputs == 1 => Ok(vec![other]),
        _ => Err(structfun_error(
            "structfun: callback did not return the requested number of outputs",
            &STRUCTFUN_ERROR_FUNCTION_ERROR,
        )),
    }
}

fn make_error_struct(error: &RuntimeError, linear_idx: usize, field_name: &str) -> Value {
    let mut st = StructValue::new();
    st.insert(
        "identifier",
        Value::String(error.identifier().unwrap_or("").to_string()),
    );
    st.insert("message", Value::String(error.message().to_string()));
    st.insert("index", Value::Num((linear_idx + 1) as f64));
    st.insert("field", Value::String(field_name.to_string()));
    Value::Struct(st)
}

enum OutputCollector {
    Uniform(UniformCollector),
    Struct(StructValue),
}

impl OutputCollector {
    fn new(uniform_output: bool, names: &[String]) -> Self {
        if uniform_output {
            OutputCollector::Uniform(UniformCollector::new(names.len()))
        } else {
            OutputCollector::Struct(StructValue::new())
        }
    }

    fn push(&mut self, field_name: &str, value: Value) -> BuiltinResult<()> {
        match self {
            OutputCollector::Uniform(collector) => collector.push(&value),
            OutputCollector::Struct(out) => {
                out.insert(field_name.to_string(), value);
                Ok(())
            }
        }
    }

    fn finish(self) -> BuiltinResult<Value> {
        match self {
            OutputCollector::Uniform(collector) => collector.finish(),
            OutputCollector::Struct(out) => Ok(Value::Struct(out)),
        }
    }
}

enum UniformCollector {
    Pending { len: usize },
    Double { data: Vec<f64>, len: usize },
    Logical { data: Vec<u8>, len: usize },
    Complex { data: Vec<(f64, f64)>, len: usize },
}

impl UniformCollector {
    fn new(len: usize) -> Self {
        UniformCollector::Pending { len }
    }

    fn push(&mut self, value: &Value) -> BuiltinResult<()> {
        match self {
            UniformCollector::Pending { len } => match classify_uniform_value(value)? {
                ClassifiedValue::Logical(flag) => {
                    *self = UniformCollector::Logical {
                        data: vec![flag as u8],
                        len: *len,
                    };
                    Ok(())
                }
                ClassifiedValue::Double(value) => {
                    *self = UniformCollector::Double {
                        data: vec![value],
                        len: *len,
                    };
                    Ok(())
                }
                ClassifiedValue::Complex(value) => {
                    *self = UniformCollector::Complex {
                        data: vec![value],
                        len: *len,
                    };
                    Ok(())
                }
            },
            UniformCollector::Logical { data, len } => match classify_uniform_value(value)? {
                ClassifiedValue::Logical(flag) => {
                    data.push(flag as u8);
                    Ok(())
                }
                ClassifiedValue::Double(value) => {
                    let mut values = data
                        .iter()
                        .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                        .collect::<Vec<_>>();
                    values.push(value);
                    *self = UniformCollector::Double {
                        data: values,
                        len: *len,
                    };
                    Ok(())
                }
                ClassifiedValue::Complex(value) => {
                    let mut values = data
                        .iter()
                        .map(|&bit| if bit != 0 { (1.0, 0.0) } else { (0.0, 0.0) })
                        .collect::<Vec<_>>();
                    values.push(value);
                    *self = UniformCollector::Complex {
                        data: values,
                        len: *len,
                    };
                    Ok(())
                }
            },
            UniformCollector::Double { data, len } => match classify_uniform_value(value)? {
                ClassifiedValue::Logical(flag) => {
                    data.push(if flag { 1.0 } else { 0.0 });
                    Ok(())
                }
                ClassifiedValue::Double(value) => {
                    data.push(value);
                    Ok(())
                }
                ClassifiedValue::Complex(value) => {
                    let mut values = data.iter().map(|&value| (value, 0.0)).collect::<Vec<_>>();
                    values.push(value);
                    *self = UniformCollector::Complex {
                        data: values,
                        len: *len,
                    };
                    Ok(())
                }
            },
            UniformCollector::Complex { data, .. } => match classify_uniform_value(value)? {
                ClassifiedValue::Logical(flag) => {
                    data.push((if flag { 1.0 } else { 0.0 }, 0.0));
                    Ok(())
                }
                ClassifiedValue::Double(value) => {
                    data.push((value, 0.0));
                    Ok(())
                }
                ClassifiedValue::Complex(value) => {
                    data.push(value);
                    Ok(())
                }
            },
        }
    }

    fn finish(self) -> BuiltinResult<Value> {
        match self {
            UniformCollector::Pending { len } => Tensor::new(Vec::new(), vec![len, 1])
                .map(Value::Tensor)
                .map_err(|err| {
                    structfun_error(format!("structfun: {err}"), &STRUCTFUN_ERROR_INTERNAL)
                }),
            UniformCollector::Double { data, len } => Tensor::new(data, vec![len, 1])
                .map(Value::Tensor)
                .map_err(|err| {
                    structfun_error(format!("structfun: {err}"), &STRUCTFUN_ERROR_INTERNAL)
                }),
            UniformCollector::Logical { data, len } => LogicalArray::new(data, vec![len, 1])
                .map(Value::LogicalArray)
                .map_err(|err| {
                    structfun_error(format!("structfun: {err}"), &STRUCTFUN_ERROR_INTERNAL)
                }),
            UniformCollector::Complex { data, len } => ComplexTensor::new(data, vec![len, 1])
                .map(Value::ComplexTensor)
                .map_err(|err| {
                    structfun_error(format!("structfun: {err}"), &STRUCTFUN_ERROR_INTERNAL)
                }),
        }
    }
}

enum ClassifiedValue {
    Logical(bool),
    Double(f64),
    Complex((f64, f64)),
}

fn classify_uniform_value(value: &Value) -> BuiltinResult<ClassifiedValue> {
    match value {
        Value::Bool(flag) => Ok(ClassifiedValue::Logical(*flag)),
        Value::Num(value) => Ok(ClassifiedValue::Double(*value)),
        Value::Int(value) => Ok(ClassifiedValue::Double(value.to_f64())),
        Value::Complex(re, im) => Ok(ClassifiedValue::Complex((*re, *im))),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(ClassifiedValue::Double(
            tensor::tensor_values_f64(tensor)[0],
        )),
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(ClassifiedValue::Logical(array.data[0] != 0))
        }
        Value::ComplexTensor(tensor) if tensor.data.len() == 1 => {
            Ok(ClassifiedValue::Complex(tensor.data[0]))
        }
        _ => Err(structfun_error(
            "structfun: callback must return scalar values when 'UniformOutput' is true",
            &STRUCTFUN_ERROR_UNIFORM_OUTPUT,
        )),
    }
}

fn wrap_callback_error(error: RuntimeError) -> RuntimeError {
    structfun_error(
        error.message().replace("cellfun:", "structfun:"),
        &STRUCTFUN_ERROR_FUNCTION_ERROR,
    )
}

fn structfun_error(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{CellArray, IntegerStorage, Tensor};
    use std::sync::Arc;

    fn call(func: Value, st: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(structfun_builtin(func, st, rest))
    }

    #[test]
    fn uniform_classifier_reads_typed_integer_tensor_storage_exactly() {
        let mut tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("integer tensor");
        tensor.data[0] = 0.0;

        match classify_uniform_value(&Value::Tensor(tensor)).expect("classify") {
            ClassifiedValue::Double(value) => {
                assert_eq!(value, 9_007_199_254_740_993_u64 as f64);
            }
            _ => panic!("expected double classification"),
        }
    }

    fn sample_struct() -> StructValue {
        let mut st = StructValue::new();
        st.insert(
            "alpha",
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
        );
        st.insert(
            "beta",
            Value::Tensor(Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap()),
        );
        st
    }

    #[test]
    fn structfun_length_uniform_default_returns_column_vector() {
        let result = call(
            Value::String("@length".into()),
            Value::Struct(sample_struct()),
            Vec::new(),
        )
        .expect("structfun");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data, vec![3.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn structfun_uniform_output_false_preserves_field_names() {
        let mut st = StructValue::new();
        st.insert("name", Value::String("runmat".into()));
        st.insert("flag", Value::Bool(true));

        let result = call(
            Value::String("@class".into()),
            Value::Struct(st),
            vec![Value::String("UniformOutput".into()), Value::Bool(false)],
        )
        .expect("structfun");
        match result {
            Value::Struct(out) => {
                assert_eq!(
                    out.fields.get("name"),
                    Some(&Value::String("string".into()))
                );
                assert_eq!(
                    out.fields.get("flag"),
                    Some(&Value::String("logical".into()))
                );
            }
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[test]
    fn structfun_multiple_outputs_collect_each_output() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 2);
                let Value::Num(input) = args[0] else {
                    panic!("expected numeric input");
                };
                Box::pin(async move {
                    Ok(Value::OutputList(vec![
                        Value::Num(input),
                        Value::Num(input + 10.0),
                    ]))
                })
            },
        )));
        let _output_guard = crate::output_count::push_output_count(Some(2));
        let mut st = StructValue::new();
        st.insert("a", Value::Num(1.0));
        st.insert("b", Value::Num(2.0));
        let handle = Value::BoundFunctionHandle {
            name: "two_outputs".to_string(),
            function: 91,
        };

        let result = call(handle, Value::Struct(st), Vec::new()).expect("structfun");
        match result {
            Value::OutputList(outputs) => {
                assert_eq!(outputs.len(), 2);
                match &outputs[0] {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0, 2.0]),
                    other => panic!("expected first tensor, got {other:?}"),
                }
                match &outputs[1] {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![11.0, 12.0]),
                    other => panic!("expected second tensor, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn structfun_multiple_outputs_with_uniform_false_return_structs() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 2);
                let Value::Num(input) = args[0] else {
                    panic!("expected numeric input");
                };
                Box::pin(async move {
                    Ok(Value::OutputList(vec![
                        Value::String(format!("v{input}")),
                        Value::Bool(input > 1.0),
                    ]))
                })
            },
        )));
        let _output_guard = crate::output_count::push_output_count(Some(2));
        let mut st = StructValue::new();
        st.insert("a", Value::Num(1.0));
        st.insert("b", Value::Num(2.0));
        let handle = Value::BoundFunctionHandle {
            name: "two_outputs".to_string(),
            function: 92,
        };

        let result = call(
            handle,
            Value::Struct(st),
            vec![Value::String("UniformOutput".into()), Value::Bool(false)],
        )
        .expect("structfun");
        match result {
            Value::OutputList(outputs) => {
                assert_eq!(outputs.len(), 2);
                match &outputs[0] {
                    Value::Struct(out) => {
                        assert_eq!(out.fields.get("a"), Some(&Value::String("v1".into())));
                        assert_eq!(out.fields.get("b"), Some(&Value::String("v2".into())));
                    }
                    other => panic!("expected first struct, got {other:?}"),
                }
                match &outputs[1] {
                    Value::Struct(out) => {
                        assert_eq!(out.fields.get("a"), Some(&Value::Bool(false)));
                        assert_eq!(out.fields.get("b"), Some(&Value::Bool(true)));
                    }
                    other => panic!("expected second struct, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn structfun_callback_errors_use_structfun_identifier() {
        let err = call(
            Value::String("@missing_structfun_callback".into()),
            Value::Struct(sample_struct()),
            Vec::new(),
        )
        .expect_err("missing callback should fail");
        assert_eq!(err.identifier(), STRUCTFUN_ERROR_FUNCTION_ERROR.identifier);
        assert!(err.message().contains("Undefined function"));
    }

    #[test]
    fn structfun_rejects_non_scalar_struct_array() {
        let array = CellArray::new(vec![Value::Struct(StructValue::new())], 1, 1).unwrap();
        let err = call(
            Value::String("@length".into()),
            Value::Cell(array),
            Vec::new(),
        )
        .expect_err("struct arrays should fail");
        assert_eq!(
            err.identifier(),
            STRUCTFUN_ERROR_NOT_SCALAR_STRUCT.identifier
        );
    }

    #[test]
    fn structfun_rejects_extra_non_option_arguments() {
        let err = call(
            Value::String("@length".into()),
            Value::Struct(sample_struct()),
            vec![Value::Num(1.0)],
        )
        .expect_err("extra positional args should fail");
        assert_eq!(err.identifier(), STRUCTFUN_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn structfun_descriptor_covers_option_forms() {
        let labels = STRUCTFUN_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"A = structfun(func, S)"));
        assert!(labels.contains(&"A = structfun(func, S, \"UniformOutput\", tf)"));
        assert!(labels.contains(&"A = structfun(func, S, \"ErrorHandler\", errfunc)"));
        assert!(labels
            .contains(&"A = structfun(func, S, \"UniformOutput\", tf, \"ErrorHandler\", errfunc)"));
    }
}
