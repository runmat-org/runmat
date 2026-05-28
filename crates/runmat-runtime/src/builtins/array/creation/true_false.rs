//! MATLAB-compatible `true`/`false` builtins for logical array creation.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    LogicalArray, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{shape::normalize_scalar_shape, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn builtin_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

const LOGICAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical output array.",
}];

const LOGICAL_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const LOGICAL_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const LOGICAL_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const LOGICAL_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const LOGICAL_SIG_PROTOTYPE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prototype",
    ty: BuiltinParamType::LikePrototype,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Prototype value when no numeric dimension arguments are provided.",
}];

const LOGICAL_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"logical\""),
        description: "Class override keyword (logical).",
    },
];

const LOGICAL_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype array used for class/device.",
    },
];

const TRUE_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "L = true()",
        inputs: &LOGICAL_SIG_EMPTY_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(n)",
        inputs: &LOGICAL_SIG_N_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(size_vector)",
        inputs: &LOGICAL_SIG_SIZE_VECTOR_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(m, n, ...)",
        inputs: &LOGICAL_SIG_DIMS_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(prototype)",
        inputs: &LOGICAL_SIG_PROTOTYPE_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(..., \"logical\")",
        inputs: &LOGICAL_SIG_CLASS_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(..., \"like\", prototype)",
        inputs: &LOGICAL_SIG_LIKE_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
];

const FALSE_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "L = false()",
        inputs: &LOGICAL_SIG_EMPTY_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(n)",
        inputs: &LOGICAL_SIG_N_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(size_vector)",
        inputs: &LOGICAL_SIG_SIZE_VECTOR_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(m, n, ...)",
        inputs: &LOGICAL_SIG_DIMS_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(prototype)",
        inputs: &LOGICAL_SIG_PROTOTYPE_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(..., \"logical\")",
        inputs: &LOGICAL_SIG_CLASS_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(..., \"like\", prototype)",
        inputs: &LOGICAL_SIG_LIKE_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
];

const TRUE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.TRUE.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "true: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.TRUE.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "true: multiple 'like' specifications are not supported",
    },
    BuiltinErrorDescriptor {
        code: "RM.TRUE.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not supported.",
        message: "true: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.TRUE.INVALID_DIMS",
        identifier: None,
        when: "Dimension arguments fail numeric/shape parsing.",
        message: "true: dimension arguments must be numeric and nonnegative",
    },
];

const FALSE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.FALSE.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "false: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.FALSE.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "false: multiple 'like' specifications are not supported",
    },
    BuiltinErrorDescriptor {
        code: "RM.FALSE.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not supported.",
        message: "false: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.FALSE.INVALID_DIMS",
        identifier: None,
        when: "Dimension arguments fail numeric/shape parsing.",
        message: "false: dimension arguments must be numeric and nonnegative",
    },
];

pub const TRUE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRUE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRUE_ERRORS,
};

pub const FALSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FALSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FALSE_ERRORS,
};

#[runtime_builtin(
    name = "true",
    category = "array/creation",
    summary = "Create logical arrays filled with true values.",
    keywords = "true,logical,array",
    accel = "array_construct",
    descriptor(crate::builtins::array::creation::true_false::TRUE_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::true_false"
)]
async fn true_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    logical_fill(rest, true, "true").await
}

#[runtime_builtin(
    name = "false",
    category = "array/creation",
    summary = "Create logical arrays filled with false values.",
    keywords = "false,logical,array",
    accel = "array_construct",
    descriptor(crate::builtins::array::creation::true_false::FALSE_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::true_false"
)]
async fn false_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    logical_fill(rest, false, "false").await
}

async fn logical_fill(args: Vec<Value>, value: bool, name: &str) -> BuiltinResult<Value> {
    let parsed = ParsedLogical::parse(args, name).await?;
    let len = tensor::element_count(&parsed.shape);
    if len == 1 {
        return Ok(Value::Bool(value));
    }
    let data = vec![if value { 1u8 } else { 0u8 }; len];
    LogicalArray::new(data, parsed.shape)
        .map(Value::LogicalArray)
        .map_err(|e| builtin_error(name, format!("{name}: {e}")))
}

struct ParsedLogical {
    shape: Vec<usize>,
}

impl ParsedLogical {
    async fn parse(args: Vec<Value>, name: &str) -> BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut saw_like = false;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        if saw_like {
                            return Err(builtin_error(
                                name,
                                format!("{name}: multiple 'like' specifications are not supported"),
                            ));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error(
                                name,
                                format!("{name}: expected prototype after 'like'"),
                            ));
                        };
                        saw_like = true;
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source =
                                Some(shape_from_value(&proto).map_err(|e| builtin_error(name, e))?);
                        }
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(builtin_error(
                            name,
                            format!("{name}: unrecognised option '{other}'"),
                        ));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg, name).await? {
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            if shape_source.is_none() {
                shape_source = Some(shape_from_value(&arg).map_err(|e| builtin_error(name, e))?);
            }
            idx += 1;
        }

        let shape = if saw_dims_arg {
            if dims.is_empty() {
                vec![0, 0]
            } else if dims.len() == 1 {
                vec![dims[0], dims[0]]
            } else {
                dims
            }
        } else if let Some(shape) = shape_source {
            shape
        } else {
            vec![1, 1]
        };

        Ok(Self { shape })
    }
}

fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            Some(text.to_ascii_lowercase())
        }
        _ => None,
    }
}

async fn extract_dims(value: &Value, name: &str) -> BuiltinResult<Option<Vec<usize>>> {
    if matches!(value, Value::LogicalArray(_)) {
        return Ok(None);
    }
    let gpu_scalar = match value {
        Value::GpuTensor(handle) => tensor::element_count(&handle.shape) == 1,
        _ => false,
    };
    match tensor::dims_from_value_async(value).await {
        Ok(dims) => Ok(dims),
        Err(err) => {
            if matches!(value, Value::Tensor(_))
                || (matches!(value, Value::GpuTensor(_)) && !gpu_scalar)
            {
                Ok(None)
            } else {
                Err(builtin_error(name, format!("{name}: {err}")))
            }
        }
    }
}

fn shape_from_value(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::ComplexTensor(t) => Ok(t.shape.clone()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(normalize_scalar_shape(&h.shape)),
        Value::CharArray(ca) => Ok(vec![ca.rows, ca.cols]),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(format!("unsupported prototype {other:?}")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn true_default_scalar() {
        let result = block_on(true_builtin(Vec::new())).expect("true");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn false_default_scalar() {
        let result = block_on(false_builtin(Vec::new())).expect("false");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn true_with_dims() {
        let args = vec![Value::Num(2.0), Value::Num(1.0)];
        let result = block_on(true_builtin(args)).expect("true");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 1]);
                assert!(logical.data.iter().all(|&x| x == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn false_from_size_vector() {
        let size_vec = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let args = vec![Value::Tensor(size_vec)];
        let result = block_on(false_builtin(args)).expect("false");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 3]);
                assert!(logical.data.iter().all(|&x| x == 0));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }
}
