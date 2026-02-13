//! MATLAB-compatible `empty` constructor used by ClassName.empty fallbacks.

use runmat_builtins::{CharArray, LogicalArray, NumericDType, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{extract_dims, keyword_of, shape_from_value};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, make_cell_with_shape, BuiltinResult, RuntimeError};

const LABEL: &str = "empty";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmptyClass {
    Double,
    Single,
    Logical,
    Char,
    String,
    Cell,
    Struct,
    GpuArray,
}

fn empty_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(LABEL).build()
}

fn class_from_keyword(keyword: &str) -> Option<EmptyClass> {
    match keyword {
        "double" => Some(EmptyClass::Double),
        "single" => Some(EmptyClass::Single),
        "logical" => Some(EmptyClass::Logical),
        "char" => Some(EmptyClass::Char),
        "string" => Some(EmptyClass::String),
        "cell" => Some(EmptyClass::Cell),
        "struct" => Some(EmptyClass::Struct),
        "gpuarray" => Some(EmptyClass::GpuArray),
        _ => None,
    }
}

fn split_class_arg(mut args: Vec<Value>) -> (EmptyClass, Vec<Value>) {
    if let Some(last) = args.last() {
        if let Some(keyword) = keyword_of(last) {
            if let Some(class) = class_from_keyword(keyword.as_str()) {
                args.pop();
                return (class, args);
            }
        }
    }
    (EmptyClass::Double, args)
}

#[runtime_builtin(
    name = "empty",
    category = "array/creation",
    summary = "Construct an empty array (used by ClassName.empty fallbacks).",
    keywords = "empty,preallocate,array",
    accel = "none",
    builtin_path = "crate::builtins::array::creation::empty"
)]
async fn empty_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (class, args) = split_class_arg(rest);

    if class == EmptyClass::String {
        return crate::call_builtin_async("string.empty", &args)
            .await
            .map_err(Into::into);
    }

    let shape = parse_shape(&args).await?;
    ensure_empty_shape(&shape)?;

    match class {
        EmptyClass::Double => {
            let tensor = tensor::zeros(&shape).map_err(|e| empty_error(format!("{LABEL}: {e}")))?;
            Ok(tensor::tensor_into_value(tensor))
        }
        EmptyClass::Single => {
            let tensor = tensor::zeros_with_dtype(&shape, NumericDType::F32)
                .map_err(|e| empty_error(format!("{LABEL}: {e}")))?;
            Ok(tensor::tensor_into_value(tensor))
        }
        EmptyClass::Logical => Ok(Value::LogicalArray(LogicalArray::zeros(shape))),
        EmptyClass::Char => {
            let (rows, cols) = char_shape(&shape)?;
            let chars = CharArray::new(Vec::new(), rows, cols)
                .map_err(|e| empty_error(format!("{LABEL}: {e}")))?;
            Ok(Value::CharArray(chars))
        }
        EmptyClass::Cell => make_cell_with_shape(Vec::new(), shape)
            .map_err(|e| empty_error(format!("{LABEL}: {e}"))),
        EmptyClass::Struct => make_cell_with_shape(Vec::new(), shape)
            .map_err(|e| empty_error(format!("{LABEL}: {e}"))),
        EmptyClass::GpuArray => {
            let zeros_args = dims_to_args(&shape);
            crate::call_builtin_async("zeros", &zeros_args)
                .await
                .map_err(Into::into)
        }
        EmptyClass::String => unreachable!("string.empty handled above"),
    }
}

async fn parse_shape(args: &[Value]) -> BuiltinResult<Vec<usize>> {
    if args.is_empty() {
        return Ok(vec![0, 0]);
    }

    let mut explicit_dims: Vec<usize> = Vec::new();
    let mut like_shape: Option<Vec<usize>> = None;
    let mut idx = 0;
    let mut saw_dims = false;

    while idx < args.len() {
        let arg = &args[idx];
        if let Some(keyword) = keyword_of(arg) {
            if keyword.as_str() == "like" {
                if like_shape.is_some() {
                    return Err(empty_error(
                        "empty: multiple 'like' prototypes are not supported",
                    ));
                }
                let Some(proto) = args.get(idx + 1) else {
                    return Err(empty_error("empty: expected prototype after 'like'"));
                };
                like_shape = Some(shape_from_value(proto, LABEL).map_err(empty_error)?);
                idx += 2;
                continue;
            }
        }

        if let Some(parsed) = extract_dims(arg, LABEL).await.map_err(empty_error)? {
            saw_dims = true;
            if explicit_dims.is_empty() {
                explicit_dims = parsed;
            } else {
                explicit_dims.extend(parsed);
            }
            idx += 1;
            continue;
        }

        return Err(empty_error(format!(
            "{LABEL}: size inputs must be numeric scalars or size vectors"
        )));
    }

    if saw_dims {
        Ok(dims_to_shape(&explicit_dims))
    } else if let Some(shape) = like_shape {
        Ok(shape)
    } else {
        Ok(vec![0, 0])
    }
}

fn dims_to_shape(dims: &[usize]) -> Vec<usize> {
    match dims.len() {
        0 => vec![0, 0],
        1 => vec![dims[0], dims[0]],
        _ => dims.to_vec(),
    }
}

fn ensure_empty_shape(shape: &[usize]) -> BuiltinResult<()> {
    if shape.iter().product::<usize>() != 0 {
        return Err(empty_error(
            "empty: at least one dimension must be zero to construct an empty array",
        ));
    }
    Ok(())
}

fn char_shape(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    match shape.len() {
        0 => Ok((0, 0)),
        1 => Ok((shape[0], shape[0])),
        2 => Ok((shape[0], shape[1])),
        _ => Err(empty_error(
            "empty: character arrays must be 2-D (use char.empty(m, n))",
        )),
    }
}

fn dims_to_args(shape: &[usize]) -> Vec<Value> {
    let mut args: Vec<Value> = shape.iter().map(|&d| Value::from(d as f64)).collect();
    args.push(Value::from("gpuArray"));
    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn empty_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::empty_builtin(rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_empty_zero_by_n() {
        let result = empty_builtin(vec![
            Value::from(0.0),
            Value::from(5.0),
            Value::from("char"),
        ])
        .expect("char.empty");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 0);
                assert_eq!(ca.cols, 5);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_empty_keeps_shape() {
        let result = empty_builtin(vec![
            Value::from(2.0),
            Value::from(0.0),
            Value::from(4.0),
            Value::from("cell"),
        ])
        .expect("cell.empty");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.shape, vec![2, 0, 4]);
                assert!(cell.data.is_empty());
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }
}
