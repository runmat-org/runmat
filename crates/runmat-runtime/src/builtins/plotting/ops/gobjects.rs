//! MATLAB-compatible `gobjects` builtin.

use crate::builtins::common::tensor;
use crate::builtins::plotting::type_resolvers::handle_array_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

const GOBJECTS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Array of invalid graphics-handle placeholders for preallocation.",
}];

const GOBJECTS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const GOBJECTS_INPUTS_DIMS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output dimensions, either as separate scalar sizes or a size vector.",
}];

const GOBJECTS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "h = gobjects()",
        inputs: &GOBJECTS_INPUTS_NONE,
        outputs: &GOBJECTS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = gobjects(sz...)",
        inputs: &GOBJECTS_INPUTS_DIMS,
        outputs: &GOBJECTS_OUTPUT,
    },
];

const GOBJECTS_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const GOBJECTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GOBJECTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GOBJECTS_ERRORS,
};

#[runtime_builtin(
    name = "gobjects",
    category = "plotting",
    summary = "Create an array of graphics-handle placeholders.",
    keywords = "gobjects,graphics,handle,preallocate,plotting",
    suppress_auto_output = true,
    type_resolver(handle_array_type),
    descriptor(crate::builtins::plotting::gobjects::GOBJECTS_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::gobjects"
)]
pub async fn gobjects_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let shape = parse_shape(args).await?;
    let len = tensor::element_count(&shape);
    let tensor = Tensor::new_with_dtype(vec![f64::NAN; len], shape, NumericDType::F64)
        .map_err(|err| format!("gobjects: {err}"))?;
    Ok(Value::Tensor(tensor))
}

async fn parse_shape(args: Vec<Value>) -> crate::BuiltinResult<Vec<usize>> {
    if args.is_empty() {
        return Ok(vec![1, 1]);
    }

    let mut dims = Vec::new();
    for arg in args {
        let Some(mut parsed) = tensor::dims_from_value_async(&arg)
            .await
            .map_err(|err| format!("gobjects: {err}"))?
        else {
            return Err(
                "gobjects: size arguments must be numeric scalar sizes or a size vector".into(),
            );
        };
        dims.append(&mut parsed);
    }

    Ok(match dims.len() {
        0 => vec![0, 0],
        1 => vec![dims[0], dims[0]],
        _ => dims,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn tensor_from(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gobjects_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = GOBJECTS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = gobjects()"));
        assert!(labels.contains(&"h = gobjects(sz...)"));
    }

    #[test]
    fn gobjects_defaults_to_scalar_placeholder() {
        let tensor = tensor_from(block_on(gobjects_builtin(Vec::new())).unwrap());
        assert_eq!(tensor.shape, vec![1, 1]);
        assert_eq!(tensor.data.len(), 1);
        assert!(tensor.data[0].is_nan());
    }

    #[test]
    fn gobjects_accepts_scalar_dims_and_size_vector() {
        let square = tensor_from(block_on(gobjects_builtin(vec![Value::Num(3.0)])).unwrap());
        assert_eq!(square.shape, vec![3, 3]);
        assert!(square.data.iter().all(|value| value.is_nan()));

        let rect = tensor_from(
            block_on(gobjects_builtin(vec![Value::Num(2.0), Value::Num(3.0)])).unwrap(),
        );
        assert_eq!(rect.shape, vec![2, 3]);

        let size_vec = Tensor::new(vec![4.0, 1.0], vec![1, 2]).unwrap();
        let vector =
            tensor_from(block_on(gobjects_builtin(vec![Value::Tensor(size_vec)])).unwrap());
        assert_eq!(vector.shape, vec![4, 1]);
    }

    #[test]
    fn gobjects_rejects_invalid_dimensions() {
        assert!(block_on(gobjects_builtin(vec![Value::Num(-1.0)])).is_err());
        assert!(block_on(gobjects_builtin(vec![Value::Num(1.5)])).is_err());
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert!(block_on(gobjects_builtin(vec![Value::Tensor(matrix)])).is_err());
    }
}
