//! MATLAB-compatible `int32` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::elementwise::integer_cast::{cast_value, CastError, IntegerTarget};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "int32";

const INT32_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "int32-converted output value.",
}];

const INT32_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar/array value to convert.",
}];

const INT32_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = int32(X)",
    inputs: &INT32_INPUTS_X,
    outputs: &INT32_OUTPUT,
}];

const INT32_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INT32.INVALID_ARGUMENT",
    identifier: Some("RunMat:int32:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "int32: invalid argument",
};

const INT32_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INT32.INVALID_INPUT",
    identifier: Some("RunMat:int32:InvalidInput"),
    when: "Input value cannot be converted to int32.",
    message: "int32: invalid input",
};

const INT32_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INT32.INTERNAL",
    identifier: Some("RunMat:int32:Internal"),
    when: "Internal conversion, gather, or provider upload failed.",
    message: "int32: internal error",
};

const INT32_ERRORS: [BuiltinErrorDescriptor; 3] = [
    INT32_ERROR_INVALID_ARGUMENT,
    INT32_ERROR_INVALID_INPUT,
    INT32_ERROR_INTERNAL,
];

pub const INT32_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INT32_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INT32_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::int32")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "int32",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("cast_to_integer")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Real gpuArray inputs use the provider resident integer-cast hook and return native int32 gpuArray storage. Complex gpuArray integer casts remain unsupported until typed complex integer provider storage exists.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::int32")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "int32",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Resident integer cast uses provider-native integer storage; fusion can target the provider hook when integer buffers are supported.",
};

fn int32_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    int32_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn int32_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn conversion_error(type_name: &str) -> RuntimeError {
    int32_error_with_detail(
        &INT32_ERROR_INVALID_INPUT,
        format!("conversion to int32 from {type_name} is not possible"),
    )
}

#[runtime_builtin(
    name = "int32",
    category = "math/elementwise",
    summary = "Convert scalars, arrays, and gpuArray values to int32 using MATLAB saturating rounding.",
    keywords = "int32,cast,integer,conversion,gpuArray",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::int32::INT32_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::int32"
)]
async fn int32_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(int32_error_with_detail(
            &INT32_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    cast_value(value, IntegerTarget::I32)
        .await
        .map_err(|cause| match cause {
            CastError::Unsupported(type_name) => conversion_error(&type_name),
            CastError::Internal(detail) => int32_error_with_detail(&INT32_ERROR_INTERNAL, detail),
        })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataOwned, HostTensorView, IntegerElementType};
    use runmat_builtins::{
        CharArray, IntValue, IntegerStorage, ResolveContext, SymbolicArray, SymbolicExpr, Tensor,
        Type,
    };

    fn int32_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::int32_builtin(value, rest))
    }

    #[test]
    fn int32_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = INT32_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = int32(X)"));
    }

    #[test]
    fn int32_type_preserves_tensor_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_scalar_saturates_and_rounds() {
        assert_eq!(
            int32_builtin(Value::Num(3.5), Vec::new()).expect("int32"),
            Value::Int(IntValue::I32(4))
        );
        assert_eq!(
            int32_builtin(Value::Num(-3.5), Vec::new()).expect("int32"),
            Value::Int(IntValue::I32(-4))
        );
        assert_eq!(
            int32_builtin(Value::Num(f64::INFINITY), Vec::new()).expect("int32"),
            Value::Int(IntValue::I32(i32::MAX))
        );
        assert_eq!(
            int32_builtin(Value::Num(f64::NEG_INFINITY), Vec::new()).expect("int32"),
            Value::Int(IntValue::I32(i32::MIN))
        );
        assert_eq!(
            int32_builtin(Value::Num(f64::NAN), Vec::new()).expect("int32"),
            Value::Int(IntValue::I32(0))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_converts_symbolic_constants() {
        let result =
            int32_builtin(Value::Symbolic(SymbolicExpr::constant(3.5)), Vec::new()).expect("int32");

        assert_eq!(result, Value::Int(IntValue::I32(4)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_converts_symbolic_array_constants() {
        let array = SymbolicArray::new(
            vec![SymbolicExpr::constant(3.5), SymbolicExpr::constant(-2.2)],
            vec![1, 2],
        )
        .unwrap();

        let result = int32_builtin(Value::SymbolicArray(array), Vec::new()).expect("int32");

        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.materialize_f64(), vec![4.0, -2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_rejects_symbolic_array_variables() {
        let array = SymbolicArray::new(vec![SymbolicExpr::variable("x")], vec![1, 1]).unwrap();

        let err = int32_builtin(Value::SymbolicArray(array), Vec::new())
            .expect_err("symbolic variable should not convert");

        assert_eq!(err.identifier(), INT32_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("conversion to int32 from sym"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_rejects_symbolic_variables() {
        let err = int32_builtin(Value::Symbolic(SymbolicExpr::variable("x")), Vec::new())
            .expect_err("symbolic variable should not convert");

        assert_eq!(err.identifier(), INT32_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("conversion to int32 from sym"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![-2.0, 2.49, 2.5, 1.0e20], vec![2, 2]).unwrap();
        let result = int32_builtin(Value::Tensor(tensor), Vec::new()).expect("int32");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.materialize_f64(), vec![-2.0, 2.0, 3.0, i32::MAX as f64]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::I32(vec![-2, 2, 3, i32::MAX]))
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_char_array_produces_codes() {
        let chars = CharArray::new_row("Az");
        let result = int32_builtin(Value::CharArray(chars), Vec::new()).expect("int32");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![65.0, 122.0]);
                assert_eq!(
                    t.integer_storage(),
                    Some(&IntegerStorage::I32(vec![65, 122]))
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_errors_on_string_input() {
        let err = int32_builtin(Value::String("hello".to_string()), Vec::new())
            .expect_err("expected error");
        assert_eq!(err.identifier(), INT32_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("string"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_too_many_arguments_has_stable_identifier() {
        let err = int32_builtin(Value::Num(1.0), vec![Value::Num(2.0)])
            .expect_err("expected too-many-args error");
        assert_eq!(err.identifier(), INT32_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-3.0, 4.4, 5.6], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = int32_builtin(Value::GpuTensor(handle), Vec::new()).expect("int32");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpuArray result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::I32)
            );
            assert_eq!(handle.shape, vec![3, 1]);
            assert_eq!(
                block_on(provider.download_integer(&handle))
                    .expect("download int32 cast")
                    .data,
                HostIntegerDataOwned::I32(vec![-3, 4, 6])
            );
        });
    }
}
