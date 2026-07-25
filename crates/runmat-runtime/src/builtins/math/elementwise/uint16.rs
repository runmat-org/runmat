//! MATLAB-compatible `uint16` builtin with GPU-aware semantics for RunMat.

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

const BUILTIN_NAME: &str = "uint16";

const UINT16_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "uint16-converted output value.",
}];

const UINT16_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar/array value to convert.",
}];

const UINT16_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = uint16(X)",
    inputs: &UINT16_INPUTS_X,
    outputs: &UINT16_OUTPUT,
}];

const UINT16_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT16.INVALID_ARGUMENT",
    identifier: Some("RunMat:uint16:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "uint16: invalid argument",
};

const UINT16_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT16.INVALID_INPUT",
    identifier: Some("RunMat:uint16:InvalidInput"),
    when: "Input value cannot be converted to uint16.",
    message: "uint16: invalid input",
};

const UINT16_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT16.INTERNAL",
    identifier: Some("RunMat:uint16:Internal"),
    when: "Internal conversion, gather, or provider upload failed.",
    message: "uint16: internal error",
};

const UINT16_ERRORS: [BuiltinErrorDescriptor; 3] = [
    UINT16_ERROR_INVALID_ARGUMENT,
    UINT16_ERROR_INVALID_INPUT,
    UINT16_ERROR_INTERNAL,
];

pub const UINT16_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UINT16_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UINT16_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::uint16")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "uint16",
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
    notes: "Real gpuArray inputs use the provider resident integer-cast hook and return native uint16 gpuArray storage. Complex gpuArray integer casts remain unsupported until typed complex integer provider storage exists.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::uint16")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "uint16",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Resident integer cast uses provider-native integer storage; fusion can target the provider hook when integer buffers are supported.",
};

fn uint16_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    uint16_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn uint16_error_with_message(
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
    uint16_error_with_detail(
        &UINT16_ERROR_INVALID_INPUT,
        format!("conversion to uint16 from {type_name} is not possible"),
    )
}

#[runtime_builtin(
    name = "uint16",
    category = "math/elementwise",
    summary = "Convert scalars, arrays, and gpuArray values to uint16 using MATLAB saturating rounding.",
    keywords = "uint16,cast,integer,conversion,gpuArray",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::uint16::UINT16_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::uint16"
)]
async fn uint16_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(uint16_error_with_detail(
            &UINT16_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    cast_value(value, IntegerTarget::U16)
        .await
        .map_err(|cause| match cause {
            CastError::Unsupported(type_name) => conversion_error(&type_name),
            CastError::Internal(detail) => uint16_error_with_detail(&UINT16_ERROR_INTERNAL, detail),
        })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataOwned, HostTensorView, IntegerElementType};
    use runmat_builtins::{
        CharArray, IntValue, IntegerStorage, ResolveContext, SymbolicExpr, Tensor, Type,
    };

    fn uint16_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::uint16_builtin(value, rest))
    }

    #[test]
    fn uint16_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = UINT16_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = uint16(X)"));
    }

    #[test]
    fn uint16_type_preserves_tensor_shape() {
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
    fn uint16_scalar_saturates_and_rounds() {
        assert_eq!(
            uint16_builtin(Value::Num(3.5), Vec::new()).expect("uint16"),
            Value::Int(IntValue::U16(4))
        );
        assert_eq!(
            uint16_builtin(Value::Num(-1.0), Vec::new()).expect("uint16"),
            Value::Int(IntValue::U16(0))
        );
        assert_eq!(
            uint16_builtin(Value::Num(f64::INFINITY), Vec::new()).expect("uint16"),
            Value::Int(IntValue::U16(u16::MAX))
        );
        assert_eq!(
            uint16_builtin(Value::Num(f64::NAN), Vec::new()).expect("uint16"),
            Value::Int(IntValue::U16(0))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint16_converts_symbolic_constants() {
        let result = uint16_builtin(Value::Symbolic(SymbolicExpr::constant(3.5)), Vec::new())
            .expect("uint16");

        assert_eq!(result, Value::Int(IntValue::U16(4)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint16_rejects_symbolic_variables() {
        let err = uint16_builtin(Value::Symbolic(SymbolicExpr::variable("x")), Vec::new())
            .expect_err("symbolic variable should not convert");

        assert_eq!(err.identifier(), UINT16_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("conversion to uint16 from sym"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint16_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![-2.0, 2.49, 2.5, 70000.0], vec![2, 2]).unwrap();
        let result = uint16_builtin(Value::Tensor(tensor), Vec::new()).expect("uint16");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0.0, 2.0, 3.0, u16::MAX as f64]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::U16(vec![0, 2, 3, u16::MAX]))
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint16_char_array_produces_codes() {
        let chars = CharArray::new_row("Az");
        let result = uint16_builtin(Value::CharArray(chars), Vec::new()).expect("uint16");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![65.0, 122.0]);
                assert_eq!(
                    t.integer_storage(),
                    Some(&IntegerStorage::U16(vec![65, 122]))
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint16_errors_on_string_input() {
        let err = uint16_builtin(Value::String("hello".to_string()), Vec::new())
            .expect_err("expected error");
        assert_eq!(err.identifier(), UINT16_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("string"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint16_too_many_arguments_has_stable_identifier() {
        let err = uint16_builtin(Value::Num(1.0), vec![Value::Num(2.0)])
            .expect_err("expected too-many-args error");
        assert_eq!(err.identifier(), UINT16_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint16_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-3.0, 4.4, 70000.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = uint16_builtin(Value::GpuTensor(handle), Vec::new()).expect("uint16");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpuArray result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U16)
            );
            assert_eq!(handle.shape, vec![3, 1]);
            assert_eq!(
                block_on(provider.download_integer(&handle))
                    .expect("download uint16 cast")
                    .data,
                HostIntegerDataOwned::U16(vec![0, 4, u16::MAX])
            );
        });
    }
}
