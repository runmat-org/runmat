//! MATLAB-compatible `uint8` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::elementwise::integer_cast::{cast_value, CastError, IntegerTarget};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "uint8";

const UINT8_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "uint8-converted output value.",
}];

const UINT8_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar/array value to convert.",
}];

const UINT8_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = uint8(X)",
    inputs: &UINT8_INPUTS_X,
    outputs: &UINT8_OUTPUT,
}];

const UINT8_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT8.INVALID_ARGUMENT",
    identifier: Some("RunMat:uint8:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "uint8: invalid argument",
};

const UINT8_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT8.INVALID_INPUT",
    identifier: Some("RunMat:uint8:InvalidInput"),
    when: "Input value cannot be converted to uint8.",
    message: "uint8: invalid input",
};

const UINT8_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT8.INTERNAL",
    identifier: Some("RunMat:uint8:Internal"),
    when: "Internal conversion, gather, or provider upload failed.",
    message: "uint8: internal error",
};

const UINT8_ERRORS: [BuiltinErrorDescriptor; 3] = [
    UINT8_ERROR_INVALID_ARGUMENT,
    UINT8_ERROR_INVALID_INPUT,
    UINT8_ERROR_INTERNAL,
];

pub const UINT8_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UINT8_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UINT8_ERRORS,
};

const UINT8_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every native integer class converts directly to authoritative uint8 storage without a floating intermediate.",
}];

pub const UINT8_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = uint8(integer_X)",
        inputs: &UINT8_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host conversion is exact and saturating. Real gpuArray conversion uses native uint8 device storage; complex-input preservation is implemented on host, while paired complex-integer device storage remains an architecture gap.",
    }];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::uint8")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "uint8",
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
    notes: "Real gpuArray inputs use the provider resident integer-cast hook and return native uint8 gpuArray storage. Complex gpuArray integer casts remain unsupported until typed complex integer provider storage exists.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::uint8")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "uint8",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Resident integer cast uses provider-native integer storage; fusion can target the provider hook when integer buffers are supported.",
};

fn uint8_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    uint8_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn uint8_error_with_message(
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
    uint8_error_with_detail(
        &UINT8_ERROR_INVALID_INPUT,
        format!("conversion to uint8 from {type_name} is not possible"),
    )
}

#[runtime_builtin(
    name = "uint8",
    category = "math/elementwise",
    summary = "Convert scalars, arrays, and gpuArray values to uint8 using MATLAB saturating rounding.",
    keywords = "uint8,cast,integer,conversion,gpuArray",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::uint8::UINT8_DESCRIPTOR),
    integer_capabilities(crate::builtins::math::elementwise::uint8::UINT8_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::uint8"
)]
async fn uint8_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(uint8_error_with_detail(
            &UINT8_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    cast_value(value, IntegerTarget::U8)
        .await
        .map_err(|cause| match cause {
            CastError::Unsupported(type_name) => conversion_error(&type_name),
            CastError::Internal(detail) => uint8_error_with_detail(&UINT8_ERROR_INTERNAL, detail),
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

    fn uint8_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::uint8_builtin(value, rest))
    }

    #[test]
    fn uint8_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = UINT8_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = uint8(X)"));
    }

    #[test]
    fn uint8_type_preserves_tensor_shape() {
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
    fn uint8_scalar_saturates_and_rounds() {
        assert_eq!(
            uint8_builtin(Value::Num(3.5), Vec::new()).expect("uint8"),
            Value::Int(IntValue::U8(4))
        );
        assert_eq!(
            uint8_builtin(Value::Num(-1.0), Vec::new()).expect("uint8"),
            Value::Int(IntValue::U8(0))
        );
        assert_eq!(
            uint8_builtin(Value::Num(f64::INFINITY), Vec::new()).expect("uint8"),
            Value::Int(IntValue::U8(u8::MAX))
        );
        assert_eq!(
            uint8_builtin(Value::Num(f64::NAN), Vec::new()).expect("uint8"),
            Value::Int(IntValue::U8(0))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_converts_symbolic_constants() {
        let result =
            uint8_builtin(Value::Symbolic(SymbolicExpr::constant(3.5)), Vec::new()).expect("uint8");

        assert_eq!(result, Value::Int(IntValue::U8(4)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_rejects_symbolic_variables() {
        let err = uint8_builtin(Value::Symbolic(SymbolicExpr::variable("x")), Vec::new())
            .expect_err("symbolic variable should not convert");

        assert_eq!(err.identifier(), UINT8_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("conversion to uint8 from sym"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![-2.0, 2.49, 2.5, 300.0], vec![2, 2]).unwrap();
        let result = uint8_builtin(Value::Tensor(tensor), Vec::new()).expect("uint8");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.materialize_f64(), vec![0.0, 2.0, 3.0, u8::MAX as f64]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::U8(vec![0, 2, 3, u8::MAX]))
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_char_array_produces_codes() {
        let chars = CharArray::new_row("Az");
        let result = uint8_builtin(Value::CharArray(chars), Vec::new()).expect("uint8");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![65.0, 122.0]);
                assert_eq!(
                    t.integer_storage(),
                    Some(&IntegerStorage::U8(vec![65, 122]))
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_errors_on_string_input() {
        let err = uint8_builtin(Value::String("hello".to_string()), Vec::new())
            .expect_err("expected error");
        assert_eq!(err.identifier(), UINT8_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("string"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_too_many_arguments_has_stable_identifier() {
        let err = uint8_builtin(Value::Num(1.0), vec![Value::Num(2.0)])
            .expect_err("expected too-many-args error");
        assert_eq!(err.identifier(), UINT8_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-3.0, 4.4, 300.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = uint8_builtin(Value::GpuTensor(handle), Vec::new()).expect("uint8");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpuArray result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U8)
            );
            assert_eq!(handle.shape, vec![3, 1]);
            assert_eq!(
                block_on(provider.download_integer(&handle))
                    .expect("download uint8 cast")
                    .data,
                HostIntegerDataOwned::U8(vec![0, 4, u8::MAX])
            );
        });
    }
}
