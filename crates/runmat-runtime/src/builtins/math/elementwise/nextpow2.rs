use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::nextpow2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "nextpow2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_nextpow2" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute nextpow2 on-device via unary_nextpow2; otherwise the runtime gathers and applies MATLAB-style scalar semantics on the host.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::nextpow2"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "nextpow2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!(
                "select(ceil(log2(abs({input}))), 0.0, abs({input}) == 0.0)"
            ))
        },
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion emits ceil(log2(abs(x))) with zero mapped to zero.",
};

const BUILTIN_NAME: &str = "nextpow2";

const NEXTPOW2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exponent p where 2^p >= abs(X).",
}];

const NEXTPOW2_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric/logical input.",
}];

const NEXTPOW2_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = nextpow2(X)",
    inputs: &NEXTPOW2_INPUTS,
    outputs: &NEXTPOW2_OUTPUT,
}];

const NEXTPOW2_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented and retain their native class in the exponent result.",
    }];

pub const NEXTPOW2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "p = nextpow2(integer_X)",
        inputs: &NEXTPOW2_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Integer magnitude and exponent selection use exact native storage, including signed minima and uint64 values; documented gpuArray execution preserves class and residency through typed fallback when no provider kernel exists.",
    }];

const NEXTPOW2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NEXTPOW2.INVALID_INPUT",
    identifier: Some("RunMat:nextpow2:InvalidInput"),
    when: "Input is not convertible to a supported real numeric tensor.",
    message: "nextpow2: invalid input",
};

const NEXTPOW2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NEXTPOW2.INTERNAL",
    identifier: Some("RunMat:nextpow2:Internal"),
    when: "Internal gather/provider/tensor construction failed.",
    message: "nextpow2: internal error",
};

const NEXTPOW2_ERRORS: [BuiltinErrorDescriptor; 2] =
    [NEXTPOW2_ERROR_INVALID_INPUT, NEXTPOW2_ERROR_INTERNAL];

pub const NEXTPOW2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NEXTPOW2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NEXTPOW2_ERRORS,
};

fn nextpow2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "nextpow2",
    category = "math/elementwise",
    summary = "Return the exponent p such that 2^p is the next power of two greater than or equal to abs(n).",
    keywords = "nextpow2,power of two,fft,zero padding,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::nextpow2::NEXTPOW2_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::math::elementwise::nextpow2::NEXTPOW2_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::math::elementwise::nextpow2"
)]
async fn nextpow2_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => nextpow2_gpu(handle).await,
        other => nextpow2_host(other),
    }
}

async fn nextpow2_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let provider = runmat_accelerate_api::provider_for_handle(&handle);
        let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        let output = nextpow2_tensor(tensor)?;
        if let Some(provider) = provider {
            if let Ok(handle) = gpu_helpers::upload_tensor(provider, &output) {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
        return Ok(tensor::tensor_into_value(output));
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_nextpow2(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    Ok(tensor::tensor_into_value(nextpow2_tensor(tensor)?))
}

fn nextpow2_host(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|e| nextpow2_error_with_detail(&NEXTPOW2_ERROR_INVALID_INPUT, e))?;
    Ok(tensor::tensor_into_value(nextpow2_tensor(tensor)?))
}

fn nextpow2_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| nextpow2_error_with_detail(&NEXTPOW2_ERROR_INTERNAL, e))?;
    let storage = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(nextpow2_scalar).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(nextpow2_scalar_f32).collect())
        }
        NumericStorage::I8(values) => NumericStorage::I8(nextpow2_signed_values(values)),
        NumericStorage::I16(values) => NumericStorage::I16(nextpow2_signed_values(values)),
        NumericStorage::I32(values) => NumericStorage::I32(nextpow2_signed_values(values)),
        NumericStorage::I64(values) => NumericStorage::I64(nextpow2_signed_values(values)),
        NumericStorage::U8(values) => NumericStorage::U8(nextpow2_unsigned_values(values)),
        NumericStorage::U16(values) => NumericStorage::U16(nextpow2_unsigned_values(values)),
        NumericStorage::U32(values) => NumericStorage::U32(nextpow2_unsigned_values(values)),
        NumericStorage::U64(values) => NumericStorage::U64(nextpow2_unsigned_values(values)),
    };
    Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| nextpow2_error_with_detail(&NEXTPOW2_ERROR_INTERNAL, e))
}

fn nextpow2_scalar(x: f64) -> f64 {
    let ax = x.abs();
    if ax == 0.0 {
        0.0
    } else {
        ax.log2().ceil()
    }
}

fn nextpow2_scalar_f32(x: f32) -> f32 {
    let absolute = x.abs();
    if absolute == 0.0 {
        0.0
    } else {
        absolute.log2().ceil()
    }
}

trait NextPow2Unsigned: Copy {
    fn into_u128(self) -> u128;
    fn from_u128(value: u128) -> Self;
}

trait NextPow2Signed: Copy {
    fn magnitude_u128(self) -> u128;
    fn from_u128(value: u128) -> Self;
}

macro_rules! impl_nextpow2_unsigned {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl NextPow2Unsigned for $ty {
                fn into_u128(self) -> u128 { self as u128 }
                fn from_u128(value: u128) -> Self { value as $ty }
            }
        )+
    };
}

macro_rules! impl_nextpow2_signed {
    ($(($signed:ty, $unsigned:ty)),+ $(,)?) => {
        $(
            impl NextPow2Signed for $signed {
                fn magnitude_u128(self) -> u128 {
                    self.unsigned_abs() as $unsigned as u128
                }
                fn from_u128(value: u128) -> Self { value as $signed }
            }
        )+
    };
}

impl_nextpow2_unsigned!(u8, u16, u32, u64);
impl_nextpow2_signed!((i8, u8), (i16, u16), (i32, u32), (i64, u64));

fn nextpow2_unsigned_values<T: NextPow2Unsigned>(values: Vec<T>) -> Vec<T> {
    values
        .into_iter()
        .map(|value| T::from_u128(nextpow2_integer(value.into_u128())))
        .collect()
}

fn nextpow2_signed_values<T: NextPow2Signed>(values: Vec<T>) -> Vec<T> {
    values
        .into_iter()
        .map(|value| T::from_u128(nextpow2_integer(value.magnitude_u128())))
        .collect()
}

fn nextpow2_integer(magnitude: u128) -> u128 {
    if magnitude == 0 {
        0
    } else {
        u128::BITS as u128 - (magnitude - 1).leading_zeros() as u128
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, ResolveContext, Type};

    #[test]
    fn nextpow2_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = NEXTPOW2_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"p = nextpow2(X)"));
    }

    #[test]
    fn nextpow2_type_preserves_tensor_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(4), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(1)])
            }
        );
    }

    #[test]
    fn nextpow2_returns_expected_scalars() {
        let Value::Num(v9) = block_on(nextpow2_builtin(Value::Num(9.0))).unwrap() else {
            panic!("expected scalar")
        };
        assert_eq!(v9, 4.0);
        let Value::Num(v0) = block_on(nextpow2_builtin(Value::Num(0.0))).unwrap() else {
            panic!("expected scalar")
        };
        assert_eq!(v0, 0.0);
        let Value::Num(vneg) = block_on(nextpow2_builtin(Value::Num(-3.0))).unwrap() else {
            panic!("expected scalar")
        };
        assert_eq!(vneg, 2.0);
    }

    #[test]
    fn nextpow2_handles_inf_and_nan() {
        let Value::Num(vinf) = block_on(nextpow2_builtin(Value::Num(f64::INFINITY))).unwrap()
        else {
            panic!("expected scalar")
        };
        assert!(vinf.is_infinite());
        let Value::Num(vnan) = block_on(nextpow2_builtin(Value::Num(f64::NAN))).unwrap() else {
            panic!("expected scalar")
        };
        assert!(vnan.is_nan());
    }

    #[test]
    fn nextpow2_tensor_matches_expected() {
        let value = block_on(super::nextpow2_builtin(Value::Tensor(
            Tensor::new(vec![0.0, 1.0, 3.0, 9.0], vec![4, 1]).unwrap(),
        )))
        .expect("nextpow2");
        let Value::Tensor(t) = value else {
            panic!("expected tensor")
        };
        assert_eq!(t.materialize_f64(), vec![0.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn nextpow2_tensor_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 3, 9]), vec![4, 1])
            .expect("typed integer tensor");

        let value = block_on(super::nextpow2_builtin(Value::Tensor(tensor))).expect("nextpow2");
        let Value::Tensor(t) = value else {
            panic!("expected tensor")
        };
        assert_eq!(
            t.integer_storage(),
            Some(&IntegerStorage::I16(vec![0, 0, 2, 4]))
        );
    }

    #[test]
    fn nextpow2_preserves_all_integer_classes_and_handles_signed_minima_exactly() {
        let cases = [
            (
                IntegerStorage::I8(vec![i8::MIN, -3, 0, 9]),
                IntegerStorage::I8(vec![7, 2, 0, 4]),
            ),
            (
                IntegerStorage::I16(vec![i16::MIN, -3, 0, 9]),
                IntegerStorage::I16(vec![15, 2, 0, 4]),
            ),
            (
                IntegerStorage::I32(vec![i32::MIN, -3, 0, 9]),
                IntegerStorage::I32(vec![31, 2, 0, 4]),
            ),
            (
                IntegerStorage::I64(vec![i64::MIN, -3, 0, 9]),
                IntegerStorage::I64(vec![63, 2, 0, 4]),
            ),
            (
                IntegerStorage::U8(vec![0, 1, 3, u8::MAX]),
                IntegerStorage::U8(vec![0, 0, 2, 8]),
            ),
            (
                IntegerStorage::U16(vec![0, 1, 3, u16::MAX]),
                IntegerStorage::U16(vec![0, 0, 2, 16]),
            ),
            (
                IntegerStorage::U32(vec![0, 1, 3, u32::MAX]),
                IntegerStorage::U32(vec![0, 0, 2, 32]),
            ),
            (
                IntegerStorage::U64(vec![0, 1, 3, u64::MAX]),
                IntegerStorage::U64(vec![0, 0, 2, 64]),
            ),
        ];
        for (input, expected) in cases {
            let tensor = Tensor::new_integer(input, vec![1, 4]).unwrap();
            let Value::Tensor(output) =
                block_on(nextpow2_builtin(Value::Tensor(tensor))).expect("nextpow2")
            else {
                panic!("expected typed tensor");
            };
            assert_eq!(output.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn nextpow2_preserves_native_single_scalar_and_empty_storage() {
        let tensor = Tensor::from_f32(vec![0.0, -3.0, 9.0], vec![1, 3]).unwrap();
        let Value::Tensor(output) =
            block_on(nextpow2_builtin(Value::Tensor(tensor))).expect("nextpow2")
        else {
            panic!("expected single tensor");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.0, 2.0, 4.0])
        );

        let scalar = Tensor::from_f32(vec![9.0], vec![1, 1]).unwrap();
        let Value::Tensor(output) =
            block_on(nextpow2_builtin(Value::Tensor(scalar))).expect("nextpow2")
        else {
            panic!("single scalar must retain tensor class");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![4.0])
        );

        let empty = Tensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::Tensor(output) =
            block_on(nextpow2_builtin(Value::Tensor(empty))).expect("nextpow2")
        else {
            panic!("expected empty single tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(Vec::new())
        );
    }

    #[test]
    fn nextpow2_integer_gpu_preserves_exact_class_and_residency() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![0, 3, u64::MAX]), vec![1, 3]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = block_on(nextpow2_builtin(Value::GpuTensor(handle))).expect("nextpow2");
            assert!(matches!(result, Value::GpuTensor(_)));
            let output = test_support::gather(result).expect("gather");
            assert_eq!(
                output.integer_storage(),
                Some(&IntegerStorage::U64(vec![0, 2, 64]))
            );
        });
    }

    #[test]
    fn nextpow2_gpu_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let host = vec![0.0, 1.0, 3.0, 9.0];
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &host,
                    shape: &[4, 1],
                })
                .expect("upload");
            let gpu =
                block_on(super::nextpow2_builtin(Value::GpuTensor(handle))).expect("gpu nextpow2");
            let t = test_support::gather(gpu).expect("gather");
            assert_eq!(t.materialize_f64(), vec![0.0, 0.0, 2.0, 4.0]);
        });
    }

    #[test]
    fn nextpow2_rejects_string_with_stable_identifier() {
        let err = block_on(nextpow2_builtin(Value::from("bad"))).expect_err("expected error");
        assert_eq!(err.identifier(), NEXTPOW2_ERROR_INVALID_INPUT.identifier);
    }
}
