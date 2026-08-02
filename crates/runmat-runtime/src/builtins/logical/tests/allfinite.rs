//! Scalar finite-value predicate for complete arrays.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, SparseTensor, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, dispatcher::download_handle_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "allfinite";

const ALLFINITE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when every element of the input is finite.",
}];

const ALLFINITE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to test for finite elements.",
}];

const ALLFINITE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = allfinite(A)",
    inputs: &ALLFINITE_INPUTS,
    outputs: &ALLFINITE_OUTPUT,
}];

const ALLFINITE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ALLFINITE.INVALID_INPUT",
    identifier: Some("RunMat:allfinite:InvalidInput"),
    when: "Input is not numeric, logical, char, or string.",
    message: "allfinite: expected numeric, logical, char, or string input",
};

const ALLFINITE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ALLFINITE.INTERNAL",
    identifier: Some("RunMat:allfinite:InternalError"),
    when: "A GPU provider, gather, download, or internal shape operation fails.",
    message: "allfinite: internal error",
};

const ALLFINITE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ALLFINITE_ERROR_INVALID_INPUT, ALLFINITE_ERROR_INTERNAL];

pub const ALLFINITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ALLFINITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ALLFINITE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::allfinite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "allfinite",
    op_kind: GpuOpKind::Custom("allfinite"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Unary {
            name: "logical_isfinite",
        },
        ProviderHook::Reduction { name: "reduce_all" },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Scalar predicate. Providers may execute isfinite plus all reduction on-device; runtimes gather to host when hooks are unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::allfinite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "allfinite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Returns a scalar predicate over the full input and is not an elementwise fusion op.",
};

pub fn allfinite_type(args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    let _ = args;
    Type::Bool
}

#[runtime_builtin(
    name = "allfinite",
    category = "logical/tests",
    summary = "Return true when every element of the input is finite.",
    keywords = "allfinite,finite,isfinite,all,logical",
    accel = "cpu",
    type_resolver(allfinite_type),
    descriptor(crate::builtins::logical::tests::allfinite::ALLFINITE_DESCRIPTOR),
    builtin_path = "crate::builtins::logical::tests::allfinite"
)]
async fn allfinite_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            if let Some(result) = allfinite_gpu(&handle).await? {
                return Ok(result);
            }
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| allfinite_error_with_message(format!("{BUILTIN_NAME}: {err}")))?;
            allfinite_host(Value::Tensor(tensor))
        }
        other => allfinite_host(other),
    }
}

async fn allfinite_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) else {
        return Ok(None);
    };
    let Ok(mask) = provider.logical_isfinite(handle) else {
        return Ok(None);
    };
    let reduced = match provider.reduce_all(&mask, false).await {
        Ok(handle) => handle,
        Err(_) => {
            let _ = provider.free(&mask);
            return Ok(None);
        }
    };
    let host = match download_handle_async(provider, &reduced).await {
        Ok(host) => host,
        Err(err) => {
            let _ = provider.free(&reduced);
            let _ = provider.free(&mask);
            return Err(allfinite_error_with_message(format!(
                "{BUILTIN_NAME}: {err}"
            )));
        }
    };
    let _ = provider.free(&reduced);
    let _ = provider.free(&mask);
    let value = host.data.first().copied().unwrap_or(1.0) != 0.0;
    Ok(Some(Value::Bool(value)))
}

fn allfinite_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(x) => Ok(Value::Bool(x.is_finite())),
        Value::Int(_) | Value::Bool(_) => Ok(Value::Bool(true)),
        Value::Complex(re, im) => Ok(Value::Bool(re.is_finite() && im.is_finite())),
        Value::Tensor(tensor) => Ok(Value::Bool(tensor_all_finite(&tensor))),
        Value::SparseTensor(sparse) => Ok(Value::Bool(sparse_all_finite(&sparse))),
        Value::ComplexTensor(tensor) => Ok(Value::Bool(complex_tensor_all_finite(&tensor))),
        Value::LogicalArray(_) | Value::CharArray(_) => Ok(Value::Bool(true)),
        Value::String(_) => Ok(Value::Bool(false)),
        Value::StringArray(array) => Ok(Value::Bool(array.data.is_empty())),
        _ => Err(allfinite_error(&ALLFINITE_ERROR_INVALID_INPUT)),
    }
}

fn tensor_all_finite(tensor: &Tensor) -> bool {
    (0..tensor.len()).all(|index| {
        tensor
            .numeric_value_at(index)
            .expect("tensor storage is structurally valid")
            .is_finite()
    })
}

fn sparse_all_finite(sparse: &SparseTensor) -> bool {
    if sparse.integer_storage().is_some() {
        return true;
    }
    sparse.values.iter().all(|value| value.is_finite())
}

fn complex_tensor_all_finite(tensor: &ComplexTensor) -> bool {
    if tensor.integer_data.is_some() {
        return true;
    }
    tensor
        .data
        .iter()
        .all(|(re, im)| re.is_finite() && im.is_finite())
}

fn allfinite_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    allfinite_error_with_descriptor(error.message, error)
}

fn allfinite_error_with_message(message: impl Into<String>) -> RuntimeError {
    allfinite_error_with_descriptor(message, &ALLFINITE_ERROR_INTERNAL)
}

fn allfinite_error_with_descriptor(
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
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        CharArray, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, ResolveContext,
        StringArray,
    };

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(allfinite_builtin(value))
    }

    fn upload_gpu(
        provider: &dyn runmat_accelerate_api::AccelProvider,
        tensor: &Tensor,
    ) -> runmat_accelerate_api::GpuTensorHandle {
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        provider.upload(&view).expect("upload")
    }

    #[test]
    fn allfinite_type_returns_bool() {
        assert_eq!(
            allfinite_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::Bool
        );
    }

    #[test]
    fn scalar_numeric_values() {
        assert_eq!(call(Value::Num(1.0)).unwrap(), Value::Bool(true));
        assert_eq!(call(Value::Num(f64::INFINITY)).unwrap(), Value::Bool(false));
        assert_eq!(call(Value::Num(f64::NAN)).unwrap(), Value::Bool(false));
        assert_eq!(
            call(Value::Int(IntValue::I32(7))).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn complex_scalar_requires_both_parts_finite() {
        assert_eq!(call(Value::Complex(1.0, -2.0)).unwrap(), Value::Bool(true));
        assert_eq!(
            call(Value::Complex(1.0, f64::NAN)).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn dense_tensor_checks_every_element() {
        let finite = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(call(Value::Tensor(finite)).unwrap(), Value::Bool(true));

        let nonfinite = Tensor::new(vec![1.0, f64::INFINITY, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(call(Value::Tensor(nonfinite)).unwrap(), Value::Bool(false));
    }

    #[test]
    fn typed_integer_tensor_checks_native_storage_not_f64_mirror() {
        let mut tensor =
            Tensor::new_integer(IntegerStorage::I32(vec![1, 2, 3, 4]), vec![2, 2]).unwrap();
        tensor.data.fill(f64::NAN);
        assert_eq!(call(Value::Tensor(tensor)).unwrap(), Value::Bool(true));
    }

    #[test]
    fn empty_numeric_arrays_are_true() {
        let empty = Tensor::zeros(vec![0, 3]);
        assert_eq!(call(Value::Tensor(empty)).unwrap(), Value::Bool(true));

        let sparse = SparseTensor::zeros(4, 5);
        assert_eq!(
            call(Value::SparseTensor(sparse)).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn sparse_tensor_checks_stored_values() {
        let finite = SparseTensor::new(3, 2, vec![0, 1, 2], vec![0, 2], vec![1.0, -2.0]).unwrap();
        assert_eq!(
            call(Value::SparseTensor(finite)).unwrap(),
            Value::Bool(true)
        );

        let nonfinite =
            SparseTensor::new(3, 2, vec![0, 1, 2], vec![0, 2], vec![1.0, f64::NAN]).unwrap();
        assert_eq!(
            call(Value::SparseTensor(nonfinite)).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn typed_integer_sparse_tensor_checks_native_storage_not_f64_mirror() {
        let mut sparse = SparseTensor::new_integer(
            3,
            2,
            vec![0, 1, 2],
            vec![0, 2],
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
        )
        .unwrap();
        sparse.values.fill(f64::NAN);
        assert_eq!(
            call(Value::SparseTensor(sparse)).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn complex_tensor_checks_real_and_imaginary_parts() {
        let finite = ComplexTensor {
            data: vec![(1.0, 0.0), (2.0, -3.0)],
            integer_data: None,
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
        };
        assert_eq!(
            call(Value::ComplexTensor(finite)).unwrap(),
            Value::Bool(true)
        );

        let nonfinite = ComplexTensor {
            data: vec![(1.0, 0.0), (2.0, f64::INFINITY)],
            integer_data: None,
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
        };
        assert_eq!(
            call(Value::ComplexTensor(nonfinite)).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn typed_complex_integer_storage_exactly_counts_as_finite() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            IntegerStorage::U64(vec![0, 7]),
        )
        .unwrap();
        let tensor = ComplexTensor {
            data: vec![(f64::NAN, f64::INFINITY), (f64::NEG_INFINITY, f64::NAN)],
            integer_data: Some(storage),
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
        };
        assert_eq!(
            call(Value::ComplexTensor(tensor)).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn logical_and_char_arrays_are_finite() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        assert_eq!(
            call(Value::LogicalArray(logical)).unwrap(),
            Value::Bool(true)
        );

        let chars = CharArray::new_row("RunMat");
        assert_eq!(call(Value::CharArray(chars)).unwrap(), Value::Bool(true));
    }

    #[test]
    fn strings_are_not_finite_values() {
        assert_eq!(
            call(Value::String("1".to_string())).unwrap(),
            Value::Bool(false)
        );

        let strings = StringArray::new(vec!["1".to_string()], vec![1, 1]).unwrap();
        assert_eq!(
            call(Value::StringArray(strings)).unwrap(),
            Value::Bool(false)
        );

        let empty = StringArray::new(Vec::<String>::new(), vec![0, 1]).unwrap();
        assert_eq!(call(Value::StringArray(empty)).unwrap(), Value::Bool(true));
    }

    #[test]
    fn rejects_non_numeric_containers() {
        let err = call(Value::Cell(
            runmat_builtins::CellArray::new(Vec::new(), 0, 0).unwrap(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:allfinite:InvalidInput"));
    }

    #[test]
    fn gpu_tensor_matches_host_path() {
        test_support::with_test_provider(|provider| {
            let finite = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let finite_handle = upload_gpu(provider, &finite);
            assert_eq!(
                call(Value::GpuTensor(finite_handle.clone())).unwrap(),
                Value::Bool(true)
            );
            provider.free(&finite_handle).ok();

            let with_inf = Tensor::new(vec![1.0, f64::INFINITY, 2.0], vec![1, 3]).unwrap();
            let inf_handle = upload_gpu(provider, &with_inf);
            assert_eq!(
                call(Value::GpuTensor(inf_handle.clone())).unwrap(),
                Value::Bool(false)
            );
            provider.free(&inf_handle).ok();

            let with_nan = Tensor::new(vec![1.0, f64::NAN, 2.0], vec![1, 3]).unwrap();
            let nan_handle = upload_gpu(provider, &with_nan);
            assert_eq!(
                call(Value::GpuTensor(nan_handle.clone())).unwrap(),
                Value::Bool(false)
            );
            provider.free(&nan_handle).ok();

            let empty = Tensor::zeros(vec![0, 3]);
            let empty_handle = upload_gpu(provider, &empty);
            assert_eq!(
                call(Value::GpuTensor(empty_handle.clone())).unwrap(),
                Value::Bool(true)
            );
            provider.free(&empty_handle).ok();
        });
    }
}
