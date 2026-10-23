//! MATLAB-compatible `ones` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, IntegerElementType};
#[cfg(test)]
use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
use runmat_builtins::{
    BuiltinErrorDescriptor, ONES_COLUMN_SIZE_VECTOR_EXTENSION, ONES_ERROR_CLASS_CONFLICT,
    ONES_ERROR_LIKE_DUPLICATE, ONES_ERROR_LIKE_EXPECTED_PROTOTYPE, ONES_ERROR_UNRECOGNIZED_OPTION,
    ONES_RESIDENT_SIZE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray, SparseTensor, Value,
};

use crate::build_runtime_error;
use crate::builtins::common::random_args::{
    extract_constructor_dimensions, normalize_constructor_shape, validate_constructor_gpu_output,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionKernelTemplate,
    GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_value::NumericDType;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::ones")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ones",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("ones"),
        ProviderHook::Custom("ones_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Allocates device ones when providers expose dedicated hooks; otherwise falls back to scalar fill or host upload.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("ones").build()
}

fn ones_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    ones_error_with_message(error.message, error)
}

fn ones_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    ones_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn ones_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("ones");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::ones")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ones",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let literal = match ctx.scalar_ty {
                            ScalarType::F32 => "1.0", ScalarType::F64 => "f64(1.0)",
                            ScalarType::I32 => "1", ScalarType::Bool => "true",
                        };
                        writer.push(literal)
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner materialises ones as inline literals; providers may substitute inexpensive fill kernels.",
};

#[runtime_builtin(
    name = "ones",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::ones"
)]
async fn ones_builtin(rest: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    let parsed = ParsedOnes::parse(rest).await?;
    build_output(parsed).await
}

struct ParsedOnes {
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    /// See zeros: host tensors are f64; honour 'single' as numeric ones and
    /// allow GPU paths to select f32 where applicable via 'like' or provider hooks.
    Single,
    Logical,
    Integer(IntegerStorage),
    GpuArray(Box<OutputTemplate>),
    Like(Value),
}

impl ParsedOnes {
    async fn parse(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<OutputTemplate> = None;
        let mut saw_size_vector = false;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) =
                crate::builtins::common::arg_tokens::scalar_keyword_of(&arg, "ones")?
            {
                if matches!(class_override.as_ref(), Some(OutputTemplate::GpuArray(_)))
                    && keyword != "like"
                {
                    return Err(ones_error(&ONES_ERROR_CLASS_CONFLICT));
                }
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err(ones_error(&ONES_ERROR_LIKE_DUPLICATE));
                        }
                        if class_override.is_some() {
                            return Err(ones_error(&ONES_ERROR_CLASS_CONFLICT));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(ones_error(&ONES_ERROR_LIKE_EXPECTED_PROTOTYPE));
                        };
                        like_proto = Some(proto.clone());
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err(ones_error_with_detail(
                                &ONES_ERROR_CLASS_CONFLICT,
                                "logical class override",
                            ));
                        }
                        if class_override.is_some() {
                            return Err(ones_error(&ONES_ERROR_CLASS_CONFLICT));
                        }
                        class_override = Some(OutputTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err(ones_error_with_detail(
                                &ONES_ERROR_CLASS_CONFLICT,
                                "double class override",
                            ));
                        }
                        if class_override.is_some() {
                            return Err(ones_error(&ONES_ERROR_CLASS_CONFLICT));
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        if like_proto.is_some() {
                            return Err(ones_error_with_detail(
                                &ONES_ERROR_CLASS_CONFLICT,
                                "single class override",
                            ));
                        }
                        if class_override.is_some() {
                            return Err(ones_error(&ONES_ERROR_CLASS_CONFLICT));
                        }
                        class_override = Some(OutputTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32"
                    | "uint64" => {
                        if like_proto.is_some() {
                            return Err(ones_error_with_detail(
                                &ONES_ERROR_CLASS_CONFLICT,
                                format!("{keyword} class override"),
                            ));
                        }
                        if class_override.is_some() {
                            return Err(ones_error(&ONES_ERROR_CLASS_CONFLICT));
                        }
                        class_override = Some(OutputTemplate::Integer(
                            integer_storage_prototype_from_keyword(keyword.as_str())
                                .expect("matched integer class keyword"),
                        ));
                        idx += 1;
                        continue;
                    }
                    "gpuarray" => {
                        if like_proto.is_some() {
                            return Err(ones_error_with_detail(
                                &ONES_ERROR_CLASS_CONFLICT,
                                "gpuArray class override",
                            ));
                        }
                        let underlying = class_override.take().unwrap_or(OutputTemplate::Double);
                        class_override = Some(OutputTemplate::GpuArray(Box::new(underlying)));
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(ones_error_with_detail(
                            &ONES_ERROR_UNRECOGNIZED_OPTION,
                            format!("'{other}'"),
                        ));
                    }
                }
            }

            if matches!(arg, Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &ONES_RESIDENT_SIZE_EXTENSION,
                    "ones",
                )?;
            }
            if let Some(parsed_dims) = extract_constructor_dimensions(&arg, "ones")
                .await
                .map_err(builtin_error)?
            {
                if parsed_dims.is_column_vector {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &ONES_COLUMN_SIZE_VECTOR_EXTENSION,
                        "ones",
                    )?;
                }
                if parsed_dims.values.len() > 1 {
                    if saw_size_vector || saw_dims_arg {
                        return Err(builtin_error(
                            "ones: a size vector must be the only dimension argument",
                        ));
                    }
                    saw_size_vector = true;
                } else if saw_size_vector {
                    return Err(builtin_error(
                        "ones: a size vector must be the only dimension argument",
                    ));
                }
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims.values;
                } else {
                    dims.extend(parsed_dims.values);
                }
                idx += 1;
                continue;
            }
            return Err(builtin_error(format!(
                "ones: unsupported dimension or option {arg:?}"
            )));
        }

        let shape = if saw_dims_arg {
            normalize_constructor_shape(dims)
        } else {
            vec![1, 1]
        };

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
        } else {
            OutputTemplate::Double
        };

        Ok(Self { shape, template })
    }
}

async fn build_output(parsed: ParsedOnes) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => ones_double(&parsed.shape),
        OutputTemplate::Single => ones_single(&parsed.shape),
        OutputTemplate::Logical => ones_logical(&parsed.shape),
        OutputTemplate::Integer(storage) => ones_integer_like(&storage, &parsed.shape),
        OutputTemplate::GpuArray(template) => ones_gpu(&parsed.shape, *template).await,
        OutputTemplate::Like(proto) => ones_like(&proto, &parsed.shape).await,
    }
}

async fn ones_gpu(shape: &[usize], template: OutputTemplate) -> crate::BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(builtin_error(
            "ones: gpuArray output requires an active provider",
        ));
    };
    let requested_precision = match &template {
        OutputTemplate::Double => Some(runmat_accelerate_api::ProviderPrecision::F64),
        OutputTemplate::Single => Some(runmat_accelerate_api::ProviderPrecision::F32),
        _ => None,
    };
    if requested_precision.is_some_and(|precision| provider.precision() != precision) {
        return Err(builtin_error(
            "ones: active provider cannot preserve requested gpuArray precision",
        ));
    }
    let host = match template {
        OutputTemplate::Double => tensor::ones_with_dtype(shape, NumericDType::F64)
            .map_err(|error| builtin_error(format!("ones: {error}")))?,
        OutputTemplate::Single => tensor::ones_with_dtype(shape, NumericDType::F32)
            .map_err(|error| builtin_error(format!("ones: {error}")))?,
        OutputTemplate::Integer(storage) => runmat_value::Tensor::new_integer(
            storage.ones_like(tensor::element_count(shape)),
            shape.to_vec(),
        )
        .map_err(|error| builtin_error(format!("ones: {error}")))?,
        OutputTemplate::Logical => {
            return super::resident_constant::logical(provider, shape, true, &[], "ones")
                .map(Value::GpuTensor);
        }
        OutputTemplate::GpuArray(_) | OutputTemplate::Like(_) => {
            return Err(builtin_error("ones: invalid gpuArray class specification"));
        }
    };
    let output = gpu_helpers::upload_tensor(provider, &host)?;
    let (precision, integer) = if let Some(storage) = host.integer_storage() {
        (None, Some(integer_element_type(storage)))
    } else {
        let precision = match host.numeric_dtype() {
            NumericDType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
            _ => runmat_accelerate_api::ProviderPrecision::F64,
        };
        (Some(precision), None)
    };
    let output = validate_constructor_gpu_output(
        "ones",
        provider,
        output,
        shape,
        runmat_accelerate_api::GpuTensorStorage::Real,
        precision,
        integer,
        false,
    )?;
    Ok(Value::GpuTensor(output))
}

fn integer_element_type(storage: &IntegerStorage) -> IntegerElementType {
    match storage {
        IntegerStorage::I8(_) => IntegerElementType::I8,
        IntegerStorage::I16(_) => IntegerElementType::I16,
        IntegerStorage::I32(_) => IntegerElementType::I32,
        IntegerStorage::I64(_) => IntegerElementType::I64,
        IntegerStorage::U8(_) => IntegerElementType::U8,
        IntegerStorage::U16(_) => IntegerElementType::U16,
        IntegerStorage::U32(_) => IntegerElementType::U32,
        IntegerStorage::U64(_) => IntegerElementType::U64,
    }
}

fn integer_storage_prototype_from_keyword(keyword: &str) -> Option<IntegerStorage> {
    Some(match keyword {
        "int8" => IntegerStorage::I8(Vec::new()),
        "int16" => IntegerStorage::I16(Vec::new()),
        "int32" => IntegerStorage::I32(Vec::new()),
        "int64" => IntegerStorage::I64(Vec::new()),
        "uint8" => IntegerStorage::U8(Vec::new()),
        "uint16" => IntegerStorage::U16(Vec::new()),
        "uint32" => IntegerStorage::U32(Vec::new()),
        "uint64" => IntegerStorage::U64(Vec::new()),
        _ => return None,
    })
}

fn ones_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let tensor = tensor::ones(shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn ones_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let tensor = tensor::ones_with_dtype(shape, NumericDType::F32)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn ones_logical(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    LogicalArray::new(vec![1u8; len], shape.to_vec())
        .map(Value::LogicalArray)
        .map_err(|e| builtin_error(format!("ones: {e}")))
}

#[async_recursion::async_recursion(?Send)]
async fn ones_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => ones_logical(shape),
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
            ones_complex_integer_like(
                tensor
                    .integer_storage()
                    .as_ref()
                    .expect("guarded typed complex integer storage"),
                shape,
            )
        }
        Value::ComplexTensor(tensor) => {
            let len = tensor::element_count(shape);
            let data = vec![(1.0, 0.0); len];
            ComplexTensor::from_f64_values_with_dtype(data, shape.to_vec(), tensor.numeric_dtype())
                .map(Value::ComplexTensor)
                .map_err(|e| builtin_error(format!("ones: {e}")))
        }
        Value::Complex(_, _) => {
            let len = tensor::element_count(shape);
            ComplexTensor::new(vec![(1.0, 0.0); len], shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|e| builtin_error(format!("ones: {e}")))
        }
        Value::GpuTensor(handle) => ones_like_gpu(handle, shape).await,
        Value::SparseTensor(sparse) => ones_sparse_like(sparse, shape),
        Value::Tensor(t) => match t.numeric_dtype() {
            NumericDType::F32 => ones_single(shape),
            NumericDType::F64 => ones_double(shape),
            dtype => tensor::ones_with_dtype(shape, dtype)
                .map(Value::Tensor)
                .map_err(|e| builtin_error(format!("ones: {e}"))),
        },
        Value::Int(value) => ones_integer_like(&IntegerStorage::from_scalar(value.clone()), shape),
        Value::Num(_) => ones_double(shape),
        _ => Err(builtin_error(
            "ones: 'like' prototype must be numeric or logical",
        )),
    }
}

fn ones_sparse_like(prototype: &SparseTensor, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if shape.len() > 2 {
        return Err(builtin_error(
            "ones: sparse 'like' output must be two-dimensional",
        ));
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| builtin_error("ones: sparse output size overflow"))?;
    let col_ptrs: Vec<_> = (0..=cols).map(|column| column * rows).collect();
    let row_indices: Vec<_> = (0..cols).flat_map(|_| 0..rows).collect();
    let sparse = if prototype.is_logical() {
        SparseTensor::new_logical(rows, cols, col_ptrs, row_indices)
    } else if let Some(storage) = prototype.integer_storage() {
        SparseTensor::new_integer(rows, cols, col_ptrs, row_indices, storage.ones_like(len))
    } else {
        match prototype.numeric_dtype() {
            Some(NumericDType::F32) => {
                SparseTensor::new_f32(rows, cols, col_ptrs, row_indices, vec![1.0_f32; len])
            }
            Some(NumericDType::F64) => {
                SparseTensor::new(rows, cols, col_ptrs, row_indices, vec![1.0; len])
            }
            _ => unreachable!("handled sparse dtype"),
        }
    }
    .map_err(|error| builtin_error(format!("ones: {error}")))?;
    Ok(Value::SparseTensor(sparse))
}

fn ones_integer_like(storage: &IntegerStorage, shape: &[usize]) -> crate::BuiltinResult<Value> {
    let tensor = runmat_value::Tensor::new_integer(
        storage.ones_like(tensor::element_count(shape)),
        shape.to_vec(),
    )
    .map_err(|e| builtin_error(format!("ones: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn ones_complex_integer_like(
    storage: &IntegerComplexStorage,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let storage = IntegerComplexStorage::new(
        storage.real().ones_like(len),
        storage.imag().zeros_like(len),
    )
    .map_err(|e| builtin_error(format!("ones: {e}")))?;
    ComplexTensor::new_integer(storage, shape.to_vec())
        .map(Value::ComplexTensor)
        .map_err(|e| builtin_error(format!("ones: {e}")))
}

async fn ones_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    super::resident_constant::like(handle, shape, super::resident_constant::Constant::One)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_default_scalar() {
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("ones");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_square_from_single_dimension() {
        let args = vec![Value::Num(3.0)];
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("ones");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![3, 3]);
                assert!(t.materialize_f64().iter().all(|&x| x == 1.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_rectangular_from_dims() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("ones");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![2, 4]);
                assert!(t.materialize_f64().iter().all(|&x| x == 1.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_from_size_vector() {
        let size_vec = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(size_vec)];
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("ones");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape(), vec![2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn ones_integer_dimensions_clamp_negative_and_normalize_trailing_singletons() {
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::I8(-2)),
                Value::Int(IntValue::U16(3)),
                Value::Int(IntValue::U64(1)),
            ]),
        ))
        .expect("ones integer dimensions");
        let Value::Tensor(tensor) = result else {
            panic!("expected empty host tensor");
        };
        assert_eq!(tensor.shape(), vec![0, 3]);
    }

    #[test]
    fn ones_column_size_vector_follows_compatibility_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let size = Tensor::new_integer(IntegerStorage::U32(vec![2, 3]), vec![2, 1])
            .expect("column size vector");
        let error = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(size)]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:OnesColumnSizeVectorExtension")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_logical_output() {
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("logical")];
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("ones");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 2]);
                assert!(logical.data.iter().all(|&x| x == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_implicit_prototype_is_rejected() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        assert!(block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(args)
        ))
        .is_err());
    }

    #[test]
    fn ones_like_preserves_every_exact_integer_class() {
        let storages = vec![
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in storages {
            let prototype = Tensor::new_integer(storage.clone(), vec![1, 2]).expect("prototype");
            let result = block_on(ones_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(1.0),
                    Value::Num(2.0),
                    Value::from("like"),
                    Value::Tensor(prototype),
                ]),
            ))
            .expect("ones like");
            let Value::Tensor(output) = result else {
                panic!("expected integer tensor");
            };
            assert_eq!(output.shape(), vec![1, 2]);
            assert_eq!(output.integer_storage(), Some(&storage.ones_like(2)));
        }

        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::Int(IntValue::U64(u64::MAX)),
            ]),
        ))
        .expect("integer scalar prototype");
        let Value::Tensor(output) = result else {
            panic!("expected uint64 tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1; 4]))
        );
    }

    #[test]
    fn ones_like_without_dimensions_returns_scalar_in_prototype_class() {
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("like"),
                Value::Int(IntValue::U64(u64::MAX)),
            ]),
        ))
        .expect("scalar ones like uint64");
        assert_eq!(result, Value::Int(IntValue::U64(1)));
    }

    #[test]
    fn ones_class_strings_create_exact_integer_storage() {
        let cases = [
            ("int8", IntegerStorage::I8(vec![1; 6])),
            ("int16", IntegerStorage::I16(vec![1; 6])),
            ("int32", IntegerStorage::I32(vec![1; 6])),
            ("int64", IntegerStorage::I64(vec![1; 6])),
            ("uint8", IntegerStorage::U8(vec![1; 6])),
            ("uint16", IntegerStorage::U16(vec![1; 6])),
            ("uint32", IntegerStorage::U32(vec![1; 6])),
            ("uint64", IntegerStorage::U64(vec![1; 6])),
        ];

        for (class_name, expected) in cases {
            let result = block_on(ones_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::Num(3.0),
                    Value::from(class_name),
                ]),
            ))
            .expect("ones integer class");
            let Value::Tensor(output) = result else {
                panic!("expected integer tensor for {class_name}");
            };
            assert_eq!(output.shape(), vec![2, 3]);
            assert_eq!(output.integer_storage(), Some(&expected));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_like_complex_scalar() {
        let args = vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ];
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("ones");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape(), vec![3, 3]);
                assert!(t
                    .materialize_f64()
                    .iter()
                    .all(|&(re, im)| (re, im) == (1.0, 0.0)));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ones_like_complex_single_preserves_native_single() {
        let prototype =
            ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).expect("complex single");
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::ComplexTensor(prototype),
            ]),
        ))
        .expect("ones complex single like");
        let Value::ComplexTensor(output) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(output.shape(), vec![2, 2]);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
    }

    #[test]
    fn ones_like_typed_complex_int64_keeps_signed_storage() {
        let prototype = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                IntegerStorage::I64(vec![i64::MAX, i64::MIN]),
            )
            .expect("typed complex prototype"),
            vec![1, 2],
        )
        .expect("typed complex tensor");
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::ComplexTensor(prototype),
            ]),
        ))
        .expect("ones like");
        let Value::ComplexTensor(output) = result else {
            panic!("expected typed complex output");
        };
        assert_eq!(
            output.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![1; 4]),
                    IntegerStorage::I64(vec![0; 4]),
                )
                .expect("typed complex ones"),
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_like_logical_array() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::LogicalArray(logical)];
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("ones");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1]);
                assert!(out.data.iter().all(|&x| x == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_gpu_like_alloc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(ones_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect("ones");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape(), vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered.materialize_f64().iter().all(|&x| x == 1.0));
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_gpu_integer_like_preserves_exact_class_resident() {
        test_support::with_test_provider(|provider| {
            let prototype_values = [u64::MAX, 9_007_199_254_740_993];
            let prototype = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&prototype_values),
                    shape: &[1, 2],
                })
                .expect("upload uint64 prototype");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(prototype),
            ];
            let result = block_on(ones_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect("ones integer gpu like");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpuArray");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U64)
            );
            assert_eq!(handle.shape(), vec![2, 2]);
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![1; 4]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ones_wgpu_like_and_gather() {
        let Ok(_provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        // Build GPU prototype via gpuArray
        let proto =
            Tensor::new_with_dtype(vec![0.0; 4], vec![2, 2], runmat_value::NumericDType::F32)
                .unwrap();
        let g = block_on(crate::call_builtin_async(
            "gpuArray",
            &[Value::Tensor(proto)],
        ))
        .expect("gpuArray");
        let args = vec![
            Value::Num(2.0),
            Value::Num(2.0),
            Value::from("like"),
            g.value().clone(),
        ];
        let result = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("ones like gpu");
        match result {
            Value::GpuTensor(h) => {
                let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather");
                assert_eq!(gathered.shape(), vec![2, 2]);
                assert!(gathered.materialize_f64().iter().all(|&x| x == 1.0));
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[test]
    fn ones_same_shape_complex_gpu_like_stays_complex_and_resident() {
        for provenance in [
            runmat_accelerate_api::GpuHandleProvenance::Automatic,
            runmat_accelerate_api::GpuHandleProvenance::Explicit,
        ] {
            test_support::with_f32_test_provider(|provider| {
                let prototype = ComplexTensor::from_f32(vec![(1.0, -1.0); 4], vec![2, 2])
                    .expect("complex single prototype");
                let handle = gpu_helpers::upload_complex_tensor(provider, &prototype)
                    .expect("upload")
                    .with_provenance(provenance)
                    .expect("prototype provenance");
                let result = block_on(ones_builtin(
                    crate::builtins::common::test_support::builtin_arguments(vec![
                        Value::Num(2.0),
                        Value::Num(2.0),
                        Value::from("like"),
                        Value::GpuTensor(handle.clone()),
                    ]),
                ))
                .expect("ones complex gpu like");
                let Value::GpuTensor(output) = result else {
                    panic!("expected resident output");
                };
                assert_eq!(
                    runmat_accelerate_api::handle_storage(&output),
                    runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                );
                assert_eq!(
                    runmat_accelerate_api::handle_precision(&output),
                    Some(runmat_accelerate_api::ProviderPrecision::F32)
                );
                assert_eq!(output.view().provenance(), Some(provenance));
                assert!(!output.shares_allocation_with(&handle));
                drop(output);
                handle.allocation_lease().validate_live().unwrap();
            });
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ones_wgpu_fusion_with_sin_and_sum() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        // Create ones on GPU (2x2), then sin, then sum along dim=1
        let args = vec![Value::Num(2.0), Value::Num(2.0)];
        let o = block_on(ones_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("ones");
        let s = block_on(crate::call_builtin_async("sin", &[o])).expect("sin");
        let summed = block_on(crate::call_builtin_async(
            "sum",
            &[s.value().clone(), Value::Num(1.0)],
        ))
        .expect("sum");
        // Gather and validate shapes; values are deterministic for sin(1)
        let gathered = test_support::gather_rooted(summed).expect("gather");
        assert_eq!(gathered.shape(), vec![1, 2]);
    }
}
