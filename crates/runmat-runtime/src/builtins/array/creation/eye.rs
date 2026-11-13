//! MATLAB-compatible `eye` builtin with GPU-aware semantics for RunMat.
use runmat_builtins::{
    EYE_COLUMN_SIZE_VECTOR_EXTENSION, EYE_IMPLICIT_PROTOTYPE_EXTENSION, EYE_ND_DIMENSIONS_EXTENSION,
};

use runmat_accelerate_api::GpuTensorHandle;
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, NumericDType,
    NumericScalar, SparseTensor, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::eye")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "eye",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("eye"),
        ProviderHook::Custom("eye_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Allocates identity tensors on the device when providers expose dedicated hooks; otherwise falls back to uploading a host-constructed identity tensor.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::eye")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "eye",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Identity tensors are materialised directly; fusion planner treats eye() as a standalone allocation.",
};

#[runtime_builtin(
    name = "eye",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::eye"
)]
async fn eye_builtin(rest: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    ensure_eye_extensions(rest.view())?;
    let parsed = ParsedEye::parse(rest).await?;
    build_output(parsed).await
}

fn ensure_eye_extensions(args: crate::call::FunctionArgumentView<'_>) -> crate::BuiltinResult<()> {
    let mut dimension_count = 0usize;
    let mut idx = 0usize;
    while idx < args.len() {
        if crate::builtins::common::arg_tokens::scalar_keyword_of(&args[idx], "eye")?.as_deref()
            == Some("like")
        {
            idx = idx.saturating_add(2);
            continue;
        }
        if crate::builtins::common::arg_tokens::scalar_keyword_of(&args[idx], "eye")?.is_some() {
            idx += 1;
            continue;
        }
        match &args[idx] {
            Value::Num(_) | Value::Int(_) => dimension_count += 1,
            Value::Tensor(value) => {
                let len = value.len();
                let row = value.shape().len() >= 2 && value.shape()[0] == 1;
                let column = value.shape().len() >= 2 && value.shape()[1] == 1;
                if len <= 1 || row || column || value.shape().len() == 1 {
                    if column && !row && len > 1 {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &EYE_COLUMN_SIZE_VECTOR_EXTENSION,
                            "eye",
                        )?;
                    }
                    dimension_count = dimension_count.saturating_add(len);
                } else {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &EYE_IMPLICIT_PROTOTYPE_EXTENSION,
                        "eye",
                    )?;
                }
            }
            Value::GpuTensor(value) => {
                let len = tensor::element_count(value.shape());
                let row = value.shape().len() >= 2 && value.shape()[0] == 1;
                let column = value.shape().len() >= 2 && value.shape()[1] == 1;
                if len <= 1 || row || column || value.shape().len() == 1 {
                    if column && !row && len > 1 {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &EYE_COLUMN_SIZE_VECTOR_EXTENSION,
                            "eye",
                        )?;
                    }
                    dimension_count = dimension_count.saturating_add(len);
                } else {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &EYE_IMPLICIT_PROTOTYPE_EXTENSION,
                        "eye",
                    )?;
                }
            }
            _ => crate::compatibility::ensure_builtin_extension_enabled(
                &EYE_IMPLICIT_PROTOTYPE_EXTENSION,
                "eye",
            )?,
        }
        idx += 1;
    }
    if dimension_count > 2 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EYE_ND_DIMENSIONS_EXTENSION,
            "eye",
        )?;
    }
    Ok(())
}

struct ParsedEye {
    shape: Vec<usize>,
    template: EyeTemplate,
}

#[derive(Clone)]
enum EyeTemplate {
    Double,
    Single,
    Logical,
    Integer(IntegerStorage),
    Like(Value),
}

impl ParsedEye {
    async fn parse(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<EyeTemplate> = None;
        let mut implicit_proto: Option<Value> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) =
                crate::builtins::common::arg_tokens::scalar_keyword_of(&arg, "eye")?
            {
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err(
                                "eye: multiple 'like' specifications are not supported".into()
                            );
                        }
                        if class_override.is_some() {
                            return Err(
                                "eye: cannot combine 'like' with other class specifiers".into()
                            );
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err("eye: expected prototype after 'like'".into());
                        };
                        ensure_eye_numeric_prototype(&proto)?;
                        like_proto = Some(proto.clone());
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err("eye: cannot combine 'like' with 'logical'".into());
                        }
                        class_override = Some(EyeTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err("eye: cannot combine 'like' with 'double'".into());
                        }
                        class_override = Some(EyeTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        if like_proto.is_some() {
                            return Err("eye: cannot combine 'like' with 'single'".into());
                        }
                        class_override = Some(EyeTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32"
                    | "uint64" => {
                        if like_proto.is_some() {
                            return Err(
                                format!("eye: cannot combine 'like' with '{keyword}'").into()
                            );
                        }
                        class_override = Some(EyeTemplate::Integer(
                            integer_storage_prototype_from_keyword(&keyword)
                                .expect("matched integer class keyword"),
                        ));
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(format!("eye: unrecognised option '{other}'").into());
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg).await? {
                if parsed_dims.len() > 1 && saw_dims_arg {
                    return Err("eye: multiple size-vector arguments are not supported".into());
                }
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            crate::compatibility::ensure_builtin_extension_enabled(
                &EYE_IMPLICIT_PROTOTYPE_EXTENSION,
                "eye",
            )?;
            if shape_source.is_none() {
                shape_source = Some(shape_from_value(&arg)?);
            }
            if implicit_proto.is_none() {
                implicit_proto = Some(arg.clone());
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

        if shape.len() > 2 {
            crate::compatibility::ensure_builtin_extension_enabled(
                &EYE_ND_DIMENSIONS_EXTENSION,
                "eye",
            )?;
        }

        let template = if let Some(proto) = like_proto {
            EyeTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
        } else if let Some(proto) = implicit_proto {
            EyeTemplate::Like(proto)
        } else {
            EyeTemplate::Double
        };

        Ok(Self { shape, template })
    }
}

fn ensure_eye_numeric_prototype(value: &Value) -> Result<(), String> {
    if matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
    ) {
        Ok(())
    } else {
        Err("eye: like prototype must be numeric or logical".into())
    }
}

async fn build_output(parsed: ParsedEye) -> crate::BuiltinResult<Value> {
    let shape = normalize_shape(&parsed.shape);
    match parsed.template {
        EyeTemplate::Double => eye_double(&shape),
        EyeTemplate::Single => eye_single(&shape),
        EyeTemplate::Logical => eye_logical(&shape),
        EyeTemplate::Integer(storage) => eye_integer_like(&storage, &shape),
        EyeTemplate::Like(proto) => return eye_like(&proto, &shape).await,
    }
    .map_err(Into::into)
}

fn eye_double(shape: &[usize]) -> Result<Value, String> {
    let tensor = identity_tensor(shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn eye_single(shape: &[usize]) -> Result<Value, String> {
    let shape = shape.to_vec();
    let mut data = vec![0.0_f32; tensor::element_count(&shape)];
    visit_identity_positions(&shape, |index| data[index] = 1.0);
    Tensor::from_f32(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|error| format!("eye: {error}"))
}

fn eye_logical(shape: &[usize]) -> Result<Value, String> {
    let shape = shape.to_vec();
    let mut logical = LogicalArray::zeros(shape.clone());
    visit_identity_positions(&shape, |idx| logical.data[idx] = 1);
    Ok(Value::LogicalArray(logical))
}

fn eye_complex(shape: &[usize], dtype: NumericDType) -> Result<Value, String> {
    let shape = shape.to_vec();
    let len = shape.iter().product();
    let mut data = vec![(0.0, 0.0); len];
    visit_identity_positions(&shape, |idx| data[idx] = (1.0, 0.0));
    let tensor = ComplexTensor::from_f64_values_with_dtype(data, shape, dtype)?;
    Ok(Value::ComplexTensor(tensor))
}

fn integer_storage_prototype_from_keyword(keyword: &str) -> Option<IntegerStorage> {
    match keyword {
        "int8" => Some(IntegerStorage::I8(Vec::new())),
        "int16" => Some(IntegerStorage::I16(Vec::new())),
        "int32" => Some(IntegerStorage::I32(Vec::new())),
        "int64" => Some(IntegerStorage::I64(Vec::new())),
        "uint8" => Some(IntegerStorage::U8(Vec::new())),
        "uint16" => Some(IntegerStorage::U16(Vec::new())),
        "uint32" => Some(IntegerStorage::U32(Vec::new())),
        "uint64" => Some(IntegerStorage::U64(Vec::new())),
        _ => None,
    }
}

#[async_recursion::async_recursion(?Send)]
async fn eye_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => eye_logical(shape),
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
            eye_complex_integer_like(
                tensor
                    .integer_storage()
                    .as_ref()
                    .expect("guarded typed complex integer storage"),
                shape,
            )
        }
        Value::ComplexTensor(tensor) => eye_complex(shape, tensor.numeric_dtype()),
        Value::Complex(_, _) => eye_complex(shape, NumericDType::F64),
        Value::GpuTensor(handle) => return eye_like_gpu(handle, shape).await,
        Value::SparseTensor(sparse) => eye_sparse_like(sparse, shape),
        Value::Tensor(tensor) => match tensor.integer_storage() {
            Some(storage) => eye_integer_like(storage, shape),
            None if tensor.numeric_dtype() == NumericDType::F32 => eye_single(shape),
            None => eye_double(shape),
        },
        Value::Int(value) => eye_integer_like(&IntegerStorage::from_scalar(value.clone()), shape),
        Value::Num(_) => eye_double(shape),
        Value::CharArray(_) | Value::Cell(_) => eye_double(shape),
        other => {
            let gathered = crate::dispatcher::gather_if_needed_async(other)
                .await
                .map_err(|e| format!("eye: {e}"))?;
            return eye_like(&gathered, shape).await;
        }
    }
    .map_err(Into::into)
}

fn eye_sparse_like(proto: &SparseTensor, shape: &[usize]) -> Result<Value, String> {
    let [rows, cols] = shape else {
        return Err("eye: sparse like output must be two-dimensional".into());
    };
    let diagonal = (*rows).min(*cols);
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::with_capacity(diagonal);
    col_ptrs.push(0);
    for col in 0..*cols {
        if col < diagonal {
            row_indices.push(col);
        }
        col_ptrs.push(row_indices.len());
    }
    let sparse = match proto.numeric_dtype() {
        Some(NumericDType::F64) => {
            SparseTensor::new(*rows, *cols, col_ptrs, row_indices, vec![1.0; diagonal])?
        }
        Some(NumericDType::F32) => {
            SparseTensor::new_f32(*rows, *cols, col_ptrs, row_indices, vec![1.0; diagonal])?
        }
        Some(_) => SparseTensor::new_integer_like(
            *rows,
            *cols,
            col_ptrs,
            row_indices,
            proto
                .integer_storage()
                .expect("integer sparse prototype has integer storage")
                .ones_like(diagonal)
                .exact_values(),
            proto
                .integer_storage()
                .expect("integer sparse prototype has integer storage"),
        )?,
        None => SparseTensor::new_logical(*rows, *cols, col_ptrs, row_indices)?,
    };
    Ok(Value::SparseTensor(sparse))
}

fn eye_integer_like(storage: &IntegerStorage, shape: &[usize]) -> Result<Value, String> {
    let shape = shape.to_vec();
    let mut values = storage.zeros_like(tensor::element_count(&shape));
    let one = storage
        .ones_like(1)
        .value_at(0)
        .expect("one-element integer storage has a value");
    visit_identity_positions(&shape, |index| {
        values
            .set_value(index, one.clone())
            .expect("identity index and integer class are valid");
    });
    let tensor = Tensor::new_integer(values, shape).map_err(|e| format!("eye: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn eye_complex_integer_like(
    storage: &IntegerComplexStorage,
    shape: &[usize],
) -> Result<Value, String> {
    let shape = shape.to_vec();
    let len = tensor::element_count(&shape);
    let mut real = storage.real().zeros_like(len);
    let one = storage
        .real()
        .ones_like(1)
        .value_at(0)
        .expect("one-element integer storage has a value");
    visit_identity_positions(&shape, |index| {
        real.set_value(index, one.clone())
            .expect("identity index and integer class are valid");
    });
    let storage = IntegerComplexStorage::new(real, storage.imag().zeros_like(len))?;
    ComplexTensor::new_integer(storage, shape).map(Value::ComplexTensor)
}

async fn eye_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    use runmat_accelerate_api::{GpuTensorStorage, HostLogicalTensorView, ProviderPrecision};
    let provider = gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| crate::RuntimeError::from("eye: GPU prototype has no owning provider"))?;
    let logical = handle.view().is_logical();
    let integer = runmat_accelerate_api::handle_integer_type(handle);
    let storage = handle.physical().storage();
    let precision = runmat_accelerate_api::handle_precision(handle);
    let output = if logical {
        let Value::LogicalArray(host) = eye_logical(shape)? else {
            unreachable!("eye_logical returns logical array")
        };
        runmat_accelerate_api::upload_logical_protecting(
            provider,
            &HostLogicalTensorView {
                data: &host.data,
                shape: &host.shape,
            },
            &[handle],
        )
        .map_err(|cause| {
            crate::build_runtime_error(format!("eye: logical upload failed: {cause}"))
                .with_builtin("eye")
                .with_source(cause)
                .build()
        })?
    } else if storage == GpuTensorStorage::ComplexInterleaved {
        let host = if let Some(integer) = integer {
            let real = integer_storage_prototype_from_element_type(integer);
            let prototype = IntegerComplexStorage::new(real.clone(), real)?;
            eye_complex_integer_like(&prototype, shape)?
        } else {
            let dtype = match precision.ok_or("eye: complex prototype is missing precision")? {
                ProviderPrecision::F32 => NumericDType::F32,
                ProviderPrecision::F64 => NumericDType::F64,
            };
            eye_complex(shape, dtype)?
        };
        let Value::ComplexTensor(host) = host else {
            unreachable!("complex identity retains tensor storage")
        };
        gpu_helpers::upload_complex_tensor_protecting(provider, &host, &[handle])?
    } else if let Some(integer) = integer {
        let prototype = integer_storage_prototype_from_element_type(integer);
        let host = integer_identity_tensor(&prototype, shape)?;
        gpu_helpers::upload_tensor_protecting(provider, &host, &[handle])?
    } else {
        let precision = precision.ok_or("eye: numeric prototype is missing precision")?;
        let direct = if precision == provider.precision() {
            match provider.eye(shape) {
                Ok(output) => Some(output),
                Err(cause)
                    if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(
                        &cause,
                    ) =>
                {
                    None
                }
                Err(cause) => {
                    return Err(crate::build_runtime_error(format!(
                        "eye: provider identity failed: {cause}"
                    ))
                    .with_builtin("eye")
                    .with_source(runmat_accelerate_api::ProviderExecutionError::from(cause))
                    .build())
                }
            }
        } else {
            None
        };
        if let Some(output) = direct {
            output
        } else {
            let host = match precision {
                ProviderPrecision::F32 => {
                    let mut data = vec![0.0f32; tensor::element_count(shape)];
                    visit_identity_positions(shape, |index| data[index] = 1.0);
                    Tensor::from_f32(data, shape.to_vec())?
                }
                ProviderPrecision::F64 => identity_tensor(shape)?,
            };
            gpu_helpers::upload_tensor_protecting(provider, &host, &[handle])?
        }
    };
    let output = crate::builtins::common::random_args::validate_constructor_gpu_output_protecting(
        "eye",
        provider,
        output,
        shape,
        storage,
        if logical { None } else { precision },
        if logical { None } else { integer },
        logical,
        &[handle],
    )?;
    let output = gpu_helpers::propagate_owned_output_provenance(output, [handle])?;
    gpu_helpers::resident_gpu_value(output)
}

fn integer_identity_tensor(storage: &IntegerStorage, shape: &[usize]) -> Result<Tensor, String> {
    let mut values = storage.zeros_like(tensor::element_count(shape));
    let one = storage
        .ones_like(1)
        .value_at(0)
        .expect("one-element integer storage");
    visit_identity_positions(shape, |index| {
        values
            .set_value(index, one.clone())
            .expect("identity index and class are valid");
    });
    Tensor::new_integer(values, shape.to_vec()).map_err(|error| format!("eye: {error}"))
}

fn integer_storage_prototype_from_element_type(
    element_type: runmat_accelerate_api::IntegerElementType,
) -> IntegerStorage {
    match element_type {
        runmat_accelerate_api::IntegerElementType::I8 => IntegerStorage::I8(Vec::new()),
        runmat_accelerate_api::IntegerElementType::I16 => IntegerStorage::I16(Vec::new()),
        runmat_accelerate_api::IntegerElementType::I32 => IntegerStorage::I32(Vec::new()),
        runmat_accelerate_api::IntegerElementType::I64 => IntegerStorage::I64(Vec::new()),
        runmat_accelerate_api::IntegerElementType::U8 => IntegerStorage::U8(Vec::new()),
        runmat_accelerate_api::IntegerElementType::U16 => IntegerStorage::U16(Vec::new()),
        runmat_accelerate_api::IntegerElementType::U32 => IntegerStorage::U32(Vec::new()),
        runmat_accelerate_api::IntegerElementType::U64 => IntegerStorage::U64(Vec::new()),
    }
}

fn identity_tensor(shape: &[usize]) -> Result<Tensor, String> {
    let shape_vec = shape.to_vec();
    let total = tensor::element_count(&shape_vec);
    let mut data = vec![0.0; total];
    visit_identity_positions(&shape_vec, |idx| data[idx] = 1.0);
    Tensor::new(data, shape_vec).map_err(|e| format!("eye: {e}"))
}

#[async_recursion::async_recursion(?Send)]
async fn extract_dims(value: &Value) -> Result<Option<Vec<usize>>, String> {
    if matches!(value, Value::LogicalArray(_)) {
        return Ok(None);
    }
    match value {
        Value::Num(value) => parse_eye_float_dimension(*value).map(|value| Some(vec![value])),
        Value::Int(value) => parse_eye_integer_dimension(value).map(|value| Some(vec![value])),
        Value::Tensor(value) => {
            let len = value.len();
            if len == 0 {
                return Ok(Some(Vec::new()));
            }
            let scalar = len == 1;
            let row = value.shape().len() >= 2 && value.shape()[0] == 1;
            let column = value.shape().len() >= 2 && value.shape()[1] == 1;
            if !(scalar || row || column || value.shape().len() == 1) {
                return Ok(None);
            }
            (0..len)
                .map(|index| {
                    value
                        .numeric_value_at(index)
                        .ok_or_else(|| "eye: missing dimension value".to_string())
                        .and_then(parse_eye_numeric_dimension)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
        }
        Value::GpuTensor(_) => {
            let gathered = crate::dispatcher::gather_if_needed_async(value)
                .await
                .map_err(|error| format!("eye: {error}"))?;
            extract_dims(&gathered).await
        }
        _ => Ok(None),
    }
}

fn parse_eye_numeric_dimension(value: NumericScalar) -> Result<usize, String> {
    match value {
        NumericScalar::F64(value) => parse_eye_float_dimension(value),
        NumericScalar::F32(value) => parse_eye_float_dimension(f64::from(value)),
        integer => parse_eye_integer_dimension(
            &integer
                .into_int_value()
                .expect("nonfloating numeric scalar is an integer"),
        ),
    }
}

fn parse_eye_float_dimension(value: f64) -> Result<usize, String> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err("eye: dimensions must be finite integer values".into());
    }
    if value <= 0.0 {
        return Ok(0);
    }
    if value >= usize::MAX as f64 {
        return Err("eye: dimension is outside the supported platform range".into());
    }
    Ok(value as usize)
}

fn parse_eye_integer_dimension(value: &IntValue) -> Result<usize, String> {
    match value {
        IntValue::I8(value) => usize::try_from((*value).max(0)),
        IntValue::I16(value) => usize::try_from((*value).max(0)),
        IntValue::I32(value) => usize::try_from((*value).max(0)),
        IntValue::I64(value) => usize::try_from((*value).max(0)),
        IntValue::U8(value) => Ok(usize::from(*value)),
        IntValue::U16(value) => Ok(usize::from(*value)),
        IntValue::U32(value) => usize::try_from(*value),
        IntValue::U64(value) => usize::try_from(*value),
    }
    .map_err(|_| "eye: dimension is outside the supported platform range".into())
}

fn shape_from_value(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape().to_vec()),
        Value::ComplexTensor(t) => Ok(t.shape().to_vec()),
        Value::LogicalArray(l) => Ok(l.shape.to_vec()),
        Value::GpuTensor(h) => Ok(h.shape().to_vec()),
        Value::CharArray(ca) => Ok(ca.shape().to_vec()),
        Value::Cell(cell) => Ok(cell.shape().to_vec()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(format!("eye: unsupported prototype {other:?}")),
    }
}

fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => {
            let n = shape[0];
            vec![n, n]
        }
        _ => shape.to_vec(),
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides
}

fn visit_identity_positions(shape: &[usize], mut f: impl FnMut(usize)) {
    if shape.len() < 2 {
        return;
    }
    let rows = shape[0];
    let cols = shape[1];
    let diag_len = rows.min(cols);
    if diag_len == 0 {
        return;
    }
    let strides = compute_strides(shape);
    let extra_dims = &shape[2..];
    let extra_count = if extra_dims.is_empty() {
        1
    } else {
        tensor::element_count(extra_dims)
    };
    let mut coords = vec![0usize; shape.len()];
    for extra_index in 0..extra_count {
        for coord in coords.iter_mut().skip(2) {
            *coord = 0;
        }
        if !extra_dims.is_empty() {
            let mut remainder = extra_index;
            for (offset, size) in extra_dims.iter().copied().enumerate() {
                let dim = offset + 2;
                if size == 0 {
                    coords[dim] = 0;
                } else {
                    coords[dim] = remainder % size;
                    remainder /= size;
                }
            }
        }
        for diag in 0..diag_len {
            coords[0] = diag;
            coords[1] = diag;
            let mut linear = 0usize;
            for (dim, &coord) in coords.iter().enumerate() {
                linear += coord * strides[dim];
            }
            f(linear);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_default_scalar() {
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("eye");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_square_from_scalar_dimension() {
        let args = vec![Value::Num(3.0)];
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[3, 3]);
                for r in 0..3 {
                    for c in 0..3 {
                        let idx = r + c * 3;
                        if r == c {
                            assert_eq!(t.materialize_f64()[idx], 1.0);
                        } else {
                            assert_eq!(t.materialize_f64()[idx], 0.0);
                        }
                    }
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_rectangular_from_two_dims() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 4]);
                assert_eq!(t.materialize_f64().len(), 8);
                for r in 0..2 {
                    for c in 0..4 {
                        let idx = r + c * 2;
                        if r == c {
                            assert_eq!(t.materialize_f64()[idx], 1.0);
                        } else {
                            assert_eq!(t.materialize_f64()[idx], 0.0);
                        }
                    }
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_accepts_size_vector_argument() {
        let size_vec = Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap();
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(size_vec)]),
        ))
        .expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 4]);
                assert_eq!(t.materialize_f64().len(), 8);
                assert_eq!(t.materialize_f64()[0], 1.0);
                assert_eq!(t.materialize_f64()[1], 0.0);
                assert_eq!(t.materialize_f64()[2], 0.0);
                assert_eq!(t.materialize_f64()[3], 1.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_zero_dimension_matrix() {
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
        ))
        .expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[0, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_uses_tensor_argument_shape_and_type() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor.clone())];
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), tensor.shape());
                assert_eq!(t.materialize_f64(), vec![1.0, 0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_logical_output() {
        let args = vec![Value::Num(4.0), Value::from("logical")];
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("eye");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![4, 4]);
                for r in 0..4 {
                    for c in 0..4 {
                        let idx = r + c * 4;
                        if r == c {
                            assert_eq!(logical.data[idx], 1);
                        } else {
                            assert_eq!(logical.data[idx], 0);
                        }
                    }
                }
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_like_bool_produces_logical() {
        let args = vec![Value::Num(3.0), Value::from("like"), Value::Bool(true)];
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("eye");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![3, 3]);
                for row in 0..3 {
                    for col in 0..3 {
                        let idx = row + col * 3;
                        if row == col {
                            assert_eq!(logical.data[idx], 1);
                        } else {
                            assert_eq!(logical.data[idx], 0);
                        }
                    }
                }
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn resident_complex_integer_eye_retains_paired_class_and_borrowed_input() {
        test_support::with_test_provider(|provider| {
            let storage = IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX]),
                IntegerStorage::U64(vec![1u64 << 63]),
            )
            .unwrap();
            let host = ComplexTensor::new_integer(storage, vec![1, 1]).unwrap();
            let input = gpu_helpers::upload_complex_tensor(provider, &host).unwrap();
            let value = block_on(eye_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::Num(3.0),
                    Value::from("like"),
                    Value::GpuTensor(input.clone()),
                ]),
            ))
            .unwrap();
            let Value::GpuTensor(output) = &value else {
                panic!("resident paired identity")
            };
            assert!(!output.shares_allocation_with(&input));
            assert_eq!(
                output.physical().storage(),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(
                output.physical().element_type(),
                runmat_accelerate_api::NumericElementType::U64
            );
            let Value::ComplexTensor(gathered) =
                block_on(gpu_helpers::gather_value_async(&value)).unwrap()
            else {
                panic!("paired host storage")
            };
            let storage = gathered.integer_storage().unwrap();
            assert_eq!(storage.real(), &IntegerStorage::U64(vec![1, 0, 0, 1, 0, 0]));
            assert_eq!(storage.imag(), &IntegerStorage::U64(vec![0; 6]));
            let Value::ComplexTensor(original) =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(input))).unwrap()
            else {
                panic!("paired prototype")
            };
            assert_eq!(original.integer_storage(), host.integer_storage());
        });
    }

    #[test]
    fn resident_logical_eye_is_produced_as_logical() {
        test_support::with_test_provider(|provider| {
            let input = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1],
                    shape: &[1, 1],
                })
                .unwrap();
            let value = block_on(eye_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::Num(3.0),
                    Value::from("like"),
                    Value::GpuTensor(input.clone()),
                ]),
            ))
            .unwrap();
            let Value::GpuTensor(output) = &value else {
                panic!("resident logical identity")
            };
            assert!(output.view().is_logical());
            assert!(!output.shares_allocation_with(&input));
            assert_eq!(
                test_support::gather(value).unwrap().materialize_f64(),
                vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            );
        });
    }

    #[test]
    fn eye_like_preserves_every_exact_integer_class() {
        let storages = vec![
            IntegerStorage::I8(vec![i8::MIN]),
            IntegerStorage::I16(vec![i16::MIN]),
            IntegerStorage::I32(vec![i32::MIN]),
            IntegerStorage::I64(vec![i64::MIN]),
            IntegerStorage::U8(vec![u8::MAX]),
            IntegerStorage::U16(vec![u16::MAX]),
            IntegerStorage::U32(vec![u32::MAX]),
            IntegerStorage::U64(vec![u64::MAX]),
        ];

        for storage in storages {
            let prototype = Tensor::new_integer(storage.clone(), vec![1, 1]).expect("prototype");
            let result = block_on(eye_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::Num(3.0),
                    Value::from("like"),
                    Value::Tensor(prototype),
                ]),
            ))
            .expect("eye like");
            let Value::Tensor(output) = result else {
                panic!("expected integer tensor");
            };
            let mut expected = storage.zeros_like(6);
            let one = storage.ones_like(1).value_at(0).expect("one");
            expected.set_value(0, one.clone()).expect("first diagonal");
            expected.set_value(3, one).expect("second diagonal");
            assert_eq!(output.shape(), &[2, 3]);
            assert_eq!(output.integer_storage(), Some(&expected));
        }

        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::Int(IntValue::I64(i64::MAX)),
            ]),
        ))
        .expect("integer scalar prototype");
        let Value::Tensor(output) = result else {
            panic!("expected int64 tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I64(vec![1, 0, 0, 1]))
        );
    }

    #[test]
    fn eye_class_strings_create_exact_integer_identity_storage() {
        let cases = [
            ("int8", IntegerStorage::I8(vec![1, 0, 0, 1, 0, 0])),
            ("int16", IntegerStorage::I16(vec![1, 0, 0, 1, 0, 0])),
            ("int32", IntegerStorage::I32(vec![1, 0, 0, 1, 0, 0])),
            ("int64", IntegerStorage::I64(vec![1, 0, 0, 1, 0, 0])),
            ("uint8", IntegerStorage::U8(vec![1, 0, 0, 1, 0, 0])),
            ("uint16", IntegerStorage::U16(vec![1, 0, 0, 1, 0, 0])),
            ("uint32", IntegerStorage::U32(vec![1, 0, 0, 1, 0, 0])),
            ("uint64", IntegerStorage::U64(vec![1, 0, 0, 1, 0, 0])),
        ];

        for (class_name, expected) in cases {
            let result = block_on(eye_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::Num(3.0),
                    Value::from(class_name),
                ]),
            ))
            .expect("eye integer class");
            let Value::Tensor(output) = result else {
                panic!("expected integer tensor for {class_name}");
            };
            assert_eq!(output.shape(), &[2, 3]);
            assert_eq!(output.integer_storage(), Some(&expected));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_prototype_with_logical_override() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let args = vec![Value::Tensor(proto), Value::from("logical")];
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("eye");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 3]);
                assert_eq!(logical.data.len(), 6);
                assert_eq!(&logical.data[..6], &[1, 0, 0, 1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_complex_like() {
        let args = vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ];
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("eye");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert_eq!(t.materialize_f64()[0], (1.0, 0.0));
                assert_eq!(t.materialize_f64()[1], (0.0, 0.0));
                assert_eq!(t.materialize_f64()[2], (0.0, 0.0));
                assert_eq!(t.materialize_f64()[3], (1.0, 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn eye_like_typed_complex_uint64_keeps_exact_integer_storage() {
        let prototype = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX]),
                IntegerStorage::U64(vec![9_007_199_254_740_993]),
            )
            .expect("typed complex prototype"),
            vec![1, 1],
        )
        .expect("typed complex tensor");
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::ComplexTensor(prototype),
            ]),
        ))
        .expect("eye like");
        let Value::ComplexTensor(output) = result else {
            panic!("expected typed complex output");
        };
        assert_eq!(
            output.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![1, 0, 0, 1]),
                    IntegerStorage::U64(vec![0; 4]),
                )
                .expect("typed complex identity"),
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_extra_dimensions_replicate_identity() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let args = vec![Value::Num(2.0), Value::Num(3.0), Value::Num(2.0)];
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 3, 2]);
                // slice 0
                assert_eq!(t.materialize_f64()[0], 1.0);
                assert_eq!(t.materialize_f64()[1], 0.0);
                assert_eq!(t.materialize_f64()[2], 0.0);
                assert_eq!(t.materialize_f64()[3], 1.0);
                // slice 1 offset = rows * cols = 6
                assert_eq!(t.materialize_f64()[6], 1.0);
                assert_eq!(t.materialize_f64()[7], 0.0);
                assert_eq!(t.materialize_f64()[8], 0.0);
                assert_eq!(t.materialize_f64()[9], 1.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_like_requires_prototype() {
        assert!(block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("like")])
        ))
        .is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_gpu_like_alloc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(3.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(eye_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect("eye");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape(), &[2, 3]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert_eq!(gathered.shape(), &[2, 3]);
                    assert_eq!(gathered.materialize_f64()[0], 1.0);
                    assert_eq!(gathered.materialize_f64()[1], 0.0);
                    assert_eq!(gathered.materialize_f64()[2], 0.0);
                    assert_eq!(gathered.materialize_f64()[3], 1.0);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_gpu_prototype_infers_shape() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let prototype = provider.upload(&view).expect("upload proto");
            let result = block_on(eye_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                    prototype,
                )]),
            ))
            .expect("eye");
            let gathered = test_support::gather(result).expect("gather gpu identity");
            assert_eq!(gathered.shape(), &[2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_like_string_first_argument() {
        let proto = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(proto)];
        let result = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("eye");
        assert!(matches!(result, Value::Num(value) if value == 1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_like_and_logical_conflict() {
        let args = vec![
            Value::Num(2.0),
            Value::from("logical"),
            Value::from("like"),
            Value::Num(1.0),
        ];
        assert!(block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args)
        ))
        .is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_negative_dimension_clamps_to_zero() {
        let args = vec![Value::Num(-1.0)];
        let Value::Tensor(result) = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("eye") else {
            panic!("expected empty tensor");
        };
        assert_eq!(result.shape(), &[0, 0]);
        assert!(result.is_empty());
    }

    #[test]
    fn eye_strict_mode_gates_implicit_and_nd_extensions() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let matrix = Tensor::new(vec![0.0; 4], vec![2, 2]).expect("matrix");
        let implicit = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(matrix)]),
        ))
        .unwrap_err();
        assert_eq!(
            implicit.identifier(),
            Some("RunMat:compatibility:EyeImplicitPrototypeExtension")
        );
        let nd = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::Num(2.0),
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            nd.identifier(),
            Some("RunMat:compatibility:EyeNdDimensionsExtension")
        );
    }

    #[test]
    fn eye_single_and_sparse_like_preserve_representation() {
        let single = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("single"),
            ]),
        ))
        .expect("single eye");
        assert!(
            matches!(single, Value::Tensor(ref tensor) if tensor.numeric_dtype() == NumericDType::F32)
        );

        let prototype = SparseTensor::zeros_f32(3, 3);
        let sparse = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::Num(3.0),
                Value::from("like"),
                Value::SparseTensor(prototype),
            ]),
        ))
        .expect("sparse eye");
        assert!(
            matches!(sparse, Value::SparseTensor(ref value) if value.shape() == vec![2, 3] && value.numeric_dtype() == Some(NumericDType::F32) && value.nnz() == 2)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_rejects_non_integer_dimension() {
        let args = vec![Value::Num(2.5)];
        assert!(block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(args)
        ))
        .is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn eye_wgpu_matches_cpu() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let cpu_value = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)]),
        ))
        .expect("cpu eye");
        let host_proto = Tensor::zeros(vec![3, 3]);
        let view = HostTensorView {
            data: &host_proto.materialize_f64(),
            shape: host_proto.shape(),
        };
        let gpu_proto = provider.upload(&view).expect("upload proto");

        let gpu_value = block_on(eye_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(3.0),
                Value::from("like"),
                Value::GpuTensor(gpu_proto),
            ]),
        ))
        .expect("gpu eye");

        let gathered = test_support::gather(gpu_value).expect("gather gpu eye");
        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor for cpu eye, got {other:?}"),
        };

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        assert_eq!(gathered.materialize_f64(), cpu_tensor.materialize_f64());
    }
}
