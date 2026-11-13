//! `createArray` builtin backed by RunMat's constant-array construction kernels.
//!
//! The implementation mirrors the modern RunMat builtin blueprint with GPU-aware semantics.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, Tensor, Value,
};

#[cfg(test)]
use crate::builtins::common::random_args::shape_from_value;
use crate::builtins::common::random_args::{extract_dims, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionKernelTemplate,
    GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::fill")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "createArray",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("fill"),
        ProviderHook::Custom("fill_like"),
    ],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs dedicated constant-fill kernels; falls back to host upload when the provider reports an error.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::fill")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "createArray",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let constant = ctx
                .constants
                .first()
                .ok_or(crate::builtins::common::spec::FusionError::MissingInput(0))?;
            writer.push(constant)
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner treats fill as a constant generator backed by a uniform parameter.",
};

#[runtime_builtin(
    name = "createArray",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::fill"
)]
async fn create_array_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    let parsed = ParsedFill::parse_create_array(args).await?;
    build_output(parsed)
}

// Retained as a source-local regression harness for the constant-fill kernel
// while public dispatch is owned by `createArray` above.
#[cfg(test)]
async fn fill_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered_value = crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(|e| format!("createArray: {e}"))?;
    let scalar = FillScalar::from_value(&gathered_value)?;
    let parsed = ParsedFill::parse(scalar, rest).await?;
    build_output(parsed)
}

struct ParsedFill {
    fill: FillScalar,
    shape: Vec<usize>,
    template: OutputTemplate,
    residency: Option<GpuTensorHandle>,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Single,
    Logical,
    Complex,
    Integer(IntegerStorage),
    Like(Value),
}

#[derive(Clone)]
enum FillScalar {
    Real(f64),
    Integer(IntValue),
    Complex(f64, f64),
    ComplexInteger(IntValue, IntValue),
    Logical(bool),
}

impl FillScalar {
    fn is_complex(&self) -> bool {
        matches!(self, Self::Complex(_, _) | Self::ComplexInteger(_, _))
    }

    fn from_value(value: &Value) -> Result<Self, String> {
        match value {
            Value::Num(n) => Ok(FillScalar::Real(*n)),
            Value::Int(i) => Ok(FillScalar::Integer(i.clone())),
            Value::Bool(b) => Ok(FillScalar::Logical(*b)),
            Value::LogicalArray(logical) => {
                if logical.data.len() != 1 {
                    return Err("fill: fill value must be a scalar".to_string());
                }
                Ok(FillScalar::Logical(logical.data[0] != 0))
            }
            Value::Tensor(tensor) => {
                if !tensor::is_scalar_tensor(tensor) {
                    return Err("fill: fill value must be a scalar".to_string());
                }
                if let Some(storage) = tensor.integer_storage() {
                    let value = storage
                        .value_at(0)
                        .ok_or_else(|| "fill: fill value must be a scalar".to_string())?;
                    Ok(FillScalar::Integer(value))
                } else {
                    Ok(FillScalar::Real(tensor::tensor_value_f64(tensor, 0)))
                }
            }
            Value::Complex(re, im) => Ok(FillScalar::Complex(*re, *im)),
            Value::ComplexTensor(tensor) => {
                if let Some(storage) = tensor.integer_storage() {
                    if storage.len() != 1 {
                        return Err("fill: fill value must be a scalar".to_string());
                    }
                    let re = storage
                        .real()
                        .value_at(0)
                        .ok_or_else(|| "fill: fill value must be a scalar".to_string())?;
                    let im = storage
                        .imag()
                        .value_at(0)
                        .ok_or_else(|| "fill: fill value must be a scalar".to_string())?;
                    return Ok(FillScalar::ComplexInteger(re, im));
                }
                if tensor.materialize_f64().len() != 1 {
                    return Err("fill: fill value must be a scalar".to_string());
                }
                Ok(FillScalar::Complex(
                    tensor.materialize_f64()[0].0,
                    tensor.materialize_f64()[0].1,
                ))
            }
            Value::CharArray(ca) => {
                if ca.data().len() != 1 {
                    return Err("fill: fill value must be a scalar".to_string());
                }
                Ok(FillScalar::Real(f64::from(ca.data()[0].raw())))
            }
            Value::StringArray(sa)
                if sa.data().len() == 1
                    && sa.data()[0].as_str().is_some_and(|text| text.len() == 1)
                    && !sa.data()[0].is_missing() =>
            {
                let ch = sa.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .chars()
                    .next()
                    .unwrap();
                Ok(FillScalar::Real(ch as u32 as f64))
            }
            Value::String(runmat_value::StringElement::Text(s)) if s.len() == 1 => {
                let ch = s.chars().next().unwrap();
                Ok(FillScalar::Real(ch as u32 as f64))
            }
            other => Err(format!(
                "fill: fill value must be numeric, logical, or complex scalar (got {other:?})"
            )),
        }
    }

    fn as_real(&self) -> Result<f64, String> {
        match self {
            FillScalar::Real(v) => Ok(*v),
            FillScalar::Integer(i) => Ok(i.to_f64()),
            FillScalar::Logical(b) => Ok(if *b { 1.0 } else { 0.0 }),
            FillScalar::Complex(re, im) => {
                if im.abs() > f64::EPSILON {
                    Err("fill: imaginary component must be zero for real outputs".to_string())
                } else {
                    Ok(*re)
                }
            }
            FillScalar::ComplexInteger(re, im) => {
                if !im.is_zero() {
                    Err("fill: imaginary component must be zero for real outputs".to_string())
                } else {
                    Ok(re.to_f64())
                }
            }
        }
    }

    fn as_complex(&self) -> (f64, f64) {
        match self {
            FillScalar::Real(v) => (*v, 0.0),
            FillScalar::Integer(i) => (i.to_f64(), 0.0),
            FillScalar::Logical(b) => (if *b { 1.0 } else { 0.0 }, 0.0),
            FillScalar::Complex(re, im) => (*re, *im),
            FillScalar::ComplexInteger(re, im) => (re.to_f64(), im.to_f64()),
        }
    }

    fn as_bool(&self) -> bool {
        match self {
            FillScalar::Real(v) => *v != 0.0,
            FillScalar::Integer(i) => !i.is_zero(),
            FillScalar::Logical(b) => *b,
            FillScalar::Complex(re, im) => *re != 0.0 || *im != 0.0,
            FillScalar::ComplexInteger(re, im) => !re.is_zero() || !im.is_zero(),
        }
    }
}

impl ParsedFill {
    async fn parse_create_array(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Self> {
        let mut dims = Vec::new();
        let mut class_name: Option<String> = None;
        let mut fill_value: Option<Value> = None;
        let mut like: Option<Value> = None;
        let mut idx = 0;
        while idx < args.len() {
            if let Some(keyword) = keyword_of(&args[idx])? {
                match keyword.as_str() {
                    "fillvalue" => {
                        let value = args.get(idx + 1).cloned().ok_or_else(|| {
                            "createArray: FillValue requires a scalar value".to_string()
                        })?;
                        if fill_value.replace(value).is_some() {
                            return Err("createArray: FillValue may be specified only once".into());
                        }
                        idx += 2;
                        continue;
                    }
                    "like" => {
                        let value = args
                            .get(idx + 1)
                            .cloned()
                            .ok_or_else(|| "createArray: Like requires a prototype".to_string())?;
                        if like.replace(value).is_some() {
                            return Err("createArray: Like may be specified only once".into());
                        }
                        idx += 2;
                        continue;
                    }
                    "double" | "single" | "logical" | "complex" | "int8" | "int16" | "int32"
                    | "int64" | "uint8" | "uint16" | "uint32" | "uint64" => {
                        if class_name.replace(keyword).is_some() {
                            return Err("createArray: class name may be specified only once".into());
                        }
                        idx += 1;
                        continue;
                    }
                    _ => {}
                }
            }
            let parsed = create_array_dims(&args[idx]).await?.ok_or_else(|| {
                "createArray: dimensions must be integer scalars or a row size vector".to_string()
            })?;
            dims.extend(parsed);
            idx += 1;
        }

        if class_name.is_some() && like.is_some() {
            return Err("createArray: classname and Like cannot be combined".into());
        }
        let shape = if dims.is_empty() {
            vec![1, 1]
        } else if dims.len() == 1 {
            vec![dims[0], dims[0]]
        } else {
            while dims.len() > 2 && dims.last() == Some(&1) {
                dims.pop();
            }
            dims
        };

        let fill_source = fill_value.unwrap_or(Value::Num(0.0));
        let fill_residency = match &fill_source {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        };
        let gathered_fill = crate::dispatcher::gather_if_needed_async(&fill_source)
            .await
            .map_err(|e| format!("createArray: {e}"))?;
        let fill = FillScalar::from_value(&gathered_fill)
            .map_err(|err| err.replacen("fill:", "createArray:", 1))?;

        let has_explicit_like = like.is_some();
        let template = if let Some(proto) = like {
            OutputTemplate::Like(proto)
        } else if let Some(class_name) = class_name {
            template_from_class_name(&class_name)?
        } else {
            template_from_fill_value(&gathered_fill, &fill)
        };
        Ok(Self {
            fill,
            shape,
            template,
            residency: if has_explicit_like {
                None
            } else {
                fill_residency
            },
        })
    }

    #[cfg(test)]
    async fn parse(fill: FillScalar, args: Vec<Value>) -> crate::BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<OutputTemplate> = None;
        let mut implicit_proto: Option<Value> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) = keyword_of(&arg)? {
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err(
                                "fill: multiple 'like' specifications are not supported".into()
                            );
                        }
                        if class_override.is_some() {
                            return Err("fill: cannot combine 'like' with class specifiers".into());
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err("fill: expected prototype after 'like'".into());
                        };
                        like_proto = Some(proto.clone());
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source = Some(shape_from_value(&proto, "fill")?);
                        }
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err("fill: cannot combine 'like' with 'logical'".into());
                        }
                        class_override = Some(OutputTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err("fill: cannot combine 'like' with 'double'".into());
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "complex" => {
                        if like_proto.is_some() {
                            return Err("fill: cannot combine 'like' with 'complex'".into());
                        }
                        class_override = Some(OutputTemplate::Complex);
                        idx += 1;
                        continue;
                    }
                    "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32"
                    | "uint64" => {
                        if like_proto.is_some() {
                            return Err(
                                format!("fill: cannot combine 'like' with '{keyword}'").into()
                            );
                        }
                        class_override = Some(OutputTemplate::Integer(
                            integer_storage_prototype_from_keyword(keyword.as_str())
                                .expect("matched integer class keyword"),
                        ));
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        return Err("fill: single precision output is not implemented yet".into());
                    }
                    other => {
                        return Err(format!("fill: unrecognised option '{other}'").into());
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg, "fill").await? {
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
                shape_source = Some(shape_from_value(&arg, "fill")?);
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

        let default_template = match &fill {
            FillScalar::Logical(_) => OutputTemplate::Logical,
            FillScalar::Complex(_, _) | FillScalar::ComplexInteger(_, _) => OutputTemplate::Complex,
            FillScalar::Real(_) | FillScalar::Integer(_) => OutputTemplate::Double,
        };

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
        } else if let Some(proto) = implicit_proto {
            OutputTemplate::Like(proto)
        } else {
            default_template
        };

        Ok(Self {
            fill,
            shape,
            template,
            residency: None,
        })
    }
}

async fn create_array_dims(value: &Value) -> Result<Option<Vec<usize>>, String> {
    match value {
        Value::Num(value) if value.is_finite() && *value < 0.0 => Ok(Some(vec![0])),
        Value::Int(value) if value.try_to_i64().is_some_and(|value| value < 0) => Ok(Some(vec![0])),
        Value::Tensor(tensor) if matches!(tensor.shape(), [_] | [1, ..]) || tensor.len() == 1 => {
            if let Some(storage) = tensor.integer_storage() {
                let dims = storage
                    .exact_values()
                    .iter()
                    .map(|value| {
                        if value.try_to_i64().is_some_and(|value| value < 0) {
                            Ok(0)
                        } else {
                            value.try_to_usize().ok_or_else(|| {
                                "createArray: dimension is outside the platform range".to_string()
                            })
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(Some(dims));
            }
            let dims = tensor
                .materialize_f64()
                .into_iter()
                .map(|value| if value < 0.0 { 0.0 } else { value })
                .map(|value| {
                    if !value.is_finite() || value.fract() != 0.0 || value > usize::MAX as f64 {
                        Err("createArray: dimensions must be finite platform integers".to_string())
                    } else {
                        Ok(value as usize)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Some(dims))
        }
        _ => extract_dims(value, "createArray").await,
    }
}

fn template_from_class_name(class_name: &str) -> Result<OutputTemplate, String> {
    Ok(match class_name {
        "double" => OutputTemplate::Double,
        "single" => OutputTemplate::Single,
        "logical" => OutputTemplate::Logical,
        "complex" => OutputTemplate::Complex,
        integer => OutputTemplate::Integer(
            integer_storage_prototype_from_keyword(integer)
                .ok_or_else(|| format!("createArray: unsupported class `{integer}`"))?,
        ),
    })
}

fn template_from_fill_value(value: &Value, fill: &FillScalar) -> OutputTemplate {
    match value {
        Value::Int(value) => OutputTemplate::Integer(IntegerStorage::from_scalar(value.clone())),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            OutputTemplate::Integer(tensor.integer_storage().expect("checked").zeros_like(0))
        }
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
            OutputTemplate::Like(value.clone())
        }
        Value::ComplexTensor(tensor)
            if tensor.numeric_dtype() == runmat_value::NumericDType::F32 =>
        {
            OutputTemplate::Like(value.clone())
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32 => {
            OutputTemplate::Single
        }
        _ => match fill {
            FillScalar::Logical(_) => OutputTemplate::Logical,
            FillScalar::Complex(_, _) | FillScalar::ComplexInteger(_, _) => OutputTemplate::Complex,
            FillScalar::Real(_) | FillScalar::Integer(_) => OutputTemplate::Double,
        },
    }
}

fn build_output(parsed: ParsedFill) -> crate::BuiltinResult<Value> {
    let residency = parsed.residency;
    let output = match parsed.template {
        OutputTemplate::Double
            if matches!(
                &parsed.fill,
                FillScalar::Complex(_, _) | FillScalar::ComplexInteger(_, _)
            ) =>
        {
            fill_complex(&parsed.fill, &parsed.shape)
        }
        OutputTemplate::Double => fill_double(&parsed.fill, &parsed.shape),
        OutputTemplate::Single
            if matches!(
                &parsed.fill,
                FillScalar::Complex(_, _) | FillScalar::ComplexInteger(_, _)
            ) =>
        {
            fill_complex_single(&parsed.fill, &parsed.shape)
        }
        OutputTemplate::Single => fill_single(&parsed.fill, &parsed.shape),
        OutputTemplate::Logical => fill_logical(&parsed.fill, &parsed.shape),
        OutputTemplate::Complex => fill_complex(&parsed.fill, &parsed.shape),
        OutputTemplate::Integer(storage)
            if matches!(
                &parsed.fill,
                FillScalar::Complex(_, _) | FillScalar::ComplexInteger(_, _)
            ) =>
        {
            let prototype = IntegerComplexStorage::new(storage.clone(), storage.zeros_like(0))?;
            fill_complex_integer_like(&parsed.fill, &parsed.shape, &prototype)
        }
        OutputTemplate::Integer(storage) => {
            fill_integer_like(&parsed.fill, &parsed.shape, &storage)
        }
        OutputTemplate::Like(proto) => Ok(fill_like(&parsed.fill, &parsed.shape, &proto)?),
    }?;
    match residency {
        Some(owner) => upload_created_output(output, &parsed.shape, &owner),
        None => Ok(output),
    }
}

fn upload_created_output(
    output: Value,
    shape: &[usize],
    owner: &GpuTensorHandle,
) -> crate::BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(owner)
        .ok_or("createArray: GPU FillValue has no registered owning provider")?;
    let handle = match output {
        Value::Tensor(tensor) => {
            gpu_helpers::upload_tensor_protecting(provider, &tensor, &[owner])?
        }
        Value::Num(value) => {
            let tensor = Tensor::new(vec![value], shape.to_vec())?;
            gpu_helpers::upload_tensor_protecting(provider, &tensor, &[owner])?
        }
        Value::Int(value) => {
            let tensor = Tensor::new_integer(IntegerStorage::from_scalar(value), shape.to_vec())?;
            gpu_helpers::upload_tensor_protecting(provider, &tensor, &[owner])?
        }
        Value::LogicalArray(array) => {
            upload_logical_fill_value(provider, &array.data, &array.shape, owner)?
        }
        Value::Bool(value) => {
            upload_logical_fill_value(provider, &[u8::from(value)], shape, owner)?
        }
        Value::ComplexTensor(tensor) => {
            gpu_helpers::upload_complex_tensor_protecting(provider, &tensor, &[owner])?
        }
        Value::Complex(real, imag) => {
            let tensor = ComplexTensor::new(vec![(real, imag)], shape.to_vec())?;
            gpu_helpers::upload_complex_tensor_protecting(provider, &tensor, &[owner])?
        }
        other => {
            return Err(format!(
                "createArray: cannot preserve GPU FillValue residency for {other:?}"
            )
            .into())
        }
    };
    if handle.shape() != shape
        || handle.shares_allocation_with(owner)
        || !gpu_helpers::exact_provider_for_handle(&handle)
            .is_some_and(|actual| runmat_accelerate_api::same_provider_instance(actual, provider))
    {
        return Err(gpu_helpers::reject_gpu_output(
            handle,
            crate::RuntimeError::from("createArray: invalid uploaded output owner, shape or alias"),
        ));
    }
    let handle = gpu_helpers::propagate_owned_output_provenance(handle, [owner])?;
    gpu_helpers::resident_gpu_value(handle)
}

fn upload_logical_fill_value(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    data: &[u8],
    shape: &[usize],
    owner: &GpuTensorHandle,
) -> crate::BuiltinResult<GpuTensorHandle> {
    runmat_accelerate_api::upload_logical_protecting(
        provider,
        &runmat_accelerate_api::HostLogicalTensorView { data, shape },
        &[owner],
    )
    .map_err(|cause| {
        crate::build_runtime_error(format!("createArray: logical upload failed: {cause}"))
            .with_builtin("createArray")
            .with_source(cause)
            .build()
    })
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

fn fill_double(fill: &FillScalar, shape: &[usize]) -> Result<Value, String> {
    let tensor = make_real_tensor(fill, shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn fill_single(fill: &FillScalar, shape: &[usize]) -> Result<Value, String> {
    let value = fill.as_real()?;
    let tensor = Tensor::from_numeric_storage(
        runmat_value::NumericStorage::F32(vec![value as f32; tensor::element_count(shape)]),
        shape.to_vec(),
    )
    .map_err(|err| format!("createArray: {err}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn fill_logical(fill: &FillScalar, shape: &[usize]) -> Result<Value, String> {
    let logical = make_logical_array(fill, shape)?;
    Ok(Value::LogicalArray(logical))
}

fn fill_complex(fill: &FillScalar, shape: &[usize]) -> Result<Value, String> {
    let tensor = make_complex_tensor(fill, shape)?;
    Ok(crate::builtins::common::random_args::complex_tensor_into_value(tensor))
}

fn fill_complex_single(fill: &FillScalar, shape: &[usize]) -> Result<Value, String> {
    let (real, imag) = fill.as_complex();
    ComplexTensor::from_f32(
        vec![(real as f32, imag as f32); tensor::element_count(shape)],
        shape.to_vec(),
    )
    .map(Value::ComplexTensor)
    .map_err(|error| format!("createArray: {error}"))
}

fn fill_like(fill: &FillScalar, shape: &[usize], proto: &Value) -> crate::BuiltinResult<Value> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) if fill.is_complex() => fill_complex(fill, shape),
        Value::LogicalArray(_) | Value::Bool(_) => fill_logical(fill, shape),
        Value::ComplexTensor(tensor) => match tensor.integer_storage() {
            Some(storage) => fill_complex_integer_like(fill, shape, storage),
            None if tensor.numeric_dtype() == runmat_value::NumericDType::F32 => {
                fill_complex_single(fill, shape)
            }
            None => fill_complex(fill, shape),
        },
        Value::Complex(_, _) => fill_complex(fill, shape),
        Value::GpuTensor(handle) => return fill_like_gpu(fill, shape, handle),
        Value::Tensor(tensor) => match tensor.integer_storage() {
            Some(storage) if fill.is_complex() => {
                let prototype =
                    IntegerComplexStorage::new(storage.zeros_like(0), storage.zeros_like(0))?;
                fill_complex_integer_like(fill, shape, &prototype)
            }
            Some(storage) => fill_integer_like(fill, shape, storage),
            None if fill.is_complex()
                && tensor.numeric_dtype() == runmat_value::NumericDType::F32 =>
            {
                fill_complex_single(fill, shape)
            }
            None if fill.is_complex() => fill_complex(fill, shape),
            None if tensor.numeric_dtype() == runmat_value::NumericDType::F32 => {
                fill_single(fill, shape)
            }
            None => fill_double(fill, shape),
        },
        Value::Int(value) => {
            let prototype = IntegerStorage::from_scalar(value.clone());
            if fill.is_complex() {
                let complex =
                    IntegerComplexStorage::new(prototype.zeros_like(0), prototype.zeros_like(0))?;
                fill_complex_integer_like(fill, shape, &complex)
            } else {
                fill_integer_like(fill, shape, &prototype)
            }
        }
        Value::Num(_) if fill.is_complex() => fill_complex(fill, shape),
        Value::Num(_) => fill_double(fill, shape),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) | Value::Cell(_) => {
            Err("fill: character, string, and cell prototypes are not supported yet".to_string())
        }
        other => Err(format!(
            "fill: unsupported prototype type {:?} for 'like' output",
            other
        )),
    }
    .map_err(Into::into)
}

fn fill_integer_like(
    fill: &FillScalar,
    shape: &[usize],
    prototype: &IntegerStorage,
) -> Result<Value, String> {
    let storage = exact_integer_fill_storage(fill, shape, prototype)?;
    Tensor::new_integer(storage, shape.to_vec())
        .map(tensor::tensor_into_value)
        .map_err(|e| format!("fill: {e}"))
}

fn fill_complex_integer_like(
    fill: &FillScalar,
    shape: &[usize],
    prototype: &IntegerComplexStorage,
) -> Result<Value, String> {
    let (real, imag) = match fill {
        FillScalar::Integer(value) => (
            prototype.real().cast_exact_assignment(value),
            prototype.imag().cast_f64_assignment(0.0),
        ),
        FillScalar::Complex(real, imag) => (
            prototype.real().cast_f64_assignment(*real),
            prototype.imag().cast_f64_assignment(*imag),
        ),
        FillScalar::ComplexInteger(real, imag) => (
            prototype.real().cast_exact_assignment(real),
            prototype.imag().cast_exact_assignment(imag),
        ),
        _ => (
            prototype.real().cast_f64_assignment(fill.as_real()?),
            prototype.imag().cast_f64_assignment(0.0),
        ),
    };
    let count = tensor::element_count(shape);
    let storage = IntegerComplexStorage::new(
        prototype.real().from_same_class_values(vec![real; count])?,
        prototype.imag().from_same_class_values(vec![imag; count])?,
    )?;
    ComplexTensor::new_integer(storage, shape.to_vec())
        .map(Value::ComplexTensor)
        .map_err(|error| format!("createArray: {error}"))
}

fn fill_like_gpu(
    fill: &FillScalar,
    shape: &[usize],
    prototype: &GpuTensorHandle,
) -> crate::BuiltinResult<Value> {
    use runmat_accelerate_api::{GpuTensorStorage, ProviderPrecision};
    let provider = gpu_helpers::exact_provider_for_handle(prototype)
        .ok_or("createArray: GPU Like prototype has no registered owning provider")?;
    let logical = prototype.view().is_logical() && !fill.is_complex();
    let complex = !logical
        && (fill.is_complex()
            || prototype.physical().storage() == GpuTensorStorage::ComplexInterleaved);
    let integer = if prototype.view().is_logical() {
        None
    } else {
        runmat_accelerate_api::handle_integer_type(prototype)
    };
    let precision = if prototype.view().is_logical() {
        Some(ProviderPrecision::F64)
    } else {
        runmat_accelerate_api::handle_precision(prototype)
    };
    let attempt = if logical {
        Some(provider.logical_fill(shape, fill.as_bool()))
    } else if !complex && integer.is_none() && precision == Some(provider.precision()) {
        let value = fill.as_real()?;
        Some(if prototype.shape() == shape {
            provider.fill_like(prototype, value)
        } else {
            provider.fill(shape, value)
        })
    } else {
        None
    };
    if let Some(attempt) = attempt {
        match attempt {
            Ok(Some(output)) => {
                let output = crate::builtins::common::random_args::validate_constructor_gpu_output_protecting(
                    "createArray", provider, output, shape, GpuTensorStorage::Real,
                    if logical { None } else { precision }, None, logical, &[prototype],
                )?;
                let output = gpu_helpers::propagate_owned_output_provenance(output, [prototype])?;
                return gpu_helpers::resident_gpu_value(output);
            }
            Ok(None) => {}
            Err(cause) => {
                return Err(crate::runtime_error::with_provider_execution_source(
                    crate::build_runtime_error(format!(
                        "createArray: provider fill failed: {cause}"
                    ))
                    .with_builtin("createArray")
                    .build(),
                    cause.with_unproven_effects(),
                ))
            }
        }
    }
    let host = if logical {
        fill_logical(fill, shape)?
    } else if let Some(integer) = integer {
        let storage = integer_storage_prototype(integer);
        if complex {
            let paired = IntegerComplexStorage::new(storage.zeros_like(0), storage.zeros_like(0))?;
            fill_complex_integer_like(fill, shape, &paired)?
        } else {
            fill_integer_like(fill, shape, &storage)?
        }
    } else {
        match (
            complex,
            precision.ok_or("createArray: floating prototype is missing precision")?,
        ) {
            (true, ProviderPrecision::F32) => fill_complex_single(fill, shape)?,
            (true, ProviderPrecision::F64) => fill_complex(fill, shape)?,
            (false, ProviderPrecision::F32) => fill_single(fill, shape)?,
            (false, ProviderPrecision::F64) => fill_double(fill, shape)?,
        }
    };
    upload_created_output(host, shape, prototype)
}

fn make_real_tensor(fill: &FillScalar, shape: &[usize]) -> Result<Tensor, String> {
    let value = fill.as_real()?;
    make_real_tensor_from_value(value, shape)
}

fn make_real_tensor_from_value(value: f64, shape: &[usize]) -> Result<Tensor, String> {
    let count = tensor::element_count(shape);
    let mut data = vec![value; count];
    if count == 0 {
        data.clear();
    }
    Tensor::new(data, shape.to_vec()).map_err(|e| format!("fill: {e}"))
}

fn make_complex_tensor(fill: &FillScalar, shape: &[usize]) -> Result<ComplexTensor, String> {
    let (re, im) = fill.as_complex();
    let count = tensor::element_count(shape);
    let data = vec![(re, im); count];
    ComplexTensor::new(data, shape.to_vec()).map_err(|e| format!("fill: {e}"))
}

fn make_logical_array(fill: &FillScalar, shape: &[usize]) -> Result<LogicalArray, String> {
    let bit = if fill.as_bool() { 1u8 } else { 0u8 };
    let count = tensor::element_count(shape);
    let mut data = vec![bit; count];
    if count == 0 {
        data.clear();
    }
    LogicalArray::new(data, shape.to_vec()).map_err(|e| format!("fill: {e}"))
}

fn exact_integer_fill_storage(
    fill: &FillScalar,
    shape: &[usize],
    prototype: &IntegerStorage,
) -> Result<IntegerStorage, String> {
    let value = match fill {
        FillScalar::Integer(value) => prototype.cast_exact_assignment(value),
        _ => prototype.cast_f64_assignment(fill.as_real()?),
    };
    prototype.from_same_class_values(vec![value; tensor::element_count(shape)])
}

fn integer_storage_prototype(
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView, HostTensorView};

    #[test]
    fn resident_single_like_complex_fill_keeps_single_components() {
        test_support::with_test_provider(|provider| {
            let host = Tensor::from_f32(vec![7.0], vec![1, 1]).unwrap();
            let input = gpu_helpers::upload_tensor(provider, &host).unwrap();
            let value = fill_like_gpu(&FillScalar::Complex(1.5, -2.25), &[2, 3], &input).unwrap();
            let Value::GpuTensor(output) = &value else {
                panic!("resident output")
            };
            assert!(!output.shares_allocation_with(&input));
            assert_eq!(
                output.physical().element_type(),
                runmat_accelerate_api::NumericElementType::F32
            );
            assert_eq!(
                output.physical().storage(),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let Value::ComplexTensor(result) =
                block_on(gpu_helpers::gather_value_async(&value)).unwrap()
            else {
                panic!("complex output")
            };
            assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
            assert_eq!(result.shape(), &[2, 3]);
            for index in 0..6 {
                let number = tensor::complex_tensor_value_complex64(&result, index);
                assert_eq!((number.re, number.im), (1.5, -2.25));
            }
            assert_eq!(
                test_support::gather(Value::GpuTensor(input))
                    .unwrap()
                    .as_f32_slice(),
                Some(&[7.0][..])
            );
        });
    }

    #[test]
    fn resident_logical_like_fill_uses_logical_producer() {
        test_support::with_test_provider(|provider| {
            let input = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[0],
                    shape: &[1, 1],
                })
                .unwrap();
            let value = fill_like_gpu(&FillScalar::Real(2.0), &[2, 3], &input).unwrap();
            let Value::GpuTensor(output) = &value else {
                panic!("resident output")
            };
            assert!(output.view().is_logical());
            assert!(!output.shares_allocation_with(&input));
            assert_eq!(
                test_support::gather(value).unwrap().materialize_f64(),
                vec![1.0; 6]
            );
            assert_eq!(
                test_support::gather(Value::GpuTensor(input))
                    .unwrap()
                    .materialize_f64(),
                vec![0.0]
            );
        });
    }

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_scalar_defaults() {
        let result = block_on(fill_builtin(Value::Num(5.0), Vec::new())).expect("fill");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_square_from_single_dimension() {
        let args = vec![Value::Num(3.0)];
        let result = block_on(fill_builtin(Value::Num(2.5), args)).expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[3, 3]);
                assert!(t.materialize_f64().iter().all(|&x| (x - 2.5).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_rectangular_dims() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = block_on(fill_builtin(Value::Num(-4.0), args)).expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 4]);
                assert!(t.materialize_f64().iter().all(|&x| (x + 4.0).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_size_vector() {
        let sz = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let result =
            block_on(fill_builtin(Value::Num(10.0), vec![Value::Tensor(sz)])).expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 3, 4]);
                assert!(t
                    .materialize_f64()
                    .iter()
                    .all(|&x| (x - 10.0).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_logical_option() {
        let args = vec![Value::Num(4.0), Value::from("logical")];
        let result = block_on(fill_builtin(Value::Num(3.0), args)).expect("fill");
        match result {
            Value::LogicalArray(l) => {
                assert_eq!(l.shape, vec![4, 4]);
                assert!(l.data.iter().all(|&b| b == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn fill_logical_option_reads_wide_uint64_truth_exactly() {
        let wide = u64::MAX;
        let result = block_on(fill_builtin(
            Value::Int(runmat_value::IntValue::U64(wide)),
            vec![Value::Num(1.0), Value::Num(3.0), Value::from("logical")],
        ))
        .expect("scalar uint64 logical fill");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(array.data, vec![1, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }

        let fill_value =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("uint64 fill");
        let result = block_on(fill_builtin(
            Value::Tensor(fill_value),
            vec![Value::Num(1.0), Value::Num(2.0), Value::from("logical")],
        ))
        .expect("typed uint64 logical fill");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 2]);
                assert_eq!(array.data, vec![1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_like_tensor_infers_shape() {
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = block_on(fill_builtin(
            Value::Num(std::f64::consts::PI),
            vec![Value::Tensor(proto)],
        ))
        .expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert!(t
                    .materialize_f64()
                    .iter()
                    .all(|&x| (x - std::f64::consts::PI).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_like_preserves_every_exact_integer_class() {
        let prototypes = [
            IntegerStorage::I8(vec![0]),
            IntegerStorage::I16(vec![0]),
            IntegerStorage::I32(vec![0]),
            IntegerStorage::I64(vec![0]),
            IntegerStorage::U8(vec![0]),
            IntegerStorage::U16(vec![0]),
            IntegerStorage::U32(vec![0]),
            IntegerStorage::U64(vec![0]),
        ];

        for storage in prototypes {
            let expected = storage
                .from_same_class_values(vec![storage.cast_f64_assignment(2.5); 4])
                .expect("expected integer storage");
            let prototype = Tensor::new_integer(storage, vec![1, 1]).expect("integer prototype");
            let result = block_on(fill_builtin(
                Value::Num(2.5),
                vec![
                    Value::Num(2.0),
                    Value::from("like"),
                    Value::Tensor(prototype),
                ],
            ))
            .expect("fill");
            let output = test_support::gather(result).expect("gather");
            assert_eq!(output.shape(), &[2, 2]);
            assert_eq!(output.integer_storage(), Some(&expected));
        }

        let scalar = Value::Int(runmat_value::IntValue::U64(u64::MAX));
        let result = block_on(fill_builtin(
            Value::Num(1.0),
            vec![Value::Num(2.0), Value::from("like"), scalar],
        ))
        .expect("fill");
        let output = test_support::gather(result).expect("gather");
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, 1, 1, 1]))
        );
    }

    #[test]
    fn fill_class_strings_create_exact_integer_storage() {
        let cases = [
            ("int8", IntegerStorage::I8(vec![3; 6])),
            ("int16", IntegerStorage::I16(vec![3; 6])),
            ("int32", IntegerStorage::I32(vec![3; 6])),
            ("int64", IntegerStorage::I64(vec![3; 6])),
            ("uint8", IntegerStorage::U8(vec![3; 6])),
            ("uint16", IntegerStorage::U16(vec![3; 6])),
            ("uint32", IntegerStorage::U32(vec![3; 6])),
            ("uint64", IntegerStorage::U64(vec![3; 6])),
        ];

        for (class_name, expected) in cases {
            let result = block_on(fill_builtin(
                Value::Int(runmat_value::IntValue::I32(3)),
                vec![Value::Num(2.0), Value::Num(3.0), Value::from(class_name)],
            ))
            .expect("fill integer class");
            let output = test_support::gather(result).expect("gather");
            assert_eq!(output.shape(), &[2, 3]);
            assert_eq!(output.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn fill_uint64_class_string_preserves_wide_integer_fill_value() {
        let wide = (1_u64 << 53) + 1;
        let result = block_on(fill_builtin(
            Value::Int(runmat_value::IntValue::U64(wide)),
            vec![Value::Num(1.0), Value::Num(3.0), Value::from("uint64")],
        ))
        .expect("fill uint64");
        let output = test_support::gather(result).expect("gather");
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide, wide, wide]))
        );
    }

    #[test]
    fn fill_integer_like_preserves_exact_wide_fill_values() {
        let wide = (1_u64 << 53) + 1;
        let prototype = Tensor::new_integer(IntegerStorage::U64(vec![0]), vec![1, 1])
            .expect("uint64 prototype");
        let result = block_on(fill_builtin(
            Value::Int(runmat_value::IntValue::U64(wide)),
            vec![
                Value::Num(1.0),
                Value::Num(3.0),
                Value::from("like"),
                Value::Tensor(prototype),
            ],
        ))
        .expect("exact uint64 fill");
        let output = test_support::gather(result).expect("gather");
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide, wide, wide]))
        );

        let fill_value = Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1])
            .expect("typed fill scalar");
        let prototype = Tensor::new_integer(IntegerStorage::U32(vec![0]), vec![1, 1])
            .expect("uint32 prototype");
        let result = block_on(fill_builtin(
            Value::Tensor(fill_value),
            vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::Tensor(prototype),
            ],
        ))
        .expect("exact typed tensor fill");
        let output = test_support::gather(result).expect("gather");
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U32(vec![0, 0, 0, 0]))
        );
    }

    #[test]
    fn fill_gpu_integer_like_preserves_exact_wide_values_resident() {
        test_support::with_test_provider(|provider| {
            let wide = (1_u64 << 53) + 1;
            let prototype_values = [0_u64, 1_u64];
            let prototype = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&prototype_values),
                    shape: &[1, 2],
                })
                .expect("upload uint64 prototype");

            let result = block_on(fill_builtin(
                Value::Int(runmat_value::IntValue::U64(wide)),
                vec![
                    Value::Num(2.0),
                    Value::from("like"),
                    Value::GpuTensor(prototype),
                ],
            ))
            .expect("exact gpu uint64 fill");

            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpuArray");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            assert_eq!(handle.shape(), vec![2, 2]);
            assert!(
                runmat_accelerate_api::provider_for_handle(&handle).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
            );
            let downloaded = block_on(provider.download_integer(&handle))
                .expect("download uint64 fill")
                .data;
            let runmat_accelerate_api::HostIntegerDataOwned::U64(values) = downloaded else {
                panic!("expected uint64 storage");
            };
            assert_eq!(values, vec![wide; 4]);
        });
    }

    #[test]
    fn create_array_gpu_like_preserves_exact_paired_complex_integer_storage() {
        test_support::with_test_provider(|provider| {
            let wide = (1_u64 << 53) + 1;
            let prototype = ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![0]),
                    IntegerStorage::U64(vec![0]),
                )
                .expect("paired prototype storage"),
                vec![1, 1],
            )
            .expect("paired prototype");
            let prototype = gpu_helpers::upload_complex_tensor(provider, &prototype)
                .expect("upload paired prototype");
            let fill_value = ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![wide]),
                    IntegerStorage::U64(vec![u64::MAX]),
                )
                .expect("paired fill storage"),
                vec![1, 1],
            )
            .expect("paired fill value");

            let result = block_on(create_array_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::from("Like"),
                    Value::GpuTensor(prototype.clone()),
                    Value::from("FillValue"),
                    Value::ComplexTensor(fill_value),
                ]),
            ))
            .expect("resident paired complex integer createArray");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident paired output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            assert!(
                runmat_accelerate_api::provider_for_handle(&output).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
            );
            let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
                output.clone(),
            )))
            .expect("gather paired output");
            let Value::ComplexTensor(gathered) = gathered else {
                panic!("expected exact paired tensor");
            };
            let storage = gathered.integer_storage().expect("paired integer storage");
            assert_eq!(storage.real(), &IntegerStorage::U64(vec![wide; 4]));
            assert_eq!(storage.imag(), &IntegerStorage::U64(vec![u64::MAX; 4]));
            drop(prototype);
            drop(output);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_like_complex() {
        let result = block_on(fill_builtin(
            Value::Complex(1.0, 2.0),
            vec![Value::Num(2.0), Value::from("complex")],
        ))
        .expect("fill");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert!(t
                    .materialize_f64()
                    .iter()
                    .all(|&(re, im)| (re - 1.0).abs() < 1e-12 && (im - 2.0).abs() < 1e-12));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fill_complex_scalar_reads_typed_integer_storage_exactly() {
        let storage = runmat_value::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![3]),
            IntegerStorage::I16(vec![-2]),
        )
        .expect("complex integer storage");
        let scalar = ComplexTensor::new_integer(storage, vec![1, 1]).expect("typed complex");

        let result = block_on(fill_builtin(
            Value::ComplexTensor(scalar),
            vec![Value::Num(2.0), Value::from("complex")],
        ))
        .expect("fill");

        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert_eq!(t.materialize_f64(), vec![(3.0, -2.0); 4]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_rejects_non_scalar_value() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = block_on(fill_builtin(Value::Tensor(tensor), Vec::new()));
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_complex_value_with_double_class_produces_complex_double() {
        let result = block_on(fill_builtin(
            Value::Complex(1.0, 1.0),
            vec![Value::Num(2.0), Value::Num(2.0), Value::from("double")],
        ))
        .expect("complex double fill");
        let Value::ComplexTensor(tensor) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(tensor.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.materialize_f64(), vec![(1.0, 1.0); 4]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_double_option_generates_real_array() {
        let args = vec![Value::Num(2.0), Value::Num(3.0), Value::from("double")];
        let result = block_on(fill_builtin(Value::Num(1.5), args)).expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 3]);
                assert!(t.materialize_f64().iter().all(|&x| (x - 1.5).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_like_infers_shape_without_dims() {
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(proto)];
        let result = block_on(fill_builtin(Value::Num(4.2), args)).expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert!(t.materialize_f64().iter().all(|&x| (x - 4.2).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_like_logical_prototype() {
        let logical = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let args = vec![Value::from("like"), Value::LogicalArray(logical)];
        let result = block_on(fill_builtin(Value::Num(0.0), args)).expect("fill");
        match result {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.shape, vec![1, 1]);
                assert_eq!(arr.data, vec![0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_rejects_single_precision_option() {
        let result = block_on(fill_builtin(
            Value::Num(1.0),
            vec![Value::Num(2.0), Value::from("single")],
        ));
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_like_logical_conflict_errors() {
        let proto = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Num(1.0),
            Value::from("logical"),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        let result = block_on(fill_builtin(Value::Num(0.0), args));
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_gpu_like_alloc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(fill_builtin(Value::Num(0.5), args)).expect("fill");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape(), vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered
                        .materialize_f64()
                        .iter()
                        .all(|&x| (x - 0.5).abs() < 1e-12));
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn create_array_uses_dimensions_first_and_preserves_integer_fill_class() {
        let result = block_on(create_array_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::U16(2)),
                Value::Int(IntValue::U32(3)),
                Value::from("FillValue"),
                Value::Int(IntValue::U64(9_007_199_254_740_993)),
            ]),
        ))
        .expect("createArray integer FillValue");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993; 6]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn create_array_dims_preserves_rank_zero_and_rank_one_row_policy() {
        let scalar = Tensor::new(vec![-2.0], vec![]).expect("rank-zero scalar");
        assert_eq!(
            block_on(create_array_dims(&Value::Tensor(scalar))).expect("scalar dimensions"),
            Some(vec![0])
        );

        let rank_one = Tensor::new(vec![-2.0, 3.0], vec![2]).expect("rank-one vector");
        assert_eq!(
            block_on(create_array_dims(&Value::Tensor(rank_one))).expect("rank-one dimensions"),
            Some(vec![0, 3])
        );

        let column = Tensor::new(vec![-2.0, 3.0], vec![2, 1]).expect("column vector");
        assert_eq!(
            block_on(create_array_dims(&Value::Tensor(column))).expect("column admission"),
            None
        );
    }

    #[test]
    fn create_array_negative_dimensions_normalize_to_zero() {
        let result = block_on(create_array_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::I16(-3)),
                Value::Num(4.0),
                Value::from("uint8"),
            ]),
        ))
        .expect("createArray negative dimension");
        let Value::Tensor(tensor) = result else {
            panic!("expected empty tensor");
        };
        assert_eq!(tensor.shape(), &[0, 4]);
        assert_eq!(tensor.integer_storage(), Some(&IntegerStorage::U8(vec![])));
    }

    #[test]
    fn create_array_classname_overrides_gpu_fillvalue_class_and_preserves_owner() {
        test_support::with_f32_test_provider(|provider| {
            let single_fill = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[7.0],
                    shape: &[1, 1],
                })
                .expect("upload single FillValue");
            let result = block_on(create_array_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::from("uint8"),
                    Value::from("FillValue"),
                    Value::GpuTensor(single_fill),
                ]),
            ))
            .expect("explicit uint8 with GPU single FillValue");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident uint8 output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(runmat_accelerate_api::IntegerElementType::U8)
            );
            assert!(
                runmat_accelerate_api::provider_for_handle(&handle).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
            );
            let downloaded = block_on(provider.download_integer(&handle))
                .expect("download uint8 output")
                .data;
            let runmat_accelerate_api::HostIntegerDataOwned::U8(values) = downloaded else {
                panic!("expected uint8 storage");
            };
            assert_eq!(values, vec![7; 4]);

            let integer_fill = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&[300]),
                    shape: &[1, 1],
                })
                .expect("upload integer FillValue");
            let result = block_on(create_array_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(1.0),
                    Value::Num(3.0),
                    Value::from("uint8"),
                    Value::from("FillValue"),
                    Value::GpuTensor(integer_fill),
                ]),
            ))
            .expect("explicit uint8 with GPU integer FillValue");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident uint8 output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(runmat_accelerate_api::IntegerElementType::U8)
            );
            assert!(
                runmat_accelerate_api::provider_for_handle(&handle).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
            );
            let downloaded = block_on(provider.download_integer(&handle))
                .expect("download uint8 output")
                .data;
            let runmat_accelerate_api::HostIntegerDataOwned::U8(values) = downloaded else {
                panic!("expected uint8 storage");
            };
            assert_eq!(values, vec![u8::MAX; 3]);
        });
    }

    #[test]
    fn create_array_rejects_classname_with_like() {
        let prototype = Tensor::new_integer(IntegerStorage::I32(vec![0]), vec![1, 1]).unwrap();
        let result = block_on(create_array_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("int32"),
                Value::from("Like"),
                Value::Tensor(prototype),
            ]),
        ));
        assert!(result.is_err());
    }

    #[test]
    fn create_array_complex_fillvalue_dominates_real_double_like() {
        let prototype = Tensor::new(vec![0.0], vec![1, 1]).expect("double prototype");
        let result = block_on(create_array_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("Like"),
                Value::Tensor(prototype),
                Value::from("FillValue"),
                Value::Complex(1.0, 2.0),
            ]),
        ))
        .expect("complex FillValue with real Like");
        let Value::ComplexTensor(tensor) = result else {
            panic!("expected complex double tensor");
        };
        assert_eq!(tensor.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.materialize_f64(), vec![(1.0, 2.0); 4]);
    }

    #[test]
    fn create_array_complex_fillvalue_preserves_real_single_like_precision() {
        let prototype = Tensor::from_f32(vec![0.0], vec![1, 1]).expect("single prototype");
        let result = block_on(create_array_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("Like"),
                Value::Tensor(prototype),
                Value::from("FillValue"),
                Value::Complex(1.25, -2.5),
            ]),
        ))
        .expect("complex FillValue with single Like");
        let Value::ComplexTensor(tensor) = result else {
            panic!("expected complex single tensor");
        };
        assert_eq!(tensor.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(tensor.materialize_f64(), vec![(1.25, -2.5); 4]);
    }

    #[test]
    fn create_array_complex_integer_fillvalue_remains_exact() {
        let wide = 9_007_199_254_740_993_u64;
        let fill_value = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![wide]),
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .and_then(|storage| ComplexTensor::new_integer(storage, vec![1, 1]))
        .expect("complex uint64 FillValue");
        let result = block_on(create_array_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("FillValue"),
                Value::ComplexTensor(fill_value),
            ]),
        ))
        .expect("exact complex integer FillValue");
        let Value::ComplexTensor(tensor) = result else {
            panic!("expected complex integer tensor");
        };
        let storage = tensor.integer_storage().expect("complex integer storage");
        assert_eq!(storage.real(), &IntegerStorage::U64(vec![wide; 4]));
        assert_eq!(storage.imag(), &IntegerStorage::U64(vec![u64::MAX; 4]));
    }

    #[test]
    fn create_array_like_preserves_complex_integer_storage() {
        let prototype =
            IntegerComplexStorage::new(IntegerStorage::U64(vec![1]), IntegerStorage::U64(vec![2]))
                .and_then(|storage| ComplexTensor::new_integer(storage, vec![1, 1]))
                .expect("complex uint64 prototype");
        let result = block_on(create_array_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("Like"),
                Value::ComplexTensor(prototype),
            ]),
        ))
        .expect("createArray complex integer Like");
        let Value::ComplexTensor(tensor) = result else {
            panic!("expected complex integer tensor");
        };
        let storage = tensor.integer_storage().expect("integer storage");
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(storage.real(), &IntegerStorage::U64(vec![0; 4]));
        assert_eq!(storage.imag(), &IntegerStorage::U64(vec![0; 4]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fill_wgpu_like_matches_cpu() {
        let provider =
            match std::panic::catch_unwind(
                || register_wgpu_provider(WgpuProviderOptions::default()),
            ) {
                Ok(Ok(_)) => {
                    if let Some(p) = runmat_accelerate_api::provider() {
                        p
                    } else {
                        tracing::warn!(
                        "skipping fill_wgpu_like_matches_cpu: provider not registered after init"
                    );
                        return;
                    }
                }
                Ok(Err(err)) => {
                    tracing::warn!(
                        "skipping fill_wgpu_like_matches_cpu: wgpu provider unavailable ({err})"
                    );
                    return;
                }
                Err(_) => {
                    tracing::warn!(
                    "skipping fill_wgpu_like_matches_cpu: wgpu provider initialisation panicked"
                );
                    return;
                }
            };
        let prototype = provider
            .fill(&[1, 1], 0.0)
            .expect("prototype allocation")
            .expect("admitted optional constructor");
        let args = vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::from("like"),
            Value::GpuTensor(prototype),
        ];
        let target = 0.75;
        let result = block_on(fill_builtin(Value::Num(target), args)).expect("fill");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert_eq!(gathered.shape(), &[2, 3]);
                for entry in gathered.materialize_f64() {
                    assert!((entry - target).abs() < 1e-12);
                }
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }
}
