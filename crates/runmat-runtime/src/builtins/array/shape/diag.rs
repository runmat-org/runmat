//! MATLAB-compatible `diag` builtin.

use crate::builtins::common::integer_conversion::{
    cast_complex_value, CastError, IntegerClass, IntegerClassExt,
};
use crate::builtins::common::{
    gpu_helpers,
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
    },
    tensor,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntegerStorage, LogicalArray, NumericDType,
    NumericScalar, NumericStorage, Tensor, Value,
};

const BUILTIN_NAME: &str = "diag";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::diag")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "diag",
    op_kind: GpuOpKind::Custom("diag"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("diag_from_vector"),
        ProviderHook::Custom("diag_from_vector_sized"),
        ProviderHook::Custom("diag_extract"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Real and logical gpuArray inputs use provider diag hooks for native vector placement, rectangular placement, vector mode, and matrix diagonal extraction. Complex, character, and conversion-heavy template forms use the host fallback.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::diag", epilogue = crate::builtins::common::spec::FusionEpilogueOperation::Diagonal)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "diag",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "diag uses provider hooks for supported real/logical gpuArray shape operations and otherwise falls back to the runtime host path.",
};

const MESSAGE_ID_INVALID_INPUT: &BuiltinErrorDescriptor = &DIAG_ERROR_INVALID_INPUT;
const MESSAGE_ID_INVALID_OFFSET: &BuiltinErrorDescriptor = &DIAG_ERROR_INVALID_OFFSET;

fn diag_error(error: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone)]
enum OutputTemplate {
    Native,
    Logical,
    Double,
    Integer(IntegerClass),
    Like(Value),
}

#[derive(Clone, Copy)]
enum ClassOverride {
    Logical,
    Double,
    Integer(IntegerClass),
}

struct ParsedDiagArgs {
    offset: isize,
    size_override: Option<(usize, usize)>,
    vector_mode: bool,
    template: OutputTemplate,
}

impl ParsedDiagArgs {
    async fn parse(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Self> {
        let mut offset: Option<isize> = None;
        let mut size_override: Option<(usize, usize)> = None;
        let mut vector_mode = false;
        let mut class_override: Option<ClassOverride> = None;
        let mut like_proto: Option<Value> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = &args[idx];
            if let Some(keyword) =
                crate::builtins::common::arg_tokens::scalar_keyword_of(arg, "diag")?
            {
                match keyword.as_str() {
                    "vector" => {
                        if vector_mode {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: duplicate 'vector' option",
                            ));
                        }
                        vector_mode = true;
                        idx += 1;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: cannot combine 'like' with 'logical'",
                            ));
                        }
                        class_override = Some(ClassOverride::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: cannot combine 'like' with 'double'",
                            ));
                        }
                        class_override = Some(ClassOverride::Double);
                        idx += 1;
                        continue;
                    }
                    "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32"
                    | "uint64" => {
                        if like_proto.is_some() {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                format!("diag: cannot combine 'like' with '{keyword}'"),
                            ));
                        }
                        let target = integer_target_from_keyword(&keyword)
                            .expect("integer class keyword must map to target");
                        class_override = Some(ClassOverride::Integer(target));
                        idx += 1;
                        continue;
                    }
                    "like" => {
                        if class_override.is_some() {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: cannot combine 'like' with class overrides",
                            ));
                        }
                        if like_proto.is_some() {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: duplicate 'like' option",
                            ));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: expected prototype after 'like'",
                            ));
                        };
                        like_proto = Some(proto);
                        idx += 2;
                        continue;
                    }
                    other => {
                        return Err(diag_error(
                            MESSAGE_ID_INVALID_INPUT,
                            format!("diag: unrecognised option '{other}'"),
                        ));
                    }
                }
            }

            if offset.is_none() {
                if let Some(parsed_offset) = try_parse_offset(arg).await? {
                    offset = Some(parsed_offset);
                    idx += 1;
                    continue;
                }
            }

            if size_override.is_none() {
                if let Some(size) = try_parse_size_override(arg).await? {
                    size_override = Some(size);
                    idx += 1;
                    continue;
                }
            }

            return Err(diag_error(
                MESSAGE_ID_INVALID_INPUT,
                format!("diag: unrecognised argument {arg:?}"),
            ));
        }

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else {
            match class_override {
                Some(ClassOverride::Logical) => OutputTemplate::Logical,
                Some(ClassOverride::Double) => OutputTemplate::Double,
                Some(ClassOverride::Integer(target)) => OutputTemplate::Integer(target),
                None => OutputTemplate::Native,
            }
        };

        Ok(Self {
            offset: offset.unwrap_or(0),
            size_override,
            vector_mode,
            template,
        })
    }
}

fn integer_target_from_keyword(keyword: &str) -> Option<IntegerClass> {
    IntegerClass::from_class_name(keyword)
}

async fn try_parse_offset(value: &Value) -> BuiltinResult<Option<isize>> {
    let gathered = gather_if_needed_async(value).await?;
    if !is_scalar_offset_candidate(&gathered) {
        return Ok(None);
    }
    scalar_to_isize(&gathered).map(Some)
}

fn is_scalar_offset_candidate(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Bool(_) => true,
        Value::Tensor(t) => tensor::is_scalar_tensor(t),
        Value::LogicalArray(array) => array.data.len() == 1,
        _ => false,
    }
}

async fn try_parse_size_override(value: &Value) -> BuiltinResult<Option<(usize, usize)>> {
    let Some(dims) = tensor::dims_from_value_async(value)
        .await
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?
    else {
        return Ok(None);
    };

    match dims.as_slice() {
        [] => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: size vector must contain one or two elements",
        )),
        [m] => Ok(Some((*m, *m))),
        [m, n] => Ok(Some((*m, *n))),
        _ => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: size vector must contain one or two elements",
        )),
    }
}

#[runtime_builtin(
    name = "diag",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::shape::diag"
)]
async fn diag_builtin(value: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &DIAG_DESCRIPTOR,
        &DIAG_ERROR_INVALID_INPUT,
        BUILTIN_NAME,
    )?;
    ensure_diag_compatibility(&value, rest.view())?;
    let parsed = ParsedDiagArgs::parse(rest.view()).await?;
    if let Value::GpuTensor(handle) = value {
        return diag_resident(handle, &parsed).await;
    }
    evaluate_diag_host(value, &parsed).await
}

fn ensure_diag_compatibility(
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::ComplexTensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(array.shape.as_slice()),
        Value::GpuTensor(handle) => Some(handle.shape()),
        _ => None,
    };
    ensure_diag_geometry_compatibility(shape, rest)
}
fn ensure_diag_geometry_compatibility(
    shape: Option<&[usize]>,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let mut keywords = Vec::new();
    for argument in rest.iter() {
        if let Some(keyword) =
            crate::builtins::common::arg_tokens::scalar_keyword_of(argument, BUILTIN_NAME)?
        {
            keywords.push(keyword);
        }
    }
    let like_only = rest.len() == 2 && keywords.iter().any(|keyword| keyword == "like");
    if rest.iter().enumerate().any(|(index, arg)| {
        matches!(arg, Value::Bool(_) | Value::LogicalArray(_)) && !(like_only && index == 1)
    }) {
        return Err(diag_error(
            MESSAGE_ID_INVALID_OFFSET,
            "diag: diagonal offset must be a real numeric integer scalar",
        ));
    }
    if keywords.iter().any(|keyword| keyword == "vector") {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DIAG_VECTOR_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if keywords.iter().any(|keyword| keyword == "like") {
        crate::compatibility::ensure_builtin_extension_enabled(&DIAG_LIKE_EXTENSION, BUILTIN_NAME)?;
    }
    if rest
        .iter()
        .enumerate()
        .any(|(index, arg)| matches!(arg, Value::GpuTensor(_)) && !(like_only && index == 1))
    {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: resident diagonal controls are not supported",
        ));
    }
    if keywords.iter().any(|keyword| {
        matches!(
            keyword.as_str(),
            "logical"
                | "double"
                | "int8"
                | "int16"
                | "int32"
                | "int64"
                | "uint8"
                | "uint16"
                | "uint32"
                | "uint64"
        )
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DIAG_CLASS_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let has_size_vector = rest.iter().any(|arg| match arg {
        Value::Tensor(tensor) => tensor.len() == 2,
        _ => false,
    });
    if has_size_vector || (rest.len() > 1 && !like_only) {
        crate::compatibility::ensure_builtin_extension_enabled(&DIAG_SIZE_EXTENSION, BUILTIN_NAME)?;
    }

    if shape.is_some_and(|shape| shape.len() > 2 && shape[2..].iter().all(|dim| *dim == 1)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DIAG_TRAILING_SINGLETON_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn evaluate_diag_host(value: Value, parsed: &ParsedDiagArgs) -> BuiltinResult<Value> {
    let input = coerce_diag_input(value)?;
    let raw = match input {
        DiagInput::Tensor(tensor) => evaluate_tensor(tensor, parsed)?,
        DiagInput::Logical(array) => evaluate_logical(array, parsed)?,
        DiagInput::Complex(tensor) => evaluate_complex(tensor, parsed)?,
        DiagInput::Char(array) => evaluate_char(array, parsed)?,
    };
    apply_output_template(raw, &parsed.template).await
}

async fn diag_resident(handle: GpuTensorHandle, parsed: &ParsedDiagArgs) -> BuiltinResult<Value> {
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: no owning acceleration provider",
        )
    })?;
    let exact_integer = runmat_accelerate_api::handle_integer_type(&handle).is_some();
    let input_precision =
        runmat_accelerate_api::handle_precision(&handle).unwrap_or(provider.precision());
    let complex =
        runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved;
    if !exact_integer && !complex {
        if let Some(output) = try_diag_gpu(handle.clone(), parsed)? {
            return Ok(output);
        }
    }
    let gathered =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await?;
    let host = evaluate_diag_host(gathered, parsed).await?;
    if matches!(parsed.template, OutputTemplate::Double)
        && provider.precision() != ProviderPrecision::F64
    {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: the owning provider cannot restore a true double-precision result",
        ));
    }
    upload_diag_result(provider, &handle, input_precision, host)
}

fn upload_diag_result(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    input_precision: ProviderPrecision,
    value: Value,
) -> BuiltinResult<Value> {
    let mut output = match value {
        Value::GpuTensor(handle) => {
            if handle.shares_allocation_with(input) {
                return Err(gpu_helpers::reject_gpu_output(
                    handle,
                    diag_error(
                        MESSAGE_ID_INVALID_INPUT,
                        "diag: restored result aliases borrowed input",
                    ),
                ));
            }
            return Ok(Value::GpuTensor(handle));
        }
        Value::Tensor(tensor) => {
            gpu_helpers::upload_tensor_protecting(provider, &tensor, &[input])?
        }
        Value::LogicalArray(array) => {
            upload_diag_logical(provider, &array.data, &array.shape, input)?
        }
        Value::Bool(value) => upload_diag_logical(provider, &[u8::from(value)], &[1, 1], input)?,
        Value::Int(value) => {
            let host = Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])?;
            gpu_helpers::upload_tensor_protecting(provider, &host, &[input])?
        }
        Value::Num(value) => {
            let host = Tensor::new(vec![value], vec![1, 1])?;
            gpu_helpers::upload_tensor_protecting(provider, &host, &[input])?
        }
        Value::ComplexTensor(tensor) => {
            gpu_helpers::upload_complex_tensor_protecting(provider, &tensor, &[input])?
        }
        Value::Complex(re, im) => {
            let host = ComplexTensor::from_f64_values_with_dtype(
                vec![(re, im)],
                vec![1, 1],
                provider_precision_to_dtype(input_precision),
            )?;
            gpu_helpers::upload_complex_tensor_protecting(provider, &host, &[input])?
        }
        other => {
            return Err(diag_error(
                MESSAGE_ID_INVALID_INPUT,
                format!("diag: cannot restore resident result {other:?}"),
            ))
        }
    };
    output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
    gpu_helpers::resident_gpu_value(output)
}

fn upload_diag_logical(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    data: &[u8],
    shape: &[usize],
    input: &GpuTensorHandle,
) -> BuiltinResult<GpuTensorHandle> {
    runmat_accelerate_api::upload_logical_protecting(
        provider,
        &runmat_accelerate_api::HostLogicalTensorView { data, shape },
        &[input],
    )
    .map_err(|cause| {
        crate::build_runtime_error(format!("diag: logical upload failed: {cause}"))
            .with_builtin("diag")
            .with_source(cause)
            .build()
    })
}

#[derive(Clone, Copy)]
struct GpuDiagTemplate {
    logical: bool,
    precision: ProviderPrecision,
}

fn try_diag_gpu(handle: GpuTensorHandle, parsed: &ParsedDiagArgs) -> BuiltinResult<Option<Value>> {
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        return Ok(None);
    }

    let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle) else {
        return Ok(None);
    };

    let input_logical = runmat_accelerate_api::handle_is_logical(&handle);
    let input_precision =
        runmat_accelerate_api::handle_precision(&handle).unwrap_or(provider.precision());
    let Some(template) =
        gpu_diag_template(parsed, input_logical, input_precision, provider.precision())
    else {
        return Ok(None);
    };

    if template.logical != input_logical
        || (!input_logical && template.precision != input_precision)
    {
        return Ok(None);
    }
    let (rows, cols) = matrix_dims(handle.shape())?;
    let is_vector = rows == 1 || cols == 1;
    validate_vector_mode(is_vector, parsed)?;

    let expected_shape = if parsed.vector_mode {
        vec![rows.max(cols), 1]
    } else if is_vector {
        let (out_rows, out_cols) = vector_output_dims(rows.max(cols), parsed)?;
        vec![out_rows, out_cols]
    } else {
        vec![diagonal_length(rows, cols, parsed.offset), 1]
    };

    let result = if parsed.vector_mode {
        let len = vector_length(rows, cols);
        provider.reshape(&handle, &[len, 1])
    } else if is_vector {
        let len = vector_length(rows, cols);
        let (out_rows, out_cols) = vector_output_dims(len, parsed)?;
        if parsed.size_override.is_some() {
            provider.diag_from_vector_sized(&handle, parsed.offset, out_rows, out_cols)
        } else {
            provider.diag_from_vector(&handle, parsed.offset)
        }
    } else {
        if parsed.size_override.is_some() {
            return Ok(None);
        }
        provider.diag_extract(&handle, parsed.offset)
    };

    match result {
        Ok(mut output)
            if native_diag_output_matches(
                &handle,
                &output,
                provider,
                &expected_shape,
                template,
            ) && (parsed.vector_mode || !output.shares_allocation_with(&handle)) =>
        {
            output = gpu_helpers::propagate_owned_output_provenance(output, [&handle])?;
            Ok(Some(Value::GpuTensor(output)))
        }
        Ok(output) => Err(gpu_helpers::reject_gpu_output(
            output,
            diag_error(
                MESSAGE_ID_INVALID_INPUT,
                "diag: invalid provider diagonal output",
            ),
        )),
        Err(cause)
            if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(&cause) =>
        {
            Ok(None)
        }
        Err(cause) => Err(crate::build_runtime_error(format!(
            "diag: provider diagonal operation failed: {cause}"
        ))
        .with_builtin("diag")
        .with_source(runmat_accelerate_api::ProviderExecutionError::from(cause))
        .build()),
    }
}

fn diagonal_length(rows: usize, cols: usize, offset: isize) -> usize {
    if offset >= 0 {
        cols.checked_sub(offset as usize)
            .map_or(0, |remaining| rows.min(remaining))
    } else {
        rows.checked_sub(offset.unsigned_abs())
            .map_or(0, |remaining| remaining.min(cols))
    }
}

fn native_diag_output_matches(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
    template: GpuDiagTemplate,
) -> bool {
    output.device_id() == input.device_id()
        && output.shape() == expected_shape
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && output.view().is_logical() == template.logical
        && (template.logical
            || (runmat_accelerate_api::handle_integer_type(output).is_none()
                && runmat_accelerate_api::handle_precision(output) == Some(template.precision)))
}

fn gpu_diag_template(
    parsed: &ParsedDiagArgs,
    input_logical: bool,
    input_precision: ProviderPrecision,
    provider_precision: ProviderPrecision,
) -> Option<GpuDiagTemplate> {
    match &parsed.template {
        OutputTemplate::Native => Some(GpuDiagTemplate {
            logical: input_logical,
            precision: input_precision,
        }),
        OutputTemplate::Double => {
            if provider_precision != ProviderPrecision::F64 {
                return None;
            }
            Some(GpuDiagTemplate {
                logical: false,
                precision: ProviderPrecision::F64,
            })
        }
        OutputTemplate::Logical => {
            if !input_logical {
                return None;
            }
            Some(GpuDiagTemplate {
                logical: true,
                precision: input_precision,
            })
        }
        // A resident `like` prototype owns both placement and class semantics.
        // Route through the validated owner-resolved fallback instead of invoking
        // an input-owner hook that could return on the wrong provider or device.
        OutputTemplate::Like(Value::GpuTensor(_)) => None,
        OutputTemplate::Like(_) => None,
        OutputTemplate::Integer(_) => None,
    }
}

enum DiagInput {
    Tensor(Tensor),
    Logical(LogicalArray),
    Complex(ComplexTensor),
    Char(CharArray),
}

fn coerce_diag_input(value: Value) -> BuiltinResult<DiagInput> {
    match value {
        Value::Tensor(tensor) => Ok(DiagInput::Tensor(tensor)),
        Value::LogicalArray(array) => Ok(DiagInput::Logical(array)),
        Value::ComplexTensor(tensor) => Ok(DiagInput::Complex(tensor)),
        Value::CharArray(array) => Ok(DiagInput::Char(array)),
        Value::Num(n) => Ok(DiagInput::Tensor(
            Tensor::new(vec![n], vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?,
        )),
        Value::Int(i) => Ok(DiagInput::Tensor(
            Tensor::new_integer(IntegerStorage::from_scalar(i), vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?,
        )),
        Value::Bool(flag) => Ok(DiagInput::Logical(
            LogicalArray::new(vec![if flag { 1 } else { 0 }], vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?,
        )),
        Value::Complex(re, im) => Ok(DiagInput::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?,
        )),
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!("diag: unsupported input {other:?}"),
        )),
    }
}

fn evaluate_tensor(tensor: Tensor, args: &ParsedDiagArgs) -> BuiltinResult<Value> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?;
    let (storage, shape) = evaluate_numeric_storage(storage, &shape, args)?;
    Tensor::from_numeric_storage(storage, shape)
        .map(Value::Tensor)
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
}

fn evaluate_numeric_storage(
    storage: NumericStorage,
    shape: &[usize],
    args: &ParsedDiagArgs,
) -> BuiltinResult<(NumericStorage, Vec<usize>)> {
    macro_rules! evaluate_storage {
        ($values:expr, $variant:ident, $zero:expr) => {{
            let (values, shape) = evaluate_column_major_diag(&$values, shape, args, $zero)?;
            Ok((NumericStorage::$variant(values), shape))
        }};
    }

    match storage {
        NumericStorage::F64(values) => evaluate_storage!(values, F64, 0.0f64),
        NumericStorage::F32(values) => evaluate_storage!(values, F32, 0.0f32),
        NumericStorage::I8(values) => evaluate_storage!(values, I8, 0i8),
        NumericStorage::I16(values) => evaluate_storage!(values, I16, 0i16),
        NumericStorage::I32(values) => evaluate_storage!(values, I32, 0i32),
        NumericStorage::I64(values) => evaluate_storage!(values, I64, 0i64),
        NumericStorage::U8(values) => evaluate_storage!(values, U8, 0u8),
        NumericStorage::U16(values) => evaluate_storage!(values, U16, 0u16),
        NumericStorage::U32(values) => evaluate_storage!(values, U32, 0u32),
        NumericStorage::U64(values) => evaluate_storage!(values, U64, 0u64),
    }
}

fn evaluate_logical(array: LogicalArray, args: &ParsedDiagArgs) -> BuiltinResult<Value> {
    let (data, shape) = evaluate_column_major_diag(&array.data, &array.shape, args, 0u8)?;
    LogicalArray::new(data, shape)
        .map(Value::LogicalArray)
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
}

fn evaluate_complex(tensor: ComplexTensor, args: &ParsedDiagArgs) -> BuiltinResult<Value> {
    let (storage, input_shape) = tensor.into_complex_storage_and_shape();
    let (storage, shape) = match storage {
        ComplexStorage::F64(values) => {
            let (values, shape) =
                evaluate_column_major_diag(&values, &input_shape, args, (0.0f64, 0.0f64).into())?;
            (ComplexStorage::F64(values.into()), shape)
        }
        ComplexStorage::F32(values) => {
            let (values, shape) =
                evaluate_column_major_diag(&values, &input_shape, args, (0.0f32, 0.0f32).into())?;
            (ComplexStorage::F32(values.into()), shape)
        }
        ComplexStorage::Integer(storage) => {
            let zero = storage
                .real()
                .zeros_like(1)
                .value_at(0)
                .expect("one typed integer zero");
            let (_, shape) = evaluate_column_major_diag(
                &storage.real().exact_values(),
                &input_shape,
                args,
                zero.clone(),
            )?;
            let storage = storage
                .reorder(|values| {
                    evaluate_column_major_diag(values, &input_shape, args, zero.clone())
                        .map(|(values, _)| values)
                        .map_err(|e| e.to_string())
                })
                .map_err(|e| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {e}")))?;
            (ComplexStorage::Integer(storage), shape)
        }
    };
    ComplexTensor::from_complex_storage(storage, shape)
        .map(Value::ComplexTensor)
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
}

fn evaluate_char(array: CharArray, args: &ParsedDiagArgs) -> BuiltinResult<Value> {
    let rows = array.rows;
    let cols = array.cols;
    let is_vector = rows == 1 || cols == 1;

    validate_vector_mode(is_vector, args)?;

    if args.vector_mode {
        let len = vector_length(rows, cols);
        let data = vector_copy(array.data(), len);
        let out = CharArray::from_code_units(data, len, 1)
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?;
        return Ok(Value::CharArray(out));
    }

    if is_vector {
        let len = vector_length(rows, cols);
        let (out_rows, out_cols) = vector_output_dims(len, args)?;
        let data = diag_matrix_from_vector_row_major(
            array.data(),
            len,
            args.offset,
            out_rows,
            out_cols,
            runmat_value::Utf16CodeUnit::from_raw(0x20),
        )?;
        let out = CharArray::from_code_units(data, out_rows, out_cols)
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?;
        return Ok(Value::CharArray(out));
    }

    if args.size_override.is_some() {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: size overrides require vector inputs",
        ));
    }

    let diag = diag_vector_from_matrix_row_major(array.data(), rows, cols, args.offset);
    let len = diag.len();
    let out = CharArray::from_code_units(diag, len, 1)
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?;
    Ok(Value::CharArray(out))
}

fn evaluate_column_major_diag<T: Clone>(
    data: &[T],
    shape: &[usize],
    args: &ParsedDiagArgs,
    zero: T,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    let (rows, cols) = matrix_dims(shape)?;
    let is_vector = rows == 1 || cols == 1;

    validate_vector_mode(is_vector, args)?;

    if args.vector_mode {
        let len = vector_length(rows, cols);
        return Ok((vector_copy(data, len), vec![len, 1]));
    }

    if is_vector {
        let len = vector_length(rows, cols);
        let (out_rows, out_cols) = vector_output_dims(len, args)?;
        let out =
            diag_matrix_from_vector_col_major(data, len, args.offset, out_rows, out_cols, zero)?;
        return Ok((out, vec![out_rows, out_cols]));
    }

    if args.size_override.is_some() {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: size overrides require vector inputs",
        ));
    }

    let diag = diag_vector_from_matrix_col_major(data, rows, cols, args.offset);
    let len = diag.len();
    Ok((diag, vec![len, 1]))
}

fn validate_vector_mode(is_vector: bool, args: &ParsedDiagArgs) -> BuiltinResult<()> {
    if !args.vector_mode {
        return Ok(());
    }
    if !is_vector {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: 'vector' requires a vector input",
        ));
    }
    if args.offset != 0 || args.size_override.is_some() {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: 'vector' cannot be combined with offsets or size overrides",
        ));
    }
    Ok(())
}

fn vector_output_dims(len: usize, args: &ParsedDiagArgs) -> BuiltinResult<(usize, usize)> {
    if let Some((rows, cols)) = args.size_override {
        return Ok((rows, cols));
    }
    let shift = args.offset.unsigned_abs();
    let size = len.checked_add(shift).ok_or_else(|| {
        diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: result dimensions exceed supported limits",
        )
    })?;
    Ok((size, size))
}

fn vector_copy<T: Clone>(data: &[T], len: usize) -> Vec<T> {
    data.iter().take(len).cloned().collect()
}

fn matrix_dims(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    if shape.len() > 2 && shape[2..].iter().any(|dim| *dim != 1) {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: only vectors and matrices are supported",
        ));
    }
    let rows = *shape.first().unwrap_or(&1);
    let cols = *shape.get(1).unwrap_or(&1);
    Ok((rows, cols))
}

fn allocate_out<T: Clone>(rows: usize, cols: usize, value: T) -> BuiltinResult<Vec<T>> {
    let count = rows.checked_mul(cols).ok_or_else(|| {
        diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: result dimensions exceed supported limits",
        )
    })?;
    Ok(vec![value; count])
}

fn diag_matrix_from_vector_col_major<T: Clone>(
    data: &[T],
    len: usize,
    offset: isize,
    rows: usize,
    cols: usize,
    zero: T,
) -> BuiltinResult<Vec<T>> {
    let mut out = allocate_out(rows, cols, zero)?;
    let shift = offset.unsigned_abs();
    let (start_row, start_col) = if offset >= 0 {
        (0usize, shift)
    } else {
        (shift, 0usize)
    };
    if start_row >= rows || start_col >= cols {
        return Ok(out);
    }

    let max_len = (rows - start_row)
        .min(cols - start_col)
        .min(len)
        .min(data.len());
    for idx in 0..max_len {
        let row = start_row + idx;
        let col = start_col + idx;
        out[row + col * rows] = data[idx].clone();
    }
    Ok(out)
}

fn diag_matrix_from_vector_row_major<T: Clone>(
    data: &[T],
    len: usize,
    offset: isize,
    rows: usize,
    cols: usize,
    zero: T,
) -> BuiltinResult<Vec<T>> {
    let mut out = allocate_out(rows, cols, zero)?;
    let shift = offset.unsigned_abs();
    let (start_row, start_col) = if offset >= 0 {
        (0usize, shift)
    } else {
        (shift, 0usize)
    };
    if start_row >= rows || start_col >= cols {
        return Ok(out);
    }

    let max_len = (rows - start_row)
        .min(cols - start_col)
        .min(len)
        .min(data.len());
    for idx in 0..max_len {
        let row = start_row + idx;
        let col = start_col + idx;
        out[row * cols + col] = data[idx].clone();
    }
    Ok(out)
}

fn diag_vector_from_matrix_col_major<T: Clone>(
    data: &[T],
    rows: usize,
    cols: usize,
    offset: isize,
) -> Vec<T> {
    let shift = offset.unsigned_abs();
    let (start_row, start_col) = if offset >= 0 {
        (0usize, shift)
    } else {
        (shift, 0usize)
    };
    if start_row >= rows || start_col >= cols {
        return Vec::new();
    }
    let max_len = (rows - start_row).min(cols - start_col);
    let mut out = Vec::with_capacity(max_len);
    for idx in 0..max_len {
        let row = start_row + idx;
        let col = start_col + idx;
        out.push(data[row + col * rows].clone());
    }
    out
}

fn diag_vector_from_matrix_row_major<T: Copy>(
    data: &[T],
    rows: usize,
    cols: usize,
    offset: isize,
) -> Vec<T> {
    let shift = offset.unsigned_abs();
    let (start_row, start_col) = if offset >= 0 {
        (0usize, shift)
    } else {
        (shift, 0usize)
    };
    if start_row >= rows || start_col >= cols {
        return Vec::new();
    }
    let max_len = (rows - start_row).min(cols - start_col);
    let mut out = Vec::with_capacity(max_len);
    for idx in 0..max_len {
        let row = start_row + idx;
        let col = start_col + idx;
        out.push(data[row * cols + col]);
    }
    out
}

fn scalar_to_isize(value: &Value) -> BuiltinResult<isize> {
    match value {
        Value::Int(i) => i.try_to_isize().ok_or_else(|| {
            diag_error(
                MESSAGE_ID_INVALID_OFFSET,
                "diag: diagonal offset is outside the supported range",
            )
        }),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(diag_error(
                    MESSAGE_ID_INVALID_OFFSET,
                    "diag: diagonal offset must be finite",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(diag_error(
                    MESSAGE_ID_INVALID_OFFSET,
                    "diag: diagonal offset must be an integer",
                ));
            }
            if rounded < isize::MIN as f64
                || rounded > isize::MAX as f64
                || (isize::BITS == 64 && rounded == isize::MAX as f64)
            {
                return Err(diag_error(
                    MESSAGE_ID_INVALID_OFFSET,
                    "diag: diagonal offset is outside the supported range",
                ));
            }
            Ok(rounded as isize)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let value = storage.value_at(0).ok_or_else(|| {
                    diag_error(
                        MESSAGE_ID_INVALID_OFFSET,
                        "diag: integer offset storage length mismatch",
                    )
                })?;
                return scalar_to_isize(&Value::Int(value));
            }
            scalar_to_isize(&Value::Num(tensor::tensor_value_f64(t, 0)))
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(if array.data[0] != 0 { 1 } else { 0 })
        }
        Value::Bool(flag) => Ok(if *flag { 1 } else { 0 }),
        other => Err(diag_error(
            MESSAGE_ID_INVALID_OFFSET,
            format!("diag: diagonal offset must be a numeric scalar, got {other:?}"),
        )),
    }
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Native => Ok(value),
        OutputTemplate::Logical => logical_array_from_value(value).map(Value::LogicalArray),
        OutputTemplate::Double => double_value_from_value(value),
        OutputTemplate::Integer(target) => integer_value_from_value(value, *target),
        OutputTemplate::Like(proto) => apply_like_template(value, proto).await,
    }
}

async fn apply_like_template(value: Value, prototype: &Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = prototype {
        return apply_gpu_like_template(value, handle).await;
    }

    let gathered_proto = gather_if_needed_async(prototype).await?;
    match gathered_proto {
        Value::LogicalArray(_) | Value::Bool(_) => {
            logical_array_from_value(value).map(Value::LogicalArray)
        }
        Value::ComplexTensor(proto_tensor) => {
            if let Some(storage) = proto_tensor.integer_storage() {
                let target = storage.real().integer_class();
                integer_value_from_value(value, target)
            } else {
                complex_tensor_from_value_with_dtype(value, proto_tensor.numeric_dtype())
                    .map(Value::ComplexTensor)
            }
        }
        Value::Complex(_, _) => complex_tensor_from_value_with_dtype(value, NumericDType::F64)
            .map(Value::ComplexTensor),
        Value::Tensor(proto_tensor) => {
            let dtype = proto_tensor.numeric_dtype();
            if let Some(target) = dtype.integer_class() {
                integer_value_from_value(value, target)
            } else {
                floating_value_from_value(value, dtype)
            }
        }
        Value::Num(_) => double_value_from_value(value),
        Value::Int(prototype) => {
            integer_value_from_value(value, prototype.integer_class())
        }
        Value::CharArray(_) => tensor_from_value(value).map(Value::Tensor),
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!(
                "diag: unsupported 'like' prototype {other:?}; expected numeric, logical, complex, or gpuArray"
            ),
        )),
    }
}

fn double_value_from_value(value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        return complex_tensor_from_value_with_dtype(value, NumericDType::F64)
            .map(Value::ComplexTensor);
    }
    let tensor = tensor_from_value(value)?;
    cast_tensor_dtype(tensor, NumericDType::F64).map(Value::Tensor)
}

fn floating_value_from_value(value: Value, dtype: NumericDType) -> BuiltinResult<Value> {
    if matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        return complex_tensor_from_value_with_dtype(value, dtype).map(Value::ComplexTensor);
    }
    let tensor = tensor_from_value(value)?;
    cast_tensor_dtype(tensor, dtype).map(Value::Tensor)
}

async fn apply_gpu_like_template(
    value: Value,
    prototype: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(prototype).ok_or_else(|| {
        diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: gpuArray 'like' prototype has no owning acceleration provider",
        )
    })?;
    let logical_target = runmat_accelerate_api::handle_is_logical(prototype);
    let precision =
        runmat_accelerate_api::handle_precision(prototype).unwrap_or(provider.precision());
    let integer_target =
        runmat_accelerate_api::handle_integer_type(prototype).map(IntegerClass::from);
    let complex_target =
        runmat_accelerate_api::handle_storage(prototype) == GpuTensorStorage::ComplexInterleaved;

    let converted = if logical_target {
        Value::LogicalArray(logical_array_from_value(value)?)
    } else if let Some(target) = integer_target {
        integer_value_from_value(value, target)?
    } else if complex_target {
        Value::ComplexTensor(complex_tensor_from_value_with_dtype(
            value,
            provider_precision_to_dtype(precision),
        )?)
    } else {
        floating_value_from_value(value, provider_precision_to_dtype(precision))?
    };

    upload_diag_result(provider, prototype, precision, converted)
}

fn provider_precision_to_dtype(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

fn cast_tensor_dtype(tensor: Tensor, dtype: NumericDType) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == dtype {
        return Ok(tensor);
    }
    Ok(tensor::coerce_tensor_dtype(tensor, dtype))
}

fn integer_value_from_value(value: Value, target: IntegerClass) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => target
            .cast_tensor(tensor)
            .map(Value::Tensor)
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array)
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?;
            target
                .cast_tensor(tensor)
                .map(Value::Tensor)
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            target
                .cast_tensor(tensor)
                .map(Value::Tensor)
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            cast_complex_value(value, target).map_err(|err| diag_cast_error("diag", err))
        }
        Value::Num(n) => {
            Tensor::new_integer(target.storage(vec![target.cast_scalar(n)]), vec![1, 1])
                .map(Value::Tensor)
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::Int(i) => Tensor::new_integer(target.storage(vec![target.cast_int(&i)]), vec![1, 1])
            .map(Value::Tensor)
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Bool(flag) => Tensor::new_integer(
            target.storage(vec![target.cast_scalar(if flag { 1.0 } else { 0.0 })]),
            vec![1, 1],
        )
        .map(Value::Tensor)
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!(
                "diag: cannot convert {other:?} to {} output",
                target.class_name()
            ),
        )),
    }
}

fn diag_cast_error(context: &str, err: CastError) -> RuntimeError {
    match err {
        CastError::Unsupported(kind) => diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!("{context}: cannot convert {kind} to integer output"),
        ),
        CastError::Internal(message) => {
            diag_error(MESSAGE_ID_INVALID_INPUT, format!("{context}: {message}"))
        }
        CastError::Runtime(source) => {
            let mut contextual = diag_error(
                MESSAGE_ID_INVALID_INPUT,
                format!("{context}: {}", source.message()),
            );
            contextual.source = Some(source);
            contextual
        }
    }
}

fn logical_array_from_value(value: Value) -> BuiltinResult<LogicalArray> {
    match value {
        Value::LogicalArray(array) => Ok(array),
        Value::Tensor(tensor) => {
            let data = (0..tensor.len())
                .map(|index| {
                    tensor
                        .numeric_value_at(index)
                        .map(|value| if numeric_scalar_is_zero(value) { 0 } else { 1 })
                        .ok_or_else(|| {
                            diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: numeric storage length mismatch",
                            )
                        })
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            LogicalArray::new(data, tensor.shape().to_vec())
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::ComplexTensor(tensor) => {
            let data: Vec<u8> = if let Some(storage) = &tensor.integer_storage() {
                storage
                    .real()
                    .exact_values()
                    .into_iter()
                    .zip(storage.imag().exact_values())
                    .map(|(re, im)| if re.is_zero() && im.is_zero() { 0 } else { 1 })
                    .collect()
            } else {
                tensor
                    .materialize_f64()
                    .iter()
                    .map(|(re, im)| if *re != 0.0 || *im != 0.0 { 1 } else { 0 })
                    .collect()
            };
            LogicalArray::new(data, tensor.shape().to_vec())
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::CharArray(chars) => {
            let data: Vec<u8> = chars
                .iter_column_major()
                .map(|ch| if ch.raw() != 0 { 1 } else { 0 })
                .collect();
            LogicalArray::new(data, vec![chars.rows, chars.cols])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::Num(n) => LogicalArray::new(vec![if n != 0.0 { 1 } else { 0 }], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Int(i) => LogicalArray::new(vec![if !i.is_zero() { 1 } else { 0 }], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Bool(flag) => LogicalArray::new(vec![if flag { 1 } else { 0 }], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Complex(re, im) => {
            let logical = if re != 0.0 || im != 0.0 { 1 } else { 0 };
            LogicalArray::new(vec![logical], vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!("diag: cannot convert {other:?} to logical output"),
        )),
    }
}

fn numeric_scalar_is_zero(value: NumericScalar) -> bool {
    match value {
        NumericScalar::F64(value) => value == 0.0,
        NumericScalar::F32(value) => value == 0.0,
        NumericScalar::I8(value) => value == 0,
        NumericScalar::I16(value) => value == 0,
        NumericScalar::I32(value) => value == 0,
        NumericScalar::I64(value) => value == 0,
        NumericScalar::U8(value) => value == 0,
        NumericScalar::U16(value) => value == 0,
        NumericScalar::U32(value) => value == 0,
        NumericScalar::U64(value) => value == 0,
    }
}

fn tensor_from_value(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::LogicalArray(array) => tensor::logical_to_tensor(&array)
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Bool(flag) => Tensor::new(vec![if flag { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::CharArray(chars) => char_array_to_tensor(&chars),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: cannot convert complex output to 'double'",
        )),
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!("diag: cannot convert {other:?} to double output"),
        )),
    }
}

fn complex_tensor_from_value(value: Value) -> BuiltinResult<ComplexTensor> {
    match value {
        Value::ComplexTensor(tensor) => Ok(tensor),
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Tensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let data: Vec<(f64, f64)> = tensor::tensor_into_values_f64(tensor)
                .into_iter()
                .map(|re| (re, 0.0))
                .collect();
            ComplexTensor::new(data, shape)
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::LogicalArray(array) => {
            let data: Vec<(f64, f64)> = array
                .data
                .iter()
                .map(|value| if *value != 0 { (1.0, 0.0) } else { (0.0, 0.0) })
                .collect();
            ComplexTensor::new(data, array.shape)
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::CharArray(chars) => {
            let data: Vec<(f64, f64)> = chars
                .iter_column_major()
                .map(|ch| (f64::from(ch.raw()), 0.0))
                .collect();
            ComplexTensor::new(data, vec![chars.rows, chars.cols])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::Num(n) => ComplexTensor::new(vec![(n, 0.0)], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Int(i) => ComplexTensor::new(vec![(i.to_f64(), 0.0)], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Bool(flag) => {
            let re = if flag { 1.0 } else { 0.0 };
            ComplexTensor::new(vec![(re, 0.0)], vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!("diag: cannot convert {other:?} to complex output"),
        )),
    }
}

fn complex_tensor_from_value_with_dtype(
    value: Value,
    dtype: NumericDType,
) -> BuiltinResult<ComplexTensor> {
    if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: complex floating conversion requires single or double precision",
        ));
    }
    ensure_integer_exact_for_complex_float(&value, dtype)?;
    let tensor = complex_tensor_from_value(value)?;
    ComplexTensor::from_f64_values_with_dtype(
        tensor.materialize_f64(),
        tensor.shape().to_vec(),
        dtype,
    )
    .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
}

fn ensure_integer_exact_for_complex_float(value: &Value, dtype: NumericDType) -> BuiltinResult<()> {
    let exact = |integer: &runmat_value::IntValue| integer_exact_for_float(integer, dtype);
    let valid = match value {
        Value::Int(integer) => exact(integer),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_none_or(|storage| {
            storage.real().exact_values().iter().all(exact)
                && storage.imag().exact_values().iter().all(exact)
        }),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!(
                "diag: integer input must be exactly representable as {} for complex conversion",
                if dtype == NumericDType::F32 {
                    "single"
                } else {
                    "double"
                }
            ),
        ))
    }
}

fn integer_exact_for_float(value: &runmat_value::IntValue, dtype: NumericDType) -> bool {
    if !crate::builtins::common::validation::integer_is_exact_f64(value) {
        return false;
    }
    if dtype == NumericDType::F64 {
        return true;
    }
    let value = value.to_f64();
    f64::from(value as f32) == value
}

fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars
        .iter_column_major()
        .map(|ch| f64::from(ch.raw()))
        .collect();
    Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
}

#[path = "diag/geometry.rs"]
mod geometry;
pub(in crate::builtins) use geometry::{diagonal_geometry, DiagonalGeometry};

fn vector_length(rows: usize, columns: usize) -> usize {
    if rows == 0 || columns == 0 {
        0
    } else {
        rows.max(columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntValue, IntegerComplexStorage};

    fn run_diag(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(diag_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    fn size_vector(rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(vec![rows as f64, cols as f64], vec![1, 2]).unwrap())
    }

    #[test]
    fn diag_offset_parser_preserves_signed_values_and_rejects_unrepresentable_uint64() {
        assert_eq!(
            scalar_to_isize(&Value::Int(IntValue::I64(-1))).expect("signed offset"),
            -1
        );
        let typed_offset =
            Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("typed offset");
        assert_eq!(
            scalar_to_isize(&Value::Tensor(typed_offset)).expect("typed tensor offset"),
            -1
        );
        let err = scalar_to_isize(&Value::Int(IntValue::U64(u64::MAX)))
            .expect_err("unrepresentable typed offset must not saturate");
        assert_eq!(err.identifier(), MESSAGE_ID_INVALID_OFFSET.identifier);
        let typed_err = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("typed offset");
        let err = scalar_to_isize(&Value::Tensor(typed_err))
            .expect_err("unrepresentable typed tensor offset must not saturate");
        assert_eq!(err.identifier(), MESSAGE_ID_INVALID_OFFSET.identifier);
    }

    #[test]
    fn diag_offset_candidate_reads_typed_integer_storage_without_mirror() {
        let typed_offset =
            Tensor::new_integer(IntegerStorage::I16(vec![-2]), vec![1, 1]).expect("typed offset");

        let offset = block_on(try_parse_offset(&Value::Tensor(typed_offset)))
            .expect("offset parse")
            .expect("typed integer scalar should be an offset candidate");
        assert_eq!(offset, -2);
    }

    #[test]
    fn diag_vector_mode_returns_column_vector() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap());
        let out = run_diag(value, vec![Value::from("vector")]).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape(), &[3, 1]);
        assert_eq!(tensor.materialize_f64(), vec![10.0, 20.0, 30.0]);
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).unwrap());
        let Value::Tensor(out) = run_diag(empty.clone(), Vec::new()).unwrap() else {
            panic!("empty diagonal matrix")
        };
        assert_eq!(out.shape(), &[0, 0]);
        let Value::Tensor(out) = run_diag(empty, vec![Value::from("vector")]).unwrap() else {
            panic!("empty vector mode")
        };
        assert_eq!(out.shape(), &[0, 1]);
    }

    #[test]
    fn diag_vector_size_override_rectangular() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let out = run_diag(value, vec![size_vector(2, 4)]).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape(), &[2, 4]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn diag_vector_offset_and_size_override() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let out = run_diag(value, vec![Value::Num(1.0), size_vector(3, 4)]).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn diag_extracts_subdiagonal() {
        let matrix = Tensor::new(
            vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let out = run_diag(Value::Tensor(matrix), vec![Value::Num(-1.0)]).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape(), &[2, 1]);
        assert_eq!(tensor.materialize_f64(), vec![4.0, 8.0]);
    }

    #[test]
    fn diag_preserves_all_exact_integer_classes_for_construction_and_extraction() {
        let storages = [
            IntegerStorage::I8(vec![-2, 7, 9]),
            IntegerStorage::I16(vec![-300, 400, 900]),
            IntegerStorage::I32(vec![i32::MIN, 0, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, 0, i64::MAX]),
            IntegerStorage::U8(vec![0, 7, u8::MAX]),
            IntegerStorage::U16(vec![0, 700, u16::MAX]),
            IntegerStorage::U32(vec![0, 9_007_199, u32::MAX]),
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in storages {
            let values = storage.exact_values();
            let zero = storage.zeros_like(1).value_at(0).expect("integer zero");
            let expected = storage
                .from_exact_values_like(vec![
                    values[0].clone(),
                    zero.clone(),
                    zero.clone(),
                    zero.clone(),
                    values[1].clone(),
                    zero.clone(),
                    zero.clone(),
                    zero,
                    values[2].clone(),
                ])
                .expect("expected diagonal storage");
            let input = Tensor::new_integer(storage.clone(), vec![1, 3]).expect("integer vector");
            let Value::Tensor(matrix) = run_diag(Value::Tensor(input), Vec::new()).expect("diag")
            else {
                panic!("expected exact integer matrix");
            };
            assert_eq!(matrix.shape(), &[3, 3]);
            assert_eq!(matrix.integer_storage(), Some(&expected));

            let Value::Tensor(extracted) =
                run_diag(Value::Tensor(matrix), Vec::new()).expect("diag extract")
            else {
                panic!("expected exact integer vector");
            };
            assert_eq!(extracted.shape(), &[3, 1]);
            assert_eq!(extracted.integer_storage(), Some(&storage));
        }

        let Value::Tensor(scalar) =
            run_diag(Value::Int(IntValue::U64(u64::MAX)), Vec::new()).expect("scalar diag")
        else {
            panic!("expected exact scalar matrix");
        };
        assert_eq!(
            scalar.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX]))
        );
    }

    #[test]
    fn diag_preserves_native_single_for_construction_and_extraction() {
        let input = Tensor::from_numeric_storage(NumericStorage::F32(vec![1.25, -2.5]), vec![1, 2])
            .expect("single vector");
        let Value::Tensor(matrix) = run_diag(Value::Tensor(input), Vec::new()).expect("diag")
        else {
            panic!("expected single matrix");
        };
        assert_eq!(matrix.shape(), &[2, 2]);
        assert_eq!(
            matrix
                .clone()
                .into_numeric_storage()
                .expect("matrix storage"),
            NumericStorage::F32(vec![1.25, 0.0, 0.0, -2.5])
        );

        let Value::Tensor(vector) =
            run_diag(Value::Tensor(matrix), Vec::new()).expect("diag extract")
        else {
            panic!("expected single vector");
        };
        assert_eq!(vector.shape(), &[2, 1]);
        assert_eq!(
            vector.into_numeric_storage().expect("vector storage"),
            NumericStorage::F32(vec![1.25, -2.5])
        );
    }

    #[test]
    fn diag_char_rectangular_and_extract() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = Value::CharArray(CharArray::new_row("ab"));
        let out = run_diag(chars, vec![size_vector(2, 4)]).expect("diag");
        let Value::CharArray(matrix) = out else {
            panic!("expected char output");
        };
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 4);
        assert_eq!(
            matrix.data(),
            runmat_value::encode_utf16_code_units("a    b  ")
        );

        let extracted = run_diag(Value::CharArray(matrix), Vec::new()).expect("diag extract");
        let Value::CharArray(vector) = extracted else {
            panic!("expected char output");
        };
        assert_eq!(vector.rows, 2);
        assert_eq!(vector.cols, 1);
        assert_eq!(vector.data(), runmat_value::encode_utf16_code_units("ab"));
    }

    #[test]
    fn diag_supports_trailing_singleton_dims() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let matrix = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2, 1, 1]).unwrap();
        let out = run_diag(Value::Tensor(matrix), Vec::new()).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape(), &[2, 1]);
        assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0]);
    }

    #[test]
    fn diag_rejects_non_singleton_trailing_dims() {
        let matrix = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = run_diag(Value::Tensor(matrix), Vec::new()).expect_err("expected error");
        assert!(
            err.message()
                .contains("only vectors and matrices are supported"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn diag_logical_output_override() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Value::Tensor(Tensor::new(vec![1.0, 0.0, 3.0], vec![1, 3]).unwrap());
        let out = run_diag(value, vec![Value::from("logical")]).expect("diag");
        let Value::LogicalArray(array) = out else {
            panic!("expected logical output");
        };
        assert_eq!(array.shape, vec![3, 3]);
        assert_eq!(array.data, vec![1, 0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn diag_logical_output_reads_wide_uint64_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 0, 7]), vec![1, 3])
            .expect("tensor");

        let out = run_diag(Value::Tensor(value), vec![Value::from("logical")]).expect("diag");
        let Value::LogicalArray(array) = out else {
            panic!("expected logical output");
        };
        assert_eq!(array.shape, vec![3, 3]);
        assert_eq!(array.data, vec![1, 0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn diag_logical_output_reads_typed_complex_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![0, -3, 0, 0]),
            IntegerStorage::I16(vec![0, 0, 0, 5]),
        )
        .expect("complex integer storage");
        let value = ComplexTensor::new_integer(storage, vec![2, 2]).expect("complex tensor");

        let out =
            run_diag(Value::ComplexTensor(value), vec![Value::from("logical")]).expect("diag");
        let Value::LogicalArray(array) = out else {
            panic!("expected logical output");
        };
        assert_eq!(array.shape, vec![2, 1]);
        assert_eq!(array.data, vec![0, 1]);
    }

    #[test]
    fn diag_integer_class_overrides_cover_all_classes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cases = [
            ("int8", IntegerStorage::I8(vec![2, 0, 0, -3])),
            ("int16", IntegerStorage::I16(vec![2, 0, 0, -3])),
            ("int32", IntegerStorage::I32(vec![2, 0, 0, -3])),
            ("int64", IntegerStorage::I64(vec![2, 0, 0, -3])),
            ("uint8", IntegerStorage::U8(vec![2, 0, 0, 0])),
            ("uint16", IntegerStorage::U16(vec![2, 0, 0, 0])),
            ("uint32", IntegerStorage::U32(vec![2, 0, 0, 0])),
            ("uint64", IntegerStorage::U64(vec![2, 0, 0, 0])),
        ];

        for (class_name, expected) in cases {
            let value = Value::Tensor(Tensor::new(vec![1.5, -2.5], vec![1, 2]).unwrap());
            let out = run_diag(value, vec![Value::from(class_name)]).expect("diag");
            let Value::Tensor(tensor) = out else {
                panic!("expected tensor output for {class_name}");
            };
            assert_eq!(tensor.shape(), &[2, 2]);
            assert_eq!(tensor.integer_storage(), Some(&expected), "{class_name}");
        }
    }

    #[test]
    fn diag_integer_class_override_reads_typed_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]),
            vec![1, 2],
        )
        .expect("tensor");

        let out = run_diag(Value::Tensor(value), vec![Value::from("int64")]).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MAX, 0, 0, i64::MAX]))
        );
    }

    #[test]
    fn diag_integer_class_override_casts_complex_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.5, -2.5), (300.0, 4.5)], vec![1, 2]).unwrap(),
        );

        let out = run_diag(value, vec![Value::from("uint8")]).expect("diag");
        let Value::ComplexTensor(tensor) = out else {
            panic!("expected complex tensor output");
        };
        assert_eq!(tensor.shape(), &[2, 2]);
        let storage = tensor.integer_storage().expect("complex integer storage");
        assert_eq!(storage.real(), &IntegerStorage::U8(vec![2, 0, 0, u8::MAX]));
        assert_eq!(storage.imag(), &IntegerStorage::U8(vec![0, 0, 0, 5]));
    }

    #[test]
    fn diag_double_override_from_logical_input() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let out =
            run_diag(Value::LogicalArray(logical), vec![Value::from("double")]).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.materialize_f64(), vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn diag_double_override_changes_typed_and_complex_storage_to_double() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let integer =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("integer");
        let Value::Tensor(integer_output) =
            run_diag(Value::Tensor(integer), vec![Value::from("double")]).expect("double")
        else {
            panic!("expected real double tensor");
        };
        assert_eq!(integer_output.numeric_dtype(), NumericDType::F64);
        assert!(integer_output.integer_storage().is_none());

        let complex = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(complex_output) =
            run_diag(Value::ComplexTensor(complex), vec![Value::from("double")])
                .expect("complex double")
        else {
            panic!("expected complex double tensor");
        };
        assert_eq!(complex_output.numeric_dtype(), NumericDType::F64);
    }

    #[test]
    fn diag_like_scalar_and_typed_prototypes_preserve_requested_class() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = || Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap());
        let Value::Tensor(scalar_like) = run_diag(
            value(),
            vec![Value::from("like"), Value::Int(IntValue::U16(0))],
        )
        .expect("scalar integer like") else {
            panic!("expected uint16 tensor");
        };
        assert_eq!(
            scalar_like.integer_storage(),
            Some(&IntegerStorage::U16(vec![2, 0, 0, 3]))
        );

        let prototype = Tensor::new_integer(IntegerStorage::I8(vec![0]), vec![1, 1]).unwrap();
        let Value::Tensor(typed_like) =
            run_diag(value(), vec![Value::from("like"), Value::Tensor(prototype)])
                .expect("typed tensor like")
        else {
            panic!("expected int8 tensor");
        };
        assert_eq!(
            typed_like.integer_storage(),
            Some(&IntegerStorage::I8(vec![2, 0, 0, 3]))
        );
    }

    #[test]
    fn diag_integer_to_complex_like_requires_exact_float_representation() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap();
        let error = run_diag(
            Value::Tensor(wide),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect_err("inexact integer-to-complex double conversion rejects");
        assert_eq!(error.identifier(), DIAG_ERROR_INVALID_INPUT.identifier);
        assert!(error.message().contains("exactly representable as double"));

        let single_proto = ComplexTensor::from_f32(vec![(0.0, 0.0)], vec![1, 1]).unwrap();
        let exact_f64_but_not_f32 =
            Tensor::new_integer(IntegerStorage::U32(vec![16_777_217]), vec![1, 1]).unwrap();
        let error = run_diag(
            Value::Tensor(exact_f64_but_not_f32),
            vec![Value::from("like"), Value::ComplexTensor(single_proto)],
        )
        .expect_err("inexact integer-to-complex single conversion rejects");
        assert!(error.message().contains("exactly representable as single"));
    }

    #[test]
    fn diag_like_logical_output() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Value::Tensor(Tensor::new(vec![2.0, 0.0], vec![1, 2]).unwrap());
        let out = run_diag(value, vec![Value::from("like"), Value::Bool(true)]).expect("diag");
        let Value::LogicalArray(array) = out else {
            panic!("expected logical output");
        };
        assert_eq!(array.shape, vec![2, 2]);
        assert_eq!(array.data, vec![1, 0, 0, 0]);
    }

    #[test]
    fn diag_like_logical_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Tensor::new_integer(
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .expect("tensor");

        let out = run_diag(
            Value::Tensor(value),
            vec![Value::from("like"), Value::Bool(true)],
        )
        .expect("diag");
        let Value::LogicalArray(array) = out else {
            panic!("expected logical output");
        };
        assert_eq!(array.shape, vec![2, 2]);
        assert_eq!(array.data, vec![0, 0, 0, 1]);
    }

    #[test]
    fn diag_like_complex_output() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Value::Tensor(Tensor::new(vec![2.0, 0.0], vec![1, 2]).unwrap());
        let out =
            run_diag(value, vec![Value::from("like"), Value::Complex(1.0, 2.0)]).expect("diag");
        let Value::ComplexTensor(tensor) = out else {
            panic!("expected complex tensor output");
        };
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![(2.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        );
    }

    #[test]
    fn diag_like_complex_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor =
            Tensor::new_integer(IntegerStorage::I64(vec![-3, 5]), vec![1, 2]).expect("tensor");
        let out = run_diag(
            Value::Tensor(tensor),
            vec![Value::from("like"), Value::Complex(1.0, 2.0)],
        )
        .expect("diag");
        let Value::ComplexTensor(tensor) = out else {
            panic!("expected complex tensor output");
        };
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![(-3.0, 0.0), (0.0, 0.0), (0.0, 0.0), (5.0, 0.0)]
        );
    }

    #[test]
    fn diag_resident_like_uses_prototype_owner_device_storage_and_precision() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let proto_tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &proto_tensor.materialize_f64(),
                shape: proto_tensor.shape(),
            };
            let proto = provider.upload(&view).expect("upload");

            let value = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
            let output = run_diag(
                value,
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("resident like");
            let Value::GpuTensor(output) = output else {
                panic!("expected resident result");
            };
            assert_eq!(output.device_id(), proto.device_id());
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                GpuTensorStorage::Real
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(provider.precision())
            );
            assert!(runmat_accelerate_api::same_provider_instance(
                runmat_accelerate_api::provider_for_handle(&output).unwrap(),
                provider
            ));
            let gathered = test_support::gather(Value::GpuTensor(output.clone())).unwrap();
            assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 0.0, 2.0]);

            drop(proto);

            drop(output);
        });
    }

    #[test]
    fn diag_resident_fallback_preserves_paired_uint64_without_like() {
        test_support::with_test_provider(|provider| {
            let wide = (1u64 << 53) + 1;
            let host = ComplexTensor::new_integer(
                runmat_value::IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![wide, u64::MAX]),
                    IntegerStorage::U64(vec![u64::MAX, wide]),
                )
                .unwrap(),
                vec![2, 1],
            )
            .unwrap();
            let input = gpu_helpers::upload_complex_tensor(provider, &host).unwrap();
            let value = run_diag(Value::GpuTensor(input.clone()), Vec::new()).unwrap();
            let Value::GpuTensor(output) = &value else {
                panic!("resident paired diagonal")
            };
            assert!(!output.shares_allocation_with(&input));
            assert_eq!(
                output.physical().element_type(),
                runmat_accelerate_api::NumericElementType::U64
            );
            let Value::ComplexTensor(result) =
                block_on(gpu_helpers::gather_value_async(&value)).unwrap()
            else {
                panic!("paired host output")
            };
            assert_eq!(result.shape(), &[2, 2]);
            let storage = result.integer_storage().unwrap();
            assert_eq!(
                storage.real(),
                &IntegerStorage::U64(vec![wide, 0, 0, u64::MAX])
            );
            assert_eq!(
                storage.imag(),
                &IntegerStorage::U64(vec![u64::MAX, 0, 0, wide])
            );
            let Value::ComplexTensor(original) =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(input))).unwrap()
            else {
                panic!("paired original")
            };
            assert_eq!(original.integer_storage(), host.integer_storage());
        });
    }

    #[test]
    fn diag_resident_like_preserves_exact_paired_complex_integer_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
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
            let input = ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![wide, u64::MAX]),
                    IntegerStorage::U64(vec![u64::MAX, wide]),
                )
                .expect("paired input storage"),
                vec![1, 2],
            )
            .expect("paired input");
            let output = run_diag(
                Value::ComplexTensor(input),
                vec![Value::from("like"), Value::GpuTensor(prototype.clone())],
            )
            .expect("resident paired diag");
            let Value::GpuTensor(output) = output else {
                panic!("expected resident paired result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let gathered = futures::executor::block_on(gpu_helpers::gather_value_async(
                &Value::GpuTensor(output.clone()),
            ))
            .expect("gather paired diag");
            let Value::ComplexTensor(gathered) = gathered else {
                panic!("expected paired diag tensor");
            };
            let storage = gathered.integer_storage().expect("paired storage");
            assert_eq!(
                storage.real(),
                &IntegerStorage::U64(vec![wide, 0, 0, u64::MAX])
            );
            assert_eq!(
                storage.imag(),
                &IntegerStorage::U64(vec![u64::MAX, 0, 0, wide])
            );

            drop(prototype);

            drop(output);
        });
    }

    #[test]
    fn diag_resident_logical_like_preserves_logical_metadata() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let proto = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[0],
                    shape: &[1, 1],
                })
                .expect("logical upload");

            let value = Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap());
            let output = run_diag(
                value,
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("logical resident like");
            let Value::GpuTensor(output) = output else {
                panic!("expected resident logical result");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&output));
            assert_eq!(output.device_id(), proto.device_id());
            assert!(runmat_accelerate_api::same_provider_instance(
                runmat_accelerate_api::provider_for_handle(&output).unwrap(),
                provider
            ));

            drop(proto);

            drop(output);
        });
    }

    #[test]
    fn diag_gpu_native_vector_paths_stay_resident() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let vector = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &vector.materialize_f64(),
                    shape: vector.shape(),
                })
                .expect("upload");

            let square = run_diag(Value::GpuTensor(handle.clone()), Vec::new()).expect("diag");
            assert!(
                matches!(square, Value::GpuTensor(_)),
                "native diag(gpuArray(vector)) should remain resident"
            );
            let gathered = test_support::gather(square).expect("gather square");
            assert_eq!(gathered.shape(), &[2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 0.0, 2.0]);

            let rectangular = run_diag(
                Value::GpuTensor(handle),
                vec![Value::Num(1.0), size_vector(3, 4)],
            )
            .expect("diag rectangular");
            assert!(
                matches!(rectangular, Value::GpuTensor(_)),
                "rectangular diag(gpuArray(vector), k, sz) should remain resident"
            );
            let gathered = test_support::gather(rectangular).expect("gather rectangular");
            assert_eq!(gathered.shape(), &[3, 4]);
            assert_eq!(
                gathered.materialize_f64(),
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
            );
        });
    }

    #[test]
    fn diag_resident_f32_double_override_returns_true_f64_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let input = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap();
            let input_handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
            let output = run_diag(
                Value::GpuTensor(input_handle.clone()),
                vec![Value::from("double")],
            )
            .expect("resident double override");
            let Value::GpuTensor(output) = output else {
                panic!("expected resident double result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(ProviderPrecision::F64)
            );
            assert!(runmat_accelerate_api::handle_integer_type(&output).is_none());
            assert!(!runmat_accelerate_api::handle_is_logical(&output));
            let gathered = test_support::gather(Value::GpuTensor(output.clone())).unwrap();
            assert_eq!(gathered.numeric_dtype(), NumericDType::F64);

            drop(input_handle);

            drop(output);
        });
    }

    #[test]
    fn diag_gpu_matrix_extract_and_logical_metadata_stay_resident() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let matrix = Tensor::new(
                vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
                vec![3, 3],
            )
            .unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &matrix.materialize_f64(),
                    shape: matrix.shape(),
                })
                .expect("upload matrix");
            let extracted =
                run_diag(Value::GpuTensor(handle), vec![Value::Num(-1.0)]).expect("diag extract");
            assert!(
                matches!(extracted, Value::GpuTensor(_)),
                "diag(gpuArray(matrix), k) should remain resident"
            );
            let gathered = test_support::gather(extracted).expect("gather extracted");
            assert_eq!(gathered.shape(), &[2, 1]);
            assert_eq!(gathered.materialize_f64(), vec![4.0, 8.0]);

            let logical_handle = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1, 0],
                    shape: &[1, 2],
                })
                .unwrap();
            // This reference provider encodes logical payloads in native F64;
            // the logical class is independent of its physical encoding.
            assert_eq!(
                runmat_accelerate_api::handle_precision(&logical_handle),
                Some(ProviderPrecision::F64)
            );
            let logical_out = run_diag(
                Value::GpuTensor(logical_handle),
                vec![Value::from("logical")],
            )
            .expect("logical diag");
            let Value::GpuTensor(logical_result) = logical_out else {
                panic!("expected logical gpu tensor output");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&logical_result));
            assert_eq!(
                runmat_accelerate_api::handle_precision(&logical_result),
                Some(ProviderPrecision::F64)
            );
            let gathered =
                test_support::gather(Value::GpuTensor(logical_result)).expect("gather logical");
            assert_eq!(gathered.shape(), &[2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 0.0, 0.0]);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn diag_wgpu_rectangular_and_extract_paths_stay_resident() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let vector = Tensor::new(vec![3.0, 5.0, 7.0], vec![3, 1]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &vector.materialize_f64(),
                shape: vector.shape(),
            })
            .expect("upload vector");
        let placed = run_diag(
            Value::GpuTensor(handle),
            vec![Value::Num(-1.0), size_vector(5, 3)],
        )
        .expect("diag placed");
        assert!(matches!(placed, Value::GpuTensor(_)));
        let gathered = test_support::gather(placed).expect("gather placed");
        assert_eq!(gathered.shape(), &[5, 3]);
        assert_eq!(
            gathered.materialize_f64(),
            vec![0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0]
        );

        let matrix = Tensor::new(
            vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &matrix.materialize_f64(),
                shape: matrix.shape(),
            })
            .expect("upload matrix");
        let extracted =
            run_diag(Value::GpuTensor(handle), vec![Value::Num(1.0)]).expect("diag extract");
        assert!(matches!(extracted, Value::GpuTensor(_)));
        let gathered = test_support::gather(extracted).expect("gather extracted");
        assert_eq!(gathered.shape(), &[2, 1]);
        assert_eq!(gathered.materialize_f64(), vec![2.0, 6.0]);

        let matrix = Tensor::new(
            vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &matrix.materialize_f64(),
                shape: matrix.shape(),
            })
            .expect("upload matrix");
        let empty = run_diag(Value::GpuTensor(handle), vec![Value::Num(10.0)]).expect("diag empty");
        assert!(matches!(empty, Value::GpuTensor(_)));
        let gathered = test_support::gather(empty).expect("gather empty");
        assert_eq!(gathered.shape(), &[0, 1]);
        assert!(gathered.materialize_f64().is_empty());
    }

    #[test]
    fn diag_vector_mode_rejects_matrix_input() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let matrix = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        let err = run_diag(matrix, vec![Value::from("vector")]).expect_err("expected error");
        assert!(
            err.message().contains("'vector' requires a vector input"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn diag_vector_mode_rejects_offset_combo() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let vector = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let err = run_diag(vector, vec![Value::Num(1.0), Value::from("vector")])
            .expect_err("expected error");
        assert!(
            err.message()
                .contains("'vector' cannot be combined with offsets or size overrides"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn diag_reports_invalid_offset_type() {
        let vector = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let err = run_diag(vector, vec![Value::CharArray(CharArray::new_row("oops"))])
            .expect_err("expected error");
        assert!(
            err.message().contains("unrecognised option")
                || err
                    .message()
                    .contains("diagonal offset must be a numeric scalar"),
            "unexpected error: {}",
            err.message()
        );
        assert!(
            err.identifier() == DIAG_ERROR_INVALID_INPUT.identifier
                || err.identifier() == DIAG_ERROR_INVALID_OFFSET.identifier,
            "unexpected identifier: {:?}",
            err.identifier()
        );
    }

    #[test]
    fn diag_resident_integer_fallback_preserves_all_classes_values_and_owner() {
        test_support::with_test_provider(|provider| {
            for storage in [
                IntegerStorage::I8(vec![-2, 7]),
                IntegerStorage::I16(vec![-300, 400]),
                IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                IntegerStorage::U8(vec![0, u8::MAX]),
                IntegerStorage::U16(vec![0, u16::MAX]),
                IntegerStorage::U32(vec![0, u32::MAX]),
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            ] {
                let input = Tensor::new_integer(storage.clone(), vec![1, 2]).unwrap();
                let input_handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
                let input_for_cleanup = input_handle.clone();
                let output = run_diag(Value::GpuTensor(input_handle), Vec::new()).unwrap();
                let Value::GpuTensor(output_handle) = output else {
                    panic!("expected resident integer result");
                };
                let owner = runmat_accelerate_api::provider_for_handle(&output_handle)
                    .expect("output owner");
                assert!(runmat_accelerate_api::same_provider_instance(
                    owner, provider
                ));
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&output_handle),
                    runmat_accelerate_api::handle_integer_type(&input_for_cleanup)
                );
                let gathered = test_support::gather(Value::GpuTensor(output_handle.clone()))
                    .expect("exact gather");
                assert_eq!(gathered.shape(), &[2, 2]);
                let zero = storage.zeros_like(1).value_at(0).unwrap();
                let values = storage.exact_values();
                let expected = storage
                    .from_exact_values_like(vec![
                        values[0].clone(),
                        zero.clone(),
                        zero,
                        values[1].clone(),
                    ])
                    .unwrap();
                assert_eq!(gathered.integer_storage(), Some(&expected));

                drop(input_for_cleanup);

                drop(output_handle);
            }
        });
    }

    #[test]
    fn diag_complex_fallback_restores_originating_provider() {
        test_support::with_test_provider(|provider| {
            let tensor = ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
            let input = gpu_helpers::upload_complex_tensor(provider, &tensor).unwrap();
            let input_for_cleanup = input.clone();
            let output = run_diag(Value::GpuTensor(input), Vec::new()).unwrap();
            let Value::GpuTensor(output) = output else {
                panic!("expected resident complex result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(ProviderPrecision::F32)
            );
            assert!(runmat_accelerate_api::same_provider_instance(
                runmat_accelerate_api::provider_for_handle(&output).unwrap(),
                provider
            ));

            drop(input_for_cleanup);

            drop(output);
        });
    }

    #[test]
    fn diag_extensions_and_resident_controls_are_strictly_bounded() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let vector = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let size_err = run_diag(vector(), vec![size_vector(2, 3)]).unwrap_err();
        assert_eq!(
            size_err.identifier.as_deref(),
            Some("RunMat:compatibility:DiagExplicitSizeExtension")
        );
        let class_err = run_diag(vector(), vec![Value::from("uint8")]).unwrap_err();
        assert_eq!(
            class_err.identifier.as_deref(),
            Some("RunMat:compatibility:DiagOutputClassExtension")
        );
        let trailing = Tensor::new(vec![1.0, 2.0], vec![1, 2, 1]).unwrap();
        let trailing_err = run_diag(Value::Tensor(trailing), Vec::new()).unwrap_err();
        assert_eq!(
            trailing_err.identifier.as_deref(),
            Some("RunMat:compatibility:DiagTrailingSingletonDimensionsExtension")
        );
        let resident_control = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("genuine unregistered allocation")
                .handle,
        );
        let control_err = run_diag(vector(), vec![resident_control]).unwrap_err();
        assert_eq!(
            control_err.identifier(),
            MESSAGE_ID_INVALID_INPUT.identifier
        );
        let resident_like = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("genuine unregistered allocation")
                .handle,
        );
        let like_err = run_diag(vector(), vec![Value::from("like"), resident_like]).unwrap_err();
        assert_eq!(like_err.identifier(), DIAG_LIKE_EXTENSION.error_identifier);
    }

    #[test]
    fn diag_descriptor_declares_only_public_forms_and_integer_contract() {
        assert_eq!(DIAG_DESCRIPTOR.signatures.len(), 9);
        assert_eq!(DIAG_EXTENSIONS.len(), 5);
        assert_eq!(DIAG_INTEGER_CAPABILITIES.len(), 2);
    }
}
