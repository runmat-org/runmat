//! MATLAB-compatible `inf` array constructor with GPU-aware semantics.

#[path = "inf/provider.rs"]
mod provider;
use provider::{inf_gpu, inf_like_gpu};

use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{
    BuiltinErrorDescriptor, INF_COLUMN_SIZE_VECTOR_EXTENSION, INF_ERROR_CLASS_CONFLICT,
    INF_ERROR_INTEGER_LIKE_PROTOTYPE, INF_ERROR_LIKE_DUPLICATE, INF_ERROR_LIKE_EXPECTED_PROTOTYPE,
    INF_ERROR_UNRECOGNIZED_OPTION, INF_RESIDENT_SIZE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, NumericDType, SparseTensor, Tensor, Value};

use crate::build_runtime_error;
use crate::builtins::common::random_args::{
    complex_tensor_into_value, extract_constructor_dimensions, normalize_constructor_shape,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionKernelTemplate,
    GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::inf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "inf",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("fill"), ProviderHook::Custom("fill_like")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Explicit gpuArray or resident like requests preserve device intent through constant-fill hooks or validated host-upload fallback; unavailable device allocation returns an error, not a host result. Ordinary size controls produce host output.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("inf").build()
}

fn inf_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    inf_error_with_message(error.message, error)
}

fn inf_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    inf_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn inf_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("inf");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::inf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "inf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let literal = match ctx.scalar_ty {
                ScalarType::F32 => "bitcast<f32>(0x7f800000u)",
                ScalarType::F64 => "bitcast<f64>(0x7ff0000000000000u)",
                ScalarType::I32 | ScalarType::Bool => {
                    return Err(crate::builtins::common::spec::FusionError::Message(
                        "inf: integer and logical fusion output is unsupported",
                    ))
                }
            };
            writer.push(literal)
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion planner materialises Inf constructors as canonical IEEE positive-infinity literals.",
};

#[runtime_builtin(
    name = "inf",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::inf"
)]
pub(crate) async fn inf_builtin(
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        let error = &runmat_builtins::INF_ERROR_TOO_MANY_OUTPUTS;
        return Err(crate::build_runtime_error(error.message)
            .with_builtin("inf")
            .with_identifier(error.identifier.expect("output count error identifier"))
            .build());
    }
    let parsed = ParsedInf::parse(rest).await?;
    build_output(parsed).await
}

struct ParsedInf {
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Single,
    GpuArray(NumericDType),
    Like(Value),
}

impl ParsedInf {
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
                crate::builtins::common::arg_tokens::scalar_keyword_of(&arg, "inf")?
            {
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err(inf_error(&INF_ERROR_LIKE_DUPLICATE));
                        }
                        if class_override.is_some() {
                            return Err(inf_error(&INF_ERROR_CLASS_CONFLICT));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(inf_error(&INF_ERROR_LIKE_EXPECTED_PROTOTYPE));
                        };
                        like_proto = Some(proto.clone());
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err(inf_error_with_detail(
                                &INF_ERROR_CLASS_CONFLICT,
                                "double class override",
                            ));
                        }
                        if class_override.is_some() {
                            return Err(inf_error(&INF_ERROR_CLASS_CONFLICT));
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        if like_proto.is_some() {
                            return Err(inf_error_with_detail(
                                &INF_ERROR_CLASS_CONFLICT,
                                "single class override",
                            ));
                        }
                        if class_override.is_some() {
                            return Err(inf_error(&INF_ERROR_CLASS_CONFLICT));
                        }
                        class_override = Some(OutputTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    "gpuArray" | "gpuarray" => {
                        if like_proto.is_some() {
                            return Err(inf_error_with_detail(
                                &INF_ERROR_CLASS_CONFLICT,
                                "gpuArray class override",
                            ));
                        }
                        let dtype = match class_override.take() {
                            Some(OutputTemplate::Single) => NumericDType::F32,
                            Some(OutputTemplate::Double) | None => NumericDType::F64,
                            Some(_) => unreachable!("inf class override is floating"),
                        };
                        class_override = Some(OutputTemplate::GpuArray(dtype));
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(inf_error_with_detail(
                            &INF_ERROR_UNRECOGNIZED_OPTION,
                            format!("'{other}'"),
                        ));
                    }
                }
            }

            if matches!(arg, Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &INF_RESIDENT_SIZE_EXTENSION,
                    "inf",
                )?;
            }
            if let Some(parsed_dims) = extract_constructor_dimensions(&arg, "inf")
                .await
                .map_err(builtin_error)?
            {
                if parsed_dims.is_column_vector {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &INF_COLUMN_SIZE_VECTOR_EXTENSION,
                        "inf",
                    )?;
                }
                if parsed_dims.values.len() > 1 {
                    if saw_size_vector || saw_dims_arg {
                        return Err(builtin_error(
                            "inf: a size vector must be the only dimension argument",
                        ));
                    }
                    saw_size_vector = true;
                } else if saw_size_vector {
                    return Err(builtin_error(
                        "inf: a size vector must be the only dimension argument",
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
                "inf: unsupported dimension or option {arg:?}"
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

async fn build_output(parsed: ParsedInf) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => inf_double(&parsed.shape),
        OutputTemplate::Single => inf_single(&parsed.shape),
        OutputTemplate::GpuArray(dtype) => inf_gpu(&parsed.shape, dtype).await,
        OutputTemplate::Like(proto) => inf_like(&proto, &parsed.shape).await,
    }
}

fn inf_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    inf_tensor(shape, NumericDType::F64).map(tensor::tensor_into_value)
}

fn inf_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    inf_tensor(shape, NumericDType::F32).map(tensor::tensor_into_value)
}

#[async_recursion::async_recursion(?Send)]
async fn inf_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::ComplexTensor(tensor) => inf_complex(shape, tensor.numeric_dtype()),
        Value::Complex(_, _) => inf_complex(shape, NumericDType::F64),
        Value::GpuTensor(handle) => inf_like_gpu(handle, shape).await,
        Value::Tensor(t) => match t.numeric_dtype() {
            NumericDType::F32 => inf_single(shape),
            NumericDType::F64 => inf_double(shape),
            NumericDType::I8
            | NumericDType::I16
            | NumericDType::I32
            | NumericDType::I64
            | NumericDType::U8
            | NumericDType::U16
            | NumericDType::U32
            | NumericDType::U64 => Err(inf_error(&INF_ERROR_INTEGER_LIKE_PROTOTYPE)),
        },
        Value::SparseTensor(sparse) if sparse.integer_storage().is_some() => {
            Err(inf_error(&INF_ERROR_INTEGER_LIKE_PROTOTYPE))
        }
        Value::SparseTensor(sparse) => match sparse.numeric_dtype() {
            Some(NumericDType::F32) => inf_sparse(shape, NumericDType::F32),
            Some(NumericDType::F64) => inf_sparse(shape, NumericDType::F64),
            None => Err(builtin_error(
                "inf: 'like' prototype must be single or double",
            )),
            Some(_) => unreachable!("integer sparse prototypes are rejected above"),
        },
        Value::Int(_) => Err(inf_error(&INF_ERROR_INTEGER_LIKE_PROTOTYPE)),
        Value::Num(_) => inf_double(shape),
        _ => Err(builtin_error(
            "inf: 'like' prototype must be single or double",
        )),
    }
}

fn inf_complex(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let data = vec![(f64::INFINITY, 0.0); len];
    ComplexTensor::from_f64_values_with_dtype(data, shape.to_vec(), dtype)
        .map(complex_tensor_into_value)
        .map_err(|e| builtin_error(format!("inf: {e}")))
}

fn inf_sparse(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Value> {
    if shape.len() > 2 {
        return Err(builtin_error(
            "inf: sparse 'like' output must be two-dimensional",
        ));
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| builtin_error("inf: sparse output size overflow"))?;
    let col_ptrs: Vec<_> = (0..=cols).map(|column| column * rows).collect();
    let row_indices: Vec<_> = (0..cols).flat_map(|_| 0..rows).collect();
    let sparse = match dtype {
        NumericDType::F32 => {
            SparseTensor::new_f32(rows, cols, col_ptrs, row_indices, vec![f32::INFINITY; len])
        }
        NumericDType::F64 => {
            SparseTensor::new(rows, cols, col_ptrs, row_indices, vec![f64::INFINITY; len])
        }
        _ => unreachable!("inf sparse output is floating"),
    }
    .map_err(|error| builtin_error(format!("inf: {error}")))?;
    Ok(Value::SparseTensor(sparse))
}

fn inf_tensor(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Tensor> {
    Tensor::new_with_dtype(
        vec![f64::INFINITY; tensor::element_count(shape)],
        shape.to_vec(),
        dtype,
    )
    .map_err(|e| builtin_error(format!("inf: {e}")))
}

fn dtype_from_precision(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

#[cfg(test)]
#[path = "inf/tests.rs"]
mod tests;

runmat_macros::register_constant!(
    "inf",
    Value::Num(f64::INFINITY),
    "crate::builtins::array::creation::inf"
);

runmat_macros::register_constant!(
    "Inf",
    Value::Num(f64::INFINITY),
    "crate::builtins::array::creation::inf"
);
