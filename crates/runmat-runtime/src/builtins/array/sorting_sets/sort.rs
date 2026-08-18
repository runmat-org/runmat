//! MATLAB-compatible `sort` builtin with multi-output and GPU-aware semantics.

use std::cmp::Ordering;

use runmat_accelerate_api::{
    GpuTensorHandle, SortComparison as ProviderSortComparison, SortOrder as ProviderSortOrder,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ComplexStorage, ComplexTensor, IntValue, IntegerStorage,
    LogicalArray, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::{float_order::SetFloat, integer_order, type_resolvers::tensor_output_type};
use crate::build_runtime_error;
use crate::builtins::common::arg_tokens::{tokens_from_values, ArgToken};
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::sort")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sort",
    op_kind: GpuOpKind::Custom("sort"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("sort_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Plain real tensors may use the provider sort hook; typed integer, logical, complex, explicit missing-placement, and unsupported-provider paths gather through authoritative host storage and restore both outputs to the owning provider.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::sorting_sets::sort")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sort",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Sorting breaks fusion chains; host fallback may gather upstream tensors before restoring new resident output handles.",
};

const BUILTIN_NAME: &str = "sort";

const SORT_OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sorted values.",
}];

const SORT_OUTPUT_BI: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sorted values.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based index permutation for each sorted slice.",
    },
];

const SORT_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const SORT_INPUTS_A_ARG1: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension selector or direction token ('ascend'/'descend').",
    },
];

const SORT_INPUTS_A_ARG1_ARG2: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension selector, placeholder, or direction token.",
    },
    BuiltinParamDescriptor {
        name: "arg2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension selector or direction token.",
    },
];

const SORT_INPUTS_COMPARISON_METHOD: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional dimension/direction arguments.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ComparisonMethod\""),
        description: "Name-value option key.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"auto\""),
        description: "Comparison method: 'auto', 'real', or 'abs'.",
    },
];

const SORT_INPUTS_MISSING_PLACEMENT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional dimension/direction arguments.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"MissingPlacement\""),
        description: "Name-value option key.",
    },
    BuiltinParamDescriptor {
        name: "placement",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"auto\""),
        description: "Missing-value placement: 'auto', 'first', or 'last'.",
    },
];

const SORT_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "B = sort(A)",
        inputs: &SORT_INPUTS_A,
        outputs: &SORT_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sort(A, arg1)",
        inputs: &SORT_INPUTS_A_ARG1,
        outputs: &SORT_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sort(A, arg1, arg2)",
        inputs: &SORT_INPUTS_A_ARG1_ARG2,
        outputs: &SORT_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sort(A, ..., \"ComparisonMethod\", method)",
        inputs: &SORT_INPUTS_COMPARISON_METHOD,
        outputs: &SORT_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sort(A, ..., \"MissingPlacement\", placement)",
        inputs: &SORT_INPUTS_MISSING_PLACEMENT,
        outputs: &SORT_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sort(A)",
        inputs: &SORT_INPUTS_A,
        outputs: &SORT_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sort(A, arg1)",
        inputs: &SORT_INPUTS_A_ARG1,
        outputs: &SORT_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sort(A, arg1, arg2)",
        inputs: &SORT_INPUTS_A_ARG1_ARG2,
        outputs: &SORT_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sort(A, ..., \"ComparisonMethod\", method)",
        inputs: &SORT_INPUTS_COMPARISON_METHOD,
        outputs: &SORT_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sort(A, ..., \"MissingPlacement\", placement)",
        inputs: &SORT_INPUTS_MISSING_PLACEMENT,
        outputs: &SORT_OUTPUT_BI,
    },
];

const SORT_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORT.INVALID_DIMENSION",
    identifier: Some("RunMat:sort:InvalidDimension"),
    when: "Dimension argument is non-positive, non-integer, or otherwise invalid.",
    message: "sort: invalid dimension argument",
};

const SORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.SORT.COMPARISON_METHOD_REQUIRES_STRING",
        identifier: Some("RunMat:sort:ComparisonMethodRequiresString"),
        when: "ComparisonMethod option value is not string-like.",
        message: "sort: 'ComparisonMethod' requires a string value",
    };

const SORT_ERROR_COMPARISON_METHOD_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORT.COMPARISON_METHOD_UNKNOWN",
    identifier: Some("RunMat:sort:ComparisonMethodUnknown"),
    when: "ComparisonMethod option value is not one of 'auto'/'real'/'abs'.",
    message: "sort: unsupported ComparisonMethod",
};

const SORT_ERROR_MISSINGPLACEMENT_REQUIRES_STRING: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.SORT.MISSINGPLACEMENT_REQUIRES_STRING",
        identifier: Some("RunMat:sort:MissingPlacementRequiresString"),
        when: "MissingPlacement option value is not string-like.",
        message: "sort: 'MissingPlacement' requires a string value",
    };

const SORT_ERROR_MISSINGPLACEMENT_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORT.MISSINGPLACEMENT_UNKNOWN",
    identifier: Some("RunMat:sort:MissingPlacementUnknown"),
    when: "MissingPlacement option value is not one of 'auto'/'first'/'last'.",
    message: "sort: unsupported MissingPlacement",
};

const SORT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORT.INVALID_ARGUMENT",
    identifier: Some("RunMat:sort:InvalidArgument"),
    when: "Parser encounters invalid or unrecognized option/value arguments.",
    message: "sort: invalid argument sequence",
};

const SORT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORT.INTERNAL",
    identifier: Some("RunMat:sort:Internal"),
    when: "Internal conversion, allocation, or provider result construction fails.",
    message: "sort: internal operation failed",
};

const SORT_ERRORS: [BuiltinErrorDescriptor; 7] = [
    SORT_ERROR_INVALID_DIMENSION,
    SORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING,
    SORT_ERROR_COMPARISON_METHOD_UNKNOWN,
    SORT_ERROR_MISSINGPLACEMENT_REQUIRES_STRING,
    SORT_ERROR_MISSINGPLACEMENT_UNKNOWN,
    SORT_ERROR_INVALID_ARGUMENT,
    SORT_ERROR_INTERNAL,
];

const SORT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented sortable input domain includes all eight real integer classes.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive integer-scalar dimension accepts every integer class and integer-valued scalar double.",
    },
];

const SORT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[B, I] = sort(integer_A, integer_dim, direction, options)",
        inputs: &SORT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "B preserves A's exact integer class and stable equal-value order; optional I is one-based double. Resident integer input uses exact typed gather fallback and restores both outputs to the owning provider.",
    }];

pub const SORT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SORT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SORT_ERRORS,
};

fn sort_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sort_internal(message: impl Into<String>) -> crate::RuntimeError {
    sort_error(&SORT_ERROR_INTERNAL, message)
}

fn sort_invalid_argument(message: impl Into<String>) -> crate::RuntimeError {
    sort_error(&SORT_ERROR_INVALID_ARGUMENT, message)
}

#[runtime_builtin(
    name = "sort",
    category = "array/sorting_sets",
    summary = "Sort array elements along a dimension with optional index outputs.",
    keywords = "sort,ascending,descending,indices,comparisonmethod,gpu",
    accel = "sink",
    sink = true,
    type_resolver(tensor_output_type),
    descriptor(crate::builtins::array::sorting_sets::sort::SORT_DESCRIPTOR),
    integer_capabilities(SORT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::sorting_sets::sort"
)]
async fn sort_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(value, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let (sorted, indices) = eval.into_values();
        if out_count == 1 {
            return Ok(Value::OutputList(vec![sorted]));
        }
        if out_count == 2 {
            return Ok(Value::OutputList(vec![sorted, indices]));
        }
        return Err(sort_invalid_argument(
            "sort: too many output arguments; maximum is 2",
        ));
    }
    Ok(eval.into_sorted_value())
}

/// Evaluate the `sort` builtin once and expose both outputs.
pub async fn evaluate(value: Value, rest: &[Value]) -> crate::BuiltinResult<SortEvaluation> {
    let args = SortArgs::parse(rest)?;
    match value {
        Value::GpuTensor(handle) => sort_gpu(handle, &args).await,
        other => sort_host(other, &args),
    }
}

async fn sort_gpu(
    handle: GpuTensorHandle,
    args: &SortArgs,
) -> crate::BuiltinResult<SortEvaluation> {
    let shape = handle.shape.clone();
    let provider = runmat_accelerate_api::provider_for_handle(&handle)
        .or_else(runmat_accelerate_api::provider);
    let dim = args.dimension.unwrap_or_else(|| default_dimension(&shape));
    if dim == 0 {
        return Err(sort_error(
            &SORT_ERROR_INVALID_DIMENSION,
            "sort: dimension must be >= 1",
        ));
    }
    let dim_len = dimension_length(&shape, dim);
    let plain_real = runmat_accelerate_api::handle_integer_type(&handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(&handle)
        && runmat_accelerate_api::handle_storage(&handle)
            == runmat_accelerate_api::GpuTensorStorage::Real;
    if dim_len > 1 && plain_real && matches!(args.missing, MissingPlacement::Auto) {
        if let Some(provider) = provider {
            let order = args.direction.to_provider();
            let comparison = args.comparison.to_provider();
            let zero_based = dim - 1;
            if let Ok(result) = provider
                .sort_dim(&handle, zero_based, order, comparison)
                .await
            {
                let sorted_tensor = Tensor::new(result.values.data, result.values.shape)
                    .map_err(|e| sort_internal(format!("sort: {e}")))?;
                let sorted_value = tensor::tensor_into_value(sorted_tensor);
                let indices_tensor = Tensor::new(result.indices.data, result.indices.shape)
                    .map_err(|e| sort_internal(format!("sort: {e}")))?;
                return upload_sort_evaluation(
                    provider,
                    SortEvaluation {
                        sorted: sorted_value,
                        indices: tensor::tensor_into_value(indices_tensor),
                    },
                );
            }
        }
    }
    let host = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
    let evaluation = sort_host(host, args)?;
    match provider {
        Some(provider) => upload_sort_evaluation(provider, evaluation),
        None => Ok(evaluation),
    }
}

fn sort_host(value: Value, args: &SortArgs) -> crate::BuiltinResult<SortEvaluation> {
    match value {
        Value::ComplexTensor(ct) => sort_complex_tensor(ct, args),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| sort_internal(format!("sort: {e}")))?;
            sort_complex_tensor(tensor, args)
        }
        Value::Int(value) => {
            let tensor = Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|e| sort_internal(format!("sort: {e}")))?;
            sort_real_tensor(tensor, args)
        }
        Value::LogicalArray(logical) => sort_logical(logical, args),
        Value::Bool(value) => {
            let logical = LogicalArray::new(vec![u8::from(value)], vec![1, 1])
                .map_err(|e| sort_internal(format!("sort: {e}")))?;
            sort_logical(logical, args)
        }
        other => {
            let tensor =
                tensor::value_into_tensor_for("sort", other).map_err(sort_invalid_argument)?;
            sort_real_tensor(tensor, args)
        }
    }
}

fn sort_logical(logical: LogicalArray, args: &SortArgs) -> crate::BuiltinResult<SortEvaluation> {
    let shape = logical.shape;
    let evaluation = sort_real_tensor(
        Tensor::new_integer(IntegerStorage::U8(logical.data), shape)
            .map_err(|e| sort_internal(format!("sort: {e}")))?,
        args,
    )?;
    let sorted = match evaluation.sorted {
        Value::Int(IntValue::U8(value)) => Value::Bool(value != 0),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let storage = tensor
                .into_numeric_storage()
                .map_err(|e| sort_internal(format!("sort: {e}")))?
                .into_integer_storage()
                .map_err(|_| sort_internal("sort: logical output lost integer storage"))?;
            let IntegerStorage::U8(values) = storage else {
                return Err(sort_internal("sort: logical output changed storage class"));
            };
            if values.len() == 1 {
                Value::Bool(values[0] != 0)
            } else {
                Value::LogicalArray(
                    LogicalArray::new(values, shape)
                        .map_err(|e| sort_internal(format!("sort: {e}")))?,
                )
            }
        }
        other => {
            return Err(sort_internal(format!(
                "sort: unexpected logical output {other:?}"
            )))
        }
    };
    Ok(SortEvaluation {
        sorted,
        indices: evaluation.indices,
    })
}

fn upload_sort_evaluation(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    evaluation: SortEvaluation,
) -> crate::BuiltinResult<SortEvaluation> {
    Ok(SortEvaluation {
        sorted: upload_sort_value(provider, evaluation.sorted)?,
        indices: upload_sort_value(provider, evaluation.indices)?,
    })
}

fn upload_sort_value(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    value: Value,
) -> crate::BuiltinResult<Value> {
    let upload_tensor = |tensor: Tensor, logical: bool| -> crate::BuiltinResult<Value> {
        let handle = gpu_helpers::upload_tensor(provider, &tensor)
            .map_err(|error| sort_internal(format!("sort: GPU upload failed: {error}")))?;
        Ok(if logical {
            gpu_helpers::logical_gpu_value(handle)
        } else {
            gpu_helpers::resident_gpu_value(handle)
        })
    };
    match value {
        Value::Tensor(tensor) => upload_tensor(tensor, false),
        Value::Num(number) => upload_tensor(
            Tensor::new(vec![number], vec![1, 1])
                .map_err(|error| sort_internal(format!("sort: {error}")))?,
            false,
        ),
        Value::Int(integer) => upload_tensor(
            Tensor::new_integer(IntegerStorage::from_scalar(integer), vec![1, 1])
                .map_err(|error| sort_internal(format!("sort: {error}")))?,
            false,
        ),
        Value::Bool(logical) => upload_tensor(
            Tensor::new(vec![if logical { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|error| sort_internal(format!("sort: {error}")))?,
            true,
        ),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(sort_internal)?;
            upload_tensor(tensor, true)
        }
        Value::Complex(real, imag) => {
            let tensor = ComplexTensor::new(vec![(real, imag)], vec![1, 1])
                .map_err(|error| sort_internal(format!("sort: {error}")))?;
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|error| sort_internal(format!("sort: {}", error.message())))?;
            Ok(gpu_helpers::complex_gpu_value(handle))
        }
        Value::ComplexTensor(tensor) => {
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|error| sort_internal(format!("sort: {}", error.message())))?;
            Ok(gpu_helpers::complex_gpu_value(handle))
        }
        other => Err(sort_internal(format!(
            "sort: cannot upload unexpected output {other:?}"
        ))),
    }
}

fn sort_real_tensor(tensor: Tensor, args: &SortArgs) -> crate::BuiltinResult<SortEvaluation> {
    let shape = tensor.shape.clone();
    let dim = args.dimension.unwrap_or_else(|| default_dimension(&shape));
    if dim == 0 {
        return Err(sort_error(
            &SORT_ERROR_INVALID_DIMENSION,
            "sort: dimension must be >= 1",
        ));
    }

    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| sort_internal(format!("sort: {e}")))?;
    match storage {
        NumericStorage::F64(values) => {
            sort_floating_tensor(values, shape, dim, args, NumericStorage::F64)
        }
        NumericStorage::F32(values) => {
            sort_floating_tensor(values, shape, dim, args, NumericStorage::F32)
        }
        storage => sort_integer_tensor(
            storage
                .into_integer_storage()
                .expect("non-floating numeric storage is integer"),
            shape,
            dim,
            args,
        ),
    }
}

fn sort_floating_tensor<T>(
    values: Vec<T>,
    shape: Vec<usize>,
    dim: usize,
    args: &SortArgs,
    wrap: fn(Vec<T>) -> NumericStorage,
) -> crate::BuiltinResult<SortEvaluation>
where
    T: SetFloat,
{
    let dim_len = dimension_length(&shape, dim);
    if values.is_empty() || dim_len <= 1 {
        let indices = vec![1.0; values.len()];
        let index_tensor =
            Tensor::new(indices, shape.clone()).map_err(|e| sort_internal(format!("sort: {e}")))?;
        let sorted_tensor = Tensor::from_numeric_storage(wrap(values), shape)
            .map_err(|e| sort_internal(format!("sort: {e}")))?;
        return Ok(SortEvaluation {
            sorted: tensor::tensor_into_value(sorted_tensor),
            indices: tensor::tensor_into_value(index_tensor),
        });
    }

    let stride_before = stride_before(&shape, dim);
    let stride_after = stride_after(&shape, dim);
    let mut sorted = values;
    let mut indices = vec![0.0f64; sorted.len()];
    let mut buffer: Vec<(usize, T)> = Vec::with_capacity(dim_len);

    for after in 0..stride_after {
        for before in 0..stride_before {
            buffer.clear();
            for k in 0..dim_len {
                let idx = before + k * stride_before + after * stride_before * dim_len;
                let value = sorted[idx];
                buffer.push((k, value));
            }
            buffer.sort_by(|a, b| compare_real_values(a.1, b.1, args));
            for (pos, (original_index, value)) in buffer.iter().enumerate() {
                let target = before + pos * stride_before + after * stride_before * dim_len;
                sorted[target] = *value;
                indices[target] = (*original_index + 1) as f64;
            }
        }
    }

    let sorted_tensor = Tensor::from_numeric_storage(wrap(sorted), shape.clone())
        .map_err(|e| sort_internal(format!("sort: {e}")))?;
    let index_tensor =
        Tensor::new(indices, shape).map_err(|e| sort_internal(format!("sort: {e}")))?;

    Ok(SortEvaluation {
        sorted: tensor::tensor_into_value(sorted_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn sort_integer_tensor(
    storage: IntegerStorage,
    shape: Vec<usize>,
    dim: usize,
    args: &SortArgs,
) -> crate::BuiltinResult<SortEvaluation> {
    let stride_before = stride_before(&shape, dim);
    let stride_after = stride_after(&shape, dim);
    let dim_len = dimension_length(&shape, dim);
    let mut sorted = storage.exact_values();
    let mut indices = vec![0.0f64; sorted.len()];
    let mut buffer: Vec<(usize, IntValue)> = Vec::with_capacity(dim_len);

    for after in 0..stride_after {
        for before in 0..stride_before {
            buffer.clear();
            for k in 0..dim_len {
                let idx = before + k * stride_before + after * stride_before * dim_len;
                buffer.push((k, sorted[idx].clone()));
            }
            buffer.sort_by(|a, b| {
                integer_order::compare(
                    &a.1,
                    &b.1,
                    matches!(args.direction, SortDirection::Descend),
                    matches!(args.comparison, ComparisonMethod::Abs),
                )
            });
            for (pos, (original_index, value)) in buffer.iter().enumerate() {
                let target = before + pos * stride_before + after * stride_before * dim_len;
                sorted[target] = value.clone();
                indices[target] = (*original_index + 1) as f64;
            }
        }
    }

    let sorted_storage = storage
        .from_exact_values_like(sorted)
        .map_err(|e| sort_internal(format!("sort: {e}")))?;
    let sorted_tensor = Tensor::new_integer(sorted_storage, shape.clone())
        .map_err(|e| sort_internal(format!("sort: {e}")))?;
    let index_tensor =
        Tensor::new(indices, shape).map_err(|e| sort_internal(format!("sort: {e}")))?;
    Ok(SortEvaluation {
        sorted: Value::Tensor(sorted_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn sort_complex_tensor(
    tensor: ComplexTensor,
    args: &SortArgs,
) -> crate::BuiltinResult<SortEvaluation> {
    let shape = tensor.shape.clone();
    let dim = args.dimension.unwrap_or_else(|| default_dimension(&shape));
    if dim == 0 {
        return Err(sort_error(
            &SORT_ERROR_INVALID_DIMENSION,
            "sort: dimension must be >= 1",
        ));
    }

    let dim_len = dimension_length(&shape, dim);
    let storage = tensor.into_complex_storage();
    if storage.is_empty() || dim_len <= 1 {
        let indices = vec![1.0; storage.len()];
        let index_tensor =
            Tensor::new(indices, shape.clone()).map_err(|e| sort_internal(format!("sort: {e}")))?;
        let sorted_tensor = ComplexTensor::from_complex_storage(storage, shape)
            .map_err(|e| sort_internal(format!("sort: {e}")))?;
        return Ok(SortEvaluation {
            sorted: complex_tensor_into_value(sorted_tensor),
            indices: tensor::tensor_into_value(index_tensor),
        });
    }

    let (source_indices, indices) = match &storage {
        ComplexStorage::F64(values) => complex_sort_permutation(values, &shape, dim, args),
        ComplexStorage::F32(values) => complex_sort_permutation(values, &shape, dim, args),
        ComplexStorage::Integer(values) => {
            complex_integer_sort_permutation(values, &shape, dim, args)
        }
    };
    let sorted_storage = storage
        .gather(&source_indices)
        .map_err(|e| sort_internal(format!("sort: {e}")))?;
    let sorted_tensor = ComplexTensor::from_complex_storage(sorted_storage, shape.clone())
        .map_err(|e| sort_internal(format!("sort: {e}")))?;
    let index_tensor =
        Tensor::new(indices, shape).map_err(|e| sort_internal(format!("sort: {e}")))?;

    Ok(SortEvaluation {
        sorted: complex_tensor_into_value(sorted_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn complex_integer_sort_permutation(
    storage: &runmat_builtins::IntegerComplexStorage,
    shape: &[usize],
    dim: usize,
    args: &SortArgs,
) -> (Vec<usize>, Vec<f64>) {
    let stride_before = stride_before(shape, dim);
    let stride_after = stride_after(shape, dim);
    let dim_len = dimension_length(shape, dim);
    let mut source_indices = vec![0usize; storage.len()];
    let mut indices = vec![0.0f64; storage.len()];
    let mut buffer: Vec<(usize, usize)> = Vec::with_capacity(dim_len);

    for after in 0..stride_after {
        for before in 0..stride_before {
            buffer.clear();
            for k in 0..dim_len {
                let source = before + k * stride_before + after * stride_before * dim_len;
                buffer.push((k, source));
            }
            buffer.sort_by(|a, b| {
                let a_real = storage.real.value_at(a.1).expect("validated complex index");
                let a_imag = storage.imag.value_at(a.1).expect("validated complex index");
                let b_real = storage.real.value_at(b.1).expect("validated complex index");
                let b_imag = storage.imag.value_at(b.1).expect("validated complex index");
                integer_order::compare_complex(
                    (&a_real, &a_imag),
                    (&b_real, &b_imag),
                    matches!(args.direction, SortDirection::Descend),
                    matches!(args.comparison, ComparisonMethod::Real),
                )
            });
            for (position, (original_index, source)) in buffer.iter().enumerate() {
                let target = before + position * stride_before + after * stride_before * dim_len;
                source_indices[target] = *source;
                indices[target] = (*original_index + 1) as f64;
            }
        }
    }

    (source_indices, indices)
}

fn complex_sort_permutation<T: SetFloat>(
    values: &[(T, T)],
    shape: &[usize],
    dim: usize,
    args: &SortArgs,
) -> (Vec<usize>, Vec<f64>) {
    let stride_before = stride_before(shape, dim);
    let stride_after = stride_after(shape, dim);
    let dim_len = dimension_length(shape, dim);
    let mut source_indices = vec![0usize; values.len()];
    let mut indices = vec![0.0f64; values.len()];
    let mut buffer: Vec<(usize, usize, (T, T))> = Vec::with_capacity(dim_len);

    for after in 0..stride_after {
        for before in 0..stride_before {
            buffer.clear();
            for k in 0..dim_len {
                let source = before + k * stride_before + after * stride_before * dim_len;
                buffer.push((k, source, values[source]));
            }
            buffer.sort_by(|a, b| compare_complex_values(a.2, b.2, args));
            for (position, (original_index, source, _)) in buffer.iter().enumerate() {
                let target = before + position * stride_before + after * stride_before * dim_len;
                source_indices[target] = *source;
                indices[target] = (*original_index + 1) as f64;
            }
        }
    }

    (source_indices, indices)
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if let Some([value]) = tensor.as_f64_slice() {
        Value::Complex(value.0, value.1)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn compare_real_values<T: SetFloat>(a: T, b: T, args: &SortArgs) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => match args.missing.resolve(args.direction) {
            MissingPlacementResolved::First => Ordering::Less,
            MissingPlacementResolved::Last => Ordering::Greater,
        },
        (false, true) => match args.missing.resolve(args.direction) {
            MissingPlacementResolved::First => Ordering::Greater,
            MissingPlacementResolved::Last => Ordering::Less,
        },
        (false, false) => compare_real_finite(a, b, args),
    }
}

fn compare_real_finite<T: SetFloat>(a: T, b: T, args: &SortArgs) -> Ordering {
    let primary = match args.comparison {
        ComparisonMethod::Abs => {
            let abs_cmp = a.abs().compare(b.abs());
            if abs_cmp != Ordering::Equal {
                return match args.direction {
                    SortDirection::Ascend => abs_cmp,
                    SortDirection::Descend => abs_cmp.reverse(),
                };
            }
            Ordering::Equal
        }
        ComparisonMethod::Auto | ComparisonMethod::Real => Ordering::Equal,
    };
    if primary != Ordering::Equal {
        return primary;
    }
    let ordering = if matches!(args.comparison, ComparisonMethod::Abs) {
        b.compare(a)
    } else {
        a.compare(b)
    };
    match args.direction {
        SortDirection::Ascend => ordering,
        SortDirection::Descend => ordering.reverse(),
    }
}

fn compare_complex_values<T: SetFloat>(a: (T, T), b: (T, T), args: &SortArgs) -> Ordering {
    match (complex_is_nan(a), complex_is_nan(b)) {
        (true, true) => Ordering::Equal,
        (true, false) => match args.missing.resolve(args.direction) {
            MissingPlacementResolved::First => Ordering::Less,
            MissingPlacementResolved::Last => Ordering::Greater,
        },
        (false, true) => match args.missing.resolve(args.direction) {
            MissingPlacementResolved::First => Ordering::Greater,
            MissingPlacementResolved::Last => Ordering::Less,
        },
        (false, false) => compare_complex_finite(a, b, args),
    }
}

fn compare_complex_finite<T: SetFloat>(a: (T, T), b: (T, T), args: &SortArgs) -> Ordering {
    match args.comparison {
        ComparisonMethod::Real => compare_complex_real_imag(a, b, args.direction),
        ComparisonMethod::Abs | ComparisonMethod::Auto => {
            let abs_cmp = complex_abs(a).compare(complex_abs(b));
            if abs_cmp != Ordering::Equal {
                return match args.direction {
                    SortDirection::Ascend => abs_cmp,
                    SortDirection::Descend => abs_cmp.reverse(),
                };
            }
            compare_complex_phase(a, b, args.direction)
        }
    }
}

fn compare_complex_phase<T: SetFloat>(a: (T, T), b: (T, T), direction: SortDirection) -> Ordering {
    let ordering = complex_phase(a).compare(complex_phase(b));
    match direction {
        SortDirection::Ascend => ordering,
        SortDirection::Descend => ordering.reverse(),
    }
}

fn complex_phase<T: SetFloat>((real, imaginary): (T, T)) -> T {
    let imaginary = if imaginary == T::default() {
        T::default()
    } else {
        imaginary
    };
    imaginary.atan2(real)
}

fn compare_complex_real_imag<T: SetFloat>(
    a: (T, T),
    b: (T, T),
    direction: SortDirection,
) -> Ordering {
    let real_cmp = match direction {
        SortDirection::Ascend => a.0.compare(b.0),
        SortDirection::Descend => b.0.compare(a.0),
    };
    if real_cmp != Ordering::Equal {
        return real_cmp;
    }
    match direction {
        SortDirection::Ascend => a.1.compare(b.1),
        SortDirection::Descend => b.1.compare(a.1),
    }
}

fn complex_is_nan<T: SetFloat>(value: (T, T)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn complex_abs<T: SetFloat>(value: (T, T)) -> T {
    value.0.hypot(value.1)
}

fn stride_before(shape: &[usize], dim: usize) -> usize {
    if dim <= 1 {
        return 1;
    }
    let mut product = 1usize;
    for i in 0..(dim - 1) {
        product = product.saturating_mul(*shape.get(i).unwrap_or(&1));
    }
    product
}

fn stride_after(shape: &[usize], dim: usize) -> usize {
    if dim >= shape.len() {
        return 1;
    }
    let mut product = 1usize;
    for extent in shape.iter().skip(dim) {
        product = product.saturating_mul(*extent);
    }
    product
}

fn dimension_length(shape: &[usize], dim: usize) -> usize {
    shape.get(dim - 1).copied().unwrap_or(1)
}

fn default_dimension(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|&extent| extent > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SortDirection {
    #[default]
    Ascend,
    Descend,
}

impl SortDirection {
    fn to_provider(self) -> ProviderSortOrder {
        match self {
            SortDirection::Ascend => ProviderSortOrder::Ascend,
            SortDirection::Descend => ProviderSortOrder::Descend,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ComparisonMethod {
    #[default]
    Auto,
    Real,
    Abs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum MissingPlacement {
    #[default]
    Auto,
    First,
    Last,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissingPlacementResolved {
    First,
    Last,
}

impl MissingPlacement {
    fn resolve(self, direction: SortDirection) -> MissingPlacementResolved {
        match self {
            Self::First => MissingPlacementResolved::First,
            Self::Last => MissingPlacementResolved::Last,
            Self::Auto => match direction {
                SortDirection::Ascend => MissingPlacementResolved::Last,
                SortDirection::Descend => MissingPlacementResolved::First,
            },
        }
    }
}

impl ComparisonMethod {
    fn to_provider(self) -> ProviderSortComparison {
        match self {
            ComparisonMethod::Auto => ProviderSortComparison::Auto,
            ComparisonMethod::Real => ProviderSortComparison::Real,
            ComparisonMethod::Abs => ProviderSortComparison::Abs,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SortArgs {
    dimension: Option<usize>,
    direction: SortDirection,
    comparison: ComparisonMethod,
    missing: MissingPlacement,
}

impl SortArgs {
    fn parse(rest: &[Value]) -> crate::BuiltinResult<Self> {
        let mut args = SortArgs::default();
        let tokens = tokens_from_values(rest);
        let mut i = 0usize;
        while i < rest.len() {
            if args.dimension.is_none() {
                if is_dimension_placeholder(&rest[i]) {
                    i += 1;
                    continue;
                }
                match tensor::parse_dimension(&rest[i], "sort") {
                    Ok(dim) => {
                        args.dimension = Some(dim);
                        i += 1;
                        continue;
                    }
                    Err(err) => {
                        if matches!(rest[i], Value::Int(_) | Value::Num(_)) {
                            return Err(sort_error(&SORT_ERROR_INVALID_DIMENSION, err));
                        }
                    }
                }
            }
            if let Some(ArgToken::String(text)) = tokens.get(i) {
                match text.as_str() {
                    "ascend" | "ascending" => {
                        args.direction = SortDirection::Ascend;
                        i += 1;
                        continue;
                    }
                    "descend" | "descending" => {
                        args.direction = SortDirection::Descend;
                        i += 1;
                        continue;
                    }
                    "comparisonmethod" => {
                        i += 1;
                        if i >= rest.len() {
                            return Err(sort_invalid_argument(
                                "sort: expected a value for 'ComparisonMethod'",
                            ));
                        }
                        let value = match tokens.get(i) {
                            Some(ArgToken::String(value)) => value.as_str(),
                            _ => {
                                return Err(sort_error(
                                    &SORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING,
                                    "sort: 'ComparisonMethod' requires a string value",
                                ))
                            }
                        };
                        args.comparison = match value {
                            "auto" => ComparisonMethod::Auto,
                            "real" => ComparisonMethod::Real,
                            "abs" | "magnitude" => ComparisonMethod::Abs,
                            other => {
                                return Err(sort_error(
                                    &SORT_ERROR_COMPARISON_METHOD_UNKNOWN,
                                    format!("sort: unsupported ComparisonMethod '{other}'"),
                                )
                                .into())
                            }
                        };
                        i += 1;
                        continue;
                    }
                    "missingplacement" => {
                        i += 1;
                        if i >= rest.len() {
                            return Err(sort_invalid_argument(
                                "sort: expected a value for 'MissingPlacement'",
                            ));
                        }
                        let value = match tokens.get(i) {
                            Some(ArgToken::String(value)) => value.as_str(),
                            _ => {
                                return Err(sort_error(
                                    &SORT_ERROR_MISSINGPLACEMENT_REQUIRES_STRING,
                                    SORT_ERROR_MISSINGPLACEMENT_REQUIRES_STRING.message,
                                ))
                            }
                        };
                        args.missing = match value {
                            "auto" => MissingPlacement::Auto,
                            "first" => MissingPlacement::First,
                            "last" => MissingPlacement::Last,
                            other => {
                                return Err(sort_error(
                                    &SORT_ERROR_MISSINGPLACEMENT_UNKNOWN,
                                    format!("sort: unsupported MissingPlacement '{other}'"),
                                ))
                            }
                        };
                        i += 1;
                        continue;
                    }
                    _ => {}
                }
            }
            if let Some(keyword) = tensor::value_to_string(&rest[i]) {
                let lowered = keyword.trim().to_ascii_lowercase();
                match lowered.as_str() {
                    "ascend" | "ascending" => {
                        args.direction = SortDirection::Ascend;
                        i += 1;
                        continue;
                    }
                    "descend" | "descending" => {
                        args.direction = SortDirection::Descend;
                        i += 1;
                        continue;
                    }
                    "comparisonmethod" => {
                        i += 1;
                        if i >= rest.len() {
                            return Err(sort_invalid_argument(
                                "sort: expected a value for 'ComparisonMethod'",
                            ));
                        }
                        let raw = &rest[i];
                        let value = match raw {
                            Value::String(s) => s.clone(),
                            Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
                            Value::CharArray(ca) if ca.rows == 1 => {
                                ca.data.iter().copied().collect()
                            }
                            _ => {
                                return Err(sort_error(
                                    &SORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING,
                                    "sort: 'ComparisonMethod' requires a string value",
                                ))
                            }
                        };
                        let lowered_value = value.trim().to_ascii_lowercase();
                        args.comparison = match lowered_value.as_str() {
                            "auto" => ComparisonMethod::Auto,
                            "real" => ComparisonMethod::Real,
                            "abs" | "magnitude" => ComparisonMethod::Abs,
                            other => {
                                return Err(sort_error(
                                    &SORT_ERROR_COMPARISON_METHOD_UNKNOWN,
                                    format!("sort: unsupported ComparisonMethod '{other}'"),
                                )
                                .into())
                            }
                        };
                        i += 1;
                        continue;
                    }
                    "missingplacement" => {
                        i += 1;
                        if i >= rest.len() {
                            return Err(sort_invalid_argument(
                                "sort: expected a value for 'MissingPlacement'",
                            ));
                        }
                        let value = tensor::value_to_string(&rest[i]).ok_or_else(|| {
                            sort_error(
                                &SORT_ERROR_MISSINGPLACEMENT_REQUIRES_STRING,
                                SORT_ERROR_MISSINGPLACEMENT_REQUIRES_STRING.message,
                            )
                        })?;
                        args.missing = match value.trim().to_ascii_lowercase().as_str() {
                            "auto" => MissingPlacement::Auto,
                            "first" => MissingPlacement::First,
                            "last" => MissingPlacement::Last,
                            other => {
                                return Err(sort_error(
                                    &SORT_ERROR_MISSINGPLACEMENT_UNKNOWN,
                                    format!("sort: unsupported MissingPlacement '{other}'"),
                                ))
                            }
                        };
                        i += 1;
                        continue;
                    }
                    _ => {}
                }
            }
            return Err(sort_invalid_argument(format!(
                "sort: unrecognised argument {:?}",
                rest[i]
            )));
        }
        Ok(args)
    }
}

fn is_dimension_placeholder(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => tensor::tensor_element_len(t) == 0,
        Value::LogicalArray(logical) => logical.data.is_empty(),
        _ => false,
    }
}

pub struct SortEvaluation {
    sorted: Value,
    indices: Value,
}

impl SortEvaluation {
    pub fn into_sorted_value(self) -> Value {
        self.sorted
    }

    pub fn into_values(self) -> (Value, Value) {
        (self.sorted, self.indices)
    }

    pub fn indices_value(&self) -> Value {
        self.indices.clone()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, NumericStorage,
        ResolveContext, Tensor, Type, Value,
    };

    fn sort_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::sort_builtin(value, rest))
    }

    fn evaluate(value: Value, rest: &[Value]) -> crate::BuiltinResult<SortEvaluation> {
        block_on(super::evaluate(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_vector_default() {
        let tensor = Tensor::new(vec![3.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let result = sort_builtin(Value::Tensor(tensor), Vec::new()).expect("sort");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0]);
                assert_eq!(t.shape, vec![3, 1]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sort_preserves_native_single_storage_and_indices() {
        let tensor = Tensor::from_f32(vec![3.0, 1.0, 2.0], vec![3, 1]).expect("input");
        let (sorted, indices) = evaluate(Value::Tensor(tensor), &[])
            .expect("sort")
            .into_values();
        let Value::Tensor(sorted) = sorted else {
            panic!("expected sorted tensor");
        };
        assert_eq!(
            sorted.into_numeric_storage().expect("storage"),
            NumericStorage::F32(vec![1.0, 2.0, 3.0])
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected index tensor");
        };
        assert_eq!(indices.materialize_f64(), vec![2.0, 3.0, 1.0]);
    }

    #[test]
    fn sort_type_resolver_tensor() {
        assert_eq!(
            tensor_output_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::tensor()
        );
    }

    #[test]
    fn sort_preserves_all_exact_real_integer_classes_and_indices() {
        let cases = [
            (
                IntegerStorage::I8(vec![i8::MAX, i8::MIN, 0, 7]),
                IntegerStorage::I8(vec![i8::MIN, 0, 7, i8::MAX]),
            ),
            (
                IntegerStorage::I16(vec![i16::MAX, i16::MIN, 0, 7]),
                IntegerStorage::I16(vec![i16::MIN, 0, 7, i16::MAX]),
            ),
            (
                IntegerStorage::I32(vec![i32::MAX, i32::MIN, 0, 7]),
                IntegerStorage::I32(vec![i32::MIN, 0, 7, i32::MAX]),
            ),
            (
                IntegerStorage::I64(vec![i64::MAX, i64::MIN, 0, 7]),
                IntegerStorage::I64(vec![i64::MIN, 0, 7, i64::MAX]),
            ),
            (
                IntegerStorage::U8(vec![u8::MAX, 0, 7, 9]),
                IntegerStorage::U8(vec![0, 7, 9, u8::MAX]),
            ),
            (
                IntegerStorage::U16(vec![u16::MAX, 0, 7, 700]),
                IntegerStorage::U16(vec![0, 7, 700, u16::MAX]),
            ),
            (
                IntegerStorage::U32(vec![u32::MAX, 0, 7, 9_007_199]),
                IntegerStorage::U32(vec![0, 7, 9_007_199, u32::MAX]),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX, 0, 7, 9_007_199_254_740_993]),
                IntegerStorage::U64(vec![0, 7, 9_007_199_254_740_993, u64::MAX]),
            ),
        ];

        for (input, expected) in cases {
            let tensor = Tensor::new_integer(input, vec![4, 1]).expect("input");
            let (sorted, indices) = evaluate(Value::Tensor(tensor), &[])
                .expect("sort")
                .into_values();
            let Value::Tensor(sorted) = sorted else {
                panic!("expected exact integer sorted values");
            };
            assert_eq!(sorted.integer_storage(), Some(&expected));
            let Value::Tensor(indices) = indices else {
                panic!("expected index tensor");
            };
            assert_eq!(indices.materialize_f64(), vec![2.0, 3.0, 4.0, 1.0]);
        }
    }

    #[test]
    fn sort_reads_exact_integer_values_without_mirror() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 0, 9_007_199_254_740_993, 7]),
            vec![4, 1],
        )
        .expect("input");

        let (sorted, indices) = evaluate(Value::Tensor(tensor), &[])
            .expect("sort")
            .into_values();
        let Value::Tensor(sorted) = sorted else {
            panic!("expected exact integer sorted values");
        };
        assert_eq!(
            sorted.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                0,
                7,
                9_007_199_254_740_993,
                u64::MAX,
            ]))
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected index tensor");
        };
        assert_eq!(indices.materialize_f64(), vec![2.0, 4.0, 3.0, 1.0]);
    }

    #[test]
    fn sort_exact_integer_honors_descend_abs_and_dimension() {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 0, 7, 9_007_199_254_740_993]),
            vec![4, 1],
        )
        .expect("input");
        let (sorted, indices) = evaluate(Value::Tensor(input), &[Value::from("descend")])
            .expect("descending sort")
            .into_values();
        let Value::Tensor(sorted) = sorted else {
            panic!("expected exact integer sorted values");
        };
        assert_eq!(
            sorted.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                u64::MAX,
                9_007_199_254_740_993,
                7,
                0,
            ]))
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected index tensor");
        };
        assert_eq!(indices.materialize_f64(), vec![1.0, 4.0, 3.0, 2.0]);

        let input = Tensor::new_integer(
            IntegerStorage::I64(vec![i64::MIN, i64::MAX, 2, -1]),
            vec![4, 1],
        )
        .expect("input");
        let (sorted, indices) = evaluate(
            Value::Tensor(input),
            &[Value::from("ComparisonMethod"), Value::from("abs")],
        )
        .expect("absolute sort")
        .into_values();
        let Value::Tensor(sorted) = sorted else {
            panic!("expected exact integer sorted values");
        };
        assert_eq!(
            sorted.integer_storage(),
            Some(&IntegerStorage::I64(vec![-1, 2, i64::MAX, i64::MIN]))
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected index tensor");
        };
        assert_eq!(indices.materialize_f64(), vec![4.0, 3.0, 2.0, 1.0]);

        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 0, 7, 9_007_199_254_740_993]),
            vec![2, 2],
        )
        .expect("matrix input");
        let (sorted, indices) = evaluate(Value::Tensor(input), &[Value::Int(IntValue::I32(2))])
            .expect("dimension sort")
            .into_values();
        let Value::Tensor(sorted) = sorted else {
            panic!("expected exact integer matrix values");
        };
        assert_eq!(
            sorted.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                7,
                0,
                u64::MAX,
                9_007_199_254_740_993,
            ]))
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected index tensor");
        };
        assert_eq!(indices.materialize_f64(), vec![2.0, 1.0, 1.0, 2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_descend_direction() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let result =
            sort_builtin(Value::Tensor(tensor), vec![Value::from("descend")]).expect("sort");
        match result {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![4.0, 3.0, 2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_matrix_default_dim1() {
        let tensor = Tensor::new(vec![4.0, 2.0, 1.0, 5.0, 6.0, 3.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![2.0, 4.0, 1.0, 5.0, 3.0, 6.0]);
                assert_eq!(t.shape, vec![2, 3]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![2.0, 1.0, 1.0, 2.0, 2.0, 1.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_matrix_along_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 2.0, 2.0, 5.0], vec![2, 3]).unwrap();
        let eval =
            evaluate(Value::Tensor(tensor), &[Value::Int(IntValue::I32(2))]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0]);
                assert_eq!(t.shape, vec![2, 3]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_dimension_placeholder_then_dim() {
        let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 2.0], vec![2, 2]).unwrap();
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[
                Value::Tensor(placeholder),
                Value::Int(IntValue::I32(2)),
                Value::from("descend"),
            ],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![4.0, 3.0, 1.0, 2.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sort_typed_integer_dimension_argument_is_not_empty_placeholder_without_mirror() {
        let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 2.0, 2.0, 5.0], vec![2, 3]).unwrap();
        let dim = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("dimension");

        let eval = evaluate(Value::Tensor(tensor), &[Value::Tensor(dim)]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_descend_then_dimension() {
        let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 2.0, 2.0, 5.0], vec![2, 3]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[Value::from("descend"), Value::Int(IntValue::I32(1))],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![3.0, 1.0, 4.0, 2.0, 5.0, 2.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_returns_indices() {
        let tensor = Tensor::new(vec![4.0, 1.0, 9.0, 2.0], vec![4, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 4.0, 9.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![2.0, 4.0, 1.0, 3.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_with_nan_handling() {
        let tensor = Tensor::new(vec![f64::NAN, 4.0, 1.0, 2.0], vec![4, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert!(t.materialize_f64()[3].is_nan());
                assert_eq!(&t.materialize_f64()[0..3], &[1.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let eval_desc =
            evaluate(Value::Tensor(tensor), &[Value::from("descend")]).expect("evaluate");
        let (sorted_desc, _) = eval_desc.into_values();
        match sorted_desc {
            Value::Tensor(t) => {
                assert!(t.materialize_f64()[0].is_nan());
                assert_eq!(&t.materialize_f64()[1..], &[4.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_by_absolute_value() {
        let tensor = Tensor::new(vec![-8.0, -1.0, 3.0, -2.0], vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[Value::from("ComparisonMethod"), Value::from("abs")],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![-1.0, -2.0, 3.0, -8.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_by_absolute_value_descend() {
        let tensor = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[
                Value::from("descend"),
                Value::from("ComparisonMethod"),
                Value::from("abs"),
            ],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![4.0, -3.0, 2.0, -1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_complex_auto_abs() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 0.5), (0.0, -1.0)], vec![3, 1]).unwrap();
        let eval = evaluate(Value::ComplexTensor(tensor), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::ComplexTensor(t) => {
                assert_eq!(
                    t.materialize_f64(),
                    vec![(0.0, -1.0), (1.0, 2.0), (-3.0, 0.5)]
                )
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![3.0, 1.0, 2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn sort_preserves_native_complex_single_storage_and_indices() {
        let tensor =
            ComplexTensor::from_f32(vec![(3.0, 4.0), (1.0, 0.0), (0.0, 2.0)], vec![3, 1]).unwrap();
        let (sorted, indices) = evaluate(Value::ComplexTensor(tensor), &[])
            .expect("sort")
            .into_values();
        let Value::ComplexTensor(sorted) = sorted else {
            panic!("expected complex single tensor");
        };
        assert_eq!(
            sorted.as_f32_slice(),
            Some(&[(1.0, 0.0), (0.0, 2.0), (3.0, 4.0)][..])
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected index tensor");
        };
        assert_eq!(indices.as_f64_slice(), Some(&[2.0, 3.0, 1.0][..]));
    }

    #[test]
    fn sort_preserves_one_element_complex_single_tensor() {
        let tensor = ComplexTensor::from_f32(vec![(1.25, -2.5)], vec![1, 1]).unwrap();
        let sorted = evaluate(Value::ComplexTensor(tensor), &[])
            .expect("sort")
            .into_sorted_value();
        let Value::ComplexTensor(sorted) = sorted else {
            panic!("expected complex single tensor");
        };
        assert_eq!(sorted.as_f32_slice(), Some(&[(1.25, -2.5)][..]));
    }

    #[test]
    fn sort_orders_typed_complex_uint64_exactly_and_preserves_storage() {
        for input in [
            (
                IntegerStorage::I8(vec![3, 1]),
                IntegerStorage::I8(vec![4, 0]),
            ),
            (
                IntegerStorage::I16(vec![3, 1]),
                IntegerStorage::I16(vec![4, 0]),
            ),
            (
                IntegerStorage::I32(vec![3, 1]),
                IntegerStorage::I32(vec![4, 0]),
            ),
            (
                IntegerStorage::I64(vec![3, 1]),
                IntegerStorage::I64(vec![4, 0]),
            ),
            (
                IntegerStorage::U8(vec![3, 1]),
                IntegerStorage::U8(vec![4, 0]),
            ),
            (
                IntegerStorage::U16(vec![3, 1]),
                IntegerStorage::U16(vec![4, 0]),
            ),
            (
                IntegerStorage::U32(vec![3, 1]),
                IntegerStorage::U32(vec![4, 0]),
            ),
            (
                IntegerStorage::U64(vec![3, 1]),
                IntegerStorage::U64(vec![4, 0]),
            ),
        ] {
            let expected_real = input
                .0
                .from_exact_values_like(vec![
                    input.0.value_at(1).unwrap(),
                    input.0.value_at(0).unwrap(),
                ])
                .unwrap();
            let expected_imag = input
                .1
                .from_exact_values_like(vec![
                    input.1.value_at(1).unwrap(),
                    input.1.value_at(0).unwrap(),
                ])
                .unwrap();
            let tensor = ComplexTensor::new_integer(
                IntegerComplexStorage::new(input.0, input.1).unwrap(),
                vec![2, 1],
            )
            .unwrap();
            let sorted = evaluate(Value::ComplexTensor(tensor), &[])
                .expect("typed complex integer sort")
                .into_sorted_value();
            let Value::ComplexTensor(sorted) = sorted else {
                panic!("expected typed complex integer tensor");
            };
            assert_eq!(
                sorted.integer_storage(),
                Some(&IntegerComplexStorage::new(expected_real, expected_imag).unwrap())
            );
        }

        let input = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, u64::MAX - 1, 0]),
            IntegerStorage::U64(vec![0, 1, u64::MAX]),
        )
        .unwrap();
        let tensor = ComplexTensor::new_integer(input, vec![3, 1]).unwrap();
        let (sorted, indices) = evaluate(Value::ComplexTensor(tensor), &[])
            .expect("typed complex integer sort")
            .into_values();
        let Value::ComplexTensor(sorted) = sorted else {
            panic!("expected typed complex integer tensor");
        };
        assert_eq!(
            sorted.integer_storage(),
            Some(
                &IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX, 0]),
                    IntegerStorage::U64(vec![1, 0, u64::MAX]),
                )
                .unwrap()
            )
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected double indices");
        };
        assert_eq!(indices.as_f64_slice(), Some(&[2.0, 1.0, 3.0][..]));
    }

    #[test]
    fn sort_restores_exact_typed_complex_integer_results_to_the_input_provider() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX, u64::MAX - 1]),
                    IntegerStorage::U64(vec![0, 1]),
                )
                .unwrap(),
                vec![2, 1],
            )
            .unwrap();
            let handle =
                crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &input)
                    .expect("typed complex upload");
            let (sorted, indices) = evaluate(Value::GpuTensor(handle), &[])
                .expect("resident typed complex sort")
                .into_values();
            let sorted = block_on(crate::builtins::common::gpu_helpers::gather_value_async(
                &sorted,
            ))
            .expect("gather sorted values");
            let indices = test_support::gather(indices).expect("gather sorted indices");
            let Value::ComplexTensor(sorted) = sorted else {
                panic!("expected typed complex integer tensor");
            };
            assert_eq!(
                sorted.integer_storage(),
                Some(
                    &IntegerComplexStorage::new(
                        IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]),
                        IntegerStorage::U64(vec![1, 0]),
                    )
                    .unwrap()
                )
            );
            assert_eq!(indices.as_f64_slice(), Some(&[2.0, 1.0][..]));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_complex_real_descend() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 0.0), (1.0, -1.0)], vec![3, 1]).unwrap();
        let eval = evaluate(
            Value::ComplexTensor(tensor),
            &[
                Value::from("descend"),
                Value::from("ComparisonMethod"),
                Value::from("real"),
            ],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::ComplexTensor(t) => {
                assert_eq!(
                    t.materialize_f64(),
                    vec![(1.0, 2.0), (1.0, -1.0), (-3.0, 0.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_stable_with_duplicates() {
        let tensor = Tensor::new(vec![2.0, 2.0, 1.0, 2.0], vec![4, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 2.0, 2.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![3.0, 1.0, 2.0, 4.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn sort_abs_ties_follow_phase_for_real_integer_and_complex_values() {
        let args = [Value::from("ComparisonMethod"), Value::from("abs")];
        let (real, real_indices) = evaluate(
            Value::Tensor(Tensor::new(vec![-3.0, 3.0], vec![2, 1]).expect("real")),
            &args,
        )
        .expect("real sort")
        .into_values();
        let Value::Tensor(real) = real else {
            panic!("expected real tensor");
        };
        assert_eq!(real.materialize_f64(), vec![3.0, -3.0]);
        let Value::Tensor(real_indices) = real_indices else {
            panic!("expected real indices");
        };
        assert_eq!(real_indices.materialize_f64(), vec![2.0, 1.0]);

        let (integer, _) = evaluate(
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I64(vec![-3, 3]), vec![2, 1]).expect("integer"),
            ),
            &args,
        )
        .expect("integer sort")
        .into_values();
        let Value::Tensor(integer) = integer else {
            panic!("expected integer tensor");
        };
        assert_eq!(
            integer.integer_storage(),
            Some(&IntegerStorage::I64(vec![3, -3]))
        );

        let (complex, complex_indices) = evaluate(
            Value::ComplexTensor(
                ComplexTensor::new(vec![(-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)], vec![3, 1])
                    .expect("complex"),
            ),
            &[],
        )
        .expect("complex sort")
        .into_values();
        let Value::ComplexTensor(complex) = complex else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            complex.materialize_f64(),
            vec![(0.0, -1.0), (0.0, 1.0), (-1.0, 0.0)]
        );
        let Value::Tensor(complex_indices) = complex_indices else {
            panic!("expected complex indices");
        };
        assert_eq!(complex_indices.materialize_f64(), vec![3.0, 2.0, 1.0]);

        let (phase_boundary, phase_boundary_indices) = evaluate(
            Value::ComplexTensor(
                ComplexTensor::new(vec![(-1.0, 0.0), (-1.0, -0.0)], vec![2, 1])
                    .expect("phase boundary"),
            ),
            &[],
        )
        .expect("phase-boundary sort")
        .into_values();
        let Value::ComplexTensor(phase_boundary) = phase_boundary else {
            panic!("expected complex phase-boundary tensor");
        };
        assert_eq!(
            phase_boundary.materialize_f64(),
            vec![(-1.0, 0.0), (-1.0, -0.0)]
        );
        let Value::Tensor(phase_boundary_indices) = phase_boundary_indices else {
            panic!("expected complex phase-boundary indices");
        };
        assert_eq!(phase_boundary_indices.materialize_f64(), vec![1.0, 2.0]);
    }

    #[test]
    fn sort_preserves_logical_class_and_accepts_every_integer_dimension_class() {
        let (logical, logical_indices) = evaluate(
            Value::LogicalArray(LogicalArray::new(vec![1, 0, 1], vec![3, 1]).expect("logical")),
            &[],
        )
        .expect("logical sort")
        .into_values();
        let Value::LogicalArray(logical) = logical else {
            panic!("expected logical array");
        };
        assert_eq!(logical.data, vec![0, 1, 1]);
        let Value::Tensor(logical_indices) = logical_indices else {
            panic!("expected logical indices");
        };
        assert_eq!(logical_indices.materialize_f64(), vec![2.0, 1.0, 3.0]);

        for dimension in [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ] {
            let input = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).expect("input");
            let dim = Value::Tensor(Tensor::new_integer(dimension, vec![1, 1]).expect("dimension"));
            let sorted = evaluate(Value::Tensor(input), &[dim])
                .expect("typed dimension")
                .into_sorted_value();
            let Value::Tensor(sorted) = sorted else {
                panic!("expected tensor");
            };
            assert_eq!(sorted.materialize_f64(), vec![2.0, 1.0, 4.0, 3.0]);
        }
    }

    #[test]
    fn sort_rejects_more_than_two_outputs() {
        let _outputs = crate::output_count::push_output_count(Some(3));
        let error = sort_builtin(
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![2, 1]).expect("input")),
            Vec::new(),
        )
        .expect_err("too many outputs");
        assert!(error.message().contains("maximum is 2"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_empty_tensor() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert!(t.materialize_f64().is_empty());
                assert_eq!(t.shape, tensor.shape);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert!(t.materialize_f64().is_empty()),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_dim_greater_than_ndims() {
        let tensor = Tensor::new(vec![4.0, 2.0, 3.0, 1.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor.clone()),
            &[Value::Int(IntValue::I32(3))],
        )
        .expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), tensor.materialize_f64()),
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert!(t
                .materialize_f64()
                .iter()
                .all(|v| (*v - 1.0).abs() < f64::EPSILON)),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_supports_missing_placement_and_rejects_unknown_values() {
        let input =
            Value::Tensor(Tensor::new(vec![2.0, f64::NAN, 1.0], vec![3, 1]).expect("input"));
        let first = sort_builtin(
            input.clone(),
            vec![Value::from("MissingPlacement"), Value::from("first")],
        )
        .expect("missing first");
        let Value::Tensor(first) = first else {
            panic!("expected tensor");
        };
        assert!(first.materialize_f64()[0].is_nan());
        assert_eq!(&first.materialize_f64()[1..], &[1.0, 2.0]);

        let last_descending = sort_builtin(
            input,
            vec![
                Value::from("descend"),
                Value::from("MissingPlacement"),
                Value::from("last"),
            ],
        )
        .expect("missing last");
        let Value::Tensor(last_descending) = last_descending else {
            panic!("expected tensor");
        };
        assert_eq!(&last_descending.materialize_f64()[..2], &[2.0, 1.0]);
        assert!(last_descending.materialize_f64()[2].is_nan());

        let err = sort_builtin(
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            vec![Value::from("MissingPlacement"), Value::from("middle")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            SORT_ERROR_MISSINGPLACEMENT_UNKNOWN.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_invalid_comparison_method_errors() {
        let err = sort_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            vec![Value::from("ComparisonMethod"), Value::from("unknown")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            SORT_ERROR_COMPARISON_METHOD_UNKNOWN.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_invalid_comparison_method_value_errors() {
        let err = sort_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            vec![
                Value::from("ComparisonMethod"),
                Value::Int(IntValue::I32(1)),
            ],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            SORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_dimension_zero_errors() {
        let err = sort_builtin(
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            vec![Value::Num(0.0)],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), SORT_ERROR_INVALID_DIMENSION.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_gpu_round_trip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("evaluate");
            let (sorted, indices) = eval.into_values();
            assert!(matches!(sorted, Value::GpuTensor(_)));
            assert!(matches!(indices, Value::GpuTensor(_)));
            let sorted = test_support::gather(sorted).expect("gather sorted");
            assert_eq!(sorted.materialize_f64(), vec![1.0, 2.0, 3.0]);
            let indices = test_support::gather(indices).expect("gather indices");
            assert_eq!(indices.materialize_f64(), vec![2.0, 3.0, 1.0]);
        });
    }

    #[test]
    fn sort_gpu_preserves_exact_wide_integer_and_logical_residency() {
        test_support::with_test_provider(|provider| {
            let integer = Tensor::new_integer(
                IntegerStorage::U64(vec![u64::MAX, 0, 9_007_199_254_740_993]),
                vec![3, 1],
            )
            .expect("integer");
            let handle = gpu_helpers::upload_tensor(provider, &integer).expect("upload integer");
            let (sorted, indices) = evaluate(Value::GpuTensor(handle), &[])
                .expect("integer GPU sort")
                .into_values();
            assert!(matches!(sorted, Value::GpuTensor(_)));
            assert!(matches!(indices, Value::GpuTensor(_)));
            let sorted = test_support::gather(sorted).expect("gather integer");
            assert_eq!(
                sorted.into_numeric_storage().expect("integer storage"),
                NumericStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX])
            );

            let logical = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0, 0.0, 1.0],
                    shape: &[3, 1],
                })
                .expect("upload logical");
            runmat_accelerate_api::set_handle_logical(&logical, true);
            let (sorted, _) = evaluate(Value::GpuTensor(logical), &[])
                .expect("logical GPU sort")
                .into_values();
            let Value::GpuTensor(sorted_handle) = &sorted else {
                panic!("expected resident logical output");
            };
            assert!(runmat_accelerate_api::handle_is_logical(sorted_handle));
            let sorted = test_support::gather(sorted).expect("gather logical");
            assert_eq!(sorted.materialize_f64(), vec![0.0, 1.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sort_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![4.0, 1.0, 3.0, 2.0], vec![4, 1]).unwrap();
        let cpu_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cpu sort");
        let (cpu_sorted, cpu_indices) = cpu_eval.into_values();

        let gpu_view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&gpu_view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("gpu sort");
        let (gpu_sorted, gpu_indices) = gpu_eval.into_values();

        let cpu_sorted_tensor = match cpu_sorted {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected CPU sorted value {other:?}"),
        };
        let cpu_indices_tensor = match cpu_indices {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected CPU indices value {other:?}"),
        };
        assert!(matches!(gpu_sorted, Value::GpuTensor(_)));
        assert!(matches!(gpu_indices, Value::GpuTensor(_)));
        let gpu_sorted_tensor = test_support::gather(gpu_sorted).expect("gather GPU sorted");
        let gpu_indices_tensor = test_support::gather(gpu_indices).expect("gather GPU indices");

        assert_eq!(
            gpu_sorted_tensor.materialize_f64(),
            cpu_sorted_tensor.materialize_f64()
        );
        assert_eq!(
            gpu_indices_tensor.materialize_f64(),
            cpu_indices_tensor.materialize_f64()
        );
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn sort_wgpu_abs_ties_follow_phase() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let input = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[-3.0, 3.0],
                shape: &[2, 1],
            })
            .expect("upload");
        let (sorted, indices) = evaluate(
            Value::GpuTensor(input),
            &[Value::from("ComparisonMethod"), Value::from("abs")],
        )
        .expect("wgpu sort")
        .into_values();
        let sorted = test_support::gather(sorted).expect("gather sorted");
        assert_eq!(sorted.materialize_f64(), vec![3.0, -3.0]);
        let indices = test_support::gather(indices).expect("gather indices");
        assert_eq!(indices.materialize_f64(), vec![2.0, 1.0]);
    }
}
