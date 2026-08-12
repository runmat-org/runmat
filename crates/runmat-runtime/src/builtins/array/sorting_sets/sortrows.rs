//! MATLAB-compatible `sortrows` builtin with GPU-aware semantics.

use std::cmp::Ordering;

use runmat_accelerate_api::{
    GpuTensorHandle, SortComparison as ProviderSortComparison, SortOrder as ProviderSortOrder,
    SortResult as ProviderSortResult, SortRowsColumnSpec as ProviderSortRowsColumnSpec,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CharArray, ComplexStorage, ComplexTensor, IntegerStorage,
    LogicalArray, NumericScalar, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::{float_order::SetFloat, integer_order, type_resolvers::tensor_output_type};
use crate::build_runtime_error;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::sortrows")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sortrows",
    op_kind: GpuOpKind::Custom("sortrows"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("sortrows")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Providers may implement a plain-real row-sort kernel; typed fallback preserves supported integer/logical storage and registered outputs return to the input handle's owner.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::sortrows"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sortrows",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`sortrows` terminates fusion chains and acts as a residency sink; unsupported provider forms use typed host fallback.",
};

const BUILTIN_NAME: &str = "sortrows";

const SORTROWS_OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sorted input rows.",
}];

const SORTROWS_OUTPUT_BI: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sorted input rows.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Permutation indices mapping sorted rows to original rows.",
    },
];

const SORTROWS_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix to sort by rows.",
}];

const SORTROWS_INPUTS_A_COLUMNS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to sort by rows.",
    },
    BuiltinParamDescriptor {
        name: "column",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column specification vector (negative entries request descending order).",
    },
];

const SORTROWS_INPUTS_A_DIRECTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to sort by rows.",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ascend\""),
        description: "Global row direction override: 'ascend' or 'descend'.",
    },
];

const SORTROWS_INPUTS_A_COLUMNS_DIRECTION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to sort by rows.",
    },
    BuiltinParamDescriptor {
        name: "column",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column specification vector (negative entries request descending order).",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ascend\""),
        description: "Global row direction override: 'ascend' or 'descend'.",
    },
];

const SORTROWS_INPUTS_COMPARISON_METHOD: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to sort by rows.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional column and direction arguments.",
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

const SORTROWS_INPUTS_MISSING_PLACEMENT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to sort by rows.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional column and direction arguments.",
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
        description: "NaN placement policy: 'auto', 'first', or 'last'.",
    },
];

const SORTROWS_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A)",
        inputs: &SORTROWS_INPUTS_A,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A, column)",
        inputs: &SORTROWS_INPUTS_A_COLUMNS,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A, direction)",
        inputs: &SORTROWS_INPUTS_A_DIRECTION,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A, column, direction)",
        inputs: &SORTROWS_INPUTS_A_COLUMNS_DIRECTION,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A, ..., \"ComparisonMethod\", method)",
        inputs: &SORTROWS_INPUTS_COMPARISON_METHOD,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A, ..., \"MissingPlacement\", placement)",
        inputs: &SORTROWS_INPUTS_MISSING_PLACEMENT,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A)",
        inputs: &SORTROWS_INPUTS_A,
        outputs: &SORTROWS_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A, column)",
        inputs: &SORTROWS_INPUTS_A_COLUMNS,
        outputs: &SORTROWS_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A, direction)",
        inputs: &SORTROWS_INPUTS_A_DIRECTION,
        outputs: &SORTROWS_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A, column, direction)",
        inputs: &SORTROWS_INPUTS_A_COLUMNS_DIRECTION,
        outputs: &SORTROWS_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A, ..., \"ComparisonMethod\", method)",
        inputs: &SORTROWS_INPUTS_COMPARISON_METHOD,
        outputs: &SORTROWS_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A, ..., \"MissingPlacement\", placement)",
        inputs: &SORTROWS_INPUTS_MISSING_PLACEMENT,
        outputs: &SORTROWS_OUTPUT_BI,
    },
];

const SORTROWS_ERROR_INVALID_COLUMN_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.INVALID_COLUMN_INDEX",
    identifier: Some("RunMat:sortrows:InvalidColumnIndex"),
    when: "Column specification indices are out of range, zero, or otherwise invalid.",
    message: "sortrows: invalid column index",
};

const SORTROWS_ERROR_MISSING_PLACEMENT_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.MISSING_PLACEMENT_UNKNOWN",
    identifier: Some("RunMat:sortrows:MissingPlacementUnknown"),
    when: "MissingPlacement option value is unsupported.",
    message: "sortrows: unsupported MissingPlacement value",
};

const SORTROWS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.INVALID_ARGUMENT",
    identifier: Some("RunMat:sortrows:InvalidArgument"),
    when: "Option parsing receives invalid argument kinds or malformed name-value pairs.",
    message: "sortrows: invalid argument",
};

const SORTROWS_ERROR_COMPARISON_METHOD_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.COMPARISON_METHOD_UNKNOWN",
    identifier: Some("RunMat:sortrows:ComparisonMethodUnknown"),
    when: "ComparisonMethod option value is unsupported.",
    message: "sortrows: unsupported ComparisonMethod value",
};

const SORTROWS_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:sortrows:UnsupportedInputType"),
    when: "Input cannot be converted to numeric, logical, complex, or char matrix domain.",
    message: "sortrows: unsupported input type",
};

const SORTROWS_ERROR_MATRIX_REQUIRED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.MATRIX_REQUIRED",
    identifier: Some("RunMat:sortrows:MatrixRequired"),
    when: "Input has rank greater than 2 where matrix input is required.",
    message: "sortrows: input must be a 2-D matrix",
};

const SORTROWS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.INTERNAL",
    identifier: Some("RunMat:sortrows:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "sortrows: internal operation failed",
};

const SORTROWS_ERRORS: [BuiltinErrorDescriptor; 7] = [
    SORTROWS_ERROR_INVALID_COLUMN_INDEX,
    SORTROWS_ERROR_MISSING_PLACEMENT_UNKNOWN,
    SORTROWS_ERROR_INVALID_ARGUMENT,
    SORTROWS_ERROR_COMPARISON_METHOD_UNKNOWN,
    SORTROWS_ERROR_UNSUPPORTED_INPUT_TYPE,
    SORTROWS_ERROR_MATRIX_REQUIRED,
    SORTROWS_ERROR_INTERNAL,
];

const SORTROWS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented sortable matrix domain includes all eight real integer classes.",
    },
    BuiltinIntegerInputCapability {
        name: "column",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented nonzero integer column scalar or vector accepts every integer class and integer-valued double.",
    },
];

const SORTROWS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[B, I] = sortrows(integer_A, integer_columns, direction, options)",
        inputs: &SORTROWS_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "B preserves A's exact integer class and stable equal-row order; optional I is one-based double. Resident integer input uses exact typed gather fallback and restores both outputs to the owning provider.",
    }];

pub const SORTROWS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SORTROWS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SORTROWS_ERRORS,
};

fn sortrows_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sortrows_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    sortrows_error_with(error, error.message)
}

fn sortrows_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    sortrows_error_with(&SORTROWS_ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "sortrows",
    category = "array/sorting_sets",
    summary = "Sort matrix rows lexicographically with column and direction controls.",
    keywords = "sortrows,row sort,lexicographic,gpu",
    accel = "sink",
    sink = true,
    type_resolver(tensor_output_type),
    descriptor(crate::builtins::array::sorting_sets::sortrows::SORTROWS_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::array::sorting_sets::sortrows::SORTROWS_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::array::sorting_sets::sortrows"
)]
async fn sortrows_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 2) {
        return Err(sortrows_error_with(
            &SORTROWS_ERROR_INVALID_ARGUMENT,
            "sortrows: too many output arguments; maximum is 2",
        ));
    }
    let provider = super::output_provider(&value);
    let eval = evaluate(value, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let (sorted, indices) = eval.into_values();
        let outputs = if out_count == 1 {
            vec![sorted]
        } else {
            vec![sorted, indices]
        };
        return Ok(Value::OutputList(super::restore_set_outputs(
            provider,
            BUILTIN_NAME,
            outputs,
            sortrows_internal_error,
        )?));
    }
    let mut outputs = super::restore_set_outputs(
        provider,
        BUILTIN_NAME,
        vec![eval.into_sorted_value()],
        sortrows_internal_error,
    )?;
    Ok(outputs.pop().expect("sortrows output"))
}

/// Evaluate the `sortrows` builtin once and expose both outputs.
pub async fn evaluate(value: Value, rest: &[Value]) -> crate::BuiltinResult<SortRowsEvaluation> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, BUILTIN_NAME)?;
    match value {
        Value::GpuTensor(handle) => sortrows_gpu(handle, rest).await,
        other => sortrows_host(other, rest),
    }
}

async fn sortrows_gpu(
    handle: GpuTensorHandle,
    rest: &[Value],
) -> crate::BuiltinResult<SortRowsEvaluation> {
    ensure_matrix_shape(&handle.shape)?;
    let (_, cols) = rows_cols_from_shape(&handle.shape);
    let args = SortRowsArgs::parse(rest, cols)?;

    let plain_real = runmat_accelerate_api::handle_integer_type(&handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(&handle)
        && runmat_accelerate_api::handle_storage(&handle)
            == runmat_accelerate_api::GpuTensorStorage::Real;
    if plain_real && args.missing_is_auto() {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle)
            .or_else(runmat_accelerate_api::provider)
        {
            let provider_columns = args.to_provider_columns();
            let provider_comparison = args.provider_comparison();
            match provider
                .sort_rows(&handle, &provider_columns, provider_comparison)
                .await
            {
                Ok(result) => return sortrows_from_provider_result(result),
                Err(_err) => {
                    // fall back to host path when provider cannot service the request
                }
            }
        }
    }

    let value = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
    sortrows_host_with_args(value, &args)
}

fn sortrows_from_provider_result(
    result: ProviderSortResult,
) -> crate::BuiltinResult<SortRowsEvaluation> {
    let sorted_tensor = Tensor::new(result.values.data, result.values.shape)
        .map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))?;
    let indices_tensor = Tensor::new(result.indices.data, result.indices.shape)
        .map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))?;
    Ok(SortRowsEvaluation {
        sorted: tensor::tensor_into_value(sorted_tensor),
        indices: indices_tensor,
    })
}

fn sortrows_host(value: Value, rest: &[Value]) -> crate::BuiltinResult<SortRowsEvaluation> {
    if matches!(&value, Value::Object(obj) if obj.is_class(crate::builtins::table::TABLE_CLASS)) {
        let (sorted, indices) = crate::builtins::table::sortrows_table(value, rest)?;
        return Ok(SortRowsEvaluation::from_parts(sorted, indices));
    }
    let shape = value_shape(&value);
    ensure_matrix_shape(&shape)?;
    let (_, cols) = rows_cols_from_shape(&shape);
    let args = SortRowsArgs::parse(rest, cols)?;
    sortrows_host_with_args(value, &args)
}

fn sortrows_host_with_args(
    value: Value,
    args: &SortRowsArgs,
) -> crate::BuiltinResult<SortRowsEvaluation> {
    match value {
        Value::Tensor(tensor) => sortrows_real_tensor_with_args(tensor, args),
        Value::LogicalArray(logical) => sortrows_logical_with_args(logical, args),
        Value::Bool(value) => {
            let logical = LogicalArray::new(vec![u8::from(value)], vec![1, 1])
                .map_err(|error| sortrows_internal_error(format!("sortrows: {error}")))?;
            sortrows_logical_with_args(logical, args)
        }
        Value::Num(_) | Value::Int(_) => {
            let tensor = tensor::value_into_tensor_for("sortrows", value)
                .map_err(|e| sortrows_internal_error(e))?;
            sortrows_real_tensor_with_args(tensor, args)
        }
        Value::ComplexTensor(ct) => sortrows_complex_tensor_with_args(ct, args),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))?;
            sortrows_complex_tensor_with_args(tensor, args)
        }
        Value::CharArray(ca) => sortrows_char_array_with_args(ca, args),
        other => Err(sortrows_error_with(
            &SORTROWS_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!(
                "sortrows: unsupported input type {:?}; expected numeric, logical, complex, or char arrays",
                other
            ),
        )
        .into()),
    }
}

fn value_shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(tensor) => tensor.shape.clone(),
        Value::LogicalArray(logical) => logical.shape.clone(),
        Value::ComplexTensor(tensor) => tensor.shape.clone(),
        Value::CharArray(array) => array.shape.clone(),
        Value::GpuTensor(handle) => handle.shape.clone(),
        _ => vec![1, 1],
    }
}

fn sortrows_logical_with_args(
    logical: LogicalArray,
    args: &SortRowsArgs,
) -> crate::BuiltinResult<SortRowsEvaluation> {
    let shape = logical.shape.clone();
    let tensor = Tensor::new_integer(IntegerStorage::U8(logical.data), shape)
        .map_err(|error| sortrows_internal_error(format!("sortrows: {error}")))?;
    let evaluation = sortrows_real_tensor_with_args(tensor, args)?;
    let sorted = match evaluation.sorted {
        Value::Int(integer) => Value::Bool(integer.to_i64() != 0),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let storage = tensor
                .into_numeric_storage()
                .map_err(|error| sortrows_internal_error(format!("sortrows: {error}")))?
                .into_integer_storage()
                .map_err(|_| sortrows_internal_error("sortrows: logical output lost storage"))?;
            let IntegerStorage::U8(values) = storage else {
                return Err(sortrows_internal_error(
                    "sortrows: logical output changed storage class",
                ));
            };
            Value::LogicalArray(
                LogicalArray::new(values, shape)
                    .map_err(|error| sortrows_internal_error(format!("sortrows: {error}")))?,
            )
        }
        other => {
            return Err(sortrows_internal_error(format!(
                "sortrows: unexpected logical output {other:?}"
            )))
        }
    };
    Ok(SortRowsEvaluation {
        sorted,
        indices: evaluation.indices,
    })
}

fn sortrows_real_tensor_with_args(
    tensor: Tensor,
    args: &SortRowsArgs,
) -> crate::BuiltinResult<SortRowsEvaluation> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))?;

    if rows <= 1 || cols == 0 || storage.is_empty() || args.columns.is_empty() {
        let tensor = Tensor::from_numeric_storage(storage, shape)
            .map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))?;
        let indices = identity_indices(rows)?;
        return Ok(SortRowsEvaluation {
            sorted: tensor::tensor_into_value(tensor),
            indices,
        });
    }

    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| compare_numeric_rows(&storage, rows, cols, args, a, b));

    let sorted_tensor = reorder_numeric_rows(storage, shape, rows, cols, &order)?;
    let indices = permutation_indices(&order)?;
    Ok(SortRowsEvaluation {
        sorted: tensor::tensor_into_value(sorted_tensor),
        indices,
    })
}

fn sortrows_complex_tensor_with_args(
    tensor: ComplexTensor,
    args: &SortRowsArgs,
) -> crate::BuiltinResult<SortRowsEvaluation> {
    let rows = tensor.rows;
    let cols = tensor.cols;
    let shape = tensor.shape.clone();
    let storage = tensor.into_complex_storage();

    if rows <= 1 || cols == 0 || storage.is_empty() || args.columns.is_empty() {
        let indices = identity_indices(rows)?;
        let tensor = ComplexTensor::from_complex_storage(storage, shape)
            .map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))?;
        return Ok(SortRowsEvaluation {
            sorted: complex_tensor_into_value(tensor),
            indices,
        });
    }

    let mut order: Vec<usize> = (0..rows).collect();
    match &storage {
        ComplexStorage::F64(values) => {
            order.sort_by(|&a, &b| compare_complex_rows(values, rows, cols, args, a, b));
        }
        ComplexStorage::F32(values) => {
            order.sort_by(|&a, &b| compare_complex_rows(values, rows, cols, args, a, b));
        }
        ComplexStorage::Integer(_) => {
            sortrows_promoted_complex_integer_order(&storage, rows, cols, args, &mut order);
        }
    }

    let sorted_tensor = reorder_complex_rows(storage, shape, rows, cols, &order)?;
    let indices = permutation_indices(&order)?;
    Ok(SortRowsEvaluation {
        sorted: complex_tensor_into_value(sorted_tensor),
        indices,
    })
}

fn sortrows_promoted_complex_integer_order(
    storage: &ComplexStorage,
    rows: usize,
    cols: usize,
    args: &SortRowsArgs,
    order: &mut [usize],
) {
    let values = storage.materialize_f64();
    order.sort_by(|&a, &b| compare_complex_rows(&values, rows, cols, args, a, b));
}

fn sortrows_char_array_with_args(
    ca: CharArray,
    args: &SortRowsArgs,
) -> crate::BuiltinResult<SortRowsEvaluation> {
    let rows = ca.rows;
    let cols = ca.cols;

    if rows <= 1 || cols == 0 || ca.data.is_empty() || args.columns.is_empty() {
        let indices = identity_indices(rows)?;
        return Ok(SortRowsEvaluation {
            sorted: Value::CharArray(ca),
            indices,
        });
    }

    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| compare_char_rows(&ca, args, a, b));

    let sorted = reorder_char_rows(&ca, rows, cols, &order)?;
    let indices = permutation_indices(&order)?;
    Ok(SortRowsEvaluation {
        sorted: Value::CharArray(sorted),
        indices,
    })
}

fn ensure_matrix_shape(shape: &[usize]) -> crate::BuiltinResult<()> {
    if shape.len() <= 2 {
        Ok(())
    } else {
        Err(sortrows_error(&SORTROWS_ERROR_MATRIX_REQUIRED))
    }
}

fn rows_cols_from_shape(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0]),
        _ => (shape[0], shape[1]),
    }
}

fn compare_numeric_rows(
    storage: &NumericStorage,
    rows: usize,
    cols: usize,
    args: &SortRowsArgs,
    a: usize,
    b: usize,
) -> Ordering {
    for spec in &args.columns {
        if spec.index >= cols {
            continue;
        }
        let idx_a = a + spec.index * rows;
        let idx_b = b + spec.index * rows;
        let va = storage
            .value_at(idx_a)
            .expect("validated sortrows numeric index");
        let vb = storage
            .value_at(idx_b)
            .expect("validated sortrows numeric index");
        let ord = compare_numeric_scalars(va, vb, spec.direction, args);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_numeric_scalars(
    a: NumericScalar,
    b: NumericScalar,
    direction: SortDirection,
    args: &SortRowsArgs,
) -> Ordering {
    match (a, b) {
        (NumericScalar::F64(a), NumericScalar::F64(b)) => compare_real_scalars(
            a,
            b,
            direction,
            args.comparison,
            args.missing_for_direction(direction),
        ),
        (NumericScalar::F32(a), NumericScalar::F32(b)) => compare_real_scalars(
            a,
            b,
            direction,
            args.comparison,
            args.missing_for_direction(direction),
        ),
        (a, b) => {
            let a = a
                .into_int_value()
                .expect("homogeneous numeric storage has matching scalar classes");
            let b = b
                .into_int_value()
                .expect("homogeneous numeric storage has matching scalar classes");
            integer_order::compare(
                &a,
                &b,
                matches!(direction, SortDirection::Descend),
                matches!(args.comparison, ComparisonMethod::Abs),
            )
        }
    }
}

fn reorder_numeric_rows(
    storage: NumericStorage,
    shape: Vec<usize>,
    rows: usize,
    cols: usize,
    order: &[usize],
) -> crate::BuiltinResult<Tensor> {
    let mut source_indices = Vec::with_capacity(storage.len());
    for col in 0..cols {
        source_indices.extend(order.iter().map(|&src_row| src_row + col * rows));
    }
    let sorted = storage
        .reorder(&source_indices)
        .map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))?;
    Tensor::from_numeric_storage(sorted, shape)
        .map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))
}

fn numeric_column_to_i64(value: NumericScalar) -> crate::BuiltinResult<i64> {
    match value {
        NumericScalar::F64(value) => floating_column_to_i64(value),
        NumericScalar::F32(value) => floating_column_to_i64(f64::from(value)),
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_i64())
            .ok_or_else(|| {
                sortrows_error_with(
                    &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
                    "sortrows: column indices must fit signed integer range",
                )
            }),
    }
}

fn floating_column_to_i64(value: f64) -> crate::BuiltinResult<i64> {
    if !value.is_finite() {
        return Err(sortrows_error_with(
            &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
            "sortrows: column indices must be finite",
        ));
    }
    let rounded = value.round();
    if rounded != value {
        return Err(sortrows_error_with(
            &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
            "sortrows: column indices must be integers",
        ));
    }
    float_to_i64_column(rounded).ok_or_else(|| {
        sortrows_error_with(
            &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
            "sortrows: column indices must fit signed integer range",
        )
    })
}

fn compare_complex_rows<T: SetFloat>(
    values: &[(T, T)],
    rows: usize,
    cols: usize,
    args: &SortRowsArgs,
    a: usize,
    b: usize,
) -> Ordering {
    for spec in &args.columns {
        if spec.index >= cols {
            continue;
        }
        let idx_a = a + spec.index * rows;
        let idx_b = b + spec.index * rows;
        let va = values[idx_a];
        let vb = values[idx_b];
        let missing = args.missing_for_direction(spec.direction);
        let ord = compare_complex_scalars(va, vb, spec.direction, args.comparison, missing);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_char_rows(ca: &CharArray, args: &SortRowsArgs, a: usize, b: usize) -> Ordering {
    for spec in &args.columns {
        if spec.index >= ca.cols {
            continue;
        }
        let idx_a = a * ca.cols + spec.index;
        let idx_b = b * ca.cols + spec.index;
        let va = ca.data[idx_a];
        let vb = ca.data[idx_b];
        let ord = match spec.direction {
            SortDirection::Ascend => va.cmp(&vb),
            SortDirection::Descend => vb.cmp(&va),
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn reorder_complex_rows(
    storage: ComplexStorage,
    shape: Vec<usize>,
    rows: usize,
    cols: usize,
    order: &[usize],
) -> crate::BuiltinResult<ComplexTensor> {
    let mut source_indices = Vec::with_capacity(storage.len());
    for col in 0..cols {
        source_indices.extend(order.iter().map(|&src_row| src_row + col * rows));
    }
    let sorted = storage
        .gather(&source_indices)
        .map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))?;
    ComplexTensor::from_complex_storage(sorted, shape)
        .map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))
}

fn reorder_char_rows(
    ca: &CharArray,
    rows: usize,
    cols: usize,
    order: &[usize],
) -> crate::BuiltinResult<CharArray> {
    let mut data = vec!['\0'; ca.data.len()];
    for (dest_row, &src_row) in order.iter().enumerate() {
        for col in 0..cols {
            let src_idx = src_row * cols + col;
            let dst_idx = dest_row * cols + col;
            data[dst_idx] = ca.data[src_idx];
        }
    }
    CharArray::new(data, rows, cols).map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))
}

fn compare_real_scalars<T: SetFloat>(
    a: T,
    b: T,
    direction: SortDirection,
    comparison: ComparisonMethod,
    missing: MissingPlacementResolved,
) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => match missing {
            MissingPlacementResolved::First => Ordering::Less,
            MissingPlacementResolved::Last => Ordering::Greater,
        },
        (false, true) => match missing {
            MissingPlacementResolved::First => Ordering::Greater,
            MissingPlacementResolved::Last => Ordering::Less,
        },
        (false, false) => compare_real_finite_scalars(a, b, direction, comparison),
    }
}

fn compare_real_finite_scalars<T: SetFloat>(
    a: T,
    b: T,
    direction: SortDirection,
    comparison: ComparisonMethod,
) -> Ordering {
    if matches!(comparison, ComparisonMethod::Abs) {
        let abs_cmp = a.abs().compare(b.abs());
        if abs_cmp != Ordering::Equal {
            return match direction {
                SortDirection::Ascend => abs_cmp,
                SortDirection::Descend => abs_cmp.reverse(),
            };
        }
    }
    let ordering = if matches!(comparison, ComparisonMethod::Abs) {
        b.compare(a)
    } else {
        a.compare(b)
    };
    match direction {
        SortDirection::Ascend => ordering,
        SortDirection::Descend => ordering.reverse(),
    }
}

fn compare_complex_scalars<T: SetFloat>(
    a: (T, T),
    b: (T, T),
    direction: SortDirection,
    comparison: ComparisonMethod,
    missing: MissingPlacementResolved,
) -> Ordering {
    match (complex_is_nan(a), complex_is_nan(b)) {
        (true, true) => Ordering::Equal,
        (true, false) => match missing {
            MissingPlacementResolved::First => Ordering::Less,
            MissingPlacementResolved::Last => Ordering::Greater,
        },
        (false, true) => match missing {
            MissingPlacementResolved::First => Ordering::Greater,
            MissingPlacementResolved::Last => Ordering::Less,
        },
        (false, false) => compare_complex_finite_scalars(a, b, direction, comparison),
    }
}

fn compare_complex_finite_scalars<T: SetFloat>(
    a: (T, T),
    b: (T, T),
    direction: SortDirection,
    comparison: ComparisonMethod,
) -> Ordering {
    match comparison {
        ComparisonMethod::Real => compare_complex_real_first(a, b, direction),
        ComparisonMethod::Auto | ComparisonMethod::Abs => {
            let abs_cmp = complex_abs(a).compare(complex_abs(b));
            if abs_cmp != Ordering::Equal {
                return match direction {
                    SortDirection::Ascend => abs_cmp,
                    SortDirection::Descend => abs_cmp.reverse(),
                };
            }
            compare_complex_phase(a, b, direction)
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

fn compare_complex_real_first<T: SetFloat>(
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

fn permutation_indices(order: &[usize]) -> crate::BuiltinResult<Tensor> {
    let rows = order.len();
    let mut data = Vec::with_capacity(rows);
    for &idx in order {
        data.push((idx + 1) as f64);
    }
    Tensor::new(data, vec![rows, 1]).map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))
}

fn identity_indices(rows: usize) -> crate::BuiltinResult<Tensor> {
    let mut data = Vec::with_capacity(rows);
    for i in 0..rows {
        data.push((i + 1) as f64);
    }
    Tensor::new(data, vec![rows, 1]).map_err(|e| sortrows_internal_error(format!("sortrows: {e}")))
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.as_f32_slice().is_some() {
        Value::ComplexTensor(tensor)
    } else if let Some([value]) = tensor.as_f64_slice() {
        Value::Complex(value.0, value.1)
    } else {
        Value::ComplexTensor(tensor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortDirection {
    Ascend,
    Descend,
}

impl SortDirection {
    fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ascend" | "ascending" => Some(SortDirection::Ascend),
            "descend" | "descending" => Some(SortDirection::Descend),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComparisonMethod {
    Auto,
    Real,
    Abs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissingPlacement {
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
            MissingPlacement::First => MissingPlacementResolved::First,
            MissingPlacement::Last => MissingPlacementResolved::Last,
            MissingPlacement::Auto => match direction {
                SortDirection::Ascend => MissingPlacementResolved::Last,
                SortDirection::Descend => MissingPlacementResolved::First,
            },
        }
    }

    fn is_auto(self) -> bool {
        matches!(self, MissingPlacement::Auto)
    }
}

#[derive(Debug, Clone)]
struct ColumnSpec {
    index: usize,
    direction: SortDirection,
}

#[derive(Debug, Clone)]
struct SortRowsArgs {
    columns: Vec<ColumnSpec>,
    comparison: ComparisonMethod,
    missing: MissingPlacement,
}

impl SortRowsArgs {
    fn parse(rest: &[Value], num_cols: usize) -> crate::BuiltinResult<Self> {
        let mut columns: Option<Vec<ColumnSpec>> = None;
        let mut override_directions: Option<Vec<SortDirection>> = None;
        let mut comparison = ComparisonMethod::Auto;
        let mut missing = MissingPlacement::Auto;
        let mut i = 0usize;

        while i < rest.len() {
            if columns.is_none() {
                if let Some(parsed) = parse_column_vector(&rest[i], num_cols)? {
                    columns = Some(parsed);
                    i += 1;
                    continue;
                }
            }
            if let Some(directions) = parse_directions(&rest[i])? {
                if override_directions.is_some() {
                    return Err(sortrows_error_with(
                        &SORTROWS_ERROR_INVALID_ARGUMENT,
                        "sortrows: sorting direction specified more than once",
                    ));
                }
                override_directions = Some(directions);
                i += 1;
                continue;
            }
            let Some(keyword) = tensor::value_to_string(&rest[i]) else {
                return Err(sortrows_error_with(
                    &SORTROWS_ERROR_INVALID_ARGUMENT,
                    format!("sortrows: invalid argument {:?}", rest[i]),
                ));
            };
            let lowered = keyword.trim().to_ascii_lowercase();
            match lowered.as_str() {
                "comparisonmethod" => {
                    i += 1;
                    if i >= rest.len() {
                        return Err(sortrows_error_with(
                            &SORTROWS_ERROR_INVALID_ARGUMENT,
                            "sortrows: expected a value for 'ComparisonMethod'",
                        ));
                    }
                    let Some(value_str) = tensor::value_to_string(&rest[i]) else {
                        return Err(sortrows_error_with(
                            &SORTROWS_ERROR_INVALID_ARGUMENT,
                            "sortrows: 'ComparisonMethod' expects a string value",
                        )
                        .into());
                    };
                    comparison = match value_str.trim().to_ascii_lowercase().as_str() {
                        "auto" => ComparisonMethod::Auto,
                        "real" => ComparisonMethod::Real,
                        "abs" | "magnitude" => ComparisonMethod::Abs,
                        other => {
                            return Err(sortrows_error_with(
                                &SORTROWS_ERROR_COMPARISON_METHOD_UNKNOWN,
                                format!("sortrows: unsupported ComparisonMethod '{other}'"),
                            )
                            .into())
                        }
                    };
                    i += 1;
                }
                "missingplacement" => {
                    i += 1;
                    if i >= rest.len() {
                        return Err(sortrows_error_with(
                            &SORTROWS_ERROR_INVALID_ARGUMENT,
                            "sortrows: expected a value for 'MissingPlacement'",
                        )
                        .into());
                    }
                    let Some(value_str) = tensor::value_to_string(&rest[i]) else {
                        return Err(sortrows_error_with(
                            &SORTROWS_ERROR_INVALID_ARGUMENT,
                            "sortrows: 'MissingPlacement' expects a string value",
                        )
                        .into());
                    };
                    missing = match value_str.trim().to_ascii_lowercase().as_str() {
                        "auto" => MissingPlacement::Auto,
                        "first" => MissingPlacement::First,
                        "last" => MissingPlacement::Last,
                        other => {
                            return Err(sortrows_error_with(
                                &SORTROWS_ERROR_MISSING_PLACEMENT_UNKNOWN,
                                format!("sortrows: unsupported MissingPlacement '{other}'"),
                            )
                            .into())
                        }
                    };
                    i += 1;
                }
                other => {
                    return Err(sortrows_error_with(
                        &SORTROWS_ERROR_INVALID_ARGUMENT,
                        format!("sortrows: unexpected argument '{other}'"),
                    ));
                }
            }
        }

        let mut columns = columns.unwrap_or_else(|| default_columns(num_cols));
        if let Some(directions) = override_directions {
            if directions.len() == 1 {
                for spec in &mut columns {
                    spec.direction = directions[0];
                }
            } else if directions.len() == columns.len() {
                for (spec, direction) in columns.iter_mut().zip(directions) {
                    spec.direction = direction;
                }
            } else {
                return Err(sortrows_error_with(
                    &SORTROWS_ERROR_INVALID_ARGUMENT,
                    format!(
                        "sortrows: direction list length {} must be 1 or match {} selected columns",
                        directions.len(),
                        columns.len()
                    ),
                ));
            }
        }
        validate_columns(&columns, num_cols)?;

        Ok(SortRowsArgs {
            columns,
            comparison,
            missing,
        })
    }

    fn to_provider_columns(&self) -> Vec<ProviderSortRowsColumnSpec> {
        self.columns
            .iter()
            .map(|spec| ProviderSortRowsColumnSpec {
                index: spec.index,
                order: match spec.direction {
                    SortDirection::Ascend => ProviderSortOrder::Ascend,
                    SortDirection::Descend => ProviderSortOrder::Descend,
                },
            })
            .collect()
    }

    fn provider_comparison(&self) -> ProviderSortComparison {
        match self.comparison {
            ComparisonMethod::Auto => ProviderSortComparison::Auto,
            ComparisonMethod::Real => ProviderSortComparison::Real,
            ComparisonMethod::Abs => ProviderSortComparison::Abs,
        }
    }

    fn missing_for_direction(&self, direction: SortDirection) -> MissingPlacementResolved {
        self.missing.resolve(direction)
    }

    fn missing_is_auto(&self) -> bool {
        self.missing.is_auto()
    }
}

fn parse_column_vector(
    value: &Value,
    num_cols: usize,
) -> crate::BuiltinResult<Option<Vec<ColumnSpec>>> {
    match value {
        Value::Int(i) => {
            let Some(column) = i.try_to_i64() else {
                return Err(sortrows_error_with(
                    &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
                    "sortrows: column indices must fit signed integer range",
                ));
            };
            parse_single_column(column, num_cols).map(Some)
        }
        Value::Num(n) => {
            let column = floating_column_to_i64(*n)?;
            parse_single_column(column, num_cols).map(Some)
        }
        Value::Tensor(tensor) => {
            if !is_vector(&tensor.shape) {
                return Err(sortrows_error_with(
                    &SORTROWS_ERROR_INVALID_ARGUMENT,
                    "sortrows: column specification must be a vector",
                ));
            }
            let mut specs = Vec::with_capacity(tensor.len());
            for index in 0..tensor.len() {
                let value = tensor.numeric_value_at(index).ok_or_else(|| {
                    sortrows_internal_error("sortrows: column tensor storage is inconsistent")
                })?;
                specs.push(parse_single_column_i64(
                    numeric_column_to_i64(value)?,
                    num_cols,
                )?);
            }
            Ok(Some(specs))
        }
        _ => Ok(None),
    }
}

fn float_to_i64_column(rounded: f64) -> Option<i64> {
    if rounded < i64::MIN as f64 || rounded >= i64::MAX as f64 {
        return None;
    }
    let parsed = rounded as i64;
    (parsed as f64 == rounded).then_some(parsed)
}

fn parse_single_column(value: i64, num_cols: usize) -> crate::BuiltinResult<Vec<ColumnSpec>> {
    parse_single_column_i64(value, num_cols).map(|spec| vec![spec])
}

fn parse_single_column_i64(value: i64, num_cols: usize) -> crate::BuiltinResult<ColumnSpec> {
    if value == 0 {
        return Err(sortrows_error_with(
            &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
            "sortrows: column indices must be non-zero",
        ));
    }
    let Some(abs) = usize::try_from(value.unsigned_abs()).ok() else {
        return Err(sortrows_error_with(
            &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
            "sortrows: column index exceeds platform index range",
        ));
    };
    if abs == 0 {
        return Err(sortrows_error_with(
            &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
            "sortrows: column indices must be >= 1",
        ));
    }
    if num_cols == 0 {
        return Err(sortrows_error_with(
            &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
            "sortrows: column index exceeds matrix with 0 columns",
        ));
    }
    if abs > num_cols {
        return Err(sortrows_error_with(
            &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
            format!(
                "sortrows: column index {} exceeds matrix with {} columns",
                abs, num_cols
            ),
        )
        .into());
    }
    let direction = if value > 0 {
        SortDirection::Ascend
    } else {
        SortDirection::Descend
    };
    Ok(ColumnSpec {
        index: abs - 1,
        direction,
    })
}

fn parse_directions(value: &Value) -> crate::BuiltinResult<Option<Vec<SortDirection>>> {
    let strings = match value {
        Value::StringArray(array) => Some(array.data.clone()),
        Value::Cell(cell) => {
            let mut strings = Vec::with_capacity(cell.data.len());
            for value in &cell.data {
                let Some(direction) = tensor::value_to_string(value) else {
                    return Err(sortrows_error_with(
                        &SORTROWS_ERROR_INVALID_ARGUMENT,
                        "sortrows: direction cell arrays must contain character vectors or strings",
                    ));
                };
                strings.push(direction);
            }
            Some(strings)
        }
        _ => tensor::value_to_string(value).map(|value| vec![value]),
    };
    let Some(strings) = strings else {
        return Ok(None);
    };
    if strings.is_empty() {
        return Err(sortrows_error_with(
            &SORTROWS_ERROR_INVALID_ARGUMENT,
            "sortrows: direction list must not be empty",
        ));
    }
    let mut directions = Vec::with_capacity(strings.len());
    for value in strings {
        let Some(direction) = SortDirection::from_str(&value) else {
            return Ok(None);
        };
        directions.push(direction);
    }
    Ok(Some(directions))
}

fn default_columns(num_cols: usize) -> Vec<ColumnSpec> {
    let mut columns = Vec::with_capacity(num_cols);
    for col in 0..num_cols {
        columns.push(ColumnSpec {
            index: col,
            direction: SortDirection::Ascend,
        });
    }
    columns
}

fn validate_columns(columns: &[ColumnSpec], num_cols: usize) -> crate::BuiltinResult<()> {
    if num_cols == 0 && columns.iter().any(|spec| spec.index > 0) {
        return Err(sortrows_error_with(
            &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
            "sortrows: column index exceeds matrix with 0 columns",
        ));
    }
    for spec in columns {
        if num_cols > 0 && spec.index >= num_cols {
            return Err(sortrows_error_with(
                &SORTROWS_ERROR_INVALID_COLUMN_INDEX,
                format!(
                    "sortrows: column index {} exceeds matrix with {} columns",
                    spec.index + 1,
                    num_cols
                ),
            )
            .into());
        }
    }
    Ok(())
}

fn is_vector(shape: &[usize]) -> bool {
    match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => false,
    }
}

#[derive(Debug)]
pub struct SortRowsEvaluation {
    sorted: Value,
    indices: Tensor,
}

impl SortRowsEvaluation {
    pub(crate) fn from_parts(sorted: Value, indices: Tensor) -> Self {
        Self { sorted, indices }
    }

    pub fn into_sorted_value(self) -> Value {
        self.sorted
    }

    pub fn into_values(self) -> (Value, Value) {
        let indices = tensor::tensor_into_value(self.indices);
        (self.sorted, indices)
    }

    pub fn indices_value(&self) -> Value {
        tensor::tensor_into_value(self.indices.clone())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{
        CellArray, IntValue, IntegerStorage, ResolveContext, StringArray, Type, Value,
    };

    fn evaluate(value: Value, rest: &[Value]) -> crate::BuiltinResult<SortRowsEvaluation> {
        futures::executor::block_on(super::evaluate(value, rest))
    }

    fn builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        futures::executor::block_on(sortrows_builtin(value, rest))
    }

    fn assert_double_values(tensor: &Tensor, expected: &[f64]) {
        assert_eq!(
            tensor.as_f64_slice().expect("expected double tensor"),
            expected
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_default_matrix() {
        let tensor = Tensor::new(vec![3.0, 1.0, 2.0, 4.0, 1.0, 5.0], vec![3, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                assert_double_values(&t, &[1.0, 2.0, 3.0, 1.0, 5.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_double_values(&t, &[2.0, 3.0, 1.0]),
            Value::Num(_) => panic!("expected tensor indices"),
            other => panic!("unexpected indices {other:?}"),
        }
    }

    #[test]
    fn sortrows_preserves_exact_integer_values_and_ordering() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 0, 1, 9_007_199_254_740_993]),
            vec![2, 2],
        )
        .expect("input");
        let (sorted, indices) = evaluate(Value::Tensor(tensor), &[])
            .expect("sortrows")
            .into_values();
        let Value::Tensor(sorted) = sorted else {
            panic!("expected exact integer output");
        };
        assert_eq!(
            sorted.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                0,
                u64::MAX,
                9_007_199_254_740_993,
                1
            ]))
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected index tensor");
        };
        assert_double_values(&indices, &[2.0, 1.0]);

        let tensor = Tensor::new_integer(
            IntegerStorage::I64(vec![i64::MIN, -1, 2, i64::MAX]),
            vec![4, 1],
        )
        .expect("input");
        let (sorted, _) = evaluate(
            Value::Tensor(tensor),
            &[Value::from("ComparisonMethod"), Value::from("abs")],
        )
        .expect("absolute sortrows")
        .into_values();
        let Value::Tensor(sorted) = sorted else {
            panic!("expected exact integer output");
        };
        assert_eq!(
            sorted.integer_storage(),
            Some(&IntegerStorage::I64(vec![-1, 2, i64::MAX, i64::MIN]))
        );
    }

    #[test]
    fn sortrows_reads_exact_integer_values_and_column_specs_without_mirrors() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 0, 1, 9_007_199_254_740_993]),
            vec![2, 2],
        )
        .expect("input");
        let columns =
            Tensor::new_integer(IntegerStorage::I16(vec![-2]), vec![1, 1]).expect("columns");

        let (sorted, indices) = evaluate(Value::Tensor(tensor), &[Value::Tensor(columns)])
            .expect("sortrows")
            .into_values();
        let Value::Tensor(sorted) = sorted else {
            panic!("expected exact integer output");
        };
        assert_eq!(
            sorted.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                0,
                u64::MAX,
                9_007_199_254_740_993,
                1,
            ]))
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected index tensor");
        };
        assert_double_values(&indices, &[2.0, 1.0]);
    }

    #[test]
    fn sortrows_column_specs_reject_unrepresentable_integer_and_double_values() {
        assert!(parse_column_vector(&Value::Int(IntValue::U64(u64::MAX)), 3).is_err());
        assert!(parse_column_vector(&Value::Num(1.0e300), 3).is_err());

        let columns =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("columns");
        assert!(parse_column_vector(&Value::Tensor(columns), 3).is_err());

        #[cfg(target_pointer_width = "32")]
        assert!(parse_single_column_i64(i64::from(u32::MAX) + 1, usize::MAX).is_err());
    }

    #[test]
    fn sortrows_preserves_every_exact_integer_storage_class() {
        let cases = [
            (
                IntegerStorage::I8(vec![i8::MAX, i8::MIN, 1, 2]),
                IntegerStorage::I8(vec![i8::MIN, i8::MAX, 2, 1]),
            ),
            (
                IntegerStorage::I16(vec![i16::MAX, i16::MIN, 1, 2]),
                IntegerStorage::I16(vec![i16::MIN, i16::MAX, 2, 1]),
            ),
            (
                IntegerStorage::I32(vec![i32::MAX, i32::MIN, 1, 2]),
                IntegerStorage::I32(vec![i32::MIN, i32::MAX, 2, 1]),
            ),
            (
                IntegerStorage::I64(vec![i64::MAX, i64::MIN, 1, 2]),
                IntegerStorage::I64(vec![i64::MIN, i64::MAX, 2, 1]),
            ),
            (
                IntegerStorage::U8(vec![u8::MAX, 0, 1, 2]),
                IntegerStorage::U8(vec![0, u8::MAX, 2, 1]),
            ),
            (
                IntegerStorage::U16(vec![u16::MAX, 0, 1, 2]),
                IntegerStorage::U16(vec![0, u16::MAX, 2, 1]),
            ),
            (
                IntegerStorage::U32(vec![u32::MAX, 0, 1, 2]),
                IntegerStorage::U32(vec![0, u32::MAX, 2, 1]),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX, 0, 1, 9_007_199_254_740_993]),
                IntegerStorage::U64(vec![0, u64::MAX, 9_007_199_254_740_993, 1]),
            ),
        ];
        for (input, expected) in cases {
            let tensor = Tensor::new_integer(input, vec![2, 2]).expect("input");
            let (sorted, indices) = evaluate(Value::Tensor(tensor), &[])
                .expect("sortrows")
                .into_values();
            let Value::Tensor(sorted) = sorted else {
                panic!("expected exact integer output");
            };
            assert_eq!(sorted.integer_storage(), Some(&expected));
            let Value::Tensor(indices) = indices else {
                panic!("expected index tensor");
            };
            assert_double_values(&indices, &[2.0, 1.0]);
        }
    }

    #[test]
    fn sortrows_preserves_logical_class() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0, 1, 0], vec![3, 2]).unwrap();
        let (sorted, indices) = evaluate(Value::LogicalArray(logical), &[])
            .expect("logical sortrows")
            .into_values();
        let Value::LogicalArray(sorted) = sorted else {
            panic!("expected logical output");
        };
        assert_eq!(sorted.shape, vec![3, 2]);
        assert_eq!(sorted.data, vec![0, 1, 1, 1, 0, 0]);
        let Value::Tensor(indices) = indices else {
            panic!("expected index tensor");
        };
        assert_double_values(&indices, &[2.0, 1.0, 3.0]);
    }

    #[test]
    fn registered_sortrows_restores_wide_integer_and_logical_outputs_and_rejects_excess_arity() {
        test_support::with_test_provider(|provider| {
            let integer = Tensor::new_integer(
                IntegerStorage::U64(vec![u64::MAX, 0, 1, 9_007_199_254_740_993]),
                vec![2, 2],
            )
            .expect("integer input");
            let handle = gpu_helpers::upload_tensor(provider, &integer).expect("typed upload");
            {
                let _guard = crate::output_count::push_output_count(Some(2));
                let Value::OutputList(outputs) =
                    builtin(Value::GpuTensor(handle), Vec::new()).expect("resident sortrows")
                else {
                    panic!("expected output list");
                };
                assert_eq!(outputs.len(), 2);
                assert!(outputs
                    .iter()
                    .all(|output| matches!(output, Value::GpuTensor(_))));
                assert_eq!(
                    test_support::gather(outputs[0].clone())
                        .expect("gather sorted")
                        .integer_storage(),
                    Some(&IntegerStorage::U64(vec![
                        0,
                        u64::MAX,
                        9_007_199_254_740_993,
                        1,
                    ]))
                );
            }

            let logical =
                Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], vec![3, 2]).expect("logical input");
            let handle = gpu_helpers::upload_tensor(provider, &logical).expect("logical upload");
            let logical = gpu_helpers::logical_gpu_value(handle);
            let Value::GpuTensor(output) =
                builtin(logical, Vec::new()).expect("resident logical sortrows")
            else {
                panic!("expected resident logical output");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&output));
        });

        let _guard = crate::output_count::push_output_count(Some(3));
        let err = builtin(Value::Num(1.0), Vec::new()).expect_err("excess outputs must reject");
        assert_eq!(err.identifier(), SORTROWS_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn sortrows_preserves_native_single_storage() {
        let tensor = Tensor::from_f32(vec![3.25, 1.5, 2.0, 4.5, 1.25, 5.75], vec![3, 2]).unwrap();
        let columns = Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap();
        let (sorted, indices) = evaluate(Value::Tensor(tensor), &[Value::Tensor(columns)])
            .expect("single sortrows")
            .into_values();
        let Value::Tensor(sorted) = sorted else {
            panic!("expected single tensor");
        };
        assert_eq!(
            sorted.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![1.5, 2.0, 3.25, 1.25, 5.75, 4.5])
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected index tensor");
        };
        assert_double_values(&indices, &[2.0, 3.0, 1.0]);
    }

    #[test]
    fn sortrows_type_resolver_tensor() {
        assert_eq!(
            tensor_output_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::tensor()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_with_column_vector() {
        let tensor = Tensor::new(
            vec![1.0, 3.0, 3.0, 4.0, 2.0, 2.0, 2.0, 5.0, 1.0],
            vec![3, 3],
        )
        .unwrap();
        let cols = Tensor::new(vec![2.0, 3.0, 1.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[Value::Tensor(cols)]).expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert_double_values(&t, &[3.0, 3.0, 1.0, 2.0, 2.0, 4.0, 1.0, 5.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_direction_descend() {
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[Value::from("descend")]).expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_double_values(&t, &[2.0, 1.0, 3.0, 4.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_mixed_directions() {
        let tensor = Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 7.0, 2.0], vec![3, 2]).unwrap();
        let cols = Tensor::new(vec![1.0, -2.0], vec![2, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[Value::Tensor(cols)]).expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_double_values(&t, &[1.0, 1.0, 1.0, 7.0, 2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn sortrows_accepts_per_column_cell_and_string_directions() {
        let tensor = Tensor::new(vec![3.0, 2.0, 1.0, 1.0, 1.0, 2.0], vec![3, 2]).unwrap();
        let columns = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let cell = Value::Cell(
            CellArray::new(vec![Value::from("descend"), Value::from("ascend")], 1, 2).unwrap(),
        );
        let (cell_sorted, _) = evaluate(
            Value::Tensor(tensor.clone()),
            &[Value::Tensor(columns.clone()), cell],
        )
        .expect("cell directions")
        .into_values();
        let strings = Value::StringArray(
            StringArray::new(
                vec!["descend".to_string(), "ascend".to_string()],
                vec![1, 2],
            )
            .unwrap(),
        );
        let (string_sorted, _) =
            evaluate(Value::Tensor(tensor), &[Value::Tensor(columns), strings])
                .expect("string directions")
                .into_values();
        assert_eq!(cell_sorted, string_sorted);
    }

    #[test]
    fn sortrows_absolute_ties_follow_phase() {
        let real = Tensor::new(vec![-1.0, 1.0], vec![2, 1]).unwrap();
        let (sorted, _) = evaluate(
            Value::Tensor(real),
            &[Value::from("ComparisonMethod"), Value::from("abs")],
        )
        .expect("real phase order")
        .into_values();
        let Value::Tensor(sorted) = sorted else {
            panic!("expected tensor");
        };
        assert_double_values(&sorted, &[1.0, -1.0]);

        let complex =
            ComplexTensor::new(vec![(0.0, -1.0), (1.0, 0.0), (0.0, 1.0)], vec![3, 1]).unwrap();
        let (sorted, _) = evaluate(Value::ComplexTensor(complex), &[])
            .expect("complex phase order")
            .into_values();
        let Value::ComplexTensor(sorted) = sorted else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            sorted.materialize_f64(),
            vec![(0.0, -1.0), (1.0, 0.0), (0.0, 1.0)]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_returns_indices() {
        let tensor = Tensor::new(vec![2.0, 1.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (_, indices) = eval.into_values();
        match indices {
            Value::Tensor(t) => assert_double_values(&t, &[2.0, 1.0]),
            Value::Num(_) => panic!("expected tensor indices"),
            other => panic!("unexpected indices {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_char_array() {
        let chars = CharArray::new(
            "bob "
                .chars()
                .chain("al  ".chars())
                .chain("ally".chars())
                .collect(),
            3,
            4,
        )
        .unwrap();
        let eval = evaluate(Value::CharArray(chars), &[]).expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 3);
                assert_eq!(ca.cols, 4);
                let strings: Vec<String> = (0..ca.rows)
                    .map(|r| {
                        ca.data[r * ca.cols..(r + 1) * ca.cols]
                            .iter()
                            .collect::<String>()
                    })
                    .collect();
                assert_eq!(
                    strings,
                    vec!["al  ".to_string(), "ally".to_string(), "bob ".to_string()]
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_complex_abs() {
        let tensor = ComplexTensor::new(vec![(1.0, 2.0), (-2.0, 1.0)], vec![2, 1]).unwrap();
        let eval = evaluate(
            Value::ComplexTensor(tensor),
            &[Value::from("ComparisonMethod"), Value::from("abs")],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.materialize_f64(), vec![(1.0, 2.0), (-2.0, 1.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn sortrows_preserves_native_complex_single_storage() {
        let tensor = ComplexTensor::from_f32(
            vec![
                (3.0, 4.0),
                (1.0, 0.0),
                (0.0, 2.0),
                (30.0, 1.0),
                (10.0, 1.0),
                (20.0, 1.0),
            ],
            vec![3, 2],
        )
        .unwrap();
        let (sorted, indices) = evaluate(Value::ComplexTensor(tensor), &[])
            .expect("evaluate")
            .into_values();
        let Value::ComplexTensor(sorted) = sorted else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            sorted.as_f32_slice(),
            Some(
                &[
                    (1.0, 0.0),
                    (0.0, 2.0),
                    (3.0, 4.0),
                    (10.0, 1.0),
                    (20.0, 1.0),
                    (30.0, 1.0),
                ][..]
            )
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected index tensor");
        };
        assert_double_values(&indices, &[2.0, 3.0, 1.0]);
    }

    #[test]
    fn sortrows_preserves_one_element_complex_single_tensor() {
        let tensor = ComplexTensor::from_f32(vec![(1.25, -2.5)], vec![1, 1]).unwrap();
        let sorted = evaluate(Value::ComplexTensor(tensor), &[])
            .expect("evaluate")
            .into_sorted_value();
        let Value::ComplexTensor(sorted) = sorted else {
            panic!("expected complex single tensor");
        };
        assert_eq!(sorted.as_f32_slice(), Some(&[(1.25, -2.5)][..]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_invalid_column_index_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = evaluate(Value::Tensor(tensor), &[Value::Int(IntValue::I32(3))]).unwrap_err();
        assert_eq!(
            err.identifier(),
            SORTROWS_ERROR_INVALID_COLUMN_INDEX.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_missingplacement_first_moves_nan_first() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 2.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[Value::from("MissingPlacement"), Value::from("first")],
        )
        .expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                let values = t.as_f64_slice().expect("expected double tensor");
                assert!(values[0].is_nan());
                assert_eq!(values[1], 1.0);
                assert_eq!(values[2], 3.0);
                assert_eq!(values[3], 2.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_double_values(&t, &[2.0, 1.0]),
            Value::Num(_) => panic!("expected tensor indices"),
            other => panic!("unexpected indices {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_missingplacement_last_descend_moves_nan_last() {
        let tensor = Tensor::new(vec![f64::NAN, 5.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[
                Value::from("descend"),
                Value::from("MissingPlacement"),
                Value::from("last"),
            ],
        )
        .expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                let values = t.as_f64_slice().expect("expected double tensor");
                assert_eq!(values[0], 5.0);
                assert!(values[1].is_nan());
                assert_eq!(values[2], 2.0);
                assert_eq!(values[3], 1.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_double_values(&t, &[2.0, 1.0]),
            Value::Num(_) => panic!("expected tensor indices"),
            other => panic!("unexpected indices {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_missingplacement_invalid_value_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = evaluate(
            Value::Tensor(tensor),
            &[Value::from("MissingPlacement"), Value::from("middle")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            SORTROWS_ERROR_MISSING_PLACEMENT_UNKNOWN.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 1.0, 2.0, 4.0, 1.0, 5.0], vec![3, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: tensor.as_f64_slice().expect("double upload tensor"),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("evaluate");
            let (sorted, indices) = eval.into_values();
            match sorted {
                Value::Tensor(t) => assert_double_values(&t, &[1.0, 2.0, 3.0, 1.0, 5.0, 4.0]),
                other => panic!("expected tensor, got {other:?}"),
            }
            match indices {
                Value::Tensor(t) => assert_double_values(&t, &[2.0, 3.0, 1.0]),
                other => panic!("unexpected indices {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sortrows_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new(vec![4.0, 2.0, 3.0, 1.0, 2.0, 5.0], vec![3, 2]).unwrap();
        let cpu_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cpu evaluate");
        let (cpu_sorted_val, cpu_indices_val) = cpu_eval.into_values();
        let cpu_sorted = match cpu_sorted_val {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        };
        let cpu_indices = match cpu_indices_val {
            Value::Tensor(t) => t,
            other => panic!("expected tensor indices, got {other:?}"),
        };

        let view = runmat_accelerate_api::HostTensorView {
            data: tensor.as_f64_slice().expect("double upload tensor"),
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle.clone()), &[]).expect("gpu evaluate");
        let (gpu_sorted_val, gpu_indices_val) = gpu_eval.into_values();
        let gpu_sorted = match gpu_sorted_val {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        };
        let gpu_indices = match gpu_indices_val {
            Value::Tensor(t) => t,
            other => panic!("expected tensor indices, got {other:?}"),
        };

        assert_eq!(gpu_sorted.shape, cpu_sorted.shape);
        assert_eq!(gpu_sorted.as_f64_slice(), cpu_sorted.as_f64_slice());
        assert_eq!(gpu_indices.shape, cpu_indices.shape);
        assert_eq!(gpu_indices.as_f64_slice(), cpu_indices.as_f64_slice());

        let _ = provider.free(&handle);
    }
}
