//! MATLAB-compatible `unique` builtin with GPU-aware semantics for RunMat.
//!
//! The implementation mirrors MathWorks MATLAB behavioural details for sorted
//! and stable orderings, row-wise uniqueness, and index outputs. GPU tensors
//! use a provider hook or typed host fallback, then public outputs are restored
//! to the owning provider.

use std::cmp::Ordering;
use std::collections::HashMap;

use runmat_accelerate_api::{
    GpuTensorHandle, UniqueOccurrence, UniqueOptions, UniqueOrder, UniqueResult,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage,
    LogicalArray, NumericStorage, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::{float_order::SetFloat, integer_order, type_resolvers::unique_values_output_type};
use crate::build_runtime_error;
use crate::builtins::common::arg_tokens::tokens_from_values;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::unique")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "unique",
    op_kind: GpuOpKind::Custom("unique"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("unique")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Providers may implement the `unique` hook; typed host fallback preserves exact supported integer storage and public outputs are restored to the input handle's owner.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::unique"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "unique",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`unique` terminates fusion chains and acts as a residency sink; upstream tensors are gathered when a provider hook is unavailable.",
};

const BUILTIN_NAME: &str = "unique";

const UNIQUE_OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Unique values or rows.",
}];

const UNIQUE_OUTPUT_C_IA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unique values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting representatives in input A.",
    },
];

const UNIQUE_OUTPUT_C_IA_IC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unique values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting representatives in input A.",
    },
    BuiltinParamDescriptor {
        name: "ic",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices mapping each input element/row to C.",
    },
];

const UNIQUE_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const UNIQUE_INPUTS_A_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option tokens plus the 'TreatMissingAsDistinct', logical name-value pair.",
    },
];

const UNIQUE_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "C = unique(A)",
        inputs: &UNIQUE_INPUTS_A,
        outputs: &UNIQUE_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = unique(A, option...)",
        inputs: &UNIQUE_INPUTS_A_OPTIONS,
        outputs: &UNIQUE_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = unique(A)",
        inputs: &UNIQUE_INPUTS_A,
        outputs: &UNIQUE_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = unique(A, option...)",
        inputs: &UNIQUE_INPUTS_A_OPTIONS,
        outputs: &UNIQUE_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ic] = unique(A)",
        inputs: &UNIQUE_INPUTS_A,
        outputs: &UNIQUE_OUTPUT_C_IA_IC,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ic] = unique(A, option...)",
        inputs: &UNIQUE_INPUTS_A_OPTIONS,
        outputs: &UNIQUE_OUTPUT_C_IA_IC,
    },
];

const UNIQUE_ERROR_LEGACY_OPTION_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.LEGACY_OPTION_UNSUPPORTED",
    identifier: Some("RunMat:unique:LegacyOptionUnsupported"),
    when: "Legacy compatibility options are requested.",
    message: "unique: the 'legacy' behaviour is not supported",
};

const UNIQUE_ERROR_CONFLICTING_ORDER_OPTIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.CONFLICTING_ORDER_OPTIONS",
    identifier: Some("RunMat:unique:ConflictingOrderOptions"),
    when: "Both 'sorted' and 'stable' options are provided.",
    message: "unique: cannot combine 'sorted' with 'stable'",
};

const UNIQUE_ERROR_CONFLICTING_OCCURRENCE_OPTIONS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.UNIQUE.CONFLICTING_OCCURRENCE_OPTIONS",
        identifier: Some("RunMat:unique:ConflictingOccurrenceOptions"),
        when: "Both 'first' and 'last' options are provided.",
        message: "unique: cannot combine 'first' with 'last'",
    };

const UNIQUE_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.UNKNOWN_OPTION",
    identifier: Some("RunMat:unique:UnknownOption"),
    when: "An unsupported option token is provided.",
    message: "unique: unrecognised option",
};

const UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.ROWS_REQUIRES_2D_MATRIX",
    identifier: Some("RunMat:unique:RowsRequiresTwoDimensionalInput"),
    when: "'rows' mode is used with non-2D data.",
    message: "unique: 'rows' option requires a 2-D matrix input",
};

const UNIQUE_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:unique:UnsupportedInputType"),
    when: "Input cannot be converted into a supported unique domain.",
    message: "unique: unsupported input type",
};

const UNIQUE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.INVALID_ARGUMENT",
    identifier: Some("RunMat:unique:InvalidArgument"),
    when: "Option arguments or name-value pairs are malformed.",
    message: "unique: invalid option arguments",
};

const UNIQUE_ERROR_GPU_OPTION_COMBINATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.GPU_OPTION_COMBINATION",
    identifier: Some("RunMat:unique:GpuOptionCombination"),
    when: "A resident input specifies both set-order and occurrence options.",
    message: "unique: GPU inputs cannot combine a set-order option with 'first' or 'last'",
};

const UNIQUE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.INTERNAL",
    identifier: Some("RunMat:unique:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "unique: internal operation failed",
};

const UNIQUE_ERRORS: [BuiltinErrorDescriptor; 9] = [
    UNIQUE_ERROR_LEGACY_OPTION_UNSUPPORTED,
    UNIQUE_ERROR_CONFLICTING_ORDER_OPTIONS,
    UNIQUE_ERROR_CONFLICTING_OCCURRENCE_OPTIONS,
    UNIQUE_ERROR_UNKNOWN_OPTION,
    UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX,
    UNIQUE_ERROR_UNSUPPORTED_INPUT_TYPE,
    UNIQUE_ERROR_INVALID_ARGUMENT,
    UNIQUE_ERROR_GPU_OPTION_COMBINATION,
    UNIQUE_ERROR_INTERNAL,
];

const UNIQUE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A accepts every real integer class and retains exact typed storage throughout host evaluation.",
    }];

const UNIQUE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[C, ia, ic] = unique(integer_A, options)",
        inputs: &UNIQUE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "C preserves A's exact integer class; ia and ic are one-based double. GPU supports integer classes through 32 bits and restores all requested outputs after typed fallback.",
    }];

pub const UNIQUE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNIQUE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UNIQUE_ERRORS,
};

fn unique_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn unique_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    unique_error_with(error, error.message)
}

fn unique_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    unique_error_with(&UNIQUE_ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "unique",
    category = "array/sorting_sets",
    summary = "Return unique elements or rows with optional index mappings.",
    keywords = "unique,set,distinct,stable,rows,indices,gpu",
    accel = "array_construct",
    sink = true,
    type_resolver(unique_values_output_type),
    descriptor(crate::builtins::array::sorting_sets::unique::UNIQUE_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::array::sorting_sets::unique::UNIQUE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::array::sorting_sets::unique"
)]
async fn unique_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 3) {
        return Err(unique_error_with(
            &UNIQUE_ERROR_INVALID_ARGUMENT,
            "unique: too many output arguments; maximum is 3",
        ));
    }
    let provider = super::output_provider(&value);
    let eval = evaluate(value, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            let outputs = super::restore_set_outputs(
                provider,
                BUILTIN_NAME,
                vec![eval.into_values_value()],
                unique_internal_error,
            )?;
            return Ok(Value::OutputList(outputs));
        }
        if out_count == 2 {
            let (values, ia) = eval.into_pair();
            let outputs = super::restore_set_outputs(
                provider,
                BUILTIN_NAME,
                vec![values, ia],
                unique_internal_error,
            )?;
            return Ok(Value::OutputList(outputs));
        }
        let (values, ia, ic) = eval.into_triple();
        let outputs = super::restore_set_outputs(
            provider,
            BUILTIN_NAME,
            vec![values, ia, ic],
            unique_internal_error,
        )?;
        return Ok(Value::OutputList(outputs));
    }
    let mut outputs = super::restore_set_outputs(
        provider,
        BUILTIN_NAME,
        vec![eval.into_values_value()],
        unique_internal_error,
    )?;
    Ok(outputs.pop().expect("unique output"))
}

/// Evaluate `unique` once and expose all outputs to the caller.
pub async fn evaluate(value: Value, rest: &[Value]) -> crate::BuiltinResult<UniqueEvaluation> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "unique")?;
    let opts = parse_options(rest)?;
    match value {
        Value::GpuTensor(handle) => unique_gpu(handle, &opts).await,
        other => unique_host(other, &opts),
    }
}

fn parse_options(rest: &[Value]) -> crate::BuiltinResult<UniqueOptions> {
    let mut opts = UniqueOptions {
        rows: false,
        order: UniqueOrder::Sorted,
        occurrence: UniqueOccurrence::First,
        treat_missing_as_distinct: true,
        explicit_order: false,
        explicit_occurrence: false,
    };
    let mut seen_order: Option<UniqueOrder> = None;
    let mut seen_occurrence: Option<UniqueOccurrence> = None;

    let tokens = tokens_from_values(rest);
    let mut index = 0;
    while index < rest.len() {
        let arg = &rest[index];
        let token = &tokens[index];
        let text = match token {
            crate::builtins::common::arg_tokens::ArgToken::String(text) => text.as_str(),
            _ => {
                let text = tensor::value_to_string(arg)
                    .ok_or_else(|| unique_error(&UNIQUE_ERROR_INVALID_ARGUMENT))?;
                let lowered = text.trim().to_ascii_lowercase();
                if lowered == "treatmissingasdistinct" {
                    index += 1;
                    let value = rest.get(index).ok_or_else(|| {
                        unique_error_with(
                            &UNIQUE_ERROR_INVALID_ARGUMENT,
                            "unique: 'TreatMissingAsDistinct' requires a logical scalar value",
                        )
                    })?;
                    opts.treat_missing_as_distinct = parse_logical_option(value)?;
                } else {
                    parse_unique_option(
                        &mut opts,
                        &mut seen_order,
                        &mut seen_occurrence,
                        &lowered,
                    )?;
                }
                index += 1;
                continue;
            }
        };
        if text.eq_ignore_ascii_case("TreatMissingAsDistinct") {
            index += 1;
            let value = rest.get(index).ok_or_else(|| {
                unique_error_with(
                    &UNIQUE_ERROR_INVALID_ARGUMENT,
                    "unique: 'TreatMissingAsDistinct' requires a logical scalar value",
                )
            })?;
            opts.treat_missing_as_distinct = parse_logical_option(value)?;
        } else {
            let lowered = text.trim().to_ascii_lowercase();
            parse_unique_option(&mut opts, &mut seen_order, &mut seen_occurrence, &lowered)?;
        }
        index += 1;
    }

    Ok(opts)
}

fn parse_logical_option(value: &Value) -> crate::BuiltinResult<bool> {
    if let Value::Bool(flag) = value {
        return Ok(*flag);
    }
    if let Value::LogicalArray(logical) = value {
        return match logical.data.as_slice() {
            [flag] => Ok(*flag != 0),
            _ => Err(unique_error_with(
                &UNIQUE_ERROR_INVALID_ARGUMENT,
                "unique: 'TreatMissingAsDistinct' must be a logical scalar",
            )),
        };
    }
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_i64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(unique_error_with(
                &UNIQUE_ERROR_INVALID_ARGUMENT,
                "unique: 'TreatMissingAsDistinct' must be logical true or false",
            )),
        };
    }
    let number = match value {
        Value::Num(number) => Some(*number),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    };
    match number {
        Some(0.0) => Ok(false),
        Some(1.0) => Ok(true),
        _ => Err(unique_error_with(
            &UNIQUE_ERROR_INVALID_ARGUMENT,
            "unique: 'TreatMissingAsDistinct' must be logical true or false",
        )),
    }
}

fn parse_unique_option(
    opts: &mut UniqueOptions,
    seen_order: &mut Option<UniqueOrder>,
    seen_occurrence: &mut Option<UniqueOccurrence>,
    lowered: &str,
) -> crate::BuiltinResult<()> {
    match lowered {
        "sorted" => {
            if let Some(prev) = seen_order {
                if *prev != UniqueOrder::Sorted {
                    return Err(unique_error_with(
                        &UNIQUE_ERROR_CONFLICTING_ORDER_OPTIONS,
                        UNIQUE_ERROR_CONFLICTING_ORDER_OPTIONS.message,
                    ));
                }
            }
            *seen_order = Some(UniqueOrder::Sorted);
            opts.order = UniqueOrder::Sorted;
            opts.explicit_order = true;
        }
        "stable" => {
            if let Some(prev) = seen_order {
                if *prev != UniqueOrder::Stable {
                    return Err(unique_error_with(
                        &UNIQUE_ERROR_CONFLICTING_ORDER_OPTIONS,
                        UNIQUE_ERROR_CONFLICTING_ORDER_OPTIONS.message,
                    ));
                }
            }
            *seen_order = Some(UniqueOrder::Stable);
            opts.order = UniqueOrder::Stable;
            opts.explicit_order = true;
        }
        "rows" => {
            opts.rows = true;
        }
        "first" => {
            if let Some(prev) = seen_occurrence {
                if *prev != UniqueOccurrence::First {
                    return Err(unique_error_with(
                        &UNIQUE_ERROR_CONFLICTING_OCCURRENCE_OPTIONS,
                        UNIQUE_ERROR_CONFLICTING_OCCURRENCE_OPTIONS.message,
                    ));
                }
            }
            *seen_occurrence = Some(UniqueOccurrence::First);
            opts.occurrence = UniqueOccurrence::First;
            opts.explicit_occurrence = true;
        }
        "last" => {
            if let Some(prev) = seen_occurrence {
                if *prev != UniqueOccurrence::Last {
                    return Err(unique_error_with(
                        &UNIQUE_ERROR_CONFLICTING_OCCURRENCE_OPTIONS,
                        UNIQUE_ERROR_CONFLICTING_OCCURRENCE_OPTIONS.message,
                    ));
                }
            }
            *seen_occurrence = Some(UniqueOccurrence::Last);
            opts.occurrence = UniqueOccurrence::Last;
            opts.explicit_occurrence = true;
        }
        "legacy" | "r2012a" => {
            return Err(unique_error(&UNIQUE_ERROR_LEGACY_OPTION_UNSUPPORTED));
        }
        other => {
            return Err(unique_error_with(
                &UNIQUE_ERROR_UNKNOWN_OPTION,
                format!("unique: unrecognised option '{other}'"),
            ));
        }
    }
    Ok(())
}

async fn unique_gpu(
    handle: GpuTensorHandle,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if super::is_unsupported_set_gpu_integer(&handle) {
        return Err(unique_error_with(
            &UNIQUE_ERROR_UNSUPPORTED_INPUT_TYPE,
            "unique: resident 64-bit integer inputs are not supported",
        ));
    }
    if opts.explicit_order && opts.explicit_occurrence {
        return Err(unique_error(&UNIQUE_ERROR_GPU_OPTION_COMBINATION));
    }
    let logical = runmat_accelerate_api::handle_is_logical(&handle);
    if runmat_accelerate_api::handle_integer_type(&handle).is_none() {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle)
            .or_else(runmat_accelerate_api::provider)
        {
            if let Ok(result) = provider.unique(&handle, opts).await {
                let evaluation = UniqueEvaluation::from_unique_result(result)?;
                return if logical {
                    evaluation.into_logical_values()
                } else {
                    Ok(evaluation)
                };
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let evaluation = unique_numeric_from_tensor(tensor, opts)?;
    if logical {
        evaluation.into_logical_values()
    } else {
        Ok(evaluation)
    }
}

fn unique_host(value: Value, opts: &UniqueOptions) -> crate::BuiltinResult<UniqueEvaluation> {
    match value {
        Value::Tensor(tensor) => unique_numeric_from_tensor(tensor, opts),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| unique_internal_error(format!("unique: {e}")))?;
            unique_numeric_from_tensor(tensor, opts)
        }
        Value::Int(i) => {
            let tensor = Tensor::new_integer(IntegerStorage::from_scalar(i), vec![1, 1])
                .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
            unique_numeric_from_tensor(tensor, opts)
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
            unique_numeric_from_tensor(tensor, opts)?.into_logical_values()
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| unique_internal_error(e))?;
            unique_numeric_from_tensor(tensor, opts)?.into_logical_values()
        }
        Value::ComplexTensor(tensor) => unique_complex_from_tensor(tensor, opts),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
            unique_complex_from_tensor(tensor, opts)
        }
        Value::CharArray(array) => unique_char_array(array, opts),
        Value::StringArray(array) => unique_string_array(array, opts),
        Value::String(s) => {
            let array = StringArray::new(vec![s], vec![1, 1]).map_err(|e| unique_internal_error(format!("unique: {e}")))?;
            unique_string_array(array, opts)
        }
        other => Err(unique_error_with(
            &UNIQUE_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!(
                "unique: unsupported input type {:?}; expected numeric, logical, char, string, or complex values",
                other
            ),
        )),
    }
}

pub fn unique_numeric_from_tensor(
    tensor: Tensor,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    let shape = tensor.shape.clone();
    match tensor
        .into_numeric_storage()
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?
    {
        NumericStorage::F64(values) => unique_floating(values, shape, opts),
        NumericStorage::F32(values) => unique_floating(values, shape, opts),
        storage => {
            let integer = storage
                .into_integer_storage()
                .map_err(|_| unique_internal_error("unique: expected integer storage"))?;
            unique_integer(&integer, shape, opts)
        }
    }
}

fn unique_floating<T: SetFloat>(
    values: Vec<T>,
    shape: Vec<usize>,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if opts.rows {
        unique_floating_rows(values, shape, opts)
    } else {
        unique_floating_elements(values, shape, opts)
    }
}

fn unique_integer(
    storage: &IntegerStorage,
    shape: Vec<usize>,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if opts.rows {
        unique_integer_rows(storage, shape, opts)
    } else {
        unique_integer_elements(storage, shape, opts)
    }
}

fn is_row_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [] => false,
        [_] => true,
        [rows, ..] if *rows != 1 => false,
        [_, _, rest @ ..] => rest.iter().all(|&dimension| dimension == 1),
    }
}

fn unique_element_values_shape(input_shape: &[usize], output_len: usize) -> Vec<usize> {
    if is_row_vector_shape(input_shape) {
        vec![1, output_len]
    } else {
        vec![output_len, 1]
    }
}

fn unique_integer_elements(
    storage: &IntegerStorage,
    shape: Vec<usize>,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    let values = storage.exact_values();
    if values.is_empty() {
        let output_shape = unique_element_values_shape(&shape, 0);
        let output = Tensor::new_integer(storage.zeros_like(0), output_shape)
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let empty = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(
            Value::Tensor(output),
            empty.clone(),
            empty,
        ));
    }

    let mut entries = Vec::<IntegerElementEntry>::new();
    let mut map = HashMap::<IntValue, usize>::new();
    let mut element_entry_index = Vec::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        match map.get(value) {
            Some(&entry_idx) => {
                entries[entry_idx].last = idx;
                element_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(IntegerElementEntry {
                    value: value.clone(),
                    first: idx,
                    last: idx,
                });
                map.insert(value.clone(), entry_idx);
                element_entry_index.push(entry_idx);
            }
        }
    }
    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| {
            integer_order::compare(&entries[a].value, &entries[b].value, false, false)
        });
    }
    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }
    let output_values: Vec<_> = order
        .iter()
        .map(|&entry_idx| entries[entry_idx].value.clone())
        .collect();
    let ia: Vec<_> = order
        .iter()
        .map(|&entry_idx| {
            let entry = &entries[entry_idx];
            (match opts.occurrence {
                UniqueOccurrence::First => entry.first,
                UniqueOccurrence::Last => entry.last,
            } + 1) as f64
        })
        .collect();
    let ic: Vec<_> = element_entry_index
        .into_iter()
        .map(|entry_idx| (entry_to_position[entry_idx] + 1) as f64)
        .collect();
    let output = Tensor::new_integer(
        storage
            .from_exact_values_like(output_values)
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?,
        unique_element_values_shape(&shape, order.len()),
    )
    .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ia = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ic = Tensor::new(ic, vec![values.len(), 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    Ok(UniqueEvaluation::new(Value::Tensor(output), ia, ic))
}

fn unique_integer_rows(
    storage: &IntegerStorage,
    shape: Vec<usize>,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if shape.len() != 2 {
        return Err(unique_error_with(
            &UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX,
            UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX.message,
        ));
    }
    let rows = shape[0];
    let cols = shape[1];
    if rows == 0 || cols == 0 {
        let output = Tensor::new_integer(storage.zeros_like(0), vec![0, cols])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![rows, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(Value::Tensor(output), ia, ic));
    }
    let values = storage.exact_values();
    let mut entries = Vec::<IntegerRowEntry>::new();
    let mut map = HashMap::<Vec<IntValue>, usize>::new();
    let mut row_entry_index = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_data: Vec<_> = (0..cols)
            .map(|col| values[row + col * rows].clone())
            .collect();
        match map.get(&row_data) {
            Some(&entry_idx) => {
                entries[entry_idx].last = row;
                row_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(IntegerRowEntry {
                    row_data: row_data.clone(),
                    first: row,
                    last: row,
                });
                map.insert(row_data, entry_idx);
                row_entry_index.push(entry_idx);
            }
        }
    }
    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_integer_rows(&entries[a].row_data, &entries[b].row_data));
    }
    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }
    let mut output_values = Vec::with_capacity(order.len() * cols);
    for col in 0..cols {
        for &entry_idx in &order {
            output_values.push(entries[entry_idx].row_data[col].clone());
        }
    }
    let ia: Vec<_> = order
        .iter()
        .map(|&entry_idx| {
            let entry = &entries[entry_idx];
            (match opts.occurrence {
                UniqueOccurrence::First => entry.first,
                UniqueOccurrence::Last => entry.last,
            } + 1) as f64
        })
        .collect();
    let ic: Vec<_> = row_entry_index
        .into_iter()
        .map(|entry_idx| (entry_to_position[entry_idx] + 1) as f64)
        .collect();
    let output = Tensor::new_integer(
        storage
            .from_exact_values_like(output_values)
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?,
        vec![order.len(), cols],
    )
    .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ia = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ic = Tensor::new(ic, vec![rows, 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    Ok(UniqueEvaluation::new(Value::Tensor(output), ia, ic))
}

fn unique_floating_elements<T: SetFloat>(
    input: Vec<T>,
    shape: Vec<usize>,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    let len = input.len();
    if len == 0 {
        let values = Tensor::from_numeric_storage(
            T::numeric_storage(Vec::new()),
            unique_element_values_shape(&shape, 0),
        )
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(
            tensor::tensor_into_value(values),
            ia,
            ic,
        ));
    }

    let mut entries = Vec::<FloatingElementEntry<T>>::new();
    let mut map: HashMap<u64, usize> = HashMap::new();
    let mut element_entry_index = Vec::with_capacity(len);

    for (idx, &value) in input.iter().enumerate() {
        if opts.treat_missing_as_distinct && value.is_nan() {
            let entry_idx = entries.len();
            entries.push(FloatingElementEntry {
                value,
                first: idx,
                last: idx,
            });
            element_entry_index.push(entry_idx);
            continue;
        }
        let key = value.canonical_key();
        match map.get(&key) {
            Some(&entry_idx) => {
                entries[entry_idx].last = idx;
                element_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(FloatingElementEntry {
                    value,
                    first: idx,
                    last: idx,
                });
                map.insert(key, entry_idx);
                element_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| entries[a].value.compare(entries[b].value));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        values.push(entry.value);
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(len);
    for entry_idx in element_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_tensor = Tensor::from_numeric_storage(
        T::numeric_storage(values),
        unique_element_values_shape(&shape, order.len()),
    )
    .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ic_tensor =
        Tensor::new(ic, vec![len, 1]).map_err(|e| unique_internal_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        tensor::tensor_into_value(value_tensor),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_floating_rows<T: SetFloat>(
    input: Vec<T>,
    shape: Vec<usize>,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if shape.len() != 2 {
        return Err(unique_error_with(
            &UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX,
            UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX.message,
        ));
    }
    let rows = shape[0];
    let cols = shape[1];

    if rows == 0 || cols == 0 {
        let values = Tensor::from_numeric_storage(T::numeric_storage(Vec::new()), vec![0, cols])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![rows, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(
            tensor::tensor_into_value(values),
            ia,
            ic,
        ));
    }

    let mut entries = Vec::<FloatingRowEntry<T>>::new();
    let mut map: HashMap<FloatingRowKey, usize> = HashMap::new();
    let mut row_entry_index = Vec::with_capacity(rows);

    for r in 0..rows {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows;
            row_values.push(input[idx]);
        }
        if opts.treat_missing_as_distinct && row_values.iter().any(|value| value.is_nan()) {
            let entry_idx = entries.len();
            entries.push(FloatingRowEntry {
                row_data: row_values,
                first: r,
                last: r,
            });
            row_entry_index.push(entry_idx);
            continue;
        }
        let key = FloatingRowKey::from_slice(&row_values);
        match map.get(&key) {
            Some(&entry_idx) => {
                entries[entry_idx].last = r;
                row_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(FloatingRowEntry {
                    row_data: row_values.clone(),
                    first: r,
                    last: r,
                });
                map.insert(key, entry_idx);
                row_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_floating_rows(&entries[a].row_data, &entries[b].row_data));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let unique_rows_count = order.len();
    let mut values = vec![T::default(); unique_rows_count * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let row = &entries[entry_idx].row_data;
        for (col, value) in row.iter().enumerate().take(cols) {
            let dest = row_pos + col * unique_rows_count;
            values[dest] = *value;
        }
    }

    let mut ia = Vec::with_capacity(unique_rows_count);
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(rows);
    for entry_idx in row_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_tensor =
        Tensor::from_numeric_storage(T::numeric_storage(values), vec![unique_rows_count, cols])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows_count, 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![rows, 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        tensor::tensor_into_value(value_tensor),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_complex_from_tensor(
    tensor: ComplexTensor,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    let shape = tensor.shape.clone();
    match tensor.into_complex_storage() {
        ComplexStorage::F64(values) => unique_floating_complex(values, shape, opts),
        ComplexStorage::F32(values) => unique_floating_complex(values, shape, opts),
        ComplexStorage::Integer(_) => Err(unique_error_with(
            &UNIQUE_ERROR_UNSUPPORTED_INPUT_TYPE,
            "unique: complex integer arrays are not supported",
        )),
    }
}

fn unique_floating_complex<T: SetFloat>(
    values: Vec<(T, T)>,
    shape: Vec<usize>,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if opts.rows {
        unique_complex_rows(values, shape, opts)
    } else {
        unique_complex_elements(values, shape, opts)
    }
}

fn unique_complex_elements<T: SetFloat>(
    input: Vec<(T, T)>,
    shape: Vec<usize>,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    let len = input.len();
    if len == 0 {
        let values = ComplexTensor::from_complex_storage(
            T::complex_storage(Vec::new()),
            unique_element_values_shape(&shape, 0),
        )
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(
            complex_tensor_into_value(values),
            ia,
            ic,
        ));
    }

    let mut entries = Vec::<ComplexElementEntry<T>>::new();
    let mut map: HashMap<ComplexKey, usize> = HashMap::new();
    let mut element_entry_index = Vec::with_capacity(len);

    for (idx, &value) in input.iter().enumerate() {
        if opts.treat_missing_as_distinct && (value.0.is_nan() || value.1.is_nan()) {
            let entry_idx = entries.len();
            entries.push(ComplexElementEntry {
                value,
                first: idx,
                last: idx,
            });
            element_entry_index.push(entry_idx);
            continue;
        }
        let key = ComplexKey::new(value);
        match map.get(&key) {
            Some(&entry_idx) => {
                entries[entry_idx].last = idx;
                element_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(ComplexElementEntry {
                    value,
                    first: idx,
                    last: idx,
                });
                map.insert(key, entry_idx);
                element_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_complex(entries[a].value, entries[b].value));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        values.push(entry.value);
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(len);
    for entry_idx in element_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_tensor = ComplexTensor::from_complex_storage(
        T::complex_storage(values),
        unique_element_values_shape(&shape, order.len()),
    )
    .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ic_tensor =
        Tensor::new(ic, vec![len, 1]).map_err(|e| unique_internal_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        complex_tensor_into_value(value_tensor),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_complex_rows<T: SetFloat>(
    input: Vec<(T, T)>,
    shape: Vec<usize>,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if shape.len() != 2 {
        return Err(unique_error_with(
            &UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX,
            UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX.message,
        ));
    }
    let rows = shape[0];
    let cols = shape[1];

    if rows == 0 || cols == 0 {
        let values =
            ComplexTensor::from_complex_storage(T::complex_storage(Vec::new()), vec![rows, cols])
                .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![rows, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(
            complex_tensor_into_value(values),
            ia,
            ic,
        ));
    }

    let mut entries = Vec::<ComplexRowEntry<T>>::new();
    let mut map: HashMap<Vec<ComplexKey>, usize> = HashMap::new();
    let mut row_entry_index = Vec::with_capacity(rows);

    for r in 0..rows {
        let mut row_values = Vec::with_capacity(cols);
        let mut key_row = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows;
            let value = input[idx];
            row_values.push(value);
            key_row.push(ComplexKey::new(value));
        }
        if opts.treat_missing_as_distinct
            && row_values
                .iter()
                .any(|value| value.0.is_nan() || value.1.is_nan())
        {
            let entry_idx = entries.len();
            entries.push(ComplexRowEntry {
                row_data: row_values,
                first: r,
                last: r,
            });
            row_entry_index.push(entry_idx);
            continue;
        }
        match map.get(&key_row) {
            Some(&entry_idx) => {
                entries[entry_idx].last = r;
                row_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(ComplexRowEntry {
                    row_data: row_values.clone(),
                    first: r,
                    last: r,
                });
                map.insert(key_row, entry_idx);
                row_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_complex_rows(&entries[a].row_data, &entries[b].row_data));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let unique_rows_count = order.len();
    let mut values = vec![(T::default(), T::default()); unique_rows_count * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let row = &entries[entry_idx].row_data;
        for (col, value) in row.iter().enumerate().take(cols) {
            let dest = row_pos + col * unique_rows_count;
            values[dest] = *value;
        }
    }

    let mut ia = Vec::with_capacity(unique_rows_count);
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(rows);
    for entry_idx in row_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_tensor = ComplexTensor::from_complex_storage(
        T::complex_storage(values),
        vec![unique_rows_count, cols],
    )
    .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows_count, 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![rows, 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        complex_tensor_into_value(value_tensor),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_char_array(
    array: CharArray,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if opts.rows {
        unique_char_rows(array, opts)
    } else {
        unique_char_elements(array, opts)
    }
}

fn unique_char_elements(
    array: CharArray,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    let shape = array.shape.clone();
    let input = array.to_column_major();
    let total = input.len();
    if total == 0 {
        let values =
            CharArray::from_column_major(Vec::new(), unique_element_values_shape(&shape, 0))
                .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(Value::CharArray(values), ia, ic));
    }

    let mut entries = Vec::<CharElementEntry>::new();
    let mut map: HashMap<u32, usize> = HashMap::new();
    let mut element_entry_index = Vec::with_capacity(total);

    for (linear_idx, &ch) in input.iter().enumerate() {
        let key = ch as u32;
        match map.get(&key) {
            Some(&entry_idx) => {
                entries[entry_idx].last = linear_idx;
                element_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(CharElementEntry {
                    ch,
                    first: linear_idx,
                    last: linear_idx,
                });
                map.insert(key, entry_idx);
                element_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| entries[a].ch.cmp(&entries[b].ch));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        values.push(entry.ch);
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(total);
    for entry_idx in element_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_array =
        CharArray::from_column_major(values, unique_element_values_shape(&shape, order.len()))
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![total, 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_char_rows(
    array: CharArray,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if array.shape.len() != 2 {
        return Err(unique_error_with(
            &UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX,
            UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX.message,
        ));
    }
    let rows = array.rows;
    let cols = array.cols;
    if rows == 0 {
        let values = CharArray::new(Vec::new(), 0, cols)
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(Value::CharArray(values), ia, ic));
    }

    let mut entries = Vec::<CharRowEntry>::new();
    let mut map: HashMap<RowCharKey, usize> = HashMap::new();
    let mut row_entry_index = Vec::with_capacity(rows);

    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let slice = &array.data[start..end];
        let key = RowCharKey::from_slice(slice);
        match map.get(&key) {
            Some(&entry_idx) => {
                entries[entry_idx].last = r;
                row_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(CharRowEntry {
                    row_data: slice.to_vec(),
                    first: r,
                    last: r,
                });
                map.insert(key, entry_idx);
                row_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_char_rows(&entries[a].row_data, &entries[b].row_data));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let unique_rows_count = order.len();
    let mut values = vec!['\0'; unique_rows_count * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let row = &entries[entry_idx].row_data;
        for col in 0..cols {
            let dest = row_pos * cols + col;
            if col < row.len() {
                values[dest] = row[col];
            }
        }
    }

    let mut ia = Vec::with_capacity(unique_rows_count);
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(rows);
    for entry_idx in row_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_array = CharArray::new(values, unique_rows_count, cols)
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows_count, 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![rows, 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_string_array(
    array: StringArray,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if opts.rows {
        unique_string_rows(array, opts)
    } else {
        unique_string_elements(array, opts)
    }
}

fn unique_string_elements(
    array: StringArray,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    let shape = array.shape.clone();
    let len = array.data.len();
    if len == 0 {
        let values = StringArray::new(Vec::new(), unique_element_values_shape(&shape, 0))
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(Value::StringArray(values), ia, ic));
    }

    let mut entries = Vec::<StringElementEntry>::new();
    let mut map: HashMap<String, usize> = HashMap::new();
    let mut element_entry_index = Vec::with_capacity(len);

    for (idx, value) in array.data.iter().enumerate() {
        match map.get(value) {
            Some(&entry_idx) => {
                entries[entry_idx].last = idx;
                element_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(StringElementEntry {
                    value: value.clone(),
                    first: idx,
                    last: idx,
                });
                map.insert(value.clone(), entry_idx);
                element_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| entries[a].value.cmp(&entries[b].value));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        values.push(entry.value.clone());
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(len);
    for entry_idx in element_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_array = StringArray::new(values, unique_element_values_shape(&shape, order.len()))
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ic_tensor =
        Tensor::new(ic, vec![len, 1]).map_err(|e| unique_internal_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_string_rows(
    array: StringArray,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if array.shape.len() != 2 {
        return Err(unique_error_with(
            &UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX,
            UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX.message,
        ));
    }
    let rows = array.shape[0];
    let cols = array.shape[1];

    if rows == 0 {
        let values = StringArray::new(Vec::new(), vec![0, cols])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1])
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(Value::StringArray(values), ia, ic));
    }

    let mut entries = Vec::<StringRowEntry>::new();
    let mut map: HashMap<RowStringKey, usize> = HashMap::new();
    let mut row_entry_index = Vec::with_capacity(rows);

    for r in 0..rows {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows;
            row_values.push(array.data[idx].clone());
        }
        let key = RowStringKey(row_values.clone());
        match map.get(&key) {
            Some(&entry_idx) => {
                entries[entry_idx].last = r;
                row_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(StringRowEntry {
                    row_data: row_values.clone(),
                    first: r,
                    last: r,
                });
                map.insert(key, entry_idx);
                row_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_string_rows(&entries[a].row_data, &entries[b].row_data));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let unique_rows_count = order.len();
    let mut values = vec![String::new(); unique_rows_count * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let row = &entries[entry_idx].row_data;
        for (col, value) in row.iter().enumerate().take(cols) {
            let dest = row_pos + col * unique_rows_count;
            values[dest] = value.clone();
        }
    }

    let mut ia = Vec::with_capacity(unique_rows_count);
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(rows);
    for entry_idx in row_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_array = StringArray::new(values, vec![unique_rows_count, cols])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows_count, 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![rows, 1])
        .map_err(|e| unique_internal_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
        ic_tensor,
    ))
}

#[derive(Debug)]
struct FloatingElementEntry<T> {
    value: T,
    first: usize,
    last: usize,
}

#[derive(Debug)]
struct IntegerElementEntry {
    value: IntValue,
    first: usize,
    last: usize,
}

#[derive(Debug)]
struct IntegerRowEntry {
    row_data: Vec<IntValue>,
    first: usize,
    last: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FloatingRowKey(Vec<u64>);

impl FloatingRowKey {
    fn from_slice<T: SetFloat>(values: &[T]) -> Self {
        Self(values.iter().map(|&value| value.canonical_key()).collect())
    }
}

#[derive(Debug, Clone)]
struct FloatingRowEntry<T> {
    row_data: Vec<T>,
    first: usize,
    last: usize,
}

#[derive(Debug)]
struct ComplexElementEntry<T> {
    value: (T, T),
    first: usize,
    last: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ComplexKey {
    re: u64,
    im: u64,
}

impl ComplexKey {
    fn new<T: SetFloat>(value: (T, T)) -> Self {
        Self {
            re: value.0.canonical_key(),
            im: value.1.canonical_key(),
        }
    }
}

#[derive(Debug, Clone)]
struct ComplexRowEntry<T> {
    row_data: Vec<(T, T)>,
    first: usize,
    last: usize,
}

#[derive(Debug)]
struct CharElementEntry {
    ch: char,
    first: usize,
    last: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowCharKey(Vec<u32>);

impl RowCharKey {
    fn from_slice(values: &[char]) -> Self {
        RowCharKey(values.iter().map(|&ch| ch as u32).collect())
    }
}

#[derive(Debug, Clone)]
struct CharRowEntry {
    row_data: Vec<char>,
    first: usize,
    last: usize,
}

#[derive(Debug, Clone)]
struct StringElementEntry {
    value: String,
    first: usize,
    last: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowStringKey(Vec<String>);

#[derive(Debug, Clone)]
struct StringRowEntry {
    row_data: Vec<String>,
    first: usize,
    last: usize,
}

fn compare_floating_rows<T: SetFloat>(a: &[T], b: &[T]) -> Ordering {
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        let ord = lhs.compare(*rhs);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_integer_rows(a: &[IntValue], b: &[IntValue]) -> Ordering {
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        let ordering = integer_order::compare(lhs, rhs, false, false);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

fn complex_is_nan<T: SetFloat>(value: (T, T)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn compare_complex<T: SetFloat>(a: (T, T), b: (T, T)) -> Ordering {
    match (complex_is_nan(a), complex_is_nan(b)) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            let mag_a = a.0.hypot(a.1);
            let mag_b = b.0.hypot(b.1);
            let mag_cmp = mag_a.compare(mag_b);
            if mag_cmp != Ordering::Equal {
                return mag_cmp;
            }
            let re_cmp = a.0.compare(b.0);
            if re_cmp != Ordering::Equal {
                return re_cmp;
            }
            a.1.compare(b.1)
        }
    }
}

fn compare_complex_rows<T: SetFloat>(a: &[(T, T)], b: &[(T, T)]) -> Ordering {
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        let ord = compare_complex(*lhs, *rhs);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_char_rows(a: &[char], b: &[char]) -> Ordering {
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        let ord = lhs.cmp(rhs);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_string_rows(a: &[String], b: &[String]) -> Ordering {
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        let ord = lhs.cmp(rhs);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

#[derive(Debug)]
pub struct UniqueEvaluation {
    values: Value,
    ia: Tensor,
    ic: Tensor,
}

impl UniqueEvaluation {
    fn new(values: Value, ia: Tensor, ic: Tensor) -> Self {
        Self { values, ia, ic }
    }

    pub fn into_values_value(self) -> Value {
        self.values
    }

    fn into_logical_values(mut self) -> crate::BuiltinResult<Self> {
        self.values = match self.values {
            Value::Num(value) => Value::Bool(value != 0.0),
            Value::Tensor(tensor) => {
                let shape = tensor.shape.clone();
                let data = tensor
                    .materialize_f64()
                    .into_iter()
                    .map(|value| u8::from(value != 0.0))
                    .collect();
                Value::LogicalArray(
                    LogicalArray::new(data, shape)
                        .map_err(|error| unique_internal_error(format!("unique: {error}")))?,
                )
            }
            other => {
                return Err(unique_internal_error(format!(
                    "unique: cannot restore logical values from {other:?}"
                )));
            }
        };
        Ok(self)
    }

    pub fn into_pair(self) -> (Value, Value) {
        let ia = tensor::tensor_into_value(self.ia);
        (self.values, ia)
    }

    pub fn into_triple(self) -> (Value, Value, Value) {
        let ia = tensor::tensor_into_value(self.ia);
        let ic = tensor::tensor_into_value(self.ic);
        (self.values, ia, ic)
    }

    pub fn from_unique_result(result: UniqueResult) -> crate::BuiltinResult<Self> {
        let UniqueResult { values, ia, ic } = result;
        let values_tensor = Tensor::new(values.data, values.shape)
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ia_tensor = Tensor::new(ia.data, ia.shape)
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        let ic_tensor = Tensor::new(ic.data, ic.shape)
            .map_err(|e| unique_internal_error(format!("unique: {e}")))?;
        Ok(UniqueEvaluation::new(
            tensor::tensor_into_value(values_tensor),
            ia_tensor,
            ic_tensor,
        ))
    }

    pub fn into_numeric_unique_result(self) -> crate::BuiltinResult<UniqueResult> {
        let UniqueEvaluation { values, ia, ic } = self;
        let values_tensor = tensor::value_into_tensor_for("unique", values)
            .map_err(|e| unique_internal_error(e))?;
        Ok(UniqueResult {
            values: tensor::tensor_into_host_f64_owned(values_tensor),
            ia: tensor::tensor_into_host_f64_owned(ia),
            ic: tensor::tensor_into_host_f64_owned(ic),
        })
    }

    pub fn ia_value(&self) -> Value {
        tensor::tensor_into_value(self.ia.clone())
    }

    pub fn ic_value(&self) -> Value {
        tensor::tensor_into_value(self.ic.clone())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{
        CharArray, IntValue, IntegerStorage, LiteralValue, LogicalArray, ResolveContext,
        StringArray, Tensor, Type, Value,
    };

    fn evaluate_sync(value: Value, rest: &[Value]) -> crate::BuiltinResult<UniqueEvaluation> {
        futures::executor::block_on(evaluate(value, rest))
    }

    fn builtin_sync(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        futures::executor::block_on(unique_builtin(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_sorted_default() {
        let tensor = Tensor::new(vec![3.0, 1.0, 3.0, 2.0], vec![4, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(tensor), &[]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0]);
                assert_eq!(t.shape, vec![3, 1]);
            }
            Value::Num(_) => panic!("expected tensor result"),
            other => panic!("unexpected result {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![2.0, 4.0, 1.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![3.0, 1.0, 3.0, 2.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[test]
    fn unique_type_resolver_numeric() {
        assert_eq!(
            unique_values_output_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(1), Some(4)])
                }],
                &ResolveContext::new(vec![LiteralValue::Unknown]),
            ),
            Type::Tensor {
                shape: Some(vec![Some(1), None])
            }
        );
        assert_eq!(
            unique_values_output_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(4), Some(1)])
                }],
                &ResolveContext::new(vec![LiteralValue::Unknown]),
            ),
            Type::Tensor {
                shape: Some(vec![None, Some(1)])
            }
        );
        assert_eq!(
            unique_values_output_type(
                &[
                    Type::Tensor {
                        shape: Some(vec![Some(4), Some(3)])
                    },
                    Type::String,
                ],
                &ResolveContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::String("rows".into()),
                ]),
            ),
            Type::Tensor {
                shape: Some(vec![None, Some(3)])
            }
        );
    }

    #[test]
    fn unique_type_resolver_string_array() {
        assert_eq!(
            unique_values_output_type(
                &[Type::cell_of(Type::String)],
                &ResolveContext::new(Vec::new()),
            ),
            Type::cell_of(Type::String)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_sorted_handles_nan() {
        let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 1.0], vec![4, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(tensor), &[]).expect("unique");
        let (values, ..) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64().len(), 4);
                assert_eq!(t.materialize_f64()[0], 1.0);
                assert_eq!(t.materialize_f64()[1], 2.0);
                assert!(t.materialize_f64()[2].is_nan());
                assert!(t.materialize_f64()[3].is_nan());
            }
            other => panic!("unexpected values {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_stable_with_nan() {
        let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 1.0], vec![4, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(tensor), &[Value::from("stable")]).expect("unique");
        let (values, ..) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert!(t.materialize_f64()[0].is_nan());
                assert_eq!(t.materialize_f64()[1], 2.0);
                assert!(t.materialize_f64()[2].is_nan());
                assert_eq!(t.materialize_f64()[3], 1.0);
            }
            other => panic!("unexpected values {other:?}"),
        }
    }

    #[test]
    fn unique_treat_missing_as_distinct_name_value_controls_nan_grouping() {
        let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN], vec![3, 1]).unwrap();
        let collapsed = evaluate_sync(
            Value::Tensor(tensor.clone()),
            &[Value::from("TreatMissingAsDistinct"), Value::Bool(false)],
        )
        .expect("collapsed missing values")
        .into_values_value();
        let Value::Tensor(collapsed) = collapsed else {
            panic!("expected tensor");
        };
        assert_eq!(collapsed.materialize_f64().len(), 2);
        assert!(collapsed.materialize_f64()[1].is_nan());

        let distinct = evaluate_sync(
            Value::Tensor(tensor),
            &[
                Value::from("TreatMissingAsDistinct"),
                Value::Int(IntValue::U8(1)),
            ],
        )
        .expect("distinct missing values")
        .into_values_value();
        let Value::Tensor(distinct) = distinct else {
            panic!("expected tensor");
        };
        assert_eq!(distinct.materialize_f64().len(), 3);

        let err = parse_options(&[Value::from("TreatMissingAsDistinct"), Value::Num(2.0)])
            .expect_err("non-logical flag must fail");
        assert_eq!(err.identifier(), UNIQUE_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_stable_preserves_order() {
        let tensor = Tensor::new(vec![4.0, 2.0, 4.0, 1.0, 2.0], vec![5, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(tensor), &[Value::from("stable")]).expect("unique");
        let (values, ia) = eval.into_pair();
        match values {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![4.0, 2.0, 1.0]),
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 4.0]),
            other => panic!("unexpected IA {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_last_occurrence() {
        let tensor = Tensor::new(vec![9.0, 8.0, 9.0, 7.0, 8.0], vec![5, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(tensor), &[Value::from("last")]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![7.0, 8.0, 9.0]),
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![4.0, 5.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![3.0, 2.0, 3.0, 1.0, 2.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_rows_sorted_default() {
        let tensor = Tensor::new(vec![1.0, 1.0, 2.0, 1.0, 3.0, 3.0, 4.0, 2.0], vec![4, 2]).unwrap();
        let eval = evaluate_sync(Value::Tensor(tensor), &[Value::from("rows")]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                assert_eq!(t.materialize_f64(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![4.0, 1.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![2.0, 2.0, 3.0, 1.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_rows_stable_last() {
        let tensor = Tensor::new(vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0], vec![3, 2]).unwrap();
        let eval = evaluate_sync(
            Value::Tensor(tensor),
            &[
                Value::from("rows"),
                Value::from("stable"),
                Value::from("last"),
            ],
        )
        .expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 1.0, 2.0]);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![2.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 1.0, 2.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_char_elements_sorted() {
        let chars = CharArray::new(vec!['m', 'z', 'm', 'a'], 2, 2).unwrap();
        let eval = evaluate_sync(Value::CharArray(chars), &[]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 3);
                assert_eq!(arr.cols, 1);
                assert_eq!(arr.data, vec!['a', 'm', 'z']);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![4.0, 1.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![2.0, 2.0, 3.0, 1.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_char_rows_last() {
        let chars = CharArray::new(vec!['a', 'b', 'a', 'b', 'a', 'c'], 3, 2).unwrap();
        let eval = evaluate_sync(
            Value::CharArray(chars),
            &[Value::from("rows"), Value::from("last")],
        )
        .expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 2);
                assert_eq!(arr.cols, 2);
                assert_eq!(arr.data, vec!['a', 'b', 'a', 'c']);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![2.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 1.0, 2.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_string_elements_stable() {
        let array = StringArray::new(
            vec!["beta".into(), "alpha".into(), "beta".into()],
            vec![3, 1],
        )
        .unwrap();
        let eval =
            evaluate_sync(Value::StringArray(array), &[Value::from("stable")]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::StringArray(sa) => {
                assert_eq!(sa.data, vec!["beta", "alpha"]);
                assert_eq!(sa.shape, vec![2, 1]);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 2.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 1.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_string_rows() {
        let array = StringArray::new(
            vec![
                "alpha".into(),
                "alpha".into(),
                "gamma".into(),
                "beta".into(),
                "beta".into(),
                "beta".into(),
            ],
            vec![3, 2],
        )
        .unwrap();
        let eval = evaluate_sync(
            Value::StringArray(array),
            &[Value::from("rows"), Value::from("stable")],
        )
        .expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(sa.data, vec!["alpha", "gamma", "beta", "beta"]);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 1.0, 2.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_complex_sorted() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (0.0, 2.0), (1.0, -1.0), (0.0, 2.0)],
            vec![4, 1],
        )
        .unwrap();
        let eval = evaluate_sync(Value::ComplexTensor(tensor), &[]).expect("unique");
        let (values, ..) = eval.into_triple();
        match values {
            Value::ComplexTensor(t) => {
                assert_eq!(t.materialize_f64().len(), 3);
                assert_eq!(t.materialize_f64()[0], (1.0, -1.0));
                assert_eq!(t.materialize_f64()[1], (1.0, 1.0));
                assert_eq!(t.materialize_f64()[2], (0.0, 2.0));
            }
            other => panic!("unexpected values {other:?}"),
        }
    }

    #[test]
    fn unique_preserves_native_single_elements_and_rows() {
        let elements = Tensor::from_f32(vec![3.0, 1.0, 3.0, 2.0], vec![4, 1]).unwrap();
        let values = evaluate_sync(Value::Tensor(elements), &[])
            .expect("unique single elements")
            .into_values_value();
        let Value::Tensor(values) = values else {
            panic!("expected native single values");
        };
        assert_eq!(
            values.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 2.0, 3.0])
        );

        let rows = Tensor::from_f32(vec![2.0, 1.0, 2.0, 20.0, 10.0, 20.0], vec![3, 2]).unwrap();
        let values = evaluate_sync(Value::Tensor(rows), &[Value::from("rows")])
            .expect("unique single rows")
            .into_values_value();
        let Value::Tensor(values) = values else {
            panic!("expected native single rows");
        };
        assert_eq!(values.shape, vec![2, 2]);
        assert_eq!(
            values.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 2.0, 10.0, 20.0])
        );
    }

    #[test]
    fn unique_preserves_native_complex_single_elements_and_rows() {
        let elements = ComplexTensor::from_f32(
            vec![(1.0, 1.0), (0.0, 2.0), (1.0, -1.0), (0.0, 2.0)],
            vec![4, 1],
        )
        .unwrap();
        let values = evaluate_sync(Value::ComplexTensor(elements), &[])
            .expect("unique complex single elements")
            .into_values_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex single values");
        };
        assert_eq!(
            values.as_f32_slice(),
            Some(&[(1.0, -1.0), (1.0, 1.0), (0.0, 2.0)][..])
        );

        let rows = ComplexTensor::from_f32(
            vec![
                (2.0, 0.0),
                (1.0, 1.0),
                (2.0, 0.0),
                (20.0, 0.0),
                (10.0, -1.0),
                (20.0, 0.0),
            ],
            vec![3, 2],
        )
        .unwrap();
        let values = evaluate_sync(
            Value::ComplexTensor(rows),
            &[Value::from("rows"), Value::from("stable")],
        )
        .expect("unique complex single rows")
        .into_values_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex single rows");
        };
        assert_eq!(values.shape, vec![2, 2]);
        assert_eq!(
            values.as_f32_slice(),
            Some(&[(2.0, 0.0), (1.0, 1.0), (20.0, 0.0), (10.0, -1.0),][..])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_handles_logical_arrays() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![1, 4]).unwrap();
        let eval = evaluate_sync(Value::LogicalArray(logical), &[]).expect("unique");
        let values = eval.into_values_value();
        match values {
            Value::LogicalArray(values) => {
                assert_eq!(values.shape, vec![1, 2]);
                assert_eq!(values.data, vec![0, 1]);
            }
            other => panic!("unexpected values {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![5.0, 3.0, 5.0, 1.0], vec![1, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval =
                evaluate_sync(Value::GpuTensor(handle), &[Value::from("stable")]).expect("unique");
            let values = eval.into_values_value();
            match values {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![1, 3]);
                    assert_eq!(t.materialize_f64(), vec![5.0, 3.0, 1.0]);
                }
                other => panic!("unexpected values {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn unique_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![5.0, 3.0, 5.0, 1.0, 2.0], vec![5, 1]).unwrap();
        let host_eval = evaluate_sync(Value::Tensor(tensor.clone()), &[]).expect("host unique");
        let (host_values, host_ia, host_ic) = host_eval.into_triple();

        let provider = runmat_accelerate_api::provider().expect("provider registered");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate_sync(Value::GpuTensor(handle.clone()), &[]).expect("gpu unique");
        let (gpu_values, gpu_ia, gpu_ic) = gpu_eval.into_triple();
        let _ = provider.free(&handle);

        let host_values = test_support::gather(host_values).expect("gather host values");
        let host_ia = test_support::gather(host_ia).expect("gather host ia");
        let host_ic = test_support::gather(host_ic).expect("gather host ic");
        let gpu_values = test_support::gather(gpu_values).expect("gather gpu values");
        let gpu_ia = test_support::gather(gpu_ia).expect("gather gpu ia");
        let gpu_ic = test_support::gather(gpu_ic).expect("gather gpu ic");

        assert_eq!(gpu_values.shape, host_values.shape);
        assert_eq!(gpu_values.materialize_f64(), host_values.materialize_f64());
        assert_eq!(gpu_ia.materialize_f64(), host_ia.materialize_f64());
        assert_eq!(gpu_ic.materialize_f64(), host_ic.materialize_f64());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_rejects_legacy_option() {
        let tensor = Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap();
        let err = evaluate_sync(Value::Tensor(tensor), &[Value::from("legacy")]).unwrap_err();
        assert_eq!(
            err.identifier(),
            UNIQUE_ERROR_LEGACY_OPTION_UNSUPPORTED.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_conflicting_order_flags() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = evaluate_sync(
            Value::Tensor(tensor),
            &[Value::from("stable"), Value::from("sorted")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            UNIQUE_ERROR_CONFLICTING_ORDER_OPTIONS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_conflicting_occurrence_flags() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = evaluate_sync(
            Value::Tensor(tensor),
            &[Value::from("first"), Value::from("last")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            UNIQUE_ERROR_CONFLICTING_OCCURRENCE_OPTIONS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_rejects_unknown_option() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = evaluate_sync(Value::Tensor(tensor), &[Value::from("bogus")]).unwrap_err();
        assert_eq!(err.identifier(), UNIQUE_ERROR_UNKNOWN_OPTION.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_rows_requires_two_dimensional_input() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1, 1]).unwrap();
        let err = evaluate_sync(Value::Tensor(tensor), &[Value::from("rows")]).unwrap_err();
        assert_eq!(
            err.identifier(),
            UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX.identifier
        );

        let chars = CharArray::from_column_major(vec!['a', 'b'], vec![1, 2, 1]).unwrap();
        let err = evaluate_sync(Value::CharArray(chars), &[Value::from("rows")]).unwrap_err();
        assert_eq!(
            err.identifier(),
            UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_handles_empty_rows() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let eval = evaluate_sync(Value::Tensor(tensor), &[Value::from("rows")]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 3]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert!(t.materialize_f64().is_empty()),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert!(t.materialize_f64().is_empty()),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_accepts_integer_scalars() {
        let eval = evaluate_sync(Value::Int(IntValue::I32(42)), &[]).expect("unique");
        let values = eval.into_values_value();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.integer_storage(), Some(&IntegerStorage::I32(vec![42])))
            }
            other => panic!("unexpected values {other:?}"),
        }
    }

    #[test]
    fn unique_preserves_exact_integer_elements_rows_and_indices() {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 0, 9_007_199_254_740_993, u64::MAX]),
            vec![4, 1],
        )
        .expect("input");
        let (values, ia, ic) = evaluate_sync(Value::Tensor(input), &[])
            .expect("unique")
            .into_triple();
        let Value::Tensor(values) = values else {
            panic!("expected exact integer values");
        };
        assert_eq!(
            values.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                0,
                9_007_199_254_740_993,
                u64::MAX
            ]))
        );
        let Value::Tensor(ia) = ia else {
            panic!("expected indices");
        };
        assert_eq!(ia.materialize_f64(), vec![2.0, 3.0, 1.0]);
        let Value::Tensor(ic) = ic else {
            panic!("expected indices");
        };
        assert_eq!(ic.materialize_f64(), vec![3.0, 1.0, 2.0, 3.0]);

        let rows = Tensor::new_integer(
            IntegerStorage::I64(vec![i64::MAX, i64::MIN, i64::MAX, 1, 2, 1]),
            vec![3, 2],
        )
        .expect("input");
        let (values, ia, ic) = evaluate_sync(
            Value::Tensor(rows),
            &[
                Value::from("rows"),
                Value::from("stable"),
                Value::from("last"),
            ],
        )
        .expect("unique rows")
        .into_triple();
        let Value::Tensor(values) = values else {
            panic!("expected exact integer rows");
        };
        assert_eq!(
            values.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MAX, i64::MIN, 1, 2]))
        );
        let Value::Tensor(ia) = ia else {
            panic!("expected indices");
        };
        assert_eq!(ia.materialize_f64(), vec![3.0, 2.0]);
        let Value::Tensor(ic) = ic else {
            panic!("expected indices");
        };
        assert_eq!(ic.materialize_f64(), vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn unique_numeric_fallback_reads_mirrorless_integer_storage() {
        let opts = parse_options(&[]).expect("options");
        let input =
            Tensor::new_integer(IntegerStorage::U16(vec![7, 2, 7, 9]), vec![4, 1]).expect("input");
        let (values, ia, ic) = unique_numeric_from_tensor(input, &opts)
            .expect("unique numeric elements")
            .into_triple();
        let Value::Tensor(values) = values else {
            panic!("expected numeric values");
        };
        assert_eq!(values.materialize_f64(), vec![2.0, 7.0, 9.0]);
        let Value::Tensor(ia) = ia else {
            panic!("expected indices");
        };
        assert_eq!(ia.materialize_f64(), vec![2.0, 1.0, 4.0]);
        let Value::Tensor(ic) = ic else {
            panic!("expected inverse indices");
        };
        assert_eq!(ic.materialize_f64(), vec![2.0, 1.0, 2.0, 3.0]);

        let rows = Tensor::new_integer(IntegerStorage::U16(vec![1, 3, 1, 2, 4, 2]), vec![3, 2])
            .expect("rows");
        let row_opts = parse_options(&[Value::from("rows")]).expect("row options");
        let (values, ia, ic) = unique_numeric_from_tensor(rows, &row_opts)
            .expect("unique numeric rows")
            .into_triple();
        let Value::Tensor(values) = values else {
            panic!("expected numeric rows");
        };
        assert_eq!(values.shape, vec![2, 2]);
        assert_eq!(values.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
        let Value::Tensor(ia) = ia else {
            panic!("expected row indices");
        };
        assert_eq!(ia.materialize_f64(), vec![1.0, 2.0]);
        let Value::Tensor(ic) = ic else {
            panic!("expected row inverse indices");
        };
        assert_eq!(ic.materialize_f64(), vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn unique_preserves_every_exact_integer_class() {
        let cases = [
            IntegerStorage::I8(vec![i8::MAX, i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MAX, i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MAX, i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MAX, i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![u8::MAX, 0, u8::MAX]),
            IntegerStorage::U16(vec![u16::MAX, 0, u16::MAX]),
            IntegerStorage::U32(vec![u32::MAX, 0, u32::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 0, u64::MAX]),
        ];
        for storage in cases {
            let expected = storage.clone();
            let tensor = Tensor::new_integer(storage, vec![3, 1]).expect("input");
            let values = evaluate_sync(Value::Tensor(tensor), &[Value::from("stable")])
                .expect("unique")
                .into_values_value();
            let Value::Tensor(values) = values else {
                panic!("expected exact integer values");
            };
            let mut expected_values = expected.exact_values();
            expected_values.truncate(2);
            assert_eq!(
                values.integer_storage(),
                Some(
                    &expected
                        .from_exact_values_like(expected_values)
                        .expect("expected")
                )
            );
        }
    }

    #[test]
    fn unique_preserves_row_orientation_for_every_exact_integer_class() {
        let cases = [
            IntegerStorage::I8(vec![3, 1, 3]),
            IntegerStorage::I16(vec![3, 1, 3]),
            IntegerStorage::I32(vec![3, 1, 3]),
            IntegerStorage::I64(vec![3, 1, 3]),
            IntegerStorage::U8(vec![3, 1, 3]),
            IntegerStorage::U16(vec![3, 1, 3]),
            IntegerStorage::U32(vec![3, 1, 3]),
            IntegerStorage::U64(vec![u64::MAX, 0, u64::MAX]),
        ];
        for storage in cases {
            let expected = storage
                .from_exact_values_like(vec![
                    storage.value_at(1).expect("second value"),
                    storage.value_at(0).expect("first value"),
                ])
                .expect("same-class expected values");
            let input = Tensor::new_integer(storage, vec![1, 3]).expect("row input");
            let (values, ia, ic) = evaluate_sync(Value::Tensor(input), &[])
                .expect("unique integer row")
                .into_triple();
            let Value::Tensor(values) = values else {
                panic!("expected exact integer values");
            };
            assert_eq!(values.shape, vec![1, 2]);
            assert_eq!(values.integer_storage(), Some(&expected));
            let Value::Tensor(ia) = ia else {
                panic!("expected ia");
            };
            let Value::Tensor(ic) = ic else {
                panic!("expected ic");
            };
            assert_eq!(ia.shape, vec![2, 1]);
            assert_eq!(ic.shape, vec![3, 1]);
        }
    }

    #[test]
    fn unique_preserves_empty_integer_row_orientation_and_column_default() {
        let cases = [
            IntegerStorage::I8(Vec::new()),
            IntegerStorage::I16(Vec::new()),
            IntegerStorage::I32(Vec::new()),
            IntegerStorage::I64(Vec::new()),
            IntegerStorage::U8(Vec::new()),
            IntegerStorage::U16(Vec::new()),
            IntegerStorage::U32(Vec::new()),
            IntegerStorage::U64(Vec::new()),
        ];
        for storage in cases {
            for (input_shape, expected_shape) in [
                (vec![1, 0], vec![1, 0]),
                (vec![0], vec![1, 0]),
                (vec![1, 0, 1], vec![1, 0]),
                (vec![0, 1], vec![0, 1]),
                (vec![0, 3], vec![0, 1]),
            ] {
                let input =
                    Tensor::new_integer(storage.clone(), input_shape).expect("empty integer input");
                let values = evaluate_sync(Value::Tensor(input), &[])
                    .expect("unique empty integer")
                    .into_values_value();
                let Value::Tensor(values) = values else {
                    panic!("expected exact integer values");
                };
                assert_eq!(values.shape, expected_shape);
                assert_eq!(values.integer_storage(), Some(&storage));
            }
        }
    }

    #[test]
    fn unique_element_orientation_is_shared_by_native_and_text_storage() {
        let single = Tensor::from_f32(vec![3.0, 1.0, 3.0], vec![1, 3]).expect("single row");
        let Value::Tensor(single) = evaluate_sync(Value::Tensor(single), &[])
            .expect("unique single row")
            .into_values_value()
        else {
            panic!("expected single tensor");
        };
        assert_eq!(single.shape, vec![1, 2]);
        assert_eq!(single.as_f32_slice(), Some(&[1.0, 3.0][..]));

        let complex = ComplexTensor::from_f32(vec![(3.0, 1.0), (1.0, 0.0), (3.0, 1.0)], vec![1, 3])
            .expect("complex row");
        let Value::ComplexTensor(complex) = evaluate_sync(Value::ComplexTensor(complex), &[])
            .expect("unique complex row")
            .into_values_value()
        else {
            panic!("expected complex tensor");
        };
        assert_eq!(complex.shape, vec![1, 2]);

        let chars = CharArray::new(vec!['z', 'a', 'z'], 1, 3).expect("character row input");
        let Value::CharArray(chars) = evaluate_sync(Value::CharArray(chars), &[])
            .expect("unique character row")
            .into_values_value()
        else {
            panic!("expected character array");
        };
        assert_eq!(chars.shape, vec![1, 2]);
        assert_eq!(chars.data, vec!['a', 'z']);

        let strings = StringArray::new(vec!["z".into(), "a".into(), "z".into()], vec![1, 3])
            .expect("string row input");
        let Value::StringArray(strings) = evaluate_sync(Value::StringArray(strings), &[])
            .expect("unique string row")
            .into_values_value()
        else {
            panic!("expected string array");
        };
        assert_eq!(strings.shape, vec![1, 2]);

        let nd_chars = CharArray::from_column_major(vec!['c', 'a', 'b', 'c'], vec![2, 1, 2])
            .expect("N-D character input");
        let Value::CharArray(nd_chars) = evaluate_sync(Value::CharArray(nd_chars), &[])
            .expect("unique N-D character input")
            .into_values_value()
        else {
            panic!("expected character array");
        };
        assert_eq!(nd_chars.shape, vec![3, 1]);
        assert_eq!(nd_chars.to_column_major(), vec!['a', 'b', 'c']);
    }

    #[test]
    fn unique_registered_builtin_restores_resident_integer_and_logical_outputs() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new_integer(IntegerStorage::I32(vec![7, 0, 7]), vec![1, 3])
                .expect("integer row");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("typed upload");
            {
                let _guard = crate::output_count::push_output_count(Some(3));
                let Value::OutputList(outputs) = builtin_sync(Value::GpuTensor(handle), Vec::new())
                    .expect("resident integer unique")
                else {
                    panic!("expected output list");
                };
                assert_eq!(outputs.len(), 3);
                assert!(outputs
                    .iter()
                    .all(|output| matches!(output, Value::GpuTensor(_))));
                assert_eq!(
                    test_support::gather(outputs[0].clone())
                        .expect("gather values")
                        .integer_storage(),
                    Some(&IntegerStorage::I32(vec![0, 7]))
                );
            };

            let logical = Tensor::new(vec![1.0, 0.0, 1.0], vec![1, 3]).unwrap();
            let logical_handle =
                gpu_helpers::upload_tensor(provider, &logical).expect("logical upload");
            let logical_input = gpu_helpers::logical_gpu_value(logical_handle);
            let Value::GpuTensor(output) =
                builtin_sync(logical_input, Vec::new()).expect("resident logical unique")
            else {
                panic!("expected resident logical result");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&output));
        });
    }

    #[test]
    fn unique_resident_restrictions_and_excess_output_arity_are_enforced() {
        test_support::with_test_provider(|provider| {
            let input =
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 0, u64::MAX]), vec![1, 3])
                    .expect("integer row");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("typed upload");
            let err = evaluate_sync(Value::GpuTensor(handle), &[])
                .expect_err("resident uint64 must fail");
            assert_eq!(
                err.identifier(),
                UNIQUE_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
            );

            let input = Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let err = evaluate_sync(
                Value::GpuTensor(handle),
                &[Value::from("stable"), Value::from("last")],
            )
            .expect_err("GPU set-order plus occurrence must fail");
            assert_eq!(
                err.identifier(),
                UNIQUE_ERROR_GPU_OPTION_COMBINATION.identifier
            );
        });

        let _guard = crate::output_count::push_output_count(Some(4));
        let err = builtin_sync(Value::Num(1.0), Vec::new()).expect_err("excess outputs must fail");
        assert_eq!(err.identifier(), UNIQUE_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn unique_rows_and_complex_values_honor_missing_distinctness() {
        let rows = Tensor::new(vec![f64::NAN, f64::NAN, 1.0, 1.0], vec![2, 2]).unwrap();
        let distinct = evaluate_sync(Value::Tensor(rows.clone()), &[Value::from("rows")])
            .expect("distinct rows")
            .into_values_value();
        let Value::Tensor(distinct) = distinct else {
            panic!("expected rows");
        };
        assert_eq!(distinct.shape, vec![2, 2]);
        let collapsed = evaluate_sync(
            Value::Tensor(rows),
            &[
                Value::from("rows"),
                Value::from("TreatMissingAsDistinct"),
                Value::Bool(false),
            ],
        )
        .expect("collapsed rows")
        .into_values_value();
        let Value::Tensor(collapsed) = collapsed else {
            panic!("expected rows");
        };
        assert_eq!(collapsed.shape, vec![1, 2]);

        let complex =
            ComplexTensor::new(vec![(f64::NAN, 1.0), (f64::NAN, 1.0)], vec![2, 1]).unwrap();
        let distinct = evaluate_sync(Value::ComplexTensor(complex.clone()), &[])
            .expect("distinct complex")
            .into_values_value();
        let Value::ComplexTensor(distinct) = distinct else {
            panic!("expected complex values");
        };
        assert_eq!(distinct.len(), 2);
        let collapsed = evaluate_sync(
            Value::ComplexTensor(complex),
            &[Value::from("TreatMissingAsDistinct"), Value::Bool(false)],
        )
        .expect("collapsed complex")
        .into_values_value();
        assert!(
            matches!(collapsed, Value::Complex(real, imaginary) if real.is_nan() && imaginary == 1.0)
        );
    }
}
