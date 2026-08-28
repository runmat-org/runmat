//! MATLAB-compatible `strfind` builtin for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, Tensor, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};

use super::text_utils::{value_to_owned_string, TextCollection, TextElement};
use crate::builtins::strings::type_resolvers::text_search_indices_type;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::search::strfind")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strfind",
    op_kind: GpuOpKind::Custom("string-search"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Executes entirely on the host; GPU-resident inputs are gathered before substring matching.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::search::strfind")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strfind",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text operation; not eligible for fusion and materialises host-side numeric or cell outputs.",
};

const BUILTIN_NAME: &str = "strfind";

const STRFIND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "idx",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Match start indices as a numeric row vector or cell array of row vectors.",
}];

const STRFIND_INPUTS_BASE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "pat",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (string/char/cell/string array).",
    },
];

const STRFIND_INPUTS_FORCE_CELL_PAIR: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "pat",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ForceCellOutput\""),
        description: "Option name (`\"ForceCellOutput\"`).",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical-ish option value controlling forced cell output.",
    },
];

const STRFIND_INPUTS_OPTION_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "pat",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option pairs (`\"ForceCellOutput\"`, value).",
    },
];

const STRFIND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "idx = strfind(str, pat)",
        inputs: &STRFIND_INPUTS_BASE,
        outputs: &STRFIND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "idx = strfind(str, pat, \"ForceCellOutput\", value)",
        inputs: &STRFIND_INPUTS_FORCE_CELL_PAIR,
        outputs: &STRFIND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "idx = strfind(str, pat, nameValuePairs...)",
        inputs: &STRFIND_INPUTS_OPTION_PAIRS,
        outputs: &STRFIND_OUTPUT,
    },
];

const STRFIND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRFIND.INVALID_INPUT",
    identifier: Some("RunMat:strfind:InvalidInput"),
    when: "Text or pattern input is not a supported text container.",
    message: "strfind: text and pattern inputs must be text values",
};

const STRFIND_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRFIND.INVALID_OPTION",
    identifier: Some("RunMat:strfind:InvalidOption"),
    when: "ForceCellOutput option arguments are invalid or malformed.",
    message: "strfind: invalid option arguments",
};

const STRFIND_ERROR_SHAPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRFIND.SHAPE_MISMATCH",
    identifier: Some("RunMat:strfind:ShapeMismatch"),
    when: "Text and pattern inputs are not broadcast-compatible.",
    message: "strfind: input sizes are not broadcast-compatible",
};

const STRFIND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRFIND.INTERNAL",
    identifier: Some("RunMat:strfind:InternalError"),
    when: "Internal output assembly failed.",
    message: "strfind: internal error",
};

const STRFIND_ERRORS: [BuiltinErrorDescriptor; 4] = [
    STRFIND_ERROR_INVALID_INPUT,
    STRFIND_ERROR_INVALID_OPTION,
    STRFIND_ERROR_SHAPE_MISMATCH,
    STRFIND_ERROR_INTERNAL,
];

pub const STRFIND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRFIND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRFIND_ERRORS,
};

const STRFIND_TYPED_FORCE_CELL_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "strfind-typed-force-cell-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "strfind with a typed-integer ForceCellOutput value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StrfindTypedForceCellOutputExtension"),
    };

const STRFIND_NONBINARY_FORCE_CELL_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "strfind-nonbinary-force-cell-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "strfind with a numeric ForceCellOutput value other than zero or one is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StrfindNonbinaryForceCellOutputExtension"),
    };

const STRFIND_RESIDENT_FORCE_CELL_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "strfind-resident-force-cell-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "strfind with a resident ForceCellOutput value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StrfindResidentForceCellOutputExtension"),
    };

pub const STRFIND_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    STRFIND_TYPED_FORCE_CELL_OUTPUT_EXTENSION,
    STRFIND_NONBINARY_FORCE_CELL_OUTPUT_EXTENSION,
    STRFIND_RESIDENT_FORCE_CELL_OUTPUT_EXTENSION,
];

const STRFIND_INTEGER_FORCE_CELL_OUTPUT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "cellOutput",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly documents false, true, 0, and 1 but does not enumerate typed integer storage classes. RunMat mode accepts every exact scalar integer class; MATLAB-compatible mode retains only logical and double zero/one.",
    }];

pub const STRFIND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "k = strfind(str, pat, 'ForceCellOutput', integer_cellOutput)",
        inputs: &STRFIND_INTEGER_FORCE_CELL_OUTPUT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The typed integer flag controls only whether double index vectors are wrapped in cells. Typed storage acceptance is mode-gated because the public page lists numeric literals but no storage classes; values other than zero and one remain a separately gated convenience.",
    }];

fn strfind_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_strfind_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "strfind",
    category = "strings/search",
    summary = "Return starting indices of substring matches within text inputs.",
    keywords = "strfind,substring,index,positions,string search",
    accel = "sink",
    type_resolver(text_search_indices_type),
    descriptor(crate::builtins::strings::search::strfind::STRFIND_DESCRIPTOR),
    extensions(crate::builtins::strings::search::strfind::STRFIND_EXTENSIONS),
    integer_capabilities(crate::builtins::strings::search::strfind::STRFIND_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::strings::search::strfind"
)]
async fn strfind_builtin(
    text: Value,
    pattern: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if crate::dispatcher::value_contains_gpu(&text)
        || crate::dispatcher::value_contains_gpu(&pattern)
    {
        return Err(strfind_error_with_message(
            STRFIND_ERROR_INVALID_INPUT.message,
            &STRFIND_ERROR_INVALID_INPUT,
        ));
    }
    let rest = validate_force_cell_output_extensions(rest).await?;
    let text = gather_if_needed_async(&text)
        .await
        .map_err(remap_strfind_flow)?;
    let pattern = gather_if_needed_async(&pattern)
        .await
        .map_err(remap_strfind_flow)?;
    let force_cell_output = parse_force_cell_output(&rest).map_err(|err| {
        strfind_error_with_message(err.message().to_string(), &STRFIND_ERROR_INVALID_OPTION)
    })?;

    let subject = TextCollection::from_subject(BUILTIN_NAME, text).map_err(|err| {
        strfind_error_with_message(err.message().to_string(), &STRFIND_ERROR_INVALID_INPUT)
    })?;
    let patterns = TextCollection::from_pattern(BUILTIN_NAME, pattern).map_err(|err| {
        strfind_error_with_message(err.message().to_string(), &STRFIND_ERROR_INVALID_INPUT)
    })?;

    evaluate_strfind(&subject, &patterns, force_cell_output)
}

async fn validate_force_cell_output_extensions(rest: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    if rest.is_empty() {
        return Ok(rest);
    }
    if rest.len() != 2 {
        return Err(strfind_error_with_message(
            "strfind: name-value pairs must be complete; expected exactly one 'ForceCellOutput' name-value pair",
            &STRFIND_ERROR_INVALID_OPTION,
        ));
    }
    let name = value_to_owned_string(&rest[0]).ok_or_else(|| {
        strfind_error_with_message(
            "strfind: option names must be text scalars",
            &STRFIND_ERROR_INVALID_OPTION,
        )
    })?;
    if !name.eq_ignore_ascii_case("ForceCellOutput") {
        return Err(strfind_error_with_message(
            format!("strfind: unknown option '{name}'; supported option is 'ForceCellOutput'"),
            &STRFIND_ERROR_INVALID_OPTION,
        ));
    }
    let value = &rest[1];
    validate_force_cell_output_shape(value)?;
    if is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STRFIND_TYPED_FORCE_CELL_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if is_nonbinary_numeric_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STRFIND_NONBINARY_FORCE_CELL_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if crate::dispatcher::value_contains_gpu(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STRFIND_RESIDENT_FORCE_CELL_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let mut host = Vec::with_capacity(rest.len());
    for value in rest {
        host.push(
            gather_if_needed_async(&value)
                .await
                .map_err(remap_strfind_flow)?,
        );
    }
    Ok(host)
}

fn validate_force_cell_output_shape(value: &Value) -> BuiltinResult<()> {
    let len = match value {
        Value::GpuTensor(handle) => tensor::element_count(&handle.shape),
        Value::Tensor(value) => value.len(),
        Value::LogicalArray(value) => value.data.len(),
        _ => 1,
    };
    if len == 1 {
        Ok(())
    } else {
        Err(strfind_error_with_message(
            "strfind: ForceCellOutput must be a scalar",
            &STRFIND_ERROR_INVALID_OPTION,
        ))
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(value) => value.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

fn is_nonbinary_numeric_value(value: &Value) -> bool {
    match value {
        Value::Num(value) => *value != 0.0 && *value != 1.0,
        Value::Int(value) => !value.is_zero() && value.try_to_usize() != Some(1),
        Value::Tensor(value) if value.len() == 1 => {
            if let Some(integer) = value
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                !integer.is_zero() && integer.try_to_usize() != Some(1)
            } else {
                let value = tensor::tensor_value_f64(value, 0);
                value != 0.0 && value != 1.0
            }
        }
        _ => false,
    }
}

fn evaluate_strfind(
    subject: &TextCollection,
    patterns: &TextCollection,
    force_cell_output: bool,
) -> BuiltinResult<Value> {
    let output_shape = broadcast_shapes(BUILTIN_NAME, &subject.shape, &patterns.shape)
        .map_err(|err| strfind_error_with_message(err, &STRFIND_ERROR_SHAPE_MISMATCH))?;
    let total = tensor::element_count(&output_shape);
    let return_cell = force_cell_output || subject.is_cell || patterns.is_cell || total != 1;

    let subject_strides = compute_strides(&subject.shape);
    let pattern_strides = compute_strides(&patterns.shape);

    let mut matches: Vec<Vec<usize>> = Vec::with_capacity(total);
    for linear in 0..total {
        let subject_idx = broadcast_index(linear, &output_shape, &subject.shape, &subject_strides);
        let pattern_idx = broadcast_index(linear, &output_shape, &patterns.shape, &pattern_strides);
        let result = match (
            &subject.elements[subject_idx],
            &patterns.elements[pattern_idx],
        ) {
            (TextElement::Missing, _) => Vec::new(),
            (_, TextElement::Missing) => Vec::new(),
            (TextElement::Text(text), TextElement::Text(pattern)) => {
                find_indices(text, pattern.as_str())
            }
        };
        matches.push(result);
    }

    if !return_cell {
        let indices = matches.into_iter().next().unwrap_or_default();
        return indices_to_numeric_value(&indices);
    }

    indices_to_cell(matches, &output_shape)
}

fn find_indices(text: &str, pattern: &str) -> Vec<usize> {
    if pattern.is_empty() {
        return Vec::new();
    }

    let text_chars: Vec<char> = text.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_len = text_chars.len();
    let pattern_len = pattern_chars.len();

    if pattern_len == 0 || pattern_len > text_len {
        return Vec::new();
    }

    let mut indices = Vec::new();
    for start in 0..=text_len - pattern_len {
        if &text_chars[start..start + pattern_len] == pattern_chars.as_slice() {
            indices.push(start + 1);
        }
    }
    indices
}

fn indices_to_numeric_value(indices: &[usize]) -> BuiltinResult<Value> {
    let data = indices.iter().map(|&pos| pos as f64).collect::<Vec<_>>();
    let cols = indices.len();
    Tensor::new(data, vec![1, cols])
        .map(Value::Tensor)
        .map_err(|e| {
            strfind_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRFIND_ERROR_INTERNAL)
        })
}

fn indices_to_tensor(indices: &[usize]) -> BuiltinResult<Value> {
    Tensor::new(
        indices.iter().map(|&pos| pos as f64).collect::<Vec<_>>(),
        vec![1, indices.len()],
    )
    .map(Value::Tensor)
    .map_err(|e| {
        strfind_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRFIND_ERROR_INTERNAL)
    })
}

fn indices_to_cell(matches: Vec<Vec<usize>>, shape: &[usize]) -> BuiltinResult<Value> {
    let total = matches.len();
    if total == 0 {
        let (rows, cols) = shape_to_rows_cols(shape);
        return CellArray::new(Vec::new(), rows, cols)
            .map(Value::Cell)
            .map_err(|e| {
                strfind_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRFIND_ERROR_INTERNAL)
            });
    }

    let (rows, cols) = shape_to_rows_cols(shape);
    if rows * cols != total {
        return Err(strfind_error_with_message(
            "strfind: internal size mismatch while constructing cell output",
            &STRFIND_ERROR_INTERNAL,
        ));
    }

    let mut values = Vec::with_capacity(total);
    for row in 0..rows {
        for col in 0..cols {
            let column_major_idx = row + rows * col;
            let indices = matches.get(column_major_idx).ok_or_else(|| {
                strfind_error_with_message(
                    "strfind: internal indexing error",
                    &STRFIND_ERROR_INTERNAL,
                )
            })?;
            let cell_value = indices_to_tensor(indices)?;
            values.push(cell_value);
        }
    }

    CellArray::new(values, rows, cols)
        .map(Value::Cell)
        .map_err(|e| {
            strfind_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRFIND_ERROR_INTERNAL)
        })
}

fn shape_to_rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => {
            let rows = shape[0];
            let cols = shape[1..]
                .iter()
                .copied()
                .fold(1usize, |acc, dim| acc.saturating_mul(dim));
            (rows, cols)
        }
    }
}

fn parse_force_cell_output(rest: &[Value]) -> BuiltinResult<bool> {
    if rest.is_empty() {
        return Ok(false);
    }
    if !rest.len().is_multiple_of(2) {
        return Err(strfind_error_with_message(
            "strfind: expected name-value pairs after the pattern (e.g., 'ForceCellOutput', true)",
            &STRFIND_ERROR_INVALID_OPTION,
        ));
    }

    let mut force_cell = None;
    for pair in rest.chunks(2) {
        let name = value_to_owned_string(&pair[0]).ok_or_else(|| {
            strfind_error_with_message(
                "strfind: option names must be text scalars",
                &STRFIND_ERROR_INVALID_OPTION,
            )
        })?;
        if !name.eq_ignore_ascii_case("forcecelloutput") {
            return Err(strfind_error_with_message(
                format!("strfind: unknown option '{name}'; supported option is 'ForceCellOutput'"),
                &STRFIND_ERROR_INVALID_OPTION,
            ));
        }
        let value = parse_bool_like(&pair[1])?;
        force_cell = Some(value);
    }
    force_cell.ok_or_else(|| {
        strfind_error_with_message(
            "strfind: expected 'ForceCellOutput' option when providing name-value arguments",
            &STRFIND_ERROR_INVALID_OPTION,
        )
    })
}

fn parse_bool_like(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => {
            if !n.is_finite() {
                Err(strfind_error_with_message(
                    "strfind: option values must be finite numeric scalars",
                    &STRFIND_ERROR_INVALID_OPTION,
                ))
            } else {
                Ok(*n != 0.0)
            }
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 {
                Err(strfind_error_with_message(
                    format!(
                        "strfind: option values must be scalar logicals (received {} elements)",
                        array.data.len()
                    ),
                    &STRFIND_ERROR_INVALID_OPTION,
                ))
            } else {
                Ok(array.data[0] != 0)
            }
        }
        Value::Tensor(tensor) => {
            let len = tensor::tensor_element_len(tensor);
            if len != 1 {
                Err(strfind_error_with_message(
                    format!(
                        "strfind: option values must be scalar numeric values (received {} elements)",
                        len
                    ),
                    &STRFIND_ERROR_INVALID_OPTION,
                ))
            } else if let Some(value) = tensor.integer_storage().and_then(|storage| storage.value_at(0)) {
                Ok(!value.is_zero())
            } else if !tensor::tensor_value_f64(tensor, 0).is_finite() {
                Err(strfind_error_with_message(
                    "strfind: option values must be finite numeric scalars",
                    &STRFIND_ERROR_INVALID_OPTION,
                ))
            } else {
                Ok(tensor::tensor_value_f64(tensor, 0) != 0.0)
            }
        }
        other => value_to_owned_string(other)
            .ok_or_else(|| {
                strfind_error_with_message(
                    "strfind: option values must be logical or numeric scalars",
                    &STRFIND_ERROR_INVALID_OPTION,
                )
            })
            .and_then(|text| match text.trim().to_ascii_lowercase().as_str() {
                "true" | "on" | "1" => Ok(true),
                "false" | "off" | "0" => Ok(false),
                _ => Err(strfind_error_with_message(
                    format!(
                        "strfind: invalid value '{text}' for 'ForceCellOutput'; expected true or false"
                    ),
                    &STRFIND_ERROR_INVALID_OPTION,
                )),
            }),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{CellArray, CharArray, IntegerStorage, StringArray, Tensor};

    fn run_strfind(text: Value, pattern: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(strfind_builtin(text, pattern, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_single_match_returns_row_vector() {
        let result = run_strfind(
            Value::String("saturn".into()),
            Value::String("sat".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                assert_eq!(tensor.materialize_f64(), vec![1.0]);
            }
            other => panic!("expected 1x1 tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_char_vector_matches() {
        let result = run_strfind(
            Value::CharArray(CharArray::new_row("abracadabra")),
            Value::CharArray(CharArray::new_row("abra")),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.materialize_f64(), vec![1.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_overlapping_matches() {
        let result = run_strfind(
            Value::String("aaaa".into()),
            Value::String("aa".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_empty_pattern_returns_empty_indices() {
        let result = run_strfind(
            Value::String("abc".into()),
            Value::String("".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 0]);
                assert!(tensor.materialize_f64().is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_string_array_returns_cell() {
        let strings = StringArray::new(
            vec!["hydrogen".into(), "helium".into(), "lithium".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = run_strfind(
            Value::StringArray(strings),
            Value::String("i".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 3);
                assert_eq!(cell.cols, 1);
                let first = cell.get(0, 0).unwrap();
                let second = cell.get(1, 0).unwrap();
                let third = cell.get(2, 0).unwrap();
                match first {
                    Value::Tensor(tensor) => assert!(tensor.materialize_f64().is_empty()),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match second {
                    Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![4.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match third {
                    Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 5.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_pattern_array_returns_cell() {
        let patterns =
            StringArray::new(vec!["sat".into(), "turn".into(), "moon".into()], vec![1, 3]).unwrap();
        let result = run_strfind(
            Value::String("saturn".into()),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 3);
                let first = cell.get(0, 0).unwrap();
                let second = cell.get(0, 1).unwrap();
                let third = cell.get(0, 2).unwrap();
                match first {
                    Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match second {
                    Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match third {
                    Value::Tensor(tensor) => assert!(tensor.materialize_f64().is_empty()),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_name_value() {
        let result = run_strfind(
            Value::CharArray(CharArray::new_row("mission")),
            Value::CharArray(CharArray::new_row("s")),
            vec![Value::String("ForceCellOutput".into()), Value::Bool(true)],
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0, 4.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_numeric_value() {
        let result = run_strfind(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![Value::String("ForceCellOutput".into()), Value::Num(1.0)],
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0, 4.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_typed_integer_tensor_reads_exact_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let flag =
            Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("integer tensor");
        let result = run_strfind(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![Value::String("ForceCellOutput".into()), Value::Tensor(flag)],
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0, 4.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_off_string() {
        let result = run_strfind(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![
                Value::String("ForceCellOutput".into()),
                Value::String("off".into()),
            ],
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.materialize_f64(), vec![3.0, 4.0]);
            }
            other => panic!("expected numeric tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_non_scalar_error() {
        let option_value =
            Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor construction for test");
        let err = run_strfind(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![
                Value::String("ForceCellOutput".into()),
                Value::Tensor(option_value),
            ],
        )
        .expect_err("strfind should error for non-scalar ForceCellOutput");
        assert!(
            err.to_string().contains("scalar"),
            "unexpected error message for non-scalar option: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_missing_value_error() {
        let err = run_strfind(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![Value::String("ForceCellOutput".into())],
        )
        .expect_err("strfind should error when ForceCellOutput value missing");
        assert!(
            err.to_string().contains("name-value pairs"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_subject_cell_scalar_returns_cell() {
        let subject = CellArray::new(vec![Value::from("needle")], 1, 1).expect("cell construction");
        let result = run_strfind(
            Value::Cell(subject),
            Value::String("needle".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_pattern_cell_scalar_returns_cell() {
        let pattern = CellArray::new(vec![Value::from("needle")], 1, 1).expect("cell construction");
        let result = run_strfind(
            Value::String("needle".into()),
            Value::Cell(pattern),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_missing_subject_returns_empty() {
        let result = run_strfind(
            Value::String("<missing>".into()),
            Value::String("abc".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 0]);
                assert!(tensor.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_missing_pattern_returns_empty_vector() {
        let patterns =
            StringArray::new(vec!["<missing>".into()], vec![1, 1]).expect("string array creation");
        let result = run_strfind(
            Value::String("planet".into()),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 0]);
                assert!(tensor.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_char_matrix_rows() {
        let data = vec!['c', 'a', 't', 'a', 'd', 'a', 'd', 'o', 'g'];
        let array = CharArray::new(data, 3, 3).unwrap();
        let result = run_strfind(
            Value::CharArray(array),
            Value::CharArray(CharArray::new_row("d")),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 3);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert!(tensor.materialize_f64().is_empty()),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match cell.get(1, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match cell.get(2, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_invalid_option_name_errors() {
        let err = run_strfind(
            Value::String("abc".into()),
            Value::String("a".into()),
            vec![Value::String("IgnoreCase".into()), Value::Bool(true)],
        )
        .expect_err("strfind should error");
        assert!(
            err.to_string().contains("unknown option"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn strfind_type_for_scalar_text_is_tensor() {
        assert_eq!(
            text_search_indices_type(
                &[Type::String, Type::String],
                &ResolveContext::new(Vec::new()),
            ),
            Type::tensor()
        );
    }
}
