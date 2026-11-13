//! MATLAB-compatible `optimizableVariable` metadata constructor.

use std::collections::{BTreeMap, HashSet};

use runmat_builtins::{
    BuiltinErrorDescriptor, OptimizableVariableOption, OptimizationVariableTransform as Transform,
    OptimizationVariableType as VariableType, OPTIMIZABLE_VARIABLE_ERROR_FLOW as ERROR_FLOW,
    OPTIMIZABLE_VARIABLE_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    OPTIMIZABLE_VARIABLE_ERROR_INVALID_OPTION as ERROR_INVALID_OPTION,
    OPTIMIZABLE_VARIABLE_ERROR_INVALID_RANGE as ERROR_INVALID_RANGE,
    OPTIMIZABLE_VARIABLE_INTEGER_RANGE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, LogicalArray, StringArray, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_helpers;
use crate::builtins::common::validation::{integer_is_exact_f64, value_has_native_integer_class};
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, GpuGatherRetry, RuntimeError,
};

mod options;
mod range;
#[cfg(test)]
mod tests;

use options::Options;
use range::OptimizableRange;

const NAME: &str = "optimizableVariable";
const CLASS_IDENTITY: runmat_types::StaticClassIdentity =
    runmat_types::standard::OPTIMIZABLE_VARIABLE;

enum AdmittedRange {
    Parsed(OptimizableRange),
    Deferred(Value),
}

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::stats::ml::optimizable_variable"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("optimizable-variable-metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host metadata construction for bayesopt; gpuArray range inputs are gathered before validation.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::stats::ml::optimizable_variable"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "optimizableVariable constructs host metadata and terminates fusion planning.",
};

#[runtime_builtin(
    name = "optimizableVariable",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::optimizable_variable",
    context = true
)]
async fn optimizable_variable_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    name: Value,
    range: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let _control_sources = rest.fork();
    let headers = Options::headers(rest)?;
    let variable_name = parse_text_scalar(&name, "Name", &ERROR_INVALID_ARGUMENT)?;
    if variable_name.trim().is_empty() {
        return Err(error(
            "optimizableVariable: Name must be a nonempty text scalar",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let admitted_options = Options::admit(headers)?;
    let host_options = Options::host_facts(&admitted_options);
    if value_has_native_integer_class(&range) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &OPTIMIZABLE_VARIABLE_INTEGER_RANGE_EXTENSION,
            NAME,
        )?;
    }
    // A known host range must not be masked by a failing download of an
    // unrelated deferred option. Scalar text needs a deferred Type before its
    // categorical meaning can be established.
    let admitted_range = if !(matches!(&range, Value::GpuTensor(_))
        || host_options.type_deferred && matches!(&range, Value::String(_) | Value::CharArray(_)))
    {
        let parsed = OptimizableRange::parse(range, host_options.var_type)?;
        if !host_options.type_deferred {
            let var_type = host_options.var_type.unwrap_or(parsed.inferred_type());
            parsed.validate_for(var_type, host_options.transform.unwrap_or(Transform::None))?;
        }
        AdmittedRange::Parsed(parsed)
    } else {
        AdmittedRange::Deferred(range)
    };
    let admitted_range = match admitted_range {
        AdmittedRange::Parsed(range) => AdmittedRange::Parsed(range),
        AdmittedRange::Deferred(range) => AdmittedRange::Deferred(
            gather_if_needed_async(&range)
                .await
                .map_err(|err| remap_flow(err, "range"))?,
        ),
    };
    let options = Options::materialize(admitted_options).await?;
    let range = match admitted_range {
        AdmittedRange::Parsed(parsed) => parsed,
        AdmittedRange::Deferred(range) => OptimizableRange::parse(range, options.var_type)?,
    };
    let var_type = options.var_type.unwrap_or(range.inferred_type());
    range.validate_for(var_type, options.transform)?;

    let fields = [
        ("Name", Value::string(variable_name)),
        ("Range", range.into_value()),
        ("Type", Value::string(var_type.public_name().to_string())),
        (
            "Transform",
            Value::string(options.transform.public_name().to_string()),
        ),
        ("Optimize", Value::Bool(options.optimize)),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime.as_ref(),
        CLASS_IDENTITY.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
}

fn parse_text_scalar(
    value: &Value,
    label: &str,
    descriptor: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<runmat_value::Utf8Text> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => {
            char_row_to_string(chars, label, descriptor).map(Into::into)
        }
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].text_ref().cloned().ok_or_else(|| {
                error(
                    format!("optimizableVariable: {label} must be a text scalar"),
                    descriptor,
                )
            })
        }
        other => Err(error(
            format!("optimizableVariable: {label} must be a text scalar, got {other:?}"),
            descriptor,
        )),
    }
}

fn char_row_to_string(
    chars: &CharArray,
    label: &str,
    descriptor: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    if chars.rows != 1 {
        return Err(error(
            format!("optimizableVariable: {label} char array must be a row vector"),
            descriptor,
        ));
    }
    crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
        error(
            format!("optimizableVariable: {label} must be valid UTF-16 text"),
            descriptor,
        )
    })
}

fn parse_bool(value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => Ok(data[0] != 0),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::Int(i) if i.to_f64() == 0.0 || i.to_f64() == 1.0 => Ok(i.to_f64() != 0.0),
        other => Err(error(
            format!("optimizableVariable: {label} must be a scalar logical value, got {other:?}"),
            &ERROR_INVALID_OPTION,
        )),
    }
}

fn error(message: impl Into<String>, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_flow(err: RuntimeError, label: &str) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "optimizableVariable: failed to gather {label}: {}",
        err.message()
    ))
    .with_builtin(NAME)
    .with_source(err)
    .with_gpu_gather_retry(GpuGatherRetry::Never);
    if let Some(identifier) = ERROR_FLOW.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
