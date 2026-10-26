//! Concrete selected-role admission shared by the public builder and solver.
//! No aggregate property/capture facts are derived for control validation.
use super::input::empty;
use crate::builtins::math::optim::common::{self, output_callback};
use crate::{context::RuntimeContext, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, OptimOptionField as Field,
    OptimPatternDefault,
};
use runmat_value::Value;

#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) struct Policy {
    pub name: &'static str,
    pub error: &'static BuiltinErrorDescriptor,
    pub integer: &'static BuiltinExtensionDescriptor,
    pub resident: Option<&'static BuiltinExtensionDescriptor>,
}
impl Policy {
    pub fn invalid(self, detail: impl std::fmt::Display) -> RuntimeError {
        let mut error = crate::build_runtime_error(format!("{}: {detail}", self.name))
            .with_builtin(self.name)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
        if let Some(identifier) = self.error.identifier {
            error = error.with_identifier(identifier);
        }
        error.build()
    }
}

pub(in crate::builtins::math::optim) fn validate_search(
    value: &Value,
    policy: Policy,
) -> BuiltinResult<()> {
    super::search::syntax(value).map_err(|error| policy.invalid(error.message()))
}
pub(in crate::builtins::math::optim) fn switch_field(field: Field) -> bool {
    matches!(
        field,
        Field::Cache
            | Field::ScaleMesh
            | Field::MeshRotate
            | Field::AccelerateMesh
            | Field::UseCompletePoll
            | Field::UseCompleteSearch
            | Field::UseVectorized
            | Field::UseParallel
    )
}
fn text_field(field: Field) -> bool {
    matches!(
        field,
        Field::Algorithm | Field::Display | Field::PollMethod | Field::PollOrderAlgorithm
    )
}

/// Numeric range semantics are role-specific: Inf is a valid mesh ceiling or
/// time limit, but never silently accepted for a finite tolerance. Native
/// integer values cross only an exact binary64 boundary.
pub(in crate::builtins::math::optim) fn number(
    field: Field,
    value: &Value,
    policy: Policy,
) -> BuiltinResult<f64> {
    if let Some(integer) = crate::builtins::common::tensor::scalar_integer_value(value) {
        if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
            return Err(policy.invalid(format!(
                "{} must be exactly representable as double",
                field.name()
            )));
        }
        return Ok(integer.to_f64());
    }
    match value {
        Value::Num(number) => Ok(*number),
        Value::Tensor(value) if value.len() == 1 && value.integer_storage().is_none() => {
            Ok(crate::builtins::common::tensor::tensor_value_f64(value, 0))
        }
        _ => Err(policy.invalid(format!("{} must be a real numeric scalar", field.name()))),
    }
}

pub(in crate::builtins::math::optim) fn metadata(
    field: Field,
    value: &Value,
    policy: Policy,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    if empty(value) {
        return Ok(());
    }
    if matches!(field, Field::OutputFcn | Field::PlotFcn) {
        return output_callback::validate_for_builtin(
            policy.name,
            value,
            if field == Field::OutputFcn {
                output_callback::CallbackForm::Handles
            } else {
                output_callback::CallbackForm::HandlesOrNames
            },
        )
        .map_err(|error| policy.invalid(error.message()));
    }
    if field == Field::SearchFcn {
        return super::search::metadata(value, policy, runtime);
    }
    if text_field(field) {
        return crate::builtins::common::validation::value_text_scalar_string(value)
            .map(|_| ())
            .ok_or_else(|| policy.invalid(format!("{} must be scalar text", field.name())));
    }
    if let Some(expression) = OptimPatternDefault::dimension_expression(field) {
        if let Some(value) = crate::builtins::common::validation::value_text_scalar_string(value) {
            return if value == expression {
                Ok(())
            } else {
                Err(policy.invalid(format!(
                    "{} is not a supported dimensional default",
                    field.name()
                )))
            };
        }
    }
    if field == Field::MaxTime && crate::builtins::duration::is_duration_object(value) {
        let tensor = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
        return if tensor.len() == 1 && tensor.as_f64_slice().is_some() {
            Ok(())
        } else {
            Err(policy.invalid("MaxTime duration must be a real double scalar"))
        };
    }
    if switch_field(field) {
        match value {
            Value::Bool(_) => return Ok(()),
            Value::LogicalArray(value) if value.data.len() == 1 => return Ok(()),
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                return crate::builtins::common::validation::value_text_scalar_string(value)
                    .map(|_| ())
                    .ok_or_else(|| {
                        policy.invalid(format!("{} must be a scalar switch", field.name()))
                    });
            }
            Value::GpuTensor(_) => {
                let fact = crate::value_fact::value_fact(value);
                if matches!(fact.kind, runmat_types::ValueKindFact::Logical)
                    && fact.shape.element_count() == Some(1)
                {
                    return common::preflight_host_residency(
                        policy.name,
                        value,
                        policy.resident,
                        runtime,
                    );
                }
            }
            _ => {}
        }
    }
    common::preflight_option_scalar(policy.name, value, field)
        .map_err(|error| policy.invalid(error.message()))?;
    common::preflight_host_residency(policy.name, value, policy.resident, runtime)?;
    if common::is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime,
            policy.integer,
            policy.name,
        )?;
    }
    Ok(())
}
