use runmat_builtins::OdeOptionField;
use runmat_value::Value;

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::common::{ode_error, OdeCompatibilityExtensions, OdeOptions};

const DEFAULT_REL_TOL: f64 = 1.0e-3;
const DEFAULT_ABS_TOL: f64 = 1.0e-6;
const DEFAULT_MAX_STEPS: usize = 100_000;

/// Only recognized, first-occurring fields cross the solver's options
/// boundary. The source struct's order matters for case-insensitive aliases.
#[derive(Default)]
pub(super) struct OdeOptionValues {
    rel_tol: Option<Value>,
    abs_tol: Option<Value>,
    initial_step: Option<Value>,
    max_step: Option<Value>,
    max_steps: Option<Value>,
    refine: Option<Value>,
}

impl OdeOptionValues {
    fn slot(&mut self, field: OdeOptionField) -> &mut Option<Value> {
        match field {
            OdeOptionField::RelTol => &mut self.rel_tol,
            OdeOptionField::AbsTol => &mut self.abs_tol,
            OdeOptionField::InitialStep => &mut self.initial_step,
            OdeOptionField::MaxStep => &mut self.max_step,
            OdeOptionField::MaxSteps => &mut self.max_steps,
            OdeOptionField::Refine => &mut self.refine,
        }
    }

    fn iter(&self) -> impl Iterator<Item = (OdeOptionField, &Value)> {
        [
            (OdeOptionField::RelTol, self.rel_tol.as_ref()),
            (OdeOptionField::AbsTol, self.abs_tol.as_ref()),
            (OdeOptionField::InitialStep, self.initial_step.as_ref()),
            (OdeOptionField::MaxStep, self.max_step.as_ref()),
            (OdeOptionField::MaxSteps, self.max_steps.as_ref()),
            (OdeOptionField::Refine, self.refine.as_ref()),
        ]
        .into_iter()
        .filter_map(|(field, value)| {
            value
                .filter(|value| !is_default_option(value))
                .map(|value| (field, value))
        })
    }

    async fn gather(self) -> BuiltinResult<Self> {
        Ok(Self {
            rel_tol: gather(self.rel_tol).await?,
            abs_tol: gather(self.abs_tol).await?,
            initial_step: gather(self.initial_step).await?,
            max_step: gather(self.max_step).await?,
            max_steps: gather(self.max_steps).await?,
            refine: gather(self.refine).await?,
        })
    }
}

async fn gather(value: Option<Value>) -> BuiltinResult<Option<Value>> {
    match value {
        Some(value) if is_default_option(&value) => Ok(None),
        Some(Value::GpuTensor(handle)) => {
            crate::dispatcher::gather_if_needed_async(&Value::GpuTensor(handle))
                .await
                .map(Some)
        }
        other => Ok(other),
    }
}

fn is_default_option(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.is_empty())
}

pub(super) fn parse(name: &str, value: Option<Value>) -> BuiltinResult<OdeOptionValues> {
    let Some(value) = value else {
        return Ok(OdeOptionValues::default());
    };
    let Value::Struct(options) = value else {
        return Err(ode_error(
            name,
            format!("{name}: options must be a struct, got {value:?}"),
        ));
    };
    let mut selected = OdeOptionValues::default();
    for (name, value) in options.into_fields_unbounded() {
        if let Some(field) = OdeOptionField::from_public_name(&name) {
            let slot = selected.slot(field);
            if slot.is_none() {
                *slot = Some(value);
            }
        }
    }
    Ok(selected)
}

pub(super) async fn admit(
    name: &str,
    selected: OdeOptionValues,
    extensions: OdeCompatibilityExtensions,
) -> BuiltinResult<OdeOptions> {
    // Check every host-visible field and compatibility gate before the first
    // resident read. Unknown and shadowed fields are not solver operands.
    for (_, value) in selected.iter() {
        if crate::builtins::common::validation::value_contains_explicit_gpu(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                extensions.resident_input,
                name,
            )?;
        }
        if crate::builtins::common::validation::value_contains_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                extensions.integer_option,
                name,
            )?;
        }
    }
    for (field, value) in selected.iter() {
        validate_host_class(name, field, value)?;
        if !matches!(value, Value::GpuTensor(_)) {
            validate_host_semantics(name, field, value)?;
        }
    }
    for (field, value) in selected.iter() {
        if field.is_floating_control()
            && crate::builtins::common::validation::value_contains_native_integer_class(value)
            && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
                .await?
        {
            return Err(ode_error(
                name,
                format!(
                    "{name}: integer option {} must be exactly representable as double",
                    field.public_name()
                ),
            ));
        }
    }
    decode(name, selected.gather().await?)
}

fn validate_host_class(name: &str, field: OdeOptionField, value: &Value) -> BuiltinResult<()> {
    let valid = match value {
        Value::Num(number) => number.is_finite(),
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::GpuTensor(_) => true,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(ode_error(
            name,
            format!(
                "{name}: option {} must be numeric, got {value:?}",
                field.public_name()
            ),
        ))
    }
}

fn validate_host_semantics(name: &str, field: OdeOptionField, value: &Value) -> BuiltinResult<()> {
    match field {
        OdeOptionField::MaxSteps | OdeOptionField::Refine => {
            count(name, field, Some(value))?;
        }
        _ => {
            if optional_floating(name, field, Some(value))?.is_some_and(|number| number <= 0.0) {
                return Err(ode_error(
                    name,
                    format!("{name}: {} must be positive", field.public_name()),
                ));
            }
        }
    }
    Ok(())
}

fn decode(name: &str, fields: OdeOptionValues) -> BuiltinResult<OdeOptions> {
    let rel_tol = floating(
        name,
        OdeOptionField::RelTol,
        fields.rel_tol.as_ref(),
        DEFAULT_REL_TOL,
    )?;
    let abs_tol = floating(
        name,
        OdeOptionField::AbsTol,
        fields.abs_tol.as_ref(),
        DEFAULT_ABS_TOL,
    )?;
    if rel_tol <= 0.0 || abs_tol <= 0.0 {
        return Err(ode_error(
            name,
            format!("{name}: RelTol and AbsTol must be positive"),
        ));
    }
    let initial_step = optional_floating(
        name,
        OdeOptionField::InitialStep,
        fields.initial_step.as_ref(),
    )?;
    let max_step = optional_floating(name, OdeOptionField::MaxStep, fields.max_step.as_ref())?;
    for (field, value) in [
        (OdeOptionField::InitialStep, initial_step),
        (OdeOptionField::MaxStep, max_step),
    ] {
        if value.is_some_and(|value| value <= 0.0) {
            return Err(ode_error(
                name,
                format!("{name}: {} must be positive", field.public_name()),
            ));
        }
    }
    let max_steps = count(name, OdeOptionField::MaxSteps, fields.max_steps.as_ref())?
        .unwrap_or(DEFAULT_MAX_STEPS);
    let refine = count(name, OdeOptionField::Refine, fields.refine.as_ref())?;
    Ok(OdeOptions {
        rel_tol,
        abs_tol,
        initial_step,
        max_step,
        max_steps,
        refine,
    })
}

fn floating(
    name: &str,
    field: OdeOptionField,
    value: Option<&Value>,
    default: f64,
) -> BuiltinResult<f64> {
    Ok(optional_floating(name, field, value)?.unwrap_or(default))
}

fn optional_floating(
    name: &str,
    field: OdeOptionField,
    value: Option<&Value>,
) -> BuiltinResult<Option<f64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value {
        Value::Num(number) if number.is_finite() => Ok(Some(*number)),
        Value::Int(integer) => Ok(Some(integer.to_f64())),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            let number = tensor::tensor_value_f64(tensor, 0);
            if number.is_finite() {
                Ok(Some(number))
            } else {
                Err(ode_error(
                    name,
                    format!("{name}: option {} must be finite", field.public_name()),
                ))
            }
        }
        other => Err(ode_error(
            name,
            format!(
                "{name}: option {} must be numeric, got {other:?}",
                field.public_name()
            ),
        )),
    }
}

fn count(name: &str, field: OdeOptionField, value: Option<&Value>) -> BuiltinResult<Option<usize>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|value| *value >= 1)
            .map(Some)
            .ok_or_else(|| {
                ode_error(
                    name,
                    format!("{name}: option {} must be at least 1", field.public_name()),
                )
            });
    }
    let parsed = match value {
        Value::Num(number) => *number,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(ode_error(
                name,
                format!(
                    "{name}: option {} must be numeric, got {other:?}",
                    field.public_name()
                ),
            ))
        }
    };
    if !parsed.is_finite()
        || parsed < 1.0
        || parsed.fract() != 0.0
        || parsed >= 2f64.powi(usize::BITS as i32)
    {
        return Err(ode_error(
            name,
            format!(
                "{name}: option {} must be a supported positive integer",
                field.public_name()
            ),
        ));
    }
    Ok(Some(parsed as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{GpuHandleProvenance, HostTensorView};

    #[test]
    fn scalar_single_controls_and_first_empty_defaults_keep_original_storage_semantics() {
        let mut source = runmat_value::UnboundedStructValueBuilder::new();
        source.insert(
            "RelTol",
            Value::Tensor(runmat_value::Tensor::from_f32(vec![0.002], vec![1, 1]).unwrap()),
        );
        source.insert(
            "AbsTol",
            Value::Tensor(runmat_value::Tensor::new(vec![1e-7], vec![1, 1]).unwrap()),
        );
        source.insert(
            "Refine",
            Value::Tensor(runmat_value::Tensor::zeros(vec![0, 0])),
        );
        source.insert("rEfInE", Value::Num(8.0));
        let options = block_on(admit(
            "ode45",
            parse("ode45", Some(Value::Struct(source.finish()))).unwrap(),
            super::super::ode45::ODE_COMPATIBILITY_EXTENSIONS,
        ))
        .unwrap();
        assert_eq!(options.rel_tol, 0.002_f32 as f64);
        assert_eq!(options.abs_tol, 1e-7);
        assert_eq!(
            options.refine, None,
            "the original first empty field selects the method default"
        );
        for value in [Value::Num(0.0), Value::Num(1.5), Value::Num(f64::INFINITY)] {
            let mut source = runmat_value::UnboundedStructValueBuilder::new();
            source.insert("Refine", value);
            assert!(block_on(admit(
                "ode45",
                parse("ode45", Some(Value::Struct(source.finish()))).unwrap(),
                super::super::ode45::ODE_COMPATIBILITY_EXTENSIONS,
            ))
            .is_err());
        }
    }

    #[test]
    fn first_case_insensitive_option_wins_and_unknown_fields_are_ignored() {
        let mut source = runmat_value::UnboundedStructValueBuilder::new();
        source.insert("rElToL", Value::Num(0.02));
        source.insert("RelTol", Value::Num(0.03));
        source.insert(
            "ignoredMetadata",
            Value::String("not a solver operand".into()),
        );
        let fields = parse("ode45", Some(Value::Struct(source.finish()))).unwrap();
        let options = block_on(admit(
            "ode45",
            fields,
            super::super::ode45::ODE_COMPATIBILITY_EXTENSIONS,
        ))
        .unwrap();
        assert_eq!(options.rel_tol, 0.02);
        assert_eq!(options.abs_tol, DEFAULT_ABS_TOL);
    }

    #[test]
    fn invalid_host_control_is_rejected_before_any_resident_transfer() {
        let mut source = runmat_value::UnboundedStructValueBuilder::new();
        source.insert("RelTol", Value::Num(-0.1));
        source.insert("MaxStep", Value::Num(0.2));
        let fields = parse("ode23", Some(Value::Struct(source.finish()))).unwrap();
        let error = block_on(admit(
            "ode23",
            fields,
            super::super::ode23::ODE_COMPATIBILITY_EXTENSIONS,
        ))
        .expect_err("negative tolerance is invalid");
        assert!(error.message().contains("RelTol must be positive"));
    }

    #[test]
    fn ignored_resident_metadata_does_not_trigger_explicit_fallback() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&HostTensorView {
                    data: &[42.0],
                    shape: &[1, 1],
                })
                .unwrap()
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap();
            let mut source = runmat_value::UnboundedStructValueBuilder::new();
            source.insert("opaqueMetadata", Value::GpuTensor(handle));
            source.insert("RelTol", Value::Num(0.02));
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let options = block_on(admit(
                "ode45",
                parse("ode45", Some(Value::Struct(source.finish()))).unwrap(),
                super::super::ode45::ODE_COMPATIBILITY_EXTENSIONS,
            ))
            .unwrap();
            assert_eq!(options.rel_tol, 0.02);
        });
    }
}
