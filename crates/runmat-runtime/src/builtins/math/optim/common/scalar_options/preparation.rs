//! Materialize only the controls consumed by the canonical solver option owner.

use super::super::{optim_error, ScalarCooperation};
use super::ScalarOptionFields;
use crate::BuiltinResult;
use runmat_builtins::{
    BuiltinExtensionDescriptor, OptimAlgorithm, OptimDisplay, OptimOptionField, OptimSolver,
};
use runmat_value::{StructValue, Value};

pub(crate) async fn prepare_solver_options(
    solver: OptimSolver,
    options: Option<StructValue>,
    numeric_extension: Option<&'static BuiltinExtensionDescriptor>,
    resident_extension: Option<&'static BuiltinExtensionDescriptor>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<StructValue>> {
    let Some(mut options) = options else {
        return Ok(None);
    };
    let fields = ScalarOptionFields::from_struct(Some(&options));
    use OptimOptionField::*;
    let selected = [
        TolX,
        TolFun,
        MaxIter,
        MaxFunEvals,
        Display,
        Algorithm,
        SpecifyObjectiveGradient,
    ]
    .into_iter()
    .filter(|field| solver.accepts(*field))
    .filter_map(|field| {
        fields
            .binding(field)
            .map(|binding| (field, binding.name.to_owned()))
    })
    .collect::<Vec<_>>();

    // Validate all host-visible control metadata before the first provider operation.
    for (field, name) in &selected {
        let value = &options.fields[name];
        if super::super::is_typed_integer_value(value) {
            if let Some(extension) = numeric_extension {
                crate::compatibility::ensure_builtin_extension_enabled(extension, solver.name())?;
            }
        }
        if matches!(value, Value::GpuTensor(_)) {
            if let Some(extension) = resident_extension {
                crate::compatibility::ensure_builtin_extension_enabled(extension, solver.name())?;
            }
        }
        match field {
            TolX | TolFun | MaxIter | MaxFunEvals => {
                super::super::preflight_option_scalar(solver.name(), value, *field)?;
                if !matches!(value, Value::GpuTensor(_)) {
                    match field {
                        TolX | TolFun => {
                            super::super::option_f64_value(
                                solver.name(),
                                Some(value),
                                field.name(),
                                0.0,
                            )?;
                        }
                        MaxIter | MaxFunEvals => {
                            super::super::option_usize_value(
                                solver.name(),
                                Some(value),
                                field.name(),
                                0,
                            )?;
                        }
                        _ => unreachable!(),
                    }
                }
            }
            Display => {
                let text = super::super::option_string_value(Some(value), field.name(), "")?;
                if !OptimDisplay::parse(&text)
                    .is_some_and(|display| solver.accepts_display(display))
                {
                    return Err(optim_error(
                        solver.name(),
                        format!("{}: unsupported Display value '{text}'", solver.name()),
                    ));
                }
            }
            Algorithm => {
                let text = super::super::option_string_value(Some(value), field.name(), "")?;
                if !OptimAlgorithm::parse(&text)
                    .is_some_and(|algorithm| solver.accepts_algorithm(algorithm))
                {
                    return Err(optim_error(
                        solver.name(),
                        format!("{}: unsupported Algorithm value '{text}'", solver.name()),
                    ));
                }
            }
            SpecifyObjectiveGradient => {
                if matches!(value, Value::GpuTensor(_)) {
                    let fact = crate::value_fact::value_fact(value);
                    if !runmat_builtins::nonlinear_real_numeric_metadata(&fact, false)
                        || fact.shape.element_count() != Some(1)
                    {
                        return Err(optim_error(
                            solver.name(),
                            format!(
                                "{}: option {} must be logical scalar",
                                solver.name(),
                                field.name()
                            ),
                        ));
                    }
                } else {
                    switch_value(solver.name(), field.name(), value)?;
                }
            }
        }
    }
    for (field, name) in selected {
        let value = options
            .fields
            .get_mut(&name)
            .expect("retained selected option key");
        if !matches!(value, Value::GpuTensor(_)) {
            continue;
        }
        let materialized = match field {
            TolX | TolFun => {
                super::super::gather_exact_floating_value_with_cooperation(
                    solver.name(),
                    value.clone(),
                    field.name(),
                    cooperation,
                )
                .await?
            }
            MaxIter | MaxFunEvals | SpecifyObjectiveGradient => {
                crate::dispatcher::gather_if_needed_async(value).await?
            }
            Display | Algorithm => {
                unreachable!("text controls were admitted before provider access")
            }
        };
        match field {
            MaxIter | MaxFunEvals => {
                super::super::option_usize_value(
                    solver.name(),
                    Some(&materialized),
                    field.name(),
                    0,
                )?;
            }
            SpecifyObjectiveGradient => {
                switch_value(solver.name(), field.name(), &materialized)?;
            }
            TolX | TolFun => {
                super::super::option_f64_value(
                    solver.name(),
                    Some(&materialized),
                    field.name(),
                    0.0,
                )?;
            }
            _ => unreachable!(),
        }
        *value = materialized;
        cooperation.completed(1).await?;
    }
    Ok(Some(options))
}

fn switch_value(name: &str, field: &str, value: &Value) -> BuiltinResult<()> {
    use crate::builtins::common::tensor;
    let valid = if let Some(integer) = tensor::scalar_integer_value(value) {
        matches!(integer.try_to_u64(), Some(0 | 1))
    } else {
        match value {
            Value::Bool(_) => true,
            Value::Num(number) => *number == 0.0 || *number == 1.0,
            Value::LogicalArray(values) => values.data.len() == 1,
            Value::Tensor(tensor_value) if tensor::is_scalar_tensor(tensor_value) => {
                let number = tensor::tensor_value_f64(tensor_value, 0);
                number == 0.0 || number == 1.0
            }
            Value::String(text) => runmat_builtins::OptimSwitch::parse(text).is_some(),
            Value::StringArray(values) if values.data.len() == 1 => {
                runmat_builtins::OptimSwitch::parse(&values.data[0]).is_some()
            }
            Value::CharArray(chars) if chars.rows == 1 => {
                runmat_builtins::OptimSwitch::parse(&chars.data.iter().collect::<String>())
                    .is_some()
            }
            _ => false,
        }
    };
    if valid {
        Ok(())
    } else {
        Err(optim_error(
            name,
            format!("{name}: option {field} must be a logical scalar or on/off switch"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn preparation_preserves_public_keys_and_ignores_unconsumed_controls() {
        let mut options = StructValue::new();
        options.insert("StepTolerance", Value::Num(0.125));
        options.insert("Algorithm", Value::Num(42.0));
        options.insert(
            "UnknownResidentLikeField",
            Value::String("untouched".into()),
        );
        let mut cooperation = ScalarCooperation::new(OptimSolver::Fsolve, None);
        let prepared = block_on(prepare_solver_options(
            OptimSolver::Fsolve,
            Some(options),
            None,
            None,
            &mut cooperation,
        ))
        .unwrap()
        .unwrap();
        assert!(matches!(
            prepared.fields.get("StepTolerance"),
            Some(Value::Num(0.125))
        ));
        assert!(matches!(
            prepared.fields.get("Algorithm"),
            Some(Value::Num(42.0))
        ));
        assert!(
            matches!(prepared.fields.get("UnknownResidentLikeField"), Some(Value::String(value)) if value == "untouched")
        );
        assert!(prepared.fields.get("TolX").is_none());
    }
}
