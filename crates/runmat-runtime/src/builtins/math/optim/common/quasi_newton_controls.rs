//! Shared value admission for the modern and legacy QN option domains.
use super::*;
use runmat_builtins::{OptimOptionField as Field, OptimQuasiNewtonUpdate as Update};

fn invalid_control(name: &str, message: impl Into<String>) -> RuntimeError {
    let mut error = optim_error(name, message);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

pub(crate) fn update(name: &str, field: Field, value: &Value) -> BuiltinResult<Update> {
    let legacy = field == Field::HessUpdate;
    if let Value::Cell(cell) = value {
        if legacy || cell.data().len() != 2 {
            return Err(invalid_control(
                name,
                "HessianApproximation must be 'bfgs', 'lbfgs', or {'lbfgs',positive integer}",
            ));
        }
        let method = option_string_value(Some(&cell.data()[0]), field.name(), "")?;
        if !method.eq_ignore_ascii_case("lbfgs") {
            return Err(invalid_control(
                name,
                "the two-element HessianApproximation form requires 'lbfgs'",
            ));
        }
        // A concrete scalar, not a rich-facts walk over an arbitrary carrier.
        // A device-resident history length is not a documented host option.
        if !matches!(
            &cell.data()[1],
            Value::Num(_) | Value::Int(_) | Value::Tensor(_)
        ) {
            return Err(invalid_control(
                name,
                "L-BFGS history must be a host positive integer scalar",
            ));
        }
        let history = option_usize_value(name, Some(&cell.data()[1]), "L-BFGS history", 0)?;
        let history = std::num::NonZeroUsize::new(history)
            .ok_or_else(|| invalid_control(name, "L-BFGS history must be positive"))?;
        return Ok(Update::LimitedMemory { history });
    }
    let text = option_string_value(Some(value), field.name(), "")?;
    Update::parse(&text, legacy).ok_or_else(|| {
        invalid_control(
            name,
            if legacy {
                "HessUpdate must be 'bfgs', 'lbfgs', 'dfp', or 'steepdesc'"
            } else {
                "HessianApproximation must be 'bfgs', 'lbfgs', or {'lbfgs',positive integer}"
            },
        )
    })
}

pub(crate) fn history_operand(value: &Value) -> Option<&Value> {
    match value {
        Value::Cell(cell) if cell.data().len() == 2 => cell.data().get(1),
        _ => None,
    }
}

pub(crate) fn objective_limit(name: &str, value: &Value) -> BuiltinResult<f64> {
    let value = match value {
        Value::Num(number) => *number,
        Value::Tensor(t) if t.len() == 1 && t.integer_storage().is_none() => {
            crate::builtins::common::tensor::tensor_value_f64(t, 0)
        }
        _ => return option_f64_value(name, Some(value), "ObjectiveLimit", -1e20),
    };
    if value.is_nan() {
        Err(invalid_control(
            name,
            "ObjectiveLimit must be a real non-NaN scalar",
        ))
    } else {
        Ok(value)
    }
}

pub(crate) fn parallel_mode(
    name: &str,
    value: &Value,
) -> BuiltinResult<crate::execution::CallbackParallelMode> {
    use crate::execution::CallbackParallelMode as Mode;
    if let Some(integer) = crate::builtins::common::tensor::scalar_integer_value(value) {
        return match integer.try_to_u64() {
            Some(0) => Ok(Mode::Off),
            Some(1) => Ok(Mode::Auto),
            _ => Err(invalid_control(name, "UseParallel must be logical 0 or 1")),
        };
    }
    let off = match value {
        Value::Bool(value) => !*value,
        Value::LogicalArray(value) if value.data.len() == 1 => value.data[0] == 0,
        Value::Num(value) if *value == 0.0 || *value == 1.0 => *value == 0.0,
        Value::Tensor(t) if t.len() == 1 && t.integer_storage().is_none() => {
            let x = crate::builtins::common::tensor::tensor_value_f64(t, 0);
            if x != 0.0 && x != 1.0 {
                return Err(invalid_control(
                    name,
                    "UseParallel must be off/on/auto or logical",
                ));
            }
            x == 0.0
        }
        _ => {
            let text = option_string_value(Some(value), "UseParallel", "")?;
            match text.to_ascii_lowercase().as_str() {
                "off" => true,
                "on" => return Ok(Mode::On),
                "auto" => false,
                _ => {
                    return Err(invalid_control(
                        name,
                        "UseParallel must be off/on/auto or logical",
                    ))
                }
            }
        }
    };
    Ok(if off { Mode::Off } else { Mode::Auto })
}

pub(crate) fn parallel_enabled(name: &str, value: &Value) -> BuiltinResult<bool> {
    Ok(parallel_mode(name, value)? != crate::execution::CallbackParallelMode::Off)
}
pub(crate) fn parallel(name: &str, value: &Value) -> BuiltinResult<()> {
    parallel_mode(name, value).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn modern_legacy_value_domains_and_inactive_roles_are_not_conflated() {
        assert_eq!(
            update("fminunc", Field::HessUpdate, &Value::from("dfp")).unwrap(),
            Update::Dfp
        );
        assert!(update("fminunc", Field::HessianApproximation, &Value::from("dfp")).is_err());
        for history in [0., -1., 1.5, f64::INFINITY] {
            let value = Value::Cell(
                runmat_value::CellArray::new(vec![Value::from("lbfgs"), Value::Num(history)], 1, 2)
                    .unwrap(),
            );
            assert!(update("fminunc", Field::HessianApproximation, &value).is_err());
        }
        assert!(parallel("fminunc", &Value::Bool(false)).is_ok());
        assert_eq!(
            parallel_mode("fminunc", &Value::from("auto")).unwrap(),
            crate::execution::CallbackParallelMode::Auto
        );
        assert_eq!(
            parallel_mode("fminunc", &Value::from("on")).unwrap(),
            crate::execution::CallbackParallelMode::On
        );
        assert_eq!(
            parallel_mode("fminunc", &Value::Bool(true)).unwrap(),
            crate::execution::CallbackParallelMode::Auto
        );
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        runtime.set_error_namespace("OriginalBatch");
        let foreign = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        foreign.set_error_namespace("ForeignBatch");
        let _foreign = foreign.enter();
        assert!(crate::execution::CallbackBatch::select(
            Some(&runtime),
            &Value::FunctionHandle("abs".into()),
            crate::execution::CallbackParallelMode::Auto
        )
        .unwrap()
        .is_none());
        let refused = crate::execution::CallbackBatch::select(
            Some(&runtime),
            &Value::FunctionHandle("abs".into()),
            crate::execution::CallbackParallelMode::On,
        )
        .err()
        .expect("serial adapter is not parallel");
        assert_eq!(refused.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            refused.identifier(),
            Some("OriginalBatch:Execution:ParallelCallbackUnavailable")
        );
        assert_eq!(
            crate::context::legacy::active().unwrap().error_namespace(),
            "ForeignBatch"
        );
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("SpecifyObjectiveGradient", Value::Bool(true));
        fields.insert("Algorithm", Value::from("trust-region"));
        fields.insert("UseParallel", Value::from("not a switch"));
        fields.insert("HessianApproximation", Value::from("not an update"));
        let mut cooperation = ScalarCooperation::new(OptimSolver::Fminunc, None);
        let prepared = futures::executor::block_on(prepare_solver_options(
            OptimSolver::Fminunc,
            Some(fields.finish()),
            None,
            None,
            None,
            &mut cooperation,
        ))
        .unwrap()
        .unwrap();
        assert!(prepared.structure().fields.contains_key("UseParallel"));
        assert!(prepared
            .structure()
            .fields
            .contains_key("HessianApproximation"));
        assert!(Field::parse_for_constructor("HessUpdate", OptimSolver::Fminunc).is_none());
        assert_eq!(
            Field::parse_for_solver("HessUpdate", OptimSolver::Fminunc),
            Some(Field::HessUpdate)
        );
    }
}
