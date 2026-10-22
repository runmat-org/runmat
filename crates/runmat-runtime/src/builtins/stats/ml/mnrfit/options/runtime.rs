use super::super::data::validate_option_weight_length;
use super::super::{
    gather_value, invalid, numeric_vector, positive_integer, positive_scalar, scalar_text,
    FitCooperation,
};
use super::{OptionName, Options};
use crate::BuiltinResult;
use runmat_builtins::{MnrfitConstant, MnrfitStatsetField};
use runmat_value::Value;

/// Public controls are decoded once. Only values that actually carry a resident
/// operand cross the provider boundary, and they retain their typed identity.
pub(in crate::builtins::stats::ml::mnrfit) struct PreparedOptions {
    actions: Vec<PreparedAction>,
}

enum PreparedAction {
    Host(OptionValue),
    Resident(OptionName, Value),
    ResidentStatset(MnrfitStatsetField, Value),
}

enum OptionValue {
    Accepted,
    Constant(bool),
    Weights(Vec<f64>),
    IterationLimit(usize),
    Tolerance(f64),
}

pub(in crate::builtins::stats::ml::mnrfit) async fn prepare_values(
    values: Vec<(OptionName, Value)>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<PreparedOptions> {
    let mut actions = Vec::new();
    for (name, value) in values {
        if name == OptionName::Options {
            let Value::Struct(statset) = value else {
                return Err(invalid("mnrfit: Options must be a struct"));
            };
            for field in MnrfitStatsetField::ALL {
                let Some(value) = statset.fields.get(field.public_name()).cloned() else {
                    continue;
                };
                if crate::dispatcher::value_contains_gpu(&value) {
                    actions.push(PreparedAction::ResidentStatset(field, value));
                } else {
                    actions.push(PreparedAction::Host(decode_statset(field, value)?));
                }
            }
        } else if crate::dispatcher::value_contains_gpu(&value) {
            actions.push(PreparedAction::Resident(name, value));
        } else {
            actions.push(PreparedAction::Host(
                decode(name, value, cooperation).await?,
            ));
        }
    }
    Ok(PreparedOptions { actions })
}

impl PreparedOptions {
    /// Validate only the effective host Weights option. Earlier valid values
    /// may be superseded, while a final resident value remains deferred until
    /// its owning provider has supplied the payload.
    pub(in crate::builtins::stats::ml::mnrfit) fn preflight_host_weights(
        &self,
        rows: usize,
    ) -> BuiltinResult<()> {
        for action in self.actions.iter().rev() {
            match action {
                PreparedAction::Host(OptionValue::Weights(weights)) => {
                    return validate_option_weight_length(weights.len(), rows)
                }
                PreparedAction::Resident(OptionName::Weights, _) => return Ok(()),
                _ => {}
            }
        }
        Ok(())
    }

    pub(in crate::builtins::stats::ml::mnrfit) async fn materialize(
        self,
        cooperation: &mut FitCooperation,
    ) -> BuiltinResult<Options> {
        let mut options = Options::default();
        for action in self.actions {
            let decoded = match action {
                PreparedAction::Host(value) => value,
                PreparedAction::Resident(name, value) => {
                    decode(name, gather_value(value).await?, cooperation).await?
                }
                PreparedAction::ResidentStatset(field, value) => {
                    decode_statset(field, gather_value(value).await?)?
                }
            };
            match decoded {
                OptionValue::Accepted => {}
                OptionValue::Constant(value) => options.constant = value,
                OptionValue::Weights(value) => options.weights = Some(value),
                OptionValue::IterationLimit(value) => options.max_iter = value,
                OptionValue::Tolerance(value) => options.tol_x = value,
            }
        }
        if options.max_iter == 0 {
            return Err(invalid("mnrfit: IterationLimit must be positive"));
        }
        if !options.tol_x.is_finite() || options.tol_x <= 0.0 {
            return Err(invalid("mnrfit: Tolerance must be positive and finite"));
        }
        Ok(options)
    }
}

async fn decode(
    name: OptionName,
    value: Value,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<OptionValue> {
    match name {
        OptionName::Model => {
            let model =
                scalar_text(&value)?.ok_or_else(|| invalid("mnrfit: Model option must be text"))?;
            match model.to_ascii_lowercase().as_str() {
                "nominal" => Ok(OptionValue::Accepted),
                other => Err(invalid(format!(
                    "mnrfit: Model '{other}' is not supported yet"
                ))),
            }
        }
        OptionName::Link => {
            let link =
                scalar_text(&value)?.ok_or_else(|| invalid("mnrfit: Link option must be text"))?;
            match link.to_ascii_lowercase().as_str() {
                "logit" => Ok(OptionValue::Accepted),
                other => Err(invalid(format!(
                    "mnrfit: Link '{other}' is not supported yet"
                ))),
            }
        }
        OptionName::Constant => {
            let constant = match text_or_logical(&value, "Constant")? {
                BoolOrText::Bool(flag) => flag,
                BoolOrText::Text(text) => MnrfitConstant::from_public(&text)
                    .map(MnrfitConstant::enabled)
                    .ok_or_else(|| {
                        invalid(format!(
                            "mnrfit: Constant must be 'on' or 'off', got '{}'",
                            text.to_ascii_lowercase()
                        ))
                    })?,
            };
            Ok(OptionValue::Constant(constant))
        }
        OptionName::Interactions => {
            let text = scalar_text(&value)?
                .ok_or_else(|| invalid("mnrfit: Interactions option must be text"))?;
            if !text.eq_ignore_ascii_case("off") {
                return Err(invalid(
                    "mnrfit: Interactions values other than 'off' are not supported yet",
                ));
            }
            Ok(OptionValue::Accepted)
        }
        OptionName::Weights => Ok(OptionValue::Weights(
            numeric_vector(value, "Weights", cooperation).await?,
        )),
        OptionName::MaxIter | OptionName::IterationLimit => Ok(OptionValue::IterationLimit(
            positive_integer(&value, "IterationLimit")?,
        )),
        OptionName::TolX | OptionName::Tolerance => Ok(OptionValue::Tolerance(positive_scalar(
            &value,
            "Tolerance",
        )?)),
        OptionName::Options => Err(invalid("mnrfit: Options must be a struct")),
        OptionName::Display => {
            let _ = scalar_text(&value)?
                .ok_or_else(|| invalid("mnrfit: Display option must be text"))?;
            Ok(OptionValue::Accepted)
        }
        OptionName::EstDisp => {
            let text = scalar_text(&value)?
                .ok_or_else(|| invalid("mnrfit: EstDisp option must be text"))?;
            if !text.eq_ignore_ascii_case("off") {
                return Err(invalid(
                    "mnrfit: overdispersion estimation is not supported yet",
                ));
            }
            Ok(OptionValue::Accepted)
        }
    }
}

fn decode_statset(field: MnrfitStatsetField, value: Value) -> BuiltinResult<OptionValue> {
    let label = match field {
        MnrfitStatsetField::MaxIter => "Options.MaxIter",
        MnrfitStatsetField::IterationLimit => "Options.IterationLimit",
        MnrfitStatsetField::TolX => "Options.TolX",
        MnrfitStatsetField::Tolerance => "Options.Tolerance",
    };
    if field.is_iteration_limit() {
        Ok(OptionValue::IterationLimit(positive_integer(
            &value, label,
        )?))
    } else {
        Ok(OptionValue::Tolerance(positive_scalar(&value, label)?))
    }
}

enum BoolOrText {
    Bool(bool),
    Text(String),
}

fn text_or_logical(value: &Value, label: &str) -> BuiltinResult<BoolOrText> {
    if let Some(text) = scalar_text(value)? {
        return Ok(BoolOrText::Text(text.into_owned()));
    }
    match value {
        Value::Bool(flag) => Ok(BoolOrText::Bool(*flag)),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(BoolOrText::Bool(logical.data[0] != 0))
        }
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(BoolOrText::Bool(*value != 0.0)),
        _ => Err(invalid(format!("mnrfit: {label} must be text or logical"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::task::noop_waker;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use runmat_value::{LogicalArray, Tensor};
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::task::{Context, Poll};

    #[test]
    fn host_weight_conversion_cooperates_before_provider_materialization() {
        for weights in [
            Value::Tensor(
                Tensor::new(vec![1.0; 20_000], vec![20_000, 1]).expect("host weight vector"),
            ),
            Value::LogicalArray(
                LogicalArray::new(vec![1; 20_000], vec![20_000, 1]).expect("host logical weights"),
            ),
        ] {
            let source = CancellationSource::new();
            let runtime = crate::context::RuntimeContext::with_cancellation(
                std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
                source.token(),
            );
            let budget = crate::execution::CpuCooperationBudget::with_test_turn(
                crate::execution::CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
                source.token(),
                || Box::pin(std::future::pending()),
            );
            let mut cooperation = FitCooperation::with_budget(&runtime, budget);
            let mut prepared = Box::pin(prepare_values(
                vec![(OptionName::Weights, weights)],
                &mut cooperation,
            ));
            let waker = noop_waker();
            assert!(prepared
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending());
            source.cancel(CancellationReason::User);
            let Poll::Ready(Err(error)) = prepared.as_mut().poll(&mut Context::from_waker(&waker))
            else {
                panic!("cancellation must stop host Weights conversion");
            };
            assert!(error.message().contains("cancelled"));
        }
    }
}
