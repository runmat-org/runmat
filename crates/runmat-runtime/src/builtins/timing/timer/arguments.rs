//! One typed timer name/value admission path for constructors and discovery.

use super::*;
use crate::builtins::common::gpu_helpers::gather_value_async;

pub(super) struct PreparedTimerPair {
    pub(super) property: TimerProperty,
    pub(super) value: Value,
}

pub(super) async fn prepare_timer_property_pairs(
    args: &[Value],
    finding: bool,
) -> BuiltinResult<Vec<PreparedTimerPair>> {
    if !args.len().is_multiple_of(2) {
        let name = if finding { "timerfind" } else { "timer" };
        return Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            format!("{name}: property filters must appear in name-value pairs"),
        ));
    }
    // Admit every property and every resident value before initiating the
    // first provider operation. A late typo or read-only assignment must not
    // download an earlier numeric argument.
    let mut admitted = Vec::with_capacity(args.len() / 2);
    for pair in args.chunks_exact(2) {
        let property = canonical_property_name(&value_to_string(&pair[0])?)?;
        if !finding && property.is_readonly() {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!("timer: property '{}' is read-only", property.name()),
            ));
        }
        if property.is_numeric() {
            if crate::builtins::common::validation::value_contains_explicit_gpu(&pair[1]) {
                let extension = if finding {
                    &TIMERFIND_EXPLICIT_GPU_NUMERIC_FILTER_EXTENSION
                } else {
                    &TIMER_EXPLICIT_GPU_NUMERIC_PROPERTY_EXTENSION
                };
                crate::compatibility::ensure_builtin_extension_enabled(
                    extension,
                    if finding { "timerfind" } else { "timer" },
                )?;
            } else if !finding {
                normalize_property_value(property, pair[1].clone())?;
            }
        } else if !finding {
            normalize_property_value(property, pair[1].clone())?;
        }
        admitted.push((property, &pair[1]));
    }
    let mut prepared = Vec::with_capacity(admitted.len());
    for (property, input) in admitted {
        let value = if property.is_numeric() {
            // Retain the provider's identifier, retry policy, source and span.
            gather_value_async(input).await?
        } else {
            input.clone()
        };
        prepared.push(PreparedTimerPair { property, value });
    }
    Ok(prepared)
}

pub(super) fn apply_name_value_pairs(
    object: &mut ObjectInstance,
    args: &[PreparedTimerPair],
    constructor: bool,
) -> BuiltinResult<()> {
    for pair in args {
        let property = pair.property;
        if property.is_readonly() {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!("timer: property '{}' is read-only", property.name()),
            ));
        }
        let value = normalize_property_value(property, pair.value.clone())?;
        if !constructor && running_is_on(object) && property.frozen_while_running() {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!(
                    "timer: property '{}' cannot be changed while Running is on",
                    property.name()
                ),
            ));
        }
        object.properties.insert(property.name().to_string(), value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_value_admission_keeps_typed_identity_through_assignment() {
        let pairs = futures::executor::block_on(prepare_timer_property_pairs(
            &[
                Value::String(" eXeCuTiOnMoDe ".into()),
                Value::String("FIXEDSPACING".into()),
                Value::String("period".into()),
                Value::Num(0.25),
            ],
            false,
        ))
        .expect("typed timer options");
        assert_eq!(pairs[0].property, TimerProperty::ExecutionMode);
        assert_eq!(pairs[1].property, TimerProperty::Period);
        let mut object = default_timer_object("timer-test".into());
        apply_name_value_pairs(&mut object, &pairs, true).expect("apply typed options");
        assert_eq!(
            object.properties.get("ExecutionMode"),
            Some(&Value::String("fixedSpacing".into()))
        );
        assert_eq!(object.properties.get("Period"), Some(&Value::Num(0.25)));
    }

    #[test]
    fn late_invalid_property_is_rejected_before_numeric_materialization() {
        let error = futures::executor::block_on(prepare_timer_property_pairs(
            &[
                Value::String("Period".into()),
                Value::Num(0.25),
                Value::String("unknown".into()),
                Value::Num(1.0),
            ],
            false,
        ))
        .err()
        .expect("invalid property");
        assert_eq!(error.identifier(), Some("RunMat:timer:InvalidProperty"));

        let error = futures::executor::block_on(prepare_timer_property_pairs(
            &[
                Value::String("Period".into()),
                Value::Num(0.25),
                Value::String("Running".into()),
                Value::String("on".into()),
            ],
            false,
        ))
        .err()
        .expect("read-only property");
        assert_eq!(error.identifier(), Some("RunMat:timer:InvalidProperty"));
    }
}
