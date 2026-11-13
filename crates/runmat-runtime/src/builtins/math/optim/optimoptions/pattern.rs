//! The public builder consumes the same typed Pattern defaults/range schema
//! as the solver, without running a solve or invoking a selected callback.
use super::*;
use crate::builtins::math::optim::{common, patternsearch::option_metadata};
use runmat_builtins::{
    OptimPatternAlgorithm, OptimPatternDefault as DefaultValue, OptimPatternOrder,
    OptimPatternPoll, OptimPatternScalarRule as Rule,
};
const POLICY: option_metadata::Policy = option_metadata::Policy {
    name: NAME,
    error: &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
    integer: &INTEGER_OPTION_EXTENSION,
    resident: Some(&RESIDENT_OPTION_EXTENSION),
};
pub(super) fn defaults(
    runtime: Option<&crate::context::RuntimeContext>,
    out: &mut OptionState,
) -> BuiltinResult<()> {
    for field in OptimOptionField::ALL
        .into_iter()
        .filter(|&field| Solver::Patternsearch.accepts(field))
    {
        let value = match DefaultValue::for_field(field).expect("declared Pattern default") {
            DefaultValue::Number(number) => Value::Num(number),
            DefaultValue::Text(text) => state::text(runtime, text)?,
            DefaultValue::Logical(flag) => Value::Bool(flag),
            DefaultValue::Empty => common::output_callback::empty(runtime)?,
            DefaultValue::DimensionCount(_) => state::text(
                runtime,
                DefaultValue::dimension_expression(field).expect("dimension default"),
            )?,
        };
        out.set(field, value);
    }
    Ok(())
}
pub(super) fn metadata(
    field: OptimOptionField,
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    option_metadata::metadata(field, value, POLICY, runtime)?;
    if matches!(value, Value::GpuTensor(_)) {
        return Ok(());
    }
    validate(field, value)
}
pub(super) fn validate(field: OptimOptionField, value: &Value) -> BuiltinResult<()> {
    use OptimOptionField as Field;
    if matches!(value, Value::Tensor(t) if t.is_empty()) {
        return Ok(());
    }
    if matches!(field, Field::OutputFcn | Field::PlotFcn) {
        return common::output_callback::validate_for_builtin(
            NAME,
            value,
            if field == Field::OutputFcn {
                common::output_callback::CallbackForm::Handles
            } else {
                common::output_callback::CallbackForm::HandlesOrNames
            },
        );
    }
    if field == Field::UseParallel {
        common::quasi_newton_controls::parallel_mode(NAME, value)?;
        return Ok(());
    }
    if option_metadata::switch_field(field) {
        logical_value(field.name(), value)?;
        return Ok(());
    }
    if field == Field::SearchFcn {
        return option_metadata::validate_search(value, POLICY);
    }
    let text = crate::builtins::common::validation::value_text_scalar_string(value);
    if matches!(
        field,
        Field::Algorithm | Field::Display | Field::PollMethod | Field::PollOrderAlgorithm
    ) {
        let text =
            text.ok_or_else(|| POLICY.invalid(format!("{} must be scalar text", field.name())))?;
        let valid = match field {
            Field::Algorithm => OptimPatternAlgorithm::parse(&text).is_some(),
            Field::Display => OptimDisplay::parse(&text)
                .is_some_and(|display| Solver::Patternsearch.accepts_display(display)),
            Field::PollMethod => OptimPatternPoll::parse(&text).is_some(),
            Field::PollOrderAlgorithm => OptimPatternOrder::parse(&text).is_some(),
            _ => unreachable!("selected text role"),
        };
        return if valid {
            Ok(())
        } else {
            Err(POLICY.invalid(format!("unrecognized {}", field.name())))
        };
    }
    if let Some(text) = text {
        return if Some(text.as_str()) == DefaultValue::dimension_expression(field) {
            Ok(())
        } else {
            Err(POLICY.invalid(format!(
                "{} has no such dimension-dependent default",
                field.name()
            )))
        };
    }
    let rule =
        Rule::for_field(field).ok_or_else(|| POLICY.invalid("undeclared Pattern scalar role"))?;
    if let Rule::Count { zero } = rule {
        let number = common::option_usize_value(NAME, Some(value), field.name(), 0)?;
        return if zero || number > 0 {
            Ok(())
        } else {
            Err(POLICY.invalid(format!("{} must be positive", field.name())))
        };
    }
    let number = if field == Field::MaxTime && crate::builtins::duration::is_duration_object(value)
    {
        let tensor = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
        if tensor.len() != 1 || tensor.as_f64_slice().is_none() {
            return Err(POLICY.invalid("MaxTime duration must be a real double scalar"));
        }
        tensor.as_f64_slice().expect("checked double")[0]
    } else {
        option_metadata::number(field, value, POLICY)?
    };
    if rule.accepts(number) {
        Ok(())
    } else {
        Err(POLICY.invalid(format!("{} is outside its supported range", field.name())))
    }
}
