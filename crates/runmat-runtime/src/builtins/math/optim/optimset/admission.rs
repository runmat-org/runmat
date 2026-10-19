use super::{
    ensure_optimset_extensions, optimset_error_with_detail, OPTIMSET_ERROR_INVALID_ARGUMENT,
    OPTIMSET_ERROR_INVALID_INPUT,
};
use crate::builtins::math::optim::common::field_name;
use crate::BuiltinResult;
use runmat_builtins::OptimOptionField;
use runmat_value::{StructValue, Value};

pub(super) enum FieldName {
    Known(OptimOptionField),
    Custom(String),
}

impl FieldName {
    pub(super) fn into_public_name(self) -> String {
        match self {
            Self::Known(field) => field.name().into(),
            Self::Custom(name) => name,
        }
    }
}

pub(super) enum Step {
    Copy(StructValue),
    Pair(FieldName, Value),
}

pub(super) fn admit(args: Vec<Value>) -> BuiltinResult<Vec<Step>> {
    ensure_optimset_extensions(&args)?;
    let pair_start = usize::from(matches!(args.first(), Some(Value::Struct(_))));
    if !(args.len() - pair_start).is_multiple_of(2) {
        return Err(optimset_error_with_detail(
            &OPTIMSET_ERROR_INVALID_ARGUMENT,
            "expected option name/value pairs",
        ));
    }
    let mut input = args.into_iter();
    let mut steps = Vec::new();
    let mut pending_name = None;
    if let Some(first) = input.next() {
        if let Value::Struct(existing) = first {
            steps.push(Step::Copy(existing));
        } else {
            pending_name = Some(decode_name(&first)?);
        }
    }
    for value in input {
        if let Some(name) = pending_name.take() {
            steps.push(Step::Pair(name, value));
        } else {
            pending_name = Some(decode_name(&value)?);
        }
    }
    if pending_name.is_some() {
        return Err(optimset_error_with_detail(
            &OPTIMSET_ERROR_INVALID_ARGUMENT,
            "expected option name/value pairs",
        ));
    }
    Ok(steps)
}

fn decode_name(value: &Value) -> BuiltinResult<FieldName> {
    let name = field_name(value).map_err(|error| {
        optimset_error_with_detail(&OPTIMSET_ERROR_INVALID_INPUT, error.message())
    })?;
    Ok(OptimOptionField::parse(&name)
        .map(FieldName::Known)
        .unwrap_or(FieldName::Custom(name)))
}
