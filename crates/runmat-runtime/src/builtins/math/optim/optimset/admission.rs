use super::{
    ensure_optimset_extensions, optimset_error_with_detail, OPTIMSET_ERROR_INVALID_ARGUMENT,
    OPTIMSET_ERROR_INVALID_INPUT,
};
use crate::builtins::math::optim::common::field_name;
use crate::BuiltinResult;
use runmat_builtins::{OptimParameterLookup, OptimParameterName};
use runmat_value::{StructValue, Value};

pub(super) enum FieldName {
    Known(OptimParameterName),
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
    Copy(StructValue, bool),
    Pair(FieldName, Value),
}

pub(super) fn admit(args: crate::call::BuiltinArguments) -> BuiltinResult<Vec<Step>> {
    ensure_optimset_extensions(args.view())?;
    let pair_start = usize::from(
        matches!(args.first(), Some(Value::Struct(_)))
            || matches!(args.first(),Some(Value::Tensor(value))if value.is_empty()),
    );
    if args.len() == 2 && pair_start == 1 && matches!(&args[1], Value::Struct(_)) {
        let mut steps = Vec::new();
        for (index, value) in args.into_iter().enumerate() {
            if let Value::Struct(value) = value {
                steps.push(Step::Copy(value, index == 1));
            }
        }
        return Ok(steps);
    }
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
            steps.push(Step::Copy(existing, false));
        } else if matches!(&first,Value::Tensor(value)if value.is_empty()) {
            // Empty oldopts has no fields, but still consumes its position.
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
    match OptimParameterName::resolve(&name) {
        OptimParameterLookup::Matched(field) => Ok(FieldName::Known(field)),
        OptimParameterLookup::Ambiguous => Err(optimset_error_with_detail(
            &OPTIMSET_ERROR_INVALID_INPUT,
            "ambiguous legacy option abbreviation",
        )),
        OptimParameterLookup::Unknown => Ok(FieldName::Custom(name)),
    }
}

pub(super) fn canonical_name(name: String) -> String {
    match OptimParameterName::resolve(&name) {
        OptimParameterLookup::Matched(known) if name != known.name() => known.name().to_owned(),
        _ => name,
    }
}
