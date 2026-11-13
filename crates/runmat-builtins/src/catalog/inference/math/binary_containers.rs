use crate::RemainderFunction;
use runmat_types::{
    standard, AliasFact, DynamicReason, MutationFact, ResidencyFact, StaticClassIdentity,
    ValueFact, ValueKindFact, ViewFact,
};

use super::super::argument_error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Operation {
    Atan2,
    Remainder(RemainderFunction),
}

pub(super) fn infer(
    operation: Operation,
    left: &ValueFact,
    right: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<ValueFact> {
    if let Some(output) = infer_tabular(operation, left, right, diagnostics) {
        return Some(output);
    }
    match operation {
        Operation::Atan2 => None,
        Operation::Remainder(function) => infer_duration(function, left, right, diagnostics),
    }
}

fn infer_tabular(
    operation: Operation,
    left: &ValueFact,
    right: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<ValueFact> {
    let left_class = object_class(left).filter(|class| is_tabular(*class));
    let right_class = object_class(right).filter(|class| is_tabular(*class));
    if left_class.is_none() && right_class.is_none() {
        return None;
    }
    let left_other_object = matches!(left.kind, ValueKindFact::Object(_)) && left_class.is_none();
    let right_other_object =
        matches!(right.kind, ValueKindFact::Object(_)) && right_class.is_none();
    let incompatible_pair =
        left_class.is_some() && right_class.is_some() && left_class != right_class;
    if left_other_object || right_other_object || incompatible_pair {
        diagnostics.push(argument_error(
            "RM-CATALOG-TABULAR-BINARY-PAIR",
            format!(
                "{} requires table and timetable operands to have the same container identity",
                operation.name()
            ),
            usize::from(left_class.is_some()),
        ));
        return Some(unsupported());
    }
    Some(materialized_object(if left_class.is_some() {
        left
    } else {
        right
    }))
}

fn infer_duration(
    function: RemainderFunction,
    left: &ValueFact,
    right: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<ValueFact> {
    let left_supported = object_class(left) == Some(standard::DURATION);
    let right_supported = object_class(right) == Some(standard::DURATION);
    let left_unsupported_object = matches!(left.kind, ValueKindFact::Object(_)) && !left_supported;
    let right_unsupported_object =
        matches!(right.kind, ValueKindFact::Object(_)) && !right_supported;
    if left_unsupported_object || right_unsupported_object {
        diagnostics.push(argument_error(
            "RM-CATALOG-REMAINDER-OBJECT",
            format!(
                "{} accepts only table, timetable, and duration objects",
                remainder_name(function)
            ),
            usize::from(!left_unsupported_object),
        ));
        return Some(unsupported());
    }
    if !left_supported && !right_supported {
        return None;
    }
    Some(materialized_object(if left_supported {
        left
    } else {
        right
    }))
}

fn materialized_object(source: &ValueFact) -> ValueFact {
    let mut output = source.clone();
    output.residency = ResidencyFact::Host;
    output.alias = AliasFact::Unique;
    output.view = ViewFact::Materialized;
    output.mutation = MutationFact::ValueSemantics;
    if let ValueKindFact::Object(object) = &mut output.kind {
        object.properties.clear();
        object.properties_complete = false;
    }
    output
}

fn object_class(fact: &ValueFact) -> Option<StaticClassIdentity> {
    let ValueKindFact::Object(object) = &fact.kind else {
        return None;
    };
    let identity = object.runtime_class.as_ref()?;
    [standard::TABLE, standard::TIMETABLE, standard::DURATION]
        .into_iter()
        .find(|candidate| identity.is(*candidate))
}

fn is_tabular(class: StaticClassIdentity) -> bool {
    class == standard::TABLE || class == standard::TIMETABLE
}

fn unsupported() -> ValueFact {
    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
}

impl Operation {
    fn name(self) -> &'static str {
        match self {
            Self::Atan2 => "atan2",
            Self::Remainder(function) => remainder_name(function),
        }
    }
}

fn remainder_name(function: RemainderFunction) -> &'static str {
    match function {
        RemainderFunction::Modulus => "mod",
        RemainderFunction::Remainder => "rem",
    }
}
