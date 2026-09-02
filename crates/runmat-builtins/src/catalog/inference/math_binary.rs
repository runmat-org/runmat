use super::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, RemainderFunction};
use runmat_types::{
    broadcast_shape, standard, AliasFact, CallInference, CallRequest, ContiguityFact,
    DynamicReason, LayoutFact, MutationFact, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer_remainder(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: RemainderFunction,
) -> CallInference {
    let name = match function {
        RemainderFunction::Modulus => "mod",
        RemainderFunction::Remainder => "rem",
    };
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-REMAINDER-ARITY",
            format!("{name} requires exactly two inputs"),
            request.arguments.len().min(1),
        ));
    }
    let Some(left) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let Some(right) = request.arguments.get(1) else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    if matches!(left.storage, StorageFact::Sparse) || matches!(right.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-REMAINDER-SPARSE",
            format!("{name} does not currently accept sparse input"),
            usize::from(!matches!(left.storage, StorageFact::Sparse)),
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    if let Some(output) = infer_object_remainder(left, right, name, &mut diagnostics) {
        return finish_fixed(entry, request, output, diagnostics);
    }

    let mut output = infer_numeric_remainder(left, right, name, &mut diagnostics);
    output.shape = match broadcast_shape(&left.shape, &right.shape) {
        Ok(shape) => shape,
        Err(diagnostic) => {
            diagnostics.push(diagnostic);
            ShapeFact::Unknown
        }
    };
    materialize(&mut output);
    output.residency = compatible_residency(&left.residency, &right.residency);
    finish_fixed(entry, request, output, diagnostics)
}

fn infer_object_remainder(
    left: &ValueFact,
    right: &ValueFact,
    name: &str,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<ValueFact> {
    let left_class = object_class(left);
    let right_class = object_class(right);
    let left_supported = left_class.is_some_and(is_supported_object);
    let right_supported = right_class.is_some_and(is_supported_object);
    if !left_supported && !right_supported {
        if matches!(left.kind, ValueKindFact::Object(_))
            || matches!(right.kind, ValueKindFact::Object(_))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-REMAINDER-OBJECT",
                format!("{name} accepts only table, timetable, and duration objects"),
                usize::from(matches!(left.kind, ValueKindFact::Object(_))),
            ));
            return Some(ValueFact::unknown(DynamicReason::UnsupportedRepresentation));
        }
        return None;
    }

    if left_supported && right_supported {
        let compatible = match (left_class, right_class) {
            (Some(a), Some(b)) if a == standard::DURATION && b == standard::DURATION => true,
            (Some(a), Some(b)) if is_tabular(a) && is_tabular(b) => a == b,
            _ => false,
        };
        if !compatible {
            diagnostics.push(argument_error(
                "RM-CATALOG-REMAINDER-OBJECT-PAIR",
                format!("{name} requires compatible table, timetable, or duration operands"),
                1,
            ));
            return Some(ValueFact::unknown(DynamicReason::UnsupportedRepresentation));
        }
    }

    let source = if left_supported { left } else { right };
    let mut output = source.clone();
    output.residency = ResidencyFact::Host;
    output.alias = AliasFact::Unique;
    output.view = ViewFact::Materialized;
    output.mutation = MutationFact::ValueSemantics;
    if let ValueKindFact::Object(object) = &mut output.kind {
        object.properties.clear();
        object.properties_complete = false;
    }
    Some(output)
}

fn object_class(fact: &ValueFact) -> Option<runmat_types::StaticClassIdentity> {
    let ValueKindFact::Object(object) = &fact.kind else {
        return None;
    };
    let identity = object.runtime_class.as_ref()?;
    [standard::TABLE, standard::TIMETABLE, standard::DURATION]
        .into_iter()
        .find(|candidate| identity.is(*candidate))
}

fn is_tabular(class: runmat_types::StaticClassIdentity) -> bool {
    class == standard::TABLE || class == standard::TIMETABLE
}

fn is_supported_object(class: runmat_types::StaticClassIdentity) -> bool {
    is_tabular(class) || class == standard::DURATION
}

fn infer_numeric_remainder(
    left: &ValueFact,
    right: &ValueFact,
    name: &str,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ValueFact {
    let left_numeric = numeric_input(&left.kind);
    let right_numeric = numeric_input(&right.kind);
    let Some(left_numeric) = left_numeric else {
        if !matches!(left.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-REMAINDER-INPUT",
                format!("{name} requires real numeric, logical, character, table, timetable, or duration input"),
                0,
            ));
        }
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let Some(right_numeric) = right_numeric else {
        if !matches!(right.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-REMAINDER-INPUT",
                format!("{name} requires real numeric, logical, character, table, timetable, or duration input"),
                1,
            ));
        }
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if left_numeric.domain == NumericDomain::Complex
        || right_numeric.domain == NumericDomain::Complex
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-REMAINDER-COMPLEX",
            format!("{name} requires real operands"),
            usize::from(right_numeric.domain == NumericDomain::Complex),
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }

    let left_integer = is_integer_class(left_numeric.class);
    let right_integer = is_integer_class(right_numeric.class);
    let class = if left_integer || right_integer {
        match (left_integer, right_integer) {
            (true, true) if left_numeric.class == right_numeric.class => left_numeric.class,
            (true, false) if right.is_scalar() && right_numeric.class == NumericClass::Double => {
                left_numeric.class
            }
            (false, true) if left.is_scalar() && left_numeric.class == NumericClass::Double => {
                right_numeric.class
            }
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-REMAINDER-INTEGER-CLASS",
                    format!("{name} requires integer operands to share a class; the other operand may be scalar double"),
                    1,
                ));
                return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            }
        }
    } else if left_numeric.class == NumericClass::Single
        || right_numeric.class == NumericClass::Single
    {
        NumericClass::Single
    } else {
        NumericClass::Double
    };
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = numeric_kind(class, NumericDomain::Real);
    output
}

fn is_integer_class(class: NumericClass) -> bool {
    !matches!(class, NumericClass::Double | NumericClass::Single)
}

fn numeric_input(kind: &ValueKindFact) -> Option<NumericFact> {
    match kind {
        ValueKindFact::Numeric(numeric) => Some(*numeric),
        ValueKindFact::Logical | ValueKindFact::Character => Some(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => None,
    }
}

fn compatible_residency(left: &ResidencyFact, right: &ResidencyFact) -> ResidencyFact {
    match (left, right) {
        (ResidencyFact::Host, ResidencyFact::Host) => ResidencyFact::Host,
        (
            ResidencyFact::Device {
                provider: left_owner,
            },
            ResidencyFact::Device {
                provider: right_owner,
            },
        ) if left_owner == right_owner => left.clone(),
        _ => ResidencyFact::Unknown,
    }
}

fn materialize(output: &mut ValueFact) {
    output.storage = if output.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.view = ViewFact::Materialized;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
}

#[cfg(test)]
mod tests;
