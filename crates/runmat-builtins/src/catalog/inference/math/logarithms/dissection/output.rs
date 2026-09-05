use super::super::super::super::support::facts::{materialize, preserve_shape_as_dynamic};
use super::super::super::super::{argument_error, numeric_kind};
use runmat_types::{
    AliasFact, DynamicReason, MutationFact, NumericClass, NumericDomain, ResidencyFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(
    input: Option<&ValueFact>,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
    diagnose_invalid: bool,
) -> ValueFact {
    let Some(input) = input else {
        diagnose(
            diagnostics,
            diagnose_invalid,
            "RM-CATALOG-LOG2-ARITY",
            "log2 requires exactly one input",
        );
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if matches!(input.residency, ResidencyFact::Device { .. }) {
        diagnose(
            diagnostics,
            diagnose_invalid,
            "RM-CATALOG-LOG2-GPU-DISSECTION",
            "two-output log2 does not support GPU-resident input",
        );
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnose(
            diagnostics,
            diagnose_invalid,
            "RM-CATALOG-LOG2-SPARSE",
            "log2 does not currently accept sparse input",
        );
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex => {
            diagnose(
                diagnostics,
                diagnose_invalid,
                "RM-CATALOG-LOG2-COMPLEX-DISSECTION",
                "two-output log2 requires real input under the current compatibility pin",
            );
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            numeric.domain = NumericDomain::Real;
            materialize(&mut output);
            output
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize(&mut output);
            output
        }
        ValueKindFact::Object(object)
            if object
                .runtime_class
                .as_ref()
                .is_some_and(runmat_types::standard::is_tabular) =>
        {
            object.properties.clear();
            object.properties_complete = false;
            output.alias = AliasFact::Unique;
            output.mutation = MutationFact::ValueSemantics;
            output
        }
        ValueKindFact::Object(_) | ValueKindFact::Unknown if !diagnose_invalid => {
            preserve_shape_as_dynamic(&mut output);
            output
        }
        _ => {
            diagnose(
                diagnostics,
                diagnose_invalid,
                "RM-CATALOG-LOG2-DISSECTION-INPUT",
                "two-output log2 requires real single, double, or supported tabular input",
            );
            ValueFact::unknown(if diagnose_invalid {
                DynamicReason::UnsupportedRepresentation
            } else {
                DynamicReason::RuntimeValue
            })
        }
    }
}

fn diagnose(
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
    enabled: bool,
    code: &'static str,
    message: &'static str,
) {
    if enabled {
        diagnostics.push(argument_error(code, message, 0));
    }
}
