use super::super::argument_error;
use super::super::support::facts::preserve_shape_as_dynamic;
use runmat_types::{
    InferenceDiagnostic, NumericDomain, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn apply_trigonometric_prototype(
    output: &mut ValueFact,
    prototype: &ValueFact,
    name: &str,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if matches!(prototype.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TRIGONOMETRIC-PROTOTYPE",
            format!("{name} does not accept a sparse prototype"),
            2,
        ));
        return;
    }
    match &prototype.kind {
        ValueKindFact::Numeric(prototype_numeric) => {
            if let ValueKindFact::Numeric(output_numeric) = &mut output.kind {
                if prototype_numeric.domain == NumericDomain::Complex {
                    output_numeric.domain = NumericDomain::Complex;
                }
            }
            output.residency = prototype.residency.clone();
        }
        ValueKindFact::Logical => {
            output.residency = prototype.residency.clone();
        }
        ValueKindFact::Unknown => {
            // An unknown prototype can request complex representation.
            preserve_shape_as_dynamic(output);
            output.residency = prototype.residency.clone();
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIGONOMETRIC-PROTOTYPE",
                format!("{name} requires a numeric or logical prototype"),
                2,
            ));
            output.residency = ResidencyFact::Unknown;
        }
    }
}
