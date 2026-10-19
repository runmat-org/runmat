use runmat_types::{InferenceDiagnostic, NumericClass, NumericDomain, ValueFact, ValueKindFact};

use super::super::super::argument_error;

pub(super) fn resolve(
    population: &ValueFact,
    selection: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<NumericClass> {
    if matches!(population.kind, ValueKindFact::Unknown)
        || matches!(selection.kind, ValueKindFact::Unknown)
    {
        return None;
    }
    let (Some(population), Some(selection)) = (population.numeric(), selection.numeric()) else {
        diagnostics.push(argument_error(
            "RM-CATALOG-NCHOOSEK-COEFFICIENT-INPUT",
            "nchoosek coefficient inputs must be real numeric scalars",
            0,
        ));
        return None;
    };
    if population.domain != NumericDomain::Real || selection.domain != NumericDomain::Real {
        diagnostics.push(argument_error(
            "RM-CATALOG-NCHOOSEK-COEFFICIENT-REAL",
            "nchoosek coefficient inputs must be real",
            0,
        ));
        return None;
    }
    match (population.class, selection.class) {
        (left, right) if left == right => Some(left),
        (NumericClass::Double, other) | (other, NumericClass::Double) => Some(other),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NCHOOSEK-COEFFICIENT-CLASS",
                "nchoosek coefficient inputs must share a class unless one input is double",
                0,
            ));
            None
        }
    }
}
