use crate::catalog::inference::argument_error;
use runmat_types::*;

pub(super) fn argument(request: &CallRequest, index: usize) -> ValueFact {
    request
        .arguments
        .get(index)
        .cloned()
        .unwrap_or_else(unknown)
}
pub(super) fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}
pub(super) fn count(
    request: &CallRequest,
    minimum: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if request.arguments.len() < minimum {
        diagnostics.push(argument_error(
            "RM-SORTING-ARITY",
            "not enough input arguments",
            0,
        ));
    }
}
pub(super) fn dimension(shape: &ShapeFact, index: usize) -> Option<usize> {
    match shape {
        ShapeFact::Scalar => Some(1),
        ShapeFact::Shaped { dims } => match dims.get(index) {
            Some(DimensionFact::Known(n)) => Some(*n),
            None => Some(1),
            _ => None,
        },
        _ => None,
    }
}
pub(super) fn column(length: Option<usize>) -> ShapeFact {
    ShapeFact::from(vec![length, Some(1)])
}
pub(super) fn index(shape: ShapeFact, residency: ResidencyFact) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    fact.residency = residency;
    fact
}
pub(super) fn logical(shape: ShapeFact, residency: ResidencyFact) -> ValueFact {
    let mut fact = ValueFact::proven(ValueKindFact::Logical, shape, StorageFact::Dense);
    fact.residency = residency;
    fact
}
pub(super) fn first_resident(a: &ValueFact, b: &ValueFact) -> ResidencyFact {
    match (&a.residency, &b.residency) {
        (ResidencyFact::Device { .. }, _) => a.residency.clone(),
        (ResidencyFact::Host, ResidencyFact::Device { .. }) => b.residency.clone(),
        (ResidencyFact::Host, ResidencyFact::Host) => ResidencyFact::Host,
        _ => ResidencyFact::Unknown,
    }
}
pub(super) fn matrix(value: &ValueFact, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if value.shape.rank().is_some_and(|rank| rank > 2) {
        diagnostics.push(argument_error(
            "RM-SORTING-MATRIX",
            "row operations require a matrix",
            index,
        ));
    }
}
pub(super) fn row_compatibility(
    a: &ValueFact,
    b: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    matrix(a, 0, diagnostics);
    matrix(b, 1, diagnostics);
    if let (Some(left), Some(right)) = (dimension(&a.shape, 1), dimension(&b.shape, 1)) {
        if left != right {
            diagnostics.push(argument_error(
                "RM-SET-COLUMNS",
                "row inputs must have equal column counts",
                1,
            ));
        }
    }
}
pub(super) fn text_option(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<String> {
    let value = request.arguments.get(index)?;
    if !matches!(
        value.kind,
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
    ) {
        diagnostics.push(argument_error(
            "RM-SORTING-OPTION",
            "expected a text option",
            index,
        ));
    }
    request
        .literals
        .literal_string_at(index)
        .map(|text| text.trim().to_owned())
}
pub(super) fn accepted_data(
    value: &ValueFact,
    text: bool,
    tables: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
    index: usize,
) {
    let accepted = match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::Numeric(_) | ValueKindFact::Logical => true,
        ValueKindFact::String | ValueKindFact::Character => text,
        ValueKindFact::Object(object) => {
            tables
                && object
                    .runtime_class
                    .as_ref()
                    .is_some_and(|class| class.is(standard::TABLE))
        }
        _ => false,
    };
    if !accepted || matches!(value.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-SORTING-INPUT",
            "unsupported input representation",
            index,
        ));
    }
    if matches!(value.kind, ValueKindFact::Numeric(NumericFact { class, domain: NumericDomain::Complex }) if class.integer_class().is_some())
    {
        diagnostics.push(argument_error(
            "RM-SORTING-COMPLEX-INTEGER",
            "complex integer inputs are not supported",
            index,
        ));
    }
}
