use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let diagnostics = diagnostics(request);
    let output = output(request).unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    finish_fixed(entry, request, output, diagnostics)
}

fn output(request: &CallRequest) -> Option<ValueFact> {
    let input = request.arguments.first()?;
    let ValueKindFact::Cell(cell) = &input.kind else {
        return None;
    };
    if input.shape.element_count() == Some(0) {
        return Some(ValueFact::proven(
            ValueKindFact::Numeric(runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(0), Some(0)]),
            StorageFact::Dense,
        ));
    }
    let element = cell.element.as_ref();
    let kind = match &element.kind {
        ValueKindFact::Numeric(numeric) => ValueKindFact::Numeric(*numeric),
        ValueKindFact::Logical => ValueKindFact::Logical,
        ValueKindFact::Character => ValueKindFact::Character,
        _ => return None,
    };
    Some(ValueFact::proven(
        kind,
        ShapeFact::Unknown,
        StorageFact::Dense,
    ))
}

fn diagnostics(request: &CallRequest) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-CELL2MAT-ARITY",
            "cell2mat expects one cell array",
            request.arguments.len().saturating_sub(1),
        ));
        return diagnostics;
    }
    if !matches!(
        request.arguments[0].kind,
        ValueKindFact::Cell(_) | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CELL2MAT-CELL",
            "cell2mat expects a cell array",
            0,
        ));
    }
    diagnostics
}
