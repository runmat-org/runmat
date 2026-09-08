use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, CellFact, DimensionFact, DynamicReason, ShapeFact, StorageFact,
    StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let diagnostics = diagnostics(request);
    let output =
        output_fact(request).unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    finish_fixed(entry, request, output, diagnostics)
}

fn output_fact(request: &CallRequest) -> Option<ValueFact> {
    let input = request.arguments.first()?;
    let ValueKindFact::Cell(_) = &input.kind else {
        return None;
    };
    let dim = match request.arguments.get(2) {
        None => 1.0,
        Some(_) => request.literals.numeric_at(2)?,
    };
    if !dim.is_finite() || dim < 1.0 || dim.fract() != 0.0 {
        return None;
    }
    if dim > usize::MAX as f64 {
        return None;
    }
    let mut dims = match &input.shape {
        ShapeFact::Scalar => vec![DimensionFact::Known(1), DimensionFact::Known(1)],
        ShapeFact::Shaped { dims } => dims.clone(),
        ShapeFact::Ranked { rank } => vec![DimensionFact::Unknown; *rank],
        ShapeFact::Unknown => return None,
    };
    let index = dim as usize - 1;
    if dims.len() <= index {
        dims.resize(index + 1, DimensionFact::Known(1));
    }
    dims[index] = DimensionFact::Known(1);
    let shape = ShapeFact::Shaped { dims };
    let structure = ValueKindFact::Struct(StructFact {
        fields: BTreeMap::new(),
        fields_complete: false,
    });
    if shape.element_count() == Some(1) {
        return Some(ValueFact::scalar(structure));
    }
    let element = ValueFact::scalar(structure);
    Some(ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: false,
        }),
        shape,
        StorageFact::Dense,
    ))
}

fn diagnostics(request: &CallRequest) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CELL2STRUCT-ARITY",
            "cell2struct expects C, fields, and optional dim",
            request.arguments.len().saturating_sub(1),
        ));
        return diagnostics;
    }
    if !matches!(
        request.arguments[0].kind,
        ValueKindFact::Cell(_) | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CELL2STRUCT-CELL",
            "cell2struct expects a cell array as its first input",
            0,
        ));
    }
    if !valid_fields(&request.arguments[1].kind) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CELL2STRUCT-FIELDS",
            "cell2struct expects character, string, or cell field names",
            1,
        ));
    }
    if let Some(dim) = request.arguments.get(2) {
        let valid = matches!(&dim.kind, ValueKindFact::Numeric(numeric) if numeric.domain == runmat_types::NumericDomain::Real)
            || matches!(dim.kind, ValueKindFact::Unknown);
        if !valid || !dim.is_scalar() {
            diagnostics.push(argument_error(
                "RM-CATALOG-CELL2STRUCT-DIM",
                "cell2struct expects a positive real scalar dimension",
                2,
            ));
        }
    }
    diagnostics
}

fn valid_fields(kind: &ValueKindFact) -> bool {
    matches!(
        kind,
        ValueKindFact::Character
            | ValueKindFact::String
            | ValueKindFact::Cell(_)
            | ValueKindFact::Unknown
    )
}
