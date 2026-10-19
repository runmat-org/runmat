use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let (shape, diagnostics) = shape_and_diagnostics(request);
    let element = empty_element(request);
    let output = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: shape.element_count() == Some(0),
        }),
        shape,
        StorageFact::Dense,
    );
    finish_fixed(entry, request, output, diagnostics)
}

fn shape_and_diagnostics(
    request: &CallRequest,
) -> (ShapeFact, Vec<runmat_types::InferenceDiagnostic>) {
    let literals = &request.literals.literal_args;
    let like = literals.iter().position(is_like);
    let dimension_end = like.unwrap_or(literals.len());
    let mut diagnostics = Vec::new();
    if let Some(index) = like {
        if index + 1 >= request.arguments.len() {
            diagnostics.push(argument_error(
                "RM-CATALOG-CELL-LIKE",
                "cell expects one prototype after the like keyword",
                index,
            ));
        }
        if request.arguments.len() > index + 2 {
            diagnostics.push(argument_error(
                "RM-CATALOG-CELL-LIKE-POSITION",
                "cell requires like and its prototype to be the final arguments",
                index,
            ));
        }
    }
    if known_invalid_sizes(literals.get(..dimension_end).unwrap_or_default()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CELL-SIZE",
            "cell sizes must be finite integer scalars or one numeric row vector",
            0,
        ));
    }
    let shape = if dimension_end == 0 {
        like.and_then(|index| request.arguments.get(index + 1))
            .map(|prototype| prototype.shape.clone())
            .unwrap_or_else(|| ShapeFact::from(vec![Some(0), Some(0)]))
    } else {
        inferred_size(request, dimension_end)
    };
    (shape, diagnostics)
}

fn inferred_size(request: &CallRequest, dimension_end: usize) -> ShapeFact {
    if dimension_end == 1 {
        if let Some(dims) = request.literals.numeric_vector_at(0) {
            let dims = dims
                .into_iter()
                .map(|dim| dim.map(normalize_signed))
                .collect::<Vec<_>>();
            return if dims.len() == 1 {
                ShapeFact::from(vec![dims[0], dims[0]])
            } else {
                ShapeFact::from(normalize_trailing(dims))
            };
        }
        if let Some(size) = request.literals.numeric_at(0).and_then(size_literal) {
            return ShapeFact::from(vec![Some(size), Some(size)]);
        }
        return ShapeFact::Ranked { rank: 2 };
    }
    let dims = request
        .literals
        .numeric_dims()
        .into_iter()
        .take(dimension_end)
        .map(|dim| dim.map(normalize_signed))
        .collect::<Vec<_>>();
    ShapeFact::from(normalize_trailing(dims))
}

fn normalize_signed(value: usize) -> usize {
    value
}

fn normalize_trailing(mut dims: Vec<Option<usize>>) -> Vec<Option<usize>> {
    while dims.len() > 2 && dims.last() == Some(&Some(1)) {
        dims.pop();
    }
    dims
}

fn size_literal(value: f64) -> Option<usize> {
    (value.is_finite() && value.fract() == 0.0 && value < usize::MAX as f64)
        .then_some(value.max(0.0) as usize)
}

fn known_invalid_sizes(values: &[LiteralValue]) -> bool {
    values.iter().any(|value| match value {
        LiteralValue::Number(value) => size_literal(*value).is_none(),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse::<f64>().ok().and_then(size_literal).is_none()
        }
        LiteralValue::Vector(values) => values.iter().any(|value| match value {
            LiteralValue::Number(value) => size_literal(*value).is_none(),
            LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
                text.parse::<f64>().ok().and_then(size_literal).is_none()
            }
            LiteralValue::Unknown => false,
            _ => true,
        }),
        LiteralValue::Unknown => false,
        _ => true,
    })
}

fn empty_element(request: &CallRequest) -> ValueFact {
    let literals = &request.literals.literal_args;
    let prototype = literals
        .iter()
        .position(is_like)
        .and_then(|index| request.arguments.get(index + 1));
    let kind = match prototype.map(|value| &value.kind) {
        Some(ValueKindFact::Logical) => ValueKindFact::Logical,
        Some(ValueKindFact::Character) => ValueKindFact::Character,
        Some(ValueKindFact::String) => ValueKindFact::String,
        Some(ValueKindFact::Cell(_)) => ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: true,
        }),
        Some(ValueKindFact::Struct(_)) => {
            ValueKindFact::Struct(StructFact::array(BTreeMap::new(), true, Vec::new(), true))
        }
        _ => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
    };
    let shape = match &kind {
        ValueKindFact::Struct(_) => ShapeFact::Scalar,
        ValueKindFact::String if prototype.is_some_and(ValueFact::is_scalar) => ShapeFact::Scalar,
        _ => ShapeFact::from(vec![Some(0), Some(0)]),
    };
    ValueFact::proven(kind, shape, StorageFact::Dense)
}

fn is_like(value: &LiteralValue) -> bool {
    matches!(value, LiteralValue::String(text) | LiteralValue::Character(text) | LiteralValue::Keyword(text) if text.eq_ignore_ascii_case("like"))
}
