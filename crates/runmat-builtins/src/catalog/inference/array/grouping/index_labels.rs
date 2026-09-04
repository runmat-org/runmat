use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CellFact, DimensionFact, DynamicReason,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

use super::super::super::{argument_error, support};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GRP2IDX-ARITY",
            "grp2idx requires exactly one grouping input",
            request.arguments.len().min(1),
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !supported(input) && !matches!(input.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GRP2IDX-INPUT",
                "grp2idx requires a supported grouping vector or character matrix",
                0,
            ));
        }
    }

    let input = request.arguments.first();
    let g = host(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![input.and_then(observation_count), Some(1)]),
    );
    let level_shape = ShapeFact::from(vec![None, Some(1)]);
    let gn = cellstr(level_shape.clone());
    let gl = input
        .map(|input| level_fact(input, level_shape))
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));

    let mut contract = CallContract::fixed(vec![g, gn, gl]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn supported(input: &ValueFact) -> bool {
    match &input.kind {
        ValueKindFact::Numeric(numeric) => {
            numeric.domain == NumericDomain::Real
                && matches!(input.storage, StorageFact::Scalar | StorageFact::Dense)
        }
        ValueKindFact::Logical
        | ValueKindFact::Character
        | ValueKindFact::String
        | ValueKindFact::Cell(_) => true,
        ValueKindFact::Object(object) => object.runtime_class.as_ref().is_none_or(|class| {
            class.is(runmat_types::standard::CATEGORICAL)
                || class.is(runmat_types::standard::DATETIME)
                || class.is(runmat_types::standard::DURATION)
        }),
        _ => false,
    }
}

fn observation_count(input: &ValueFact) -> Option<usize> {
    if matches!(input.kind, ValueKindFact::Character) {
        return match &input.shape {
            ShapeFact::Scalar => Some(1),
            ShapeFact::Shaped { dims } => dims.first().and_then(|dimension| match dimension {
                DimensionFact::Known(rows) => Some(*rows),
                DimensionFact::Symbolic(_) | DimensionFact::Unknown => None,
            }),
            ShapeFact::Unknown | ShapeFact::Ranked { .. } => None,
        };
    }
    input.shape.element_count()
}

fn level_fact(input: &ValueFact, shape: ShapeFact) -> ValueFact {
    match &input.kind {
        ValueKindFact::String => cellstr(shape),
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
            host(ValueKindFact::Numeric(*numeric), shape)
        }
        ValueKindFact::Logical => host(ValueKindFact::Logical, shape),
        ValueKindFact::Character => host(ValueKindFact::Character, ShapeFact::Ranked { rank: 2 }),
        ValueKindFact::Cell(cell) => host(ValueKindFact::Cell(cell.clone()), shape),
        ValueKindFact::Object(object) => host(ValueKindFact::Object(object.clone()), shape),
        ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
    }
}

fn cellstr(shape: ShapeFact) -> ValueFact {
    let character = host(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
    );
    host(
        ValueKindFact::Cell(CellFact {
            element: Box::new(character),
            elements: Vec::new(),
            elements_complete: false,
        }),
        shape,
    )
}

fn host(kind: ValueKindFact, shape: ShapeFact) -> ValueFact {
    let mut fact = ValueFact::proven(kind, shape, StorageFact::Dense);
    support::facts::materialize(&mut fact);
    fact.residency = ResidencyFact::Host;
    fact
}
