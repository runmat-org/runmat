//! A table always has three columns; its row count depends on observed values.
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CellFact, DimensionFact, DynamicReason,
    InferenceDiagnostic, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let shape = ShapeFact::from(vec![None, Some(3)]);
    let mut diagnostics = Vec::new();
    let kind = match request.arguments.first().map(|value| &value.kind) {
        Some(ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        })) => {
            if class.integer_class().is_some() {
                cell_kind()
            } else {
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })
            }
        }
        Some(ValueKindFact::String) => cell_kind(),
        Some(ValueKindFact::Character) => match character_row(&request.arguments[0].shape) {
            Admission::Valid => cell_kind(),
            Admission::Invalid => {
                diagnostics.push(InferenceDiagnostic::error(
                    "RM.TABULATE.INPUT",
                    "tabulate character input must have one row",
                ));
                ValueKindFact::Unknown
            }
            Admission::Unknown => ValueKindFact::Unknown,
        },
        Some(ValueKindFact::Cell(cell)) => match cell_labels(cell) {
            Admission::Valid => cell_kind(),
            Admission::Invalid => {
                diagnostics.push(InferenceDiagnostic::error(
                    "RM.TABULATE.INPUT",
                    "tabulate cell entries must be scalar strings or scalars",
                ));
                ValueKindFact::Unknown
            }
            Admission::Unknown => ValueKindFact::Unknown,
        },
        Some(ValueKindFact::Logical) => ValueKindFact::Unknown,
        Some(ValueKindFact::Unknown) => ValueKindFact::Unknown,
        Some(_) => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.TABULATE.INPUT",
                "tabulate requires real numeric, logical, or text labels",
            ));
            ValueKindFact::Unknown
        }
        None => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.TABULATE.ARITY",
                "tabulate requires one input",
            ));
            ValueKindFact::Unknown
        }
    };
    if request.arguments.len() > 1 {
        diagnostics.push(InferenceDiagnostic::error(
            "RM.TABULATE.ARITY",
            "tabulate accepts one input",
        ));
    }
    let mut output = if matches!(kind, ValueKindFact::Unknown) {
        let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
        fact.shape = shape;
        fact
    } else {
        let storage = if matches!(kind, ValueKindFact::Cell(_)) {
            StorageFact::Unknown
        } else {
            StorageFact::Dense
        };
        ValueFact::proven(kind, shape, storage)
    };
    output.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !diagnostics.is_empty() {
        for output in &mut result.outputs {
            *output = ValueFact::unknown(DynamicReason::RuntimeValue);
        }
    }
    result.diagnostics.extend(diagnostics);
    result
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Admission {
    Valid,
    Invalid,
    Unknown,
}

fn character_row(shape: &ShapeFact) -> Admission {
    match shape {
        ShapeFact::Scalar => Admission::Valid,
        ShapeFact::Shaped { dims } => match dims.first() {
            Some(DimensionFact::Known(1)) => Admission::Valid,
            Some(DimensionFact::Known(_)) => Admission::Invalid,
            _ => Admission::Unknown,
        },
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => Admission::Unknown,
    }
}

fn cell_labels(cell: &CellFact) -> Admission {
    if !cell.elements_complete {
        return Admission::Unknown;
    }
    let mut result = Admission::Valid;
    for element in &cell.elements {
        match cell_label(element) {
            Admission::Invalid => return Admission::Invalid,
            Admission::Unknown => result = Admission::Unknown,
            Admission::Valid => {}
        }
    }
    result
}

fn cell_label(element: &ValueFact) -> Admission {
    match &element.kind {
        ValueKindFact::String => match element.shape.element_count() {
            Some(1) => Admission::Valid,
            Some(_) => Admission::Invalid,
            None => Admission::Unknown,
        },
        ValueKindFact::Character => character_row(&element.shape),
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical => match element.shape.element_count() {
            Some(1) | None => Admission::Unknown,
            Some(_) => Admission::Invalid,
        },
        ValueKindFact::Unknown => Admission::Unknown,
        _ => Admission::Invalid,
    }
}

fn cell_kind() -> ValueKindFact {
    ValueKindFact::Cell(CellFact {
        element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
        elements: Vec::new(),
        elements_complete: false,
    })
}
