//! A table always has three columns; its row count depends on observed values.
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CellFact, DynamicReason,
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
        Some(ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Cell(_)) => {
            cell_kind()
        }
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

fn cell_kind() -> ValueKindFact {
    ValueKindFact::Cell(CellFact {
        element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
        elements: Vec::new(),
        elements_complete: false,
    })
}
