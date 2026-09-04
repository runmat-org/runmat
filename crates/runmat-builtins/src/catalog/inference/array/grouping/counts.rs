use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CellFact, ClassIdentity, DynamicReason,
    NumericClass, NumericDomain, NumericFact, ObjectFact, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

use super::super::super::{argument_error, support};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = validate(request);
    let first = request.arguments.first();
    let table = first.is_some_and(is_tabular);
    let outputs = if table {
        vec![table_fact()]
    } else {
        vec![double_column(), group_labels(first), double_column()]
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn validate(request: &CallRequest) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-GROUPCOUNTS-ARITY",
            "groupcounts requires grouping data",
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        let supported = matches!(
            input.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::String
                | ValueKindFact::Cell(_)
                | ValueKindFact::Object(_)
                | ValueKindFact::Unknown
        );
        if !supported {
            diagnostics.push(argument_error(
                "RM-CATALOG-GROUPCOUNTS-INPUT",
                "groupcounts requires array, cell-array, table, or timetable grouping data",
                0,
            ));
        }
    }
    diagnostics
}

fn group_labels(input: Option<&ValueFact>) -> ValueFact {
    let Some(input) = input else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let kind = match &input.kind {
        ValueKindFact::Cell(cell) => ValueKindFact::Cell(cell.clone()),
        ValueKindFact::Numeric(value) => ValueKindFact::Numeric(*value),
        ValueKindFact::Logical => ValueKindFact::Logical,
        ValueKindFact::String => ValueKindFact::String,
        ValueKindFact::Object(value) => ValueKindFact::Object(value.clone()),
        ValueKindFact::Unknown => return ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => return ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
    };
    if super::roles::is_known_matrix(input) {
        let element = host(kind, ShapeFact::from(vec![None, Some(1)]));
        let Some(columns) = super::roles::known_expanded_count(input) else {
            return ValueFact::unknown(DynamicReason::RuntimeValue);
        };
        return host(
            ValueKindFact::Cell(CellFact {
                element: Box::new(element),
                elements: Vec::new(),
                elements_complete: false,
            }),
            ShapeFact::from(vec![Some(1), Some(columns)]),
        );
    }
    host(kind, ShapeFact::from(vec![None, Some(1)]))
}

fn double_column() -> ValueFact {
    host(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![None, Some(1)]),
    )
}

fn table_fact() -> ValueFact {
    host(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(ClassIdentity::from(runmat_types::standard::TABLE)),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Unknown,
    )
}

fn is_tabular(input: &ValueFact) -> bool {
    matches!(&input.kind, ValueKindFact::Object(value) if value.runtime_class.as_ref().is_some_and(|class| class.is(runmat_types::standard::TABLE) || class.is(runmat_types::standard::TIMETABLE)))
}

fn host(kind: ValueKindFact, shape: ShapeFact) -> ValueFact {
    let mut fact = ValueFact::proven(kind, shape, StorageFact::Dense);
    support::facts::materialize(&mut fact);
    fact.residency = ResidencyFact::Host;
    fact
}
