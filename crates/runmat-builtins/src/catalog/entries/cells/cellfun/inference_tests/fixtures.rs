use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    BuiltinId, CallRequest, CallableFact, CallableIdentity, CapabilitySet, CellFact, DimensionFact,
    LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn numeric(class: NumericClass) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }))
}

pub(super) fn cell(element: ValueFact, dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::Shaped {
            dims: dims.iter().copied().map(DimensionFact::Known).collect(),
        },
        StorageFact::Dense,
    )
}

pub(super) fn builtin(name: &str) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: Some(CallableIdentity::Builtin(BuiltinId(name.into()))),
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: Vec::new(),
        captures_complete: true,
    }))
}

pub(super) fn infer(
    arguments: Vec<ValueFact>,
    literals: LiteralContext,
) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name("cellfun").expect("cellfun entry"),
        &CallRequest {
            arguments,
            literals,
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}
