use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

mod invalid;
mod known;
mod unknown;

fn infer(
    name: &str,
    arguments: Vec<ValueFact>,
    outputs: RequestedOutputCount,
) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name(name).expect("environment catalog entry"),
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(outputs),
        },
    )
}

fn shaped(kind: ValueKindFact, dimensions: &[usize]) -> ValueFact {
    ValueFact::proven(
        kind,
        ShapeFact::from(dimensions.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn character_row() -> ValueFact {
    shaped(ValueKindFact::Character, &[1, 8])
}
