use super::*;
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, LiteralContext, LiteralValue,
    NumericClass, NumericDomain, NumericFact, ObjectFact, OutputSelection, RequestedOutputCount,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn call(input: ValueFact, controls: Vec<LiteralValue>, outputs: usize) -> CallInference {
    let mut arguments = vec![input];
    arguments.extend(controls.iter().map(|literal| match literal {
        LiteralValue::String(_) | LiteralValue::Character(_) | LiteralValue::Keyword(_) => {
            ValueFact::scalar(ValueKindFact::String)
        }
        LiteralValue::Empty | LiteralValue::Unknown => {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        _ => numeric(NumericClass::Double, ShapeFact::Scalar),
    }));
    let mut literals = vec![LiteralValue::Unknown];
    literals.extend(controls);
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &MAX_CATALOG_ENTRY,
    )
}

mod descriptor;
mod reduction;
mod control;
mod pairwise;
