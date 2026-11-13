use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    BuiltinId, CallRequest, CallableFact, CallableIdentity, CapabilitySet, LiteralContext,
    NumericClass, NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

fn callable(identity: CallableIdentity, output: Option<ValueFact>) -> ValueFact {
    let output_is_known = output.is_some();
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: Some(identity),
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: output.into_iter().collect(),
        outputs_complete: output_is_known,
        variadic_inputs: true,
        variadic_outputs: !output_is_known,
        captures: Vec::new(),
        captures_complete: true,
    }))
}

fn builtin(name: &str) -> ValueFact {
    callable(CallableIdentity::Builtin(BuiltinId(name.into())), None)
}

fn infer(arguments: Vec<ValueFact>, literals: LiteralContext) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name("arrayfun").expect("arrayfun entry"),
        &CallRequest {
            arguments,
            literals,
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

mod output;
mod validation;
