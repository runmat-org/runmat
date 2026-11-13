use super::*;
use runmat_types::*;
fn infer(arguments: Vec<ValueFact>, count: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
        &REGISTER_TEST_CLASSES_CATALOG_ENTRY,
    )
}
#[test]
fn fixed_arity_and_single_output_are_owned() {
    let args = Vec::new();
    assert!(infer(args.clone(), 1).diagnostics.is_empty());
    assert!(!infer(args, 2).diagnostics.is_empty());
    assert!(!infer(vec![ValueFact::scalar(ValueKindFact::String); 1], 1)
        .diagnostics
        .is_empty());
}
