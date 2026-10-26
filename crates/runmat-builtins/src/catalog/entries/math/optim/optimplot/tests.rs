use super::*;
use runmat_types::*;
#[test]
fn all_three_real_callbacks_have_logical_stop_and_truthful_graphics_contracts() {
    for entry in ENTRIES {
        let request = CallRequest {
            arguments: vec![
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                    Default::default(),
                    false,
                ))),
                ValueFact::scalar(ValueKindFact::Character),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let result = inference::infer(&request, entry);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(result.outputs[0].shape.element_count(), Some(1));
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
        assert!(result
            .capabilities
            .0
            .contains(&CapabilityRequirement::UserInterface));
        assert_eq!(entry.documentation.examples.len(), 1);
        assert_eq!(entry.contract.maturity, BuiltinContractMaturity::Complete);
        assert_eq!(entry.descriptor.signatures[0].inputs.len(), 3);
    }
    assert_eq!(OptimPlotKind::CurrentPoint.name(), "optimplotx");
    assert_eq!(OptimPlotKind::FunctionCount.name(), "optimplotfunccount");
    assert_eq!(OptimPlotKind::FunctionValue.name(), "optimplotfval");
}
