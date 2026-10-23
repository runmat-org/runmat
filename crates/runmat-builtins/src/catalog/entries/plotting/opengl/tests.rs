use super::*;
use runmat_types::*;
#[test]
fn canonical_output_contract_and_host_result() {
    let arguments = vec![];
    for count in [0, 1, 2] {
        let request = CallRequest {
            arguments: arguments.clone(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        };
        let result = inference::infer(&request, &OPENGL_CATALOG_ENTRY);
        assert_eq!(result.diagnostics.is_empty(), count <= 1);
        assert_eq!(result.effects, OPENGL_CATALOG_ENTRY.contract.effect_set());
        if count == 1 {
            assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        }
    }
}

#[test]
fn typed_report_has_exact_fields_and_no_hardware_capability_claim() {
    let request = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String)],
        literals: LiteralContext {
            literal_args: vec![LiteralValue::String("info".into())],
        },
        outputs: OutputSelection::default(),
    };
    let result = inference::infer(&request, &OPENGL_CATALOG_ENTRY);
    let ValueKindFact::Struct(fields) = &result.outputs[0].kind else {
        panic!("expected report");
    };
    assert!(fields.fields_complete);
    assert_eq!(fields.fields.len(), 10);
    assert_eq!(fields.fields["Software"].kind, ValueKindFact::Logical);
    assert!(OPENGL_CATALOG_ENTRY.contract.capabilities.is_empty());
    assert_eq!(OpenGlCommand::parse(" DATA "), Some(OpenGlCommand::Report));
    assert_eq!(OpenGlSaveMode::parse("neverselect"), None);
}
