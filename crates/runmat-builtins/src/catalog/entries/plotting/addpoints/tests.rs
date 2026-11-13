use super::*;
use runmat_types::*;
#[test]
fn canonical_output_contract_and_host_result() {
    let double = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let arguments = vec![double.clone(), double.clone(), double.clone()];
    for count in [0, 1, 2] {
        let request = CallRequest {
            arguments: arguments.clone(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        };
        let result = inference::infer(&request, &ADDPOINTS_CATALOG_ENTRY);
        assert_eq!(result.diagnostics.is_empty(), count == 0);
        assert_eq!(
            result.effects,
            ADDPOINTS_CATALOG_ENTRY.contract.effect_set()
        );
    }
}
