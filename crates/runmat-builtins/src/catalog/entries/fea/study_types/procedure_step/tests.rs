use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn procedure_step_constructors_follow_their_positional_contract() {
    assert_constructor_contract(&configure::ENTRY, &inference::CONFIGURE);
}

#[test]
fn configure_returns_the_input_family_step_class() {
    use crate::fea_contract::study_types::{
        classes::*,
        test_support::{infer, object, valid_fact},
    };
    use runmat_types::{StaticClassIdentity, ValueKindFact};
    let thermal = StaticClassIdentity::new("fea.ThermalStep");
    let mut arguments: Vec<_> = inference::CONFIGURE
        .arguments
        .iter()
        .copied()
        .map(valid_fact)
        .collect();
    arguments[0] = object(Some(thermal));
    let inferred = infer(&configure::ENTRY, arguments.clone());
    assert!(inferred.diagnostics.is_empty());
    let ValueKindFact::Object(result) = &inferred.outputs[0].kind else {
        panic!("object")
    };
    assert_eq!(result.runtime_class, Some(thermal.owned()));
    arguments[0] = object(Some(FEA_STATE_OUTPUT_CLASS));
    assert!(!infer(&configure::ENTRY, arguments).diagnostics.is_empty());
}
