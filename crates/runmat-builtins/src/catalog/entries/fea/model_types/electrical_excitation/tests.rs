use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `terminalVoltage`, `current`, `charge`, `currentDensity`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&terminal_voltage::ENTRY, &inference::TERMINAL_VOLTAGE);
    assert_model_object_contract(&current::ENTRY, &inference::CURRENT);
    assert_model_object_contract(&charge::ENTRY, &inference::CHARGE);
    assert_model_object_contract(&current_density::ENTRY, &inference::CURRENT_DENSITY);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.ElectricalExcitation."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}
