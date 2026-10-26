use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `beam`, `shell`, `solid`, `cell`, `allBeams`, `allShells`, `allSolids`, `allCells`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&beam::ENTRY, &inference::BEAM);
    assert_model_object_contract(&shell::ENTRY, &inference::SHELL);
    assert_model_object_contract(&solid::ENTRY, &inference::SOLID);
    assert_model_object_contract(&cell::ENTRY, &inference::CELL);
    assert_model_object_contract(&all_beams::ENTRY, &inference::ALL_BEAMS);
    assert_model_object_contract(&all_shells::ENTRY, &inference::ALL_SHELLS);
    assert_model_object_contract(&all_solids::ENTRY, &inference::ALL_SOLIDS);
    assert_model_object_contract(&all_cells::ENTRY, &inference::ALL_CELLS);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.PropertyRegionTarget."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}
