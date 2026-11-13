use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn state_input_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&initial_conditions::ENTRY, &inference::INITIAL_CONDITIONS),
        (&upstream::ENTRY, &inference::UPSTREAM),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
