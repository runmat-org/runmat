use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn reduction_policy_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&none::ENTRY, &inference::NONE),
        (&new::ENTRY, &inference::NEW),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
