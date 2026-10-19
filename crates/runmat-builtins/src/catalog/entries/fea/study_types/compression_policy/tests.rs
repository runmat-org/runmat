use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn compression_policy_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&lossless::ENTRY, &inference::LOSSLESS),
        (&none::ENTRY, &inference::NONE),
        (&error_bounded::ENTRY, &inference::ERROR_BOUNDED),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
