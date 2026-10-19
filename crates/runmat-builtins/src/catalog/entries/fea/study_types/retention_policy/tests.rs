use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn retention_policy_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&canonical::ENTRY, &inference::CANONICAL),
        (
            &until_dependents_complete::ENTRY,
            &inference::UNTIL_DEPENDENTS_COMPLETE,
        ),
        (&latest_samples::ENTRY, &inference::LATEST_SAMPLES),
        (&explicit_samples::ENTRY, &inference::EXPLICIT_SAMPLES),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
