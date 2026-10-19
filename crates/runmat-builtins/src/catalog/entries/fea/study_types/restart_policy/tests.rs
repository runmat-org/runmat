use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn restart_policy_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&fresh::ENTRY, &inference::FRESH),
        (
            &continue_from_completed_state::ENTRY,
            &inference::CONTINUE_FROM_COMPLETED_STATE,
        ),
        (&resume_checkpoint::ENTRY, &inference::RESUME_CHECKPOINT),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
