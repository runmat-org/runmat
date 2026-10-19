use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn monitor_cadence_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&accepted_increments::ENTRY, &inference::ACCEPTED_INCREMENTS),
        (&wall_seconds::ENTRY, &inference::WALL_SECONDS),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
