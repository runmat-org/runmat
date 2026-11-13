use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn observation_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&field::ENTRY, &inference::FIELD),
        (&history::ENTRY, &inference::HISTORY),
        (&probe::ENTRY, &inference::PROBE),
        (&reduction::ENTRY, &inference::REDUCTION),
        (&diagnostic::ENTRY, &inference::DIAGNOSTIC),
        (&monitor::ENTRY, &inference::MONITOR),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
