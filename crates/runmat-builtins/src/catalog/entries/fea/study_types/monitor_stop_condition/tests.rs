use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn monitor_stop_condition_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&none::ENTRY, &inference::NONE),
        (&less_than::ENTRY, &inference::LESS_THAN),
        (&greater_than::ENTRY, &inference::GREATER_THAN),
        (
            &absolute_change_below::ENTRY,
            &inference::ABSOLUTE_CHANGE_BELOW,
        ),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
