use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn analysis_sequence_constructors_follow_their_positional_contract() {
    assert_constructor_contract(&new::ENTRY, &inference::NEW);
}
