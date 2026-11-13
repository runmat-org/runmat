use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn observation_event_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&first_yield::ENTRY, &inference::FIRST_YIELD),
        (
            &contact_status_change::ENTRY,
            &inference::CONTACT_STATUS_CHANGE,
        ),
        (&peak_response::ENTRY, &inference::PEAK_RESPONSE),
        (&convergence_failure::ENTRY, &inference::CONVERGENCE_FAILURE),
        (
            &accuracy_criterion_crossing::ENTRY,
            &inference::ACCURACY_CRITERION_CROSSING,
        ),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
