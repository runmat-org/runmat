use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn sampling_policy_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&final_sample::ENTRY, &inference::FINAL_SAMPLE),
        (
            &every_accepted_increment::ENTRY,
            &inference::EVERY_ACCEPTED_INCREMENT,
        ),
        (&time_seconds::ENTRY, &inference::TIME_SECONDS),
        (
            &time_interval_seconds::ENTRY,
            &inference::TIME_INTERVAL_SECONDS,
        ),
        (&frequency_hertz::ENTRY, &inference::FREQUENCY_HERTZ),
        (
            &frequency_interval_hertz::ENTRY,
            &inference::FREQUENCY_INTERVAL_HERTZ,
        ),
        (&load_factors::ENTRY, &inference::LOAD_FACTORS),
        (
            &load_factor_interval::ENTRY,
            &inference::LOAD_FACTOR_INTERVAL,
        ),
        (
            &study_parameter_values::ENTRY,
            &inference::STUDY_PARAMETER_VALUES,
        ),
        (&mode_indices::ENTRY, &inference::MODE_INDICES),
        (&event::ENTRY, &inference::EVENT),
        (&adaptive::ENTRY, &inference::ADAPTIVE),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
