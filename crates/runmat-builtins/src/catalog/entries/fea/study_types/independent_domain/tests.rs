use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn independent_domain_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&static_domain::ENTRY, &inference::STATIC_DOMAIN),
        (&time_seconds::ENTRY, &inference::TIME_SECONDS),
        (&frequency_hertz::ENTRY, &inference::FREQUENCY_HERTZ),
        (&load_factor::ENTRY, &inference::LOAD_FACTOR),
        (&eigenvalue::ENTRY, &inference::EIGENVALUE),
        (&study_parameter::ENTRY, &inference::STUDY_PARAMETER),
        (&response_spectra::ENTRY, &inference::RESPONSE_SPECTRA),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
