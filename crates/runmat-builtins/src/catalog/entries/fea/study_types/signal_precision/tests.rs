use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn signal_precision_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&real32::ENTRY, &inference::REAL32),
        (&real64::ENTRY, &inference::REAL64),
        (&complex64::ENTRY, &inference::COMPLEX64),
        (&complex128::ENTRY, &inference::COMPLEX128),
    ] {
        assert_constructor_contract(entry, signature);
    }
}
