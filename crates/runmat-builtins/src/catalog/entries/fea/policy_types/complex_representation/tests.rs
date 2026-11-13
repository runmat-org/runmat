use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn complex_representation_identities_conform() {
    assert_family(&[
        (
            &real::FEA_COMPLEX_REPRESENTATION_REAL_CATALOG_ENTRY,
            &real::POLICY_CALL,
        ),
        (
            &imaginary::FEA_COMPLEX_REPRESENTATION_IMAGINARY_CATALOG_ENTRY,
            &imaginary::POLICY_CALL,
        ),
        (
            &magnitude::FEA_COMPLEX_REPRESENTATION_MAGNITUDE_CATALOG_ENTRY,
            &magnitude::POLICY_CALL,
        ),
        (
            &phase_radians::FEA_COMPLEX_REPRESENTATION_PHASE_RADIANS_CATALOG_ENTRY,
            &phase_radians::POLICY_CALL,
        ),
        (
            &complex_value::FEA_COMPLEX_REPRESENTATION_COMPLEX_VALUE_CATALOG_ENTRY,
            &complex_value::POLICY_CALL,
        ),
    ]);
}
