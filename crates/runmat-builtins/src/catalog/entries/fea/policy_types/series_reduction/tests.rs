use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn series_reduction_identities_conform() {
    assert_family(&[
        (
            &none_method::FEA_SERIES_REDUCTION_NONE_CATALOG_ENTRY,
            &none_method::POLICY_CALL,
        ),
        (
            &minimum::FEA_SERIES_REDUCTION_MINIMUM_CATALOG_ENTRY,
            &minimum::POLICY_CALL,
        ),
        (
            &maximum::FEA_SERIES_REDUCTION_MAXIMUM_CATALOG_ENTRY,
            &maximum::POLICY_CALL,
        ),
        (
            &absolute_maximum::FEA_SERIES_REDUCTION_ABSOLUTE_MAXIMUM_CATALOG_ENTRY,
            &absolute_maximum::POLICY_CALL,
        ),
        (
            &envelope::FEA_SERIES_REDUCTION_ENVELOPE_CATALOG_ENTRY,
            &envelope::POLICY_CALL,
        ),
        (
            &root_mean_square::FEA_SERIES_REDUCTION_ROOT_MEAN_SQUARE_CATALOG_ENTRY,
            &root_mean_square::POLICY_CALL,
        ),
        (
            &mean::FEA_SERIES_REDUCTION_MEAN_CATALOG_ENTRY,
            &mean::POLICY_CALL,
        ),
        (
            &variance::FEA_SERIES_REDUCTION_VARIANCE_CATALOG_ENTRY,
            &variance::POLICY_CALL,
        ),
    ]);
}
