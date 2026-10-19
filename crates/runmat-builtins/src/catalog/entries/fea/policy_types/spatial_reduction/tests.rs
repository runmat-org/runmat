use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn spatial_reduction_identities_conform() {
    assert_family(&[
        (
            &none_method::FEA_SPATIAL_REDUCTION_NONE_CATALOG_ENTRY,
            &none_method::POLICY_CALL,
        ),
        (
            &integral::FEA_SPATIAL_REDUCTION_INTEGRAL_CATALOG_ENTRY,
            &integral::POLICY_CALL,
        ),
        (
            &average::FEA_SPATIAL_REDUCTION_AVERAGE_CATALOG_ENTRY,
            &average::POLICY_CALL,
        ),
        (
            &root_mean_square::FEA_SPATIAL_REDUCTION_ROOT_MEAN_SQUARE_CATALOG_ENTRY,
            &root_mean_square::POLICY_CALL,
        ),
        (
            &euclidean_norm::FEA_SPATIAL_REDUCTION_EUCLIDEAN_NORM_CATALOG_ENTRY,
            &euclidean_norm::POLICY_CALL,
        ),
        (
            &minimum::FEA_SPATIAL_REDUCTION_MINIMUM_CATALOG_ENTRY,
            &minimum::POLICY_CALL,
        ),
        (
            &maximum::FEA_SPATIAL_REDUCTION_MAXIMUM_CATALOG_ENTRY,
            &maximum::POLICY_CALL,
        ),
        (
            &resultant::FEA_SPATIAL_REDUCTION_RESULTANT_CATALOG_ENTRY,
            &resultant::POLICY_CALL,
        ),
        (
            &moment::FEA_SPATIAL_REDUCTION_MOMENT_CATALOG_ENTRY,
            &moment::POLICY_CALL,
        ),
    ]);
}
