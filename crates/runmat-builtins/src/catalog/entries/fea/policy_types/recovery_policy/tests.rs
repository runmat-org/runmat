use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn recovery_policy_identities_conform() {
    assert_family(&[
        (&native::FEA_RECOVERY_POLICY_NATIVE_CATALOG_ENTRY, &native::POLICY_CALL),
        (&extrapolate::FEA_RECOVERY_POLICY_EXTRAPOLATE_CATALOG_ENTRY, &extrapolate::POLICY_CALL),
        (&volume_weighted_average::FEA_RECOVERY_POLICY_VOLUME_WEIGHTED_AVERAGE_CATALOG_ENTRY, &volume_weighted_average::POLICY_CALL),
        (&area_weighted_average::FEA_RECOVERY_POLICY_AREA_WEIGHTED_AVERAGE_CATALOG_ENTRY, &area_weighted_average::POLICY_CALL),
        (&conservative_projection::FEA_RECOVERY_POLICY_CONSERVATIVE_PROJECTION_CATALOG_ENTRY, &conservative_projection::POLICY_CALL),
        (&shape_function_interpolation::FEA_RECOVERY_POLICY_SHAPE_FUNCTION_INTERPOLATION_CATALOG_ENTRY, &shape_function_interpolation::POLICY_CALL),
    ]);
}
