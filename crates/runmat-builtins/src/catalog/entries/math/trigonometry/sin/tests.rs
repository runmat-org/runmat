use crate::*;
use runmat_types::EffectKind;

#[test]
fn trigonometric_provider_routes_declare_suspension_and_fixed_output_errors() {
    for entry in [
        &ACOS_CATALOG_ENTRY,
        &ACOSH_CATALOG_ENTRY,
        &ASIN_CATALOG_ENTRY,
        &ASINH_CATALOG_ENTRY,
        &ATAN_CATALOG_ENTRY,
        &ATAN2_CATALOG_ENTRY,
        &ATANH_CATALOG_ENTRY,
        &COS_CATALOG_ENTRY,
        &COSD_CATALOG_ENTRY,
        &COSH_CATALOG_ENTRY,
        &COSPI_CATALOG_ENTRY,
        &SIN_CATALOG_ENTRY,
        &SIND_CATALOG_ENTRY,
        &SINH_CATALOG_ENTRY,
        &SINPI_CATALOG_ENTRY,
        &TAN_CATALOG_ENTRY,
        &TAND_CATALOG_ENTRY,
        &TANH_CATALOG_ENTRY,
        &DEG2RAD_CATALOG_ENTRY,
        &RAD2DEG_CATALOG_ENTRY,
    ] {
        assert_eq!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::MaySuspend,
            "{}",
            entry.identity.name
        );
        assert!(
            entry.contract.effects.contains(&EffectKind::MaySuspend),
            "{}",
            entry.identity.name
        );
        let identifier = format!("RunMat:{}:TooManyOutputs", entry.identity.name);
        assert!(
            entry
                .descriptor
                .errors
                .iter()
                .any(|error| error.identifier == Some(identifier.as_str())),
            "{}",
            entry.identity.name
        );
    }
}
