use super::*;

#[test]
fn signed_power_balance_checks_arithmetic_and_one_watt_normalization() {
    let balanced = ThermalEnergyBalance::from_powers(4.0, 3.0, 9.0, 2.0);
    assert_eq!(balanced.residual_w, 0.0);
    assert_eq!(balanced.relative_residual, 0.0);
    balanced.validate().unwrap();
    let unbalanced = ThermalEnergyBalance::from_powers(4.0, 3.0, 8.0, 2.0);
    assert_eq!(unbalanced.residual_w, 1.0);
    assert_eq!(unbalanced.relative_residual, 0.125);
    unbalanced.validate().unwrap(); // arithmetic consistency is distinct from acceptance
    let small = ThermalEnergyBalance::from_powers(0.25, 0.0, 0.0, 0.0);
    assert_eq!(small.relative_residual, 0.25);
    small.validate().unwrap();
    let mut forged = balanced;
    forged.applied_power_w += 1.0;
    assert_eq!(
        forged.validate(),
        Err(ThermalEnergyBalanceError::Inconsistent)
    );
    forged = balanced;
    forged.relative_residual = -1.0;
    assert_eq!(
        forged.validate(),
        Err(ThermalEnergyBalanceError::Inconsistent)
    );
}

#[test]
fn energy_balance_rejects_nonfinite_and_overflow_without_rewriting_evidence() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            ThermalEnergyBalance::from_powers(value, 0.0, 0.0, 0.0).validate(),
            Err(ThermalEnergyBalanceError::NonFinite)
        );
    }
    let mut overflow = ThermalEnergyBalance::from_powers(f64::MAX, f64::MAX, 0.0, 0.0);
    overflow.residual_w = 0.0;
    overflow.relative_residual = 0.0;
    assert_eq!(
        overflow.validate(),
        Err(ThermalEnergyBalanceError::NonFinite)
    );
    let balance = ThermalEnergyBalance::from_powers(1.25, -0.5, 0.25, -0.5);
    let encoded = serde_json::to_vec(&balance).unwrap();
    let decoded: ThermalEnergyBalance = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(balance, decoded);
    decoded.validate().unwrap();
}
