use super::*;

#[test]
fn relative_conservation_is_invariant_under_power_scaling() {
    // Powers of two avoid introducing decimal scaling roundoff into this contract test.
    // The independently calculated imbalance is 1/8 of the largest power term.
    for exponent in [-900, -100, -40, 0, 40, 100, 900] {
        let scale = 2.0_f64.powi(exponent);
        let balance = ThermalEnergyBalance::from_powers(
            runmat_types::physical::Power::new(4.0 * scale),
            runmat_types::physical::Power::new(3.0 * scale),
            runmat_types::physical::Power::new(8.0 * scale),
            runmat_types::physical::Power::new(2.0 * scale),
        );
        balance.validate().unwrap();
        assert_eq!(balance.relative_residual, 0.125, "power scale {scale:e}");
        for sign in [-1.0, 1.0] {
            let unbalanced = ThermalEnergyBalance::from_powers(
                runmat_types::physical::Power::new(0.0),
                runmat_types::physical::Power::new(0.0),
                runmat_types::physical::Power::new(sign * scale),
                runmat_types::physical::Power::new(0.0),
            );
            assert_eq!(unbalanced.relative_residual, 1.0, "unbalanced {scale:e} W");
        }
    }
    let zero = ThermalEnergyBalance::from_powers(
        runmat_types::physical::Power::new(0.0),
        runmat_types::physical::Power::new(0.0),
        runmat_types::physical::Power::new(0.0),
        runmat_types::physical::Power::new(0.0),
    );
    zero.validate().unwrap();
    assert_eq!(zero.relative_residual, 0.0);
}

#[test]
fn signed_power_balance_checks_arithmetic_and_relative_normalization() {
    let balanced = ThermalEnergyBalance::from_powers(
        runmat_types::physical::Power::new(4.0),
        runmat_types::physical::Power::new(3.0),
        runmat_types::physical::Power::new(9.0),
        runmat_types::physical::Power::new(2.0),
    );
    assert_eq!(balanced.residual_w.canonical_value(), 0.0);
    assert_eq!(balanced.relative_residual, 0.0);
    balanced.validate().unwrap();
    let unbalanced = ThermalEnergyBalance::from_powers(
        runmat_types::physical::Power::new(4.0),
        runmat_types::physical::Power::new(3.0),
        runmat_types::physical::Power::new(8.0),
        runmat_types::physical::Power::new(2.0),
    );
    assert_eq!(unbalanced.residual_w.canonical_value(), 1.0);
    assert_eq!(unbalanced.relative_residual, 0.125);
    unbalanced.validate().unwrap(); // arithmetic consistency is distinct from acceptance
    let small = ThermalEnergyBalance::from_powers(
        runmat_types::physical::Power::new(0.25),
        runmat_types::physical::Power::new(0.0),
        runmat_types::physical::Power::new(0.0),
        runmat_types::physical::Power::new(0.0),
    );
    assert_eq!(small.relative_residual, 1.0);
    small.validate().unwrap();
    let mut forged = balanced;
    forged.applied_power_w = forged
        .applied_power_w
        .clone()
        .try_map_canonical_storage(|value| value + (1.0))
        .unwrap();
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
            ThermalEnergyBalance::from_powers(
                runmat_types::physical::Power::new(value),
                runmat_types::physical::Power::new(0.0),
                runmat_types::physical::Power::new(0.0),
                runmat_types::physical::Power::new(0.0)
            )
            .validate(),
            Err(ThermalEnergyBalanceError::NonFinite)
        );
    }
    let mut overflow = ThermalEnergyBalance::from_powers(
        runmat_types::physical::Power::new(f64::MAX),
        runmat_types::physical::Power::new(f64::MAX),
        runmat_types::physical::Power::new(0.0),
        runmat_types::physical::Power::new(0.0),
    );
    overflow.residual_w = runmat_types::physical::Power::new(0.0);
    overflow.relative_residual = 0.0;
    assert_eq!(
        overflow.validate(),
        Err(ThermalEnergyBalanceError::NonFinite)
    );
    let balance = ThermalEnergyBalance::from_powers(
        runmat_types::physical::Power::new(1.25),
        runmat_types::physical::Power::new(-0.5),
        runmat_types::physical::Power::new(0.25),
        runmat_types::physical::Power::new(-0.5),
    );
    let encoded = serde_json::to_vec(&balance).unwrap();
    let decoded: ThermalEnergyBalance = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(balance, decoded);
    decoded.validate().unwrap();
}

#[test]
fn persisted_balance_rejects_a_dimensional_floor_disguised_as_relative_evidence() {
    let balance = ThermalEnergyBalance::from_powers(
        runmat_types::physical::Power::new(0.0),
        runmat_types::physical::Power::new(0.0),
        runmat_types::physical::Power::new(2.0_f64.powi(-40)),
        runmat_types::physical::Power::new(0.0),
    );
    let mut encoded = serde_json::to_value(balance).unwrap();
    encoded["relative_residual"] = serde_json::json!(balance.residual_w.canonical_value().abs());
    let forged: ThermalEnergyBalance = serde_json::from_value(encoded).unwrap();
    assert_eq!(
        forged.validate(),
        Err(ThermalEnergyBalanceError::Inconsistent)
    );
    // Validation must not silently normalize persisted numerical claims into new evidence.
    assert_eq!(
        forged.relative_residual,
        balance.residual_w.canonical_value().abs()
    );
    assert_eq!(balance.relative_residual, 1.0);
}

#[test]
fn opposing_physical_transfers_retain_a_scale_before_signed_cancellation() {
    // Unit-conductance three-node bar, fixed 300/400 K, center 350+epsilon K:
    // outward prescribed reactions are +(50+epsilon) and -(50-epsilon) W.
    let epsilon = 2.0_f64.powi(-20);
    for exponent in [-100, 0, 100] {
        let factor = 2.0_f64.powi(exponent);
        let balance = ThermalEnergyBalance::from_power_transfers(
            runmat_types::physical::Power::new(0.0),
            runmat_types::physical::Power::new(0.0),
            runmat_types::physical::Power::new(0.0),
            runmat_types::physical::Power::new(2.0 * epsilon * factor),
            ThermalPowerScale {
                storage_w: runmat_types::physical::Power::new(0.0),
                boundary_w: runmat_types::physical::Power::new(0.0),
                applied_w: runmat_types::physical::Power::new(0.0),
                prescribed_w: runmat_types::physical::Power::new(100.0 * factor),
            },
            ThermalPrescribedHeatFlow {
                entering_w: runmat_types::physical::Power::new((50.0 - epsilon) * factor),
                leaving_w: runmat_types::physical::Power::new((50.0 + epsilon) * factor),
            },
        );
        balance.validate().unwrap();
        assert_eq!(balance.relative_residual, 2.0 * epsilon / 100.0);
        // The gross flows through the two fixed ends survive the net cancellation.
        assert_eq!(
            balance.prescribed_heat_flow.entering_w.canonical_value(),
            (50.0 - epsilon) * factor
        );
        for flow in [-1.0, 101.0, f64::NAN] {
            let mut invalid = balance;
            invalid.prescribed_heat_flow.leaving_w =
                runmat_types::physical::Power::new(flow * factor);
            assert!(invalid.validate().is_err(), "{flow}");
        }
        let decoded: ThermalEnergyBalance =
            serde_json::from_slice(&serde_json::to_vec(&balance).unwrap()).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded, balance);
        let mut forged = balance;
        forged.scale.prescribed_w = forged
            .scale
            .prescribed_w
            .clone()
            .try_map_canonical_storage(|value| value * (2.0))
            .unwrap();
        assert_eq!(
            forged.validate(),
            Err(ThermalEnergyBalanceError::Inconsistent)
        );
        for scale in [-1.0, 0.0, f64::NAN, f64::INFINITY] {
            let mut invalid = balance;
            invalid.scale.prescribed_w = runmat_types::physical::Power::new(scale);
            assert!(invalid.validate().is_err());
        }
    }
}
