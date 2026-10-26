use super::*;
use runmat_analysis_core::{MaterialElasticityModel, MaterialMechanicalModel, MaterialProvenance};

const YOUNGS: f64 = 210.0e9;
const POISSON: f64 = 0.3;
const YIELD: f64 = 250.0e6;
const HARDENING: f64 = 2.0e9;

fn material(plastic: bool, hardening: f64) -> MaterialModel {
    MaterialModel {
        fluid: None,
        material_id: "j2-test".into(),
        name: "independent constitutive fixture".into(),
        provenance: MaterialProvenance::Authored,
        mass_density_kg_per_m3: (None).map(runmat_types::physical::MassDensity::new),
        elasticity: Some(MaterialElasticityModel::IsotropicLinear(
            MaterialMechanicalModel {
                youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(YOUNGS),
                poisson_ratio: runmat_types::physical::PoissonRatio::new(POISSON),
            },
        )),
        damping: None,
        thermal: None,
        acoustic: None,
        electrical: None,
        plastic: plastic.then_some(MaterialPlasticModel::J2IsotropicLinearHardening {
            yield_stress_pa: runmat_types::physical::YieldStress::new(YIELD),
            isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(
                hardening,
            ),
        }),
    }
}

fn close(actual: f64, expected: f64, scale: f64) {
    assert!(
        (actual - expected).abs() <= 2.0e-12 * scale,
        "actual={actual:e}, expected={expected:e}, scale={scale:e}"
    );
}

fn equivalent(stress: [f64; 6]) -> f64 {
    // Independent principal-difference expression, not the return-map trial invariant.
    (0.5 * ((stress[0] - stress[1]).powi(2)
        + (stress[1] - stress[2]).powi(2)
        + (stress[2] - stress[0]).powi(2))
        + 3.0 * (stress[3].powi(2) + stress[4].powi(2) + stress[5].powi(2)))
    .sqrt()
}

#[test]
fn independent_elastic_hydrostatic_and_engineering_shear_responses() {
    let model = IsotropicConstitutiveModel::from_material(&material(false, 0.0)).unwrap();
    let state = PlasticityPointState::default();
    let hydrostatic = model
        .update([0.001, 0.001, 0.001, 0.0, 0.0, 0.0], &state)
        .unwrap();
    let pressure = YOUNGS * 0.001 / (1.0 - 2.0 * POISSON);
    for stress in hydrostatic.stress_pa.iter().take(3) {
        close(*stress, pressure, pressure);
    }
    for stress in hydrostatic.stress_pa.iter().skip(3) {
        assert_eq!(*stress, 0.0);
    }
    assert_eq!(hydrostatic.state, state);
    close(
        hydrostatic.elastic_energy_density_pa,
        1.5 * pressure * 0.001,
        pressure * 0.001,
    );
    let shear = YOUNGS / (2.0 * (1.0 + POISSON));
    for index in 3..6 {
        let mut strain = [0.0; 6];
        strain[index] = 0.002;
        let response = model.update(strain, &state).unwrap();
        for (i, stress) in response.stress_pa.iter().enumerate() {
            close(
                *stress,
                if i == index { shear * 0.002 } else { 0.0 },
                shear * 0.002,
            );
        }
        close(response.tangent_pa[index][index], shear, shear);
        close(
            response.elastic_energy_density_pa,
            0.5 * shear * 0.002_f64.powi(2),
            shear * 0.002_f64.powi(2),
        );
    }
}

#[test]
fn hydrostatic_pressure_does_not_activate_j2_or_change_plastic_history() {
    let model = IsotropicConstitutiveModel::from_material(&material(true, HARDENING)).unwrap();
    let response = model
        .update(
            [0.01, 0.01, 0.01, 0.0, 0.0, 0.0],
            &PlasticityPointState::default(),
        )
        .unwrap();
    assert_eq!(response.state, PlasticityPointState::default());
    assert_eq!(response.plastic_dissipation_increment_pa, 0.0);
    close(equivalent(response.stress_pa), 0.0, YIELD);
    assert_eq!(response.tangent_pa, model.elastic_tangent_pa());
}

#[test]
fn pure_shear_yield_flow_and_tangent_match_scalar_return_solution() {
    let shear = YOUNGS / (2.0 * (1.0 + POISSON));
    for hardening in [0.0, HARDENING] {
        let model = IsotropicConstitutiveModel::from_material(&material(true, hardening)).unwrap();
        let gamma = 0.008;
        let alpha = ((3.0_f64).sqrt() * shear * gamma - YIELD) / (3.0 * shear + hardening);
        for index in 3..6 {
            let mut strain = [0.0; 6];
            strain[index] = gamma;
            let response = model
                .update(strain, &PlasticityPointState::default())
                .unwrap();
            close(response.state.equivalent_plastic_strain, alpha, alpha);
            close(
                response.state.engineering_plastic_strain[index],
                (3.0_f64).sqrt() * alpha,
                gamma,
            );
            close(
                response.stress_pa[index],
                (YIELD + hardening * alpha) / (3.0_f64).sqrt(),
                YIELD,
            );
            close(
                response.tangent_pa[index][index],
                shear * hardening / (3.0 * shear + hardening),
                shear,
            );
            close(
                equivalent(response.stress_pa),
                YIELD + hardening * alpha,
                YIELD,
            );
            for ep in response.state.engineering_plastic_strain.iter().take(3) {
                assert_eq!(*ep, 0.0);
            }
            close(
                response.plastic_dissipation_increment_pa,
                YIELD * alpha,
                YIELD * alpha,
            );
            close(
                response.hardening_energy_density_pa,
                0.5 * hardening * alpha.powi(2),
                (YIELD * alpha).max(1.0),
            );
        }
    }
}

#[test]
fn hydrostatic_bias_does_not_change_shear_plastic_flow() {
    let model = IsotropicConstitutiveModel::from_material(&material(true, HARDENING)).unwrap();
    let state = PlasticityPointState::default();
    let shear = model
        .update([0.0, 0.0, 0.0, 0.008, 0.0, 0.0], &state)
        .unwrap();
    let biased = model
        .update([0.1, 0.1, 0.1, 0.008, 0.0, 0.0], &state)
        .unwrap();
    close(
        biased.state.equivalent_plastic_strain,
        shear.state.equivalent_plastic_strain,
        0.008,
    );
    for i in 0..6 {
        close(
            biased.state.engineering_plastic_strain[i],
            shear.state.engineering_plastic_strain[i],
            0.008,
        );
        for j in 0..6 {
            close(biased.tangent_pa[i][j], shear.tangent_pa[i][j], YOUNGS);
        }
    }
    close(biased.stress_pa[3], shear.stress_pa[3], YIELD);
    close(
        biased.state.engineering_plastic_strain.iter().take(3).sum(),
        0.0,
        0.008,
    );
}

fn uniaxial_strain(axial: f64, stress: f64, plastic_axial: f64) -> [f64; 6] {
    let lateral = -POISSON * stress / YOUNGS - 0.5 * plastic_axial;
    [axial, lateral, lateral, 0.0, 0.0, 0.0]
}

#[test]
fn monotonic_uniaxial_stress_has_independent_bilinear_response_and_energy() {
    let model = IsotropicConstitutiveModel::from_material(&material(true, HARDENING)).unwrap();
    let mut committed = PlasticityPointState::default();
    let mut accumulated_dissipation = 0.0;
    for axial in [0.0005, 0.002, 0.005, 0.01] {
        let alpha = ((YOUNGS * axial - YIELD) / (YOUNGS + HARDENING)).max(0.0);
        let stress = YOUNGS * (axial - alpha);
        let old = committed.clone();
        let response = model
            .update(uniaxial_strain(axial, stress, alpha), &committed)
            .unwrap();
        assert_eq!(committed, old, "a trial evaluation cannot commit itself");
        close(response.stress_pa[0], stress, stress);
        for value in response.stress_pa.iter().skip(1) {
            close(*value, 0.0, stress);
        }
        close(response.state.equivalent_plastic_strain, alpha, axial);
        close(response.state.engineering_plastic_strain[0], alpha, axial);
        close(
            response.state.engineering_plastic_strain[1],
            -alpha / 2.0,
            axial,
        );
        close(
            response.state.engineering_plastic_strain[2],
            -alpha / 2.0,
            axial,
        );
        close(
            response.elastic_energy_density_pa,
            stress.powi(2) / (2.0 * YOUNGS),
            stress * axial,
        );
        assert!(response.plastic_dissipation_increment_pa >= 0.0);
        accumulated_dissipation += response.plastic_dissipation_increment_pa;
        close(
            accumulated_dissipation,
            YIELD * alpha,
            (stress * axial).max(1.0),
        );
        close(
            response.hardening_energy_density_pa,
            0.5 * HARDENING * alpha.powi(2),
            (stress * axial).max(1.0),
        );
        committed = response.state;
    }
}

#[test]
fn unloading_reloading_and_reverse_yield_preserve_committed_history_and_expand_surface() {
    let model = IsotropicConstitutiveModel::from_material(&material(true, HARDENING)).unwrap();
    let axial = 0.006;
    let alpha = (YOUNGS * axial - YIELD) / (YOUNGS + HARDENING);
    let stress = YIELD + HARDENING * alpha;
    let committed = model
        .update(
            uniaxial_strain(axial, stress, alpha),
            &PlasticityPointState::default(),
        )
        .unwrap()
        .state;
    let original = committed.clone();
    for elastic_stress in [100.0e6, -100.0e6, 200.0e6] {
        let strain = std::array::from_fn(|i| {
            committed.engineering_plastic_strain[i]
                + match i {
                    0 => elastic_stress / YOUNGS,
                    1 | 2 => -POISSON * elastic_stress / YOUNGS,
                    _ => 0.0,
                }
        });
        let response = model.update(strain, &committed).unwrap();
        assert_eq!(response.state, committed);
        close(response.stress_pa[0], elastic_stress, YIELD);
        assert_eq!(response.plastic_dissipation_increment_pa, 0.0);
        assert_eq!(response.tangent_pa, model.elastic_tangent_pa());
    }
    let reverse_alpha_increment = 0.002;
    let reverse_stress = -(YIELD + HARDENING * (alpha + reverse_alpha_increment));
    let plastic_axial = alpha - reverse_alpha_increment;
    let reverse_axial = plastic_axial + reverse_stress / YOUNGS;
    let response = model
        .update(
            uniaxial_strain(reverse_axial, reverse_stress, plastic_axial),
            &committed,
        )
        .unwrap();
    close(response.stress_pa[0], reverse_stress, YIELD);
    close(
        response.state.equivalent_plastic_strain,
        alpha + reverse_alpha_increment,
        alpha,
    );
    close(
        response.state.engineering_plastic_strain[0],
        plastic_axial,
        alpha,
    );
    close(
        response.plastic_dissipation_increment_pa,
        YIELD * reverse_alpha_increment,
        YIELD * alpha,
    );
    assert_eq!(committed, original);
}

#[test]
fn mixed_stress_tangent_is_symmetric_and_all_six_columns_match_finite_differences() {
    let model = IsotropicConstitutiveModel::from_material(&material(true, HARDENING)).unwrap();
    let committed = model
        .update(
            [0.003, -0.001, -0.002, 0.001, -0.002, 0.002],
            &PlasticityPointState::default(),
        )
        .unwrap()
        .state;
    let total = [0.007, -0.0015, -0.003, 0.003, -0.001, 0.004];
    let original = committed.clone();
    let response = model.update(total, &committed).unwrap();
    assert!(response.state.equivalent_plastic_strain > committed.equivalent_plastic_strain);
    close(
        equivalent(response.stress_pa),
        YIELD + HARDENING * response.state.equivalent_plastic_strain,
        YIELD,
    );
    close(
        response
            .state
            .engineering_plastic_strain
            .iter()
            .take(3)
            .sum::<f64>(),
        0.0,
        0.01,
    );
    for column in 0..6 {
        for h in [2.0e-8, 1.0e-8] {
            let mut plus = total;
            let mut minus = total;
            plus[column] += h;
            minus[column] -= h;
            let plus = model.update(plus, &committed).unwrap();
            let minus = model.update(minus, &committed).unwrap();
            for row in 0..6 {
                let derivative = (plus.stress_pa[row] - minus.stress_pa[row]) / (2.0 * h);
                // Perturbation/strain ~1e-6, central truncation ~1e-12; allow 5e-8 of E
                // for floating-point subtraction. No finite-difference tangent is used by update.
                assert!(
                    (derivative - response.tangent_pa[row][column]).abs() / YOUNGS < 5.0e-8,
                    "row={row}, column={column}, h={h:e}, fd={derivative:e}, tangent={:e}",
                    response.tangent_pa[row][column]
                );
                close(
                    response.tangent_pa[row][column],
                    response.tangent_pa[column][row],
                    YOUNGS,
                );
            }
        }
    }
    assert_eq!(committed, original);
}

#[test]
fn malformed_laws_strains_and_history_are_rejected() {
    let mut value = material(true, HARDENING);
    value.plastic = Some(MaterialPlasticModel::J2IsotropicLinearHardening {
        yield_stress_pa: runmat_types::physical::YieldStress::new(-1.0),
        isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(HARDENING),
    });
    assert!(matches!(
        IsotropicConstitutiveModel::from_material(&value),
        Err(PlasticityError::InvalidPlasticLaw)
    ));
    value.plastic = Some(MaterialPlasticModel::J2IsotropicLinearHardening {
        yield_stress_pa: runmat_types::physical::YieldStress::new(YIELD),
        isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(-1.0),
    });
    assert!(matches!(
        IsotropicConstitutiveModel::from_material(&value),
        Err(PlasticityError::InvalidPlasticLaw)
    ));
    for poisson in [-1.0, 0.5, f64::NAN] {
        let MaterialElasticityModel::IsotropicLinear(elasticity) =
            value.elasticity.as_mut().unwrap()
        else {
            panic!("isotropic fixture");
        };
        elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(poisson);
        assert!(matches!(
            IsotropicConstitutiveModel::from_material(&value),
            Err(PlasticityError::InvalidElasticity)
        ));
    }
    let model = IsotropicConstitutiveModel::from_material(&material(true, HARDENING)).unwrap();
    assert!(matches!(
        model.update([f64::NAN; 6], &PlasticityPointState::default()),
        Err(PlasticityError::NonFiniteStrain)
    ));
    for state in [
        PlasticityPointState {
            engineering_plastic_strain: [0.0; 6],
            equivalent_plastic_strain: -1.0,
        },
        PlasticityPointState {
            engineering_plastic_strain: [0.001; 6],
            equivalent_plastic_strain: 0.1,
        },
        PlasticityPointState {
            engineering_plastic_strain: [0.002, -0.001, -0.001, 0.0, 0.0, 0.0],
            equivalent_plastic_strain: 0.001,
        },
        PlasticityPointState {
            engineering_plastic_strain: [f64::INFINITY; 6],
            equivalent_plastic_strain: 0.0,
        },
    ] {
        assert!(model.update([0.0; 6], &state).is_err());
    }
    let elastic = IsotropicConstitutiveModel::from_material(&material(false, 0.0)).unwrap();
    let history = PlasticityPointState {
        engineering_plastic_strain: [0.001, -0.0005, -0.0005, 0.0, 0.0, 0.0],
        equivalent_plastic_strain: 0.001,
    };
    assert!(matches!(
        elastic.update([0.0; 6], &history),
        Err(PlasticityError::PlasticHistoryWithoutLaw)
    ));
}

#[test]
fn accumulated_dissipation_is_law_bound_and_not_hardening_energy() {
    let model = IsotropicConstitutiveModel::from_material(&material(true, HARDENING)).unwrap();
    let state = PlasticityPointState {
        engineering_plastic_strain: [0.001, -0.0005, -0.0005, 0.0, 0.0, 0.0],
        equivalent_plastic_strain: 0.002,
    };
    close(
        model
            .committed_plastic_dissipation_density_pa(&state)
            .unwrap(),
        YIELD * 0.002,
        YIELD * 0.002,
    );
    let elastic = IsotropicConstitutiveModel::from_material(&material(false, 0.0)).unwrap();
    assert_eq!(
        elastic
            .committed_plastic_dissipation_density_pa(&PlasticityPointState::default())
            .unwrap(),
        0.0
    );
    assert!(matches!(
        elastic.committed_plastic_dissipation_density_pa(&state),
        Err(PlasticityError::PlasticHistoryWithoutLaw)
    ));
    assert!(matches!(
        elastic.replay_increment_roundoff_bound(&state, [0.0; 6]),
        Err(PlasticityError::PlasticHistoryWithoutLaw)
    ));
}

#[test]
fn near_first_yield_replay_is_bounded_without_an_arbitrary_plastic_floor() {
    let model = IsotropicConstitutiveModel::from_material(&material(true, HARDENING)).unwrap();
    let shear = YOUNGS / (2.0 * (1.0 + POISSON));
    let yield_gamma = YIELD / ((3.0_f64).sqrt() * shear);
    for relative_excess in [4.0 * f64::EPSILON, 16.0 * f64::EPSILON, 1.0e-13] {
        let strain = [
            0.0,
            0.0,
            0.0,
            yield_gamma * (1.0 + relative_excess),
            0.0,
            0.0,
        ];
        let accepted = model
            .update(strain, &PlasticityPointState::default())
            .unwrap()
            .state;
        assert!(accepted.equivalent_plastic_strain > 0.0);
        assert!(accepted.equivalent_plastic_strain < 1.0e-12);
        let original = accepted.clone();
        let replayed = model.update(strain, &accepted).unwrap();
        let bound = model
            .replay_increment_roundoff_bound(&accepted, strain)
            .unwrap();
        let alpha_increment =
            (replayed.state.equivalent_plastic_strain - accepted.equivalent_plastic_strain).abs();
        let plastic_increment_norm = replayed
            .state
            .engineering_plastic_strain
            .iter()
            .zip(accepted.engineering_plastic_strain)
            .enumerate()
            .fold(0.0_f64, |norm, (i, (new, old))| {
                norm.hypot(
                    (new - old)
                        * if i < 3 {
                            (2.0_f64 / 3.0).sqrt()
                        } else {
                            (1.0_f64 / 3.0).sqrt()
                        },
                )
            });
        assert!(alpha_increment <= bound && plastic_increment_norm <= bound);
        assert!(
            bound > 256.0 * f64::EPSILON * accepted.equivalent_plastic_strain,
            "yield-stress roundoff is not proportional to a vanishing plastic history"
        );
        let mut physically_changed = strain;
        physically_changed[3] += 1.0e-6;
        let additional = model.update(physically_changed, &accepted).unwrap();
        let changed_bound = model
            .replay_increment_roundoff_bound(&accepted, physically_changed)
            .unwrap();
        assert!(
            additional.state.equivalent_plastic_strain - accepted.equivalent_plastic_strain
                > changed_bound,
            "roundoff evidence cannot excuse a real missing plastic increment"
        );
        assert_eq!(accepted, original);
    }
}

#[test]
fn replay_bound_is_invariant_to_consistent_stress_unit_scaling_and_rejects_nonfinite_inputs() {
    let model = IsotropicConstitutiveModel::from_material(&material(true, HARDENING)).unwrap();
    let total = [0.005, -0.002, -0.003, 0.003, -0.002, 0.001];
    let state = model
        .update(total, &PlasticityPointState::default())
        .unwrap()
        .state;
    let bound = model
        .replay_increment_roundoff_bound(&state, total)
        .unwrap();
    let mut scaled = material(true, HARDENING);
    let MaterialElasticityModel::IsotropicLinear(elasticity) = scaled.elasticity.as_mut().unwrap()
    else {
        panic!("isotropic fixture");
    };
    elasticity.youngs_modulus_pa = runmat_types::physical::Quantity::new(
        elasticity.youngs_modulus_pa.canonical_value() * (1000.0),
    );
    scaled.plastic = Some(MaterialPlasticModel::J2IsotropicLinearHardening {
        yield_stress_pa: runmat_types::physical::YieldStress::new(YIELD * 1000.0),
        isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(
            HARDENING * 1000.0,
        ),
    });
    let scaled = IsotropicConstitutiveModel::from_material(&scaled).unwrap();
    close(
        scaled
            .replay_increment_roundoff_bound(&state, total)
            .unwrap(),
        bound,
        bound,
    );
    assert!(model
        .replay_increment_roundoff_bound(&state, [f64::NAN; 6])
        .is_err());
    let malformed = PlasticityPointState {
        engineering_plastic_strain: [0.0; 6],
        equivalent_plastic_strain: -1.0,
    };
    assert!(model
        .replay_increment_roundoff_bound(&malformed, total)
        .is_err());
    assert!(model
        .committed_plastic_dissipation_density_pa(&malformed)
        .is_err());
}
