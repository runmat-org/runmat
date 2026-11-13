use runmat_analysis_core::{
    validate_material_definitions, MaterialElasticityModel, MaterialMechanicalModel, MaterialModel,
    MaterialOrthotropicLaminaModel, MaterialPlasticModel, MaterialProvenance,
};

fn material(yield_stress_pa: f64, isotropic_hardening_modulus_pa: f64) -> MaterialModel {
    MaterialModel {
        fluid: None,
        material_id: "steel".into(),
        name: "Steel".into(),
        provenance: MaterialProvenance::Authored,
        mass_density_kg_per_m3: (None).map(runmat_types::physical::MassDensity::new),
        elasticity: Some(MaterialElasticityModel::IsotropicLinear(
            MaterialMechanicalModel {
                youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(200e9),
                poisson_ratio: runmat_types::physical::PoissonRatio::new(0.3),
            },
        )),
        damping: None,
        thermal: None,
        acoustic: None,
        electrical: None,
        plastic: Some(MaterialPlasticModel::J2IsotropicLinearHardening {
            yield_stress_pa: runmat_types::physical::YieldStress::new(yield_stress_pa),
            isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(
                isotropic_hardening_modulus_pa,
            ),
        }),
    }
}

#[test]
fn j2_requires_finite_dimensional_parameters_and_isotropic_elasticity() {
    for hardening in [0.0, 1e9] {
        validate_material_definitions(&[material(250e6, hardening)]).unwrap();
    }
    for stress in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(validate_material_definitions(&[material(stress, 1e9)]).is_err());
    }
    for hardening in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(validate_material_definitions(&[material(250e6, hardening)]).is_err());
    }
    let mut missing_elasticity = material(250e6, 1e9);
    missing_elasticity.elasticity = None;
    assert!(validate_material_definitions(&[missing_elasticity]).is_err());
    let mut wrong_elasticity = material(250e6, 1e9);
    wrong_elasticity.elasticity = Some(MaterialElasticityModel::OrthotropicLamina(
        MaterialOrthotropicLaminaModel {
            longitudinal_modulus_pa: runmat_types::physical::YoungsModulus::new(200e9),
            transverse_modulus_pa: runmat_types::physical::YoungsModulus::new(200e9),
            major_poisson_ratio: runmat_types::physical::PoissonRatio::new(0.3),
            in_plane_shear_modulus_pa: runmat_types::physical::ShearModulus::new(80e9),
            transverse_shear_modulus_13_pa: runmat_types::physical::ShearModulus::new(80e9),
            transverse_shear_modulus_23_pa: runmat_types::physical::ShearModulus::new(80e9),
        },
    ));
    assert!(validate_material_definitions(&[wrong_elasticity]).is_err());
    for (youngs, poisson) in [(0.0, 0.3), (200e9, 0.5), (f64::NAN, 0.3)] {
        let mut invalid = material(250e6, 1e9);
        invalid.elasticity = Some(MaterialElasticityModel::IsotropicLinear(
            MaterialMechanicalModel {
                youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(youngs),
                poisson_ratio: runmat_types::physical::PoissonRatio::new(poisson),
            },
        ));
        assert!(validate_material_definitions(&[invalid]).is_err());
    }
}

#[test]
fn j2_wire_contract_is_closed_and_does_not_decode_proxy_fields() {
    let law = material(250e6, 1e9).plastic.unwrap();
    let encoded = serde_json::to_value(&law).unwrap();
    assert_eq!(encoded["law"], "j2_isotropic_linear_hardening");
    assert_eq!(
        serde_json::from_value::<MaterialPlasticModel>(encoded.clone()).unwrap(),
        law
    );
    for field in [
        "yield_strain",
        "hardening_modulus_ratio",
        "saturation_exponent",
        "extra",
    ] {
        let mut invalid = encoded.clone();
        invalid[field] = serde_json::json!(1.0);
        assert!(serde_json::from_value::<MaterialPlasticModel>(invalid).is_err());
    }
    for invalid in [
        serde_json::json!({"yield_strain": 0.001, "hardening_modulus_ratio": 0.1, "saturation_exponent": 1.0}),
        serde_json::json!({"law": "j2_kinematic_hardening", "yield_stress_pa": 250e6, "isotropic_hardening_modulus_pa": 1e9}),
        serde_json::json!({"law": "j2_isotropic_linear_hardening", "yield_stress_pa": 250e6}),
    ] {
        assert!(serde_json::from_value::<MaterialPlasticModel>(invalid).is_err());
    }
}
