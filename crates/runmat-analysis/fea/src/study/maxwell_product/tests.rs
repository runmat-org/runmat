use super::*;
use crate::study::maxwell_planning::tests::{compiled, mesh, model, policy};
use runmat_analysis_core::*;
use runmat_meshing_core::ElementOrder;

#[test]
fn both_affine_orders_preserve_exact_compiled_model_properties_and_complex_source() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order, 2);
        let study = compiled(model(), &mesh);
        let before = study.clone();
        let product = plan_maxwell_study_product(study, &mesh, policy()).unwrap();
        assert_eq!(product.compiled, before);
        assert_eq!(product.workload.model, before.model);
        let formulation = product.workload.formulation(&mesh).unwrap();
        assert_eq!(
            formulation.realization(),
            &crate::admission::canonical_maxwell_realization(order)
        );
        let physical = product.workload.compile_input(&mesh).unwrap();
        assert_eq!(
            physical.angular_frequency_rad_per_s.canonical_value(),
            std::f64::consts::TAU * 1e8
        );
        assert!(physical.source_current.iter().all(|sample| sample
            .current_density_a_per_m2
            .canonical_value()[1]
            .real
            == 0.6
            && sample.current_density_a_per_m2.canonical_value()[1].imaginary == 0.8));
        product.compiled.validate_against(&mesh).unwrap();
    }
}

#[test]
fn foreign_mesh_and_stale_property_ownership_fail_before_product() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let original = mesh(order, 2);
        let study = compiled(model(), &original);
        assert!(matches!(
            plan_maxwell_study_product(study.clone(), &mesh(order, 3), policy()),
            Err(MaxwellStudyProductError::Compilation(_))
        ));
        let mut changed = study;
        changed.model.materials[0]
            .electrical
            .as_mut()
            .unwrap()
            .relative_permittivity = runmat_types::physical::Quantity::new(
            changed.model.materials[0]
                .electrical
                .as_mut()
                .unwrap()
                .relative_permittivity
                .canonical_value()
                * (2.0),
        );
        assert!(matches!(
            plan_maxwell_study_product(changed, &original, policy()),
            Err(MaxwellStudyProductError::Compilation(_))
        ));
    }
}

#[test]
fn explicit_pec_positive_loss_and_original_current_are_not_inferred() {
    let mesh = mesh(ElementOrder::Tet4, 2);
    let original = model();
    let mut missing_pec = original.clone();
    missing_pec.constraints.remove(0);
    let mut magnetic_ground = original.clone();
    magnetic_ground.constraints =
        crate::physics::magnetic::test_support::authored_model(1.0).constraints;
    let mut lossless = original.clone();
    lossless.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .conductivity_s_per_m = runmat_types::physical::ElectricalConductivity::new(0.0);
    let mut foreign_source = original.clone();
    let ExcitationAction::MagneticVolumeCurrentDensity(source) = &mut foreign_source.excitations[0]
    else {
        panic!()
    };
    source.signal_id = SignalId("missing-current".into());
    // Zero conductivity (air) needs no artificial loss.
    assert!(plan_maxwell_study_product(compiled(lossless, &mesh), &mesh, policy()).is_ok());
    // Faces without a ground are symmetry planes of the current (n x H = 0).
    assert!(plan_maxwell_study_product(compiled(missing_pec, &mesh), &mesh, policy()).is_ok());
    for changed in [magnetic_ground, foreign_source] {
        assert!(plan_maxwell_study_product(compiled(changed, &mesh), &mesh, policy()).is_err());
    }
}
