//! Independent one-dimensional conduction/radiation reference; no solver-derived temperatures.
use super::*;
use crate::topology::test_meshes::slab_mesh;

fn slab() -> (AnalysisModel, runmat_meshing_core::SolverMeshArtifact) {
    let mesh = slab_mesh(1);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::all_solids();
    model.materials[0]
        .thermal
        .as_mut()
        .unwrap()
        .conductivity_w_per_mk = runmat_types::physical::ThermalConductivity::new(1.0);
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "gray_surface".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::ThermalRadiation {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(300.0)),
            emissivity: runmat_analysis_core::SurfaceEmissivity::new(0.8),
        },
    });
    (model, mesh)
}

// witness: FEA-THERMAL-INPUT-001/boundaries @ thermal_steady/tet4/numerical_kernel;
//   boundary: Tet4 slab exact gray-surface law, finite-difference tangent and physical power; does
//   not cover arbitrary enclosure radiation.
// witness: FEA-THERMAL-PUBLIC-001/radiation_to_ambient @ thermal_steady/tet4/numerical_kernel;
//   coverage: partial;
//   boundary: A Tet4 slab with a gray surface matches the Stefan-Boltzmann law with emissivity 0.8
//   at 350 K surface against a 300 K ambient; convection on the same face and transient radiation
//   are not covered.
#[test]
fn exact_steady_slab_radiation_residual_tangent_and_prescribed_power_match_continuum() {
    let (mut model, mesh) = slab();
    let surface: f64 = 350.0;
    // Independent SI Stefan–Boltzmann law and unit-area/unit-length k=1 resistance.
    let sigma = 5.670_374_419e-8;
    let flux = 0.8 * sigma * (surface.powi(4) - 300.0_f64.powi(4));
    model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "hot_face".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Constant(Temperature::new(surface + flux)),
            },
        }));
    let assembly = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[])
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let reference = vec![surface; assembly.projection.nodes.len()];
    let departure = assembly
        .projection
        .nodes
        .iter()
        .map(|n| flux * (1.0 - n.coordinates_m.canonical_values()[0]))
        .collect::<Vec<_>>();
    let physical = assembly
        .nonlinear_residual(
            &model,
            ExcitationSamplePoint::Static,
            &reference,
            &departure,
            None,
            0.0,
        )
        .unwrap();
    for (node, &power) in physical.residual_w.canonical_values().iter().enumerate() {
        if assembly.prescribed_temperature_k.canonical_values()[node].is_none() {
            assert!(power.abs() < 1e-10, "node={node}, power={power}");
        }
    }
    assert!((physical.physical_power_scale_w.canonical_value() - flux).abs() < 1e-10);
    let operator = assembly
        .nonlinear_increment_operator(
            &model,
            ExcitationSamplePoint::Static,
            &reference,
            &departure,
            None,
            0.0,
        )
        .unwrap();
    assert!(operator
        .with_canonical_system(|numerical| numerical.nullspace_evidence().cloned())
        .is_some());
    for (node, &power) in operator.load().iter().enumerate() {
        if let Some(target) = assembly.prescribed_temperature_k.canonical_values()[node] {
            // Preserve the original authored operands: (surface+flux)-surface
            // need not round to the independently constructed flux exactly.
            assert_eq!(power, (target - reference[node]) - departure[node]);
        }
    }
    let tangent_action = apply_matrix(
        operator.operators().stiffness(),
        &vec![1.0; reference.len()],
    );
    assert!(
        (tangent_action.iter().sum::<f64>() - 4.0 * 0.8 * sigma * surface.powi(3)).abs() < 1e-10
    );
    // Check the actual full residual Jacobian, including conduction and radiation.
    let direction = assembly
        .projection
        .nodes
        .iter()
        .map(|n| 0.2 + n.coordinates_m.canonical_values()[0] * 0.1)
        .collect::<Vec<_>>();
    let h = 1e-3;
    let plus = departure
        .iter()
        .zip(&direction)
        .map(|(t, d)| t + h * d)
        .collect::<Vec<_>>();
    let minus = departure
        .iter()
        .zip(&direction)
        .map(|(t, d)| t - h * d)
        .collect::<Vec<_>>();
    let rp = assembly
        .nonlinear_residual(
            &model,
            ExcitationSamplePoint::Static,
            &reference,
            &plus,
            None,
            0.0,
        )
        .unwrap();
    let rm = assembly
        .nonlinear_residual(
            &model,
            ExcitationSamplePoint::Static,
            &reference,
            &minus,
            None,
            0.0,
        )
        .unwrap();
    let action = apply_matrix(operator.operators().stiffness(), &direction);
    for ((p, m), j) in rp
        .residual_w
        .canonical_values()
        .iter()
        .zip(rm.residual_w.canonical_values())
        .zip(action)
    {
        assert!(((p - m) / (2.0 * h) - j).abs() < 1e-7);
    }
    // Backward Euler adds exactly the original consistent capacity divided by dt.
    let be = assembly
        .nonlinear_increment_operator(
            &model,
            ExcitationSamplePoint::Static,
            &reference,
            &departure,
            Some(&departure),
            0.25,
        )
        .unwrap();
    let be_action = apply_matrix(be.operators().stiffness(), &direction);
    let steady_action = apply_matrix(operator.operators().stiffness(), &direction);
    let capacity_action = apply_matrix(&assembly.capacity, &direction);
    for ((actual, steady), capacity) in be_action.iter().zip(steady_action).zip(capacity_action) {
        assert!((actual - steady - capacity / 0.25).abs() < 1e-8);
    }
    let energy = assembly
        .energy_balance(
            &model,
            ExcitationSamplePoint::Static,
            &departure,
            &departure,
            1.0,
            surface,
        )
        .unwrap();
    energy.validate().unwrap();
    assert!((energy.boundary_loss_w.canonical_value() - flux).abs() < 1e-10);
    assert!((energy.prescribed_boundary_power_w.canonical_value() + flux).abs() < 1e-10);
    assert!(energy.relative_residual < 1e-12);
}

#[test]
fn radiation_anchor_is_actual_positive_tangent_not_nonzero_emissivity_alone() {
    let (model, mesh) = slab();
    let assembly = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[])
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let zeros = vec![0.0; assembly.projection.nodes.len()];
    let zero = assembly
        .nonlinear_increment_operator(
            &model,
            ExcitationSamplePoint::Static,
            &zeros,
            &zeros,
            None,
            0.0,
        )
        .unwrap();
    assert!(zero
        .with_canonical_system(|numerical| numerical.nullspace_evidence().cloned())
        .is_none());
    let warm = vec![300.0; zeros.len()];
    let positive = assembly
        .nonlinear_increment_operator(
            &model,
            ExcitationSamplePoint::Static,
            &warm,
            &zeros,
            None,
            0.0,
        )
        .unwrap();
    assert!(positive
        .with_canonical_system(|numerical| numerical.nullspace_evidence().cloned())
        .is_some());
    let negative = vec![-1.0; zeros.len()];
    assert!(assembly
        .nonlinear_residual(
            &model,
            ExcitationSamplePoint::Static,
            &zeros,
            &negative,
            None,
            0.0
        )
        .is_err());
}
