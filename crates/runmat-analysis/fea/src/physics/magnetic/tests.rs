use super::test_support::authored_model;
use super::*;
use crate::topology::{
    test_meshes::{quadratic_slab_mesh, slab_mesh},
    AnalysisProjection,
};
use runmat_analysis_core::*;
use runmat_meshing_core::ElementOrder;

fn problem(
    order: ElementOrder,
    cells: usize,
    mu: f64,
    current: f64,
) -> (AnalysisProjection, Vec<MagneticVolumeData>) {
    let mesh = match order {
        ElementOrder::Tet4 => slab_mesh(cells),
        ElementOrder::Tet10 => quadratic_slab_mesh(cells),
    };
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let data = projection
        .elements
        .iter()
        .map(|element| MagneticVolumeData {
            element_identity: element.stable_identity,
            permeability_h_per_m: runmat_types::physical::MagneticPermeability::new(mu),
            current_density_a_per_m2: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::ElectricCurrentDensity,
                _,
            >::new([0.0, 0.0, current]),
        })
        .collect();
    (projection, data)
}

// Independent separated solution of -Δa=μJ on a unit square, a=0
// on its perimeter, A=(0,0,a), extruded along z. These are continuous
// Fourier/hyperbolic series, not values from RunMat's operator or another solve.
// Stable ratios avoid cosh overflow. Uniform omitted-tail bounds:
// |a_tail| <= 2μ|J|/(π³ N²), |grad_tail| <= 4μ|J|/(π² N).
fn reference(x: f64, y: f64, mu_j: f64, maximum: usize) -> ([f64; 3], [f64; 3]) {
    let pi = std::f64::consts::PI;
    let mut a = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for n in (1..=maximum).step_by(2) {
        let k = n as f64 * pi;
        let left = (-k * x).exp();
        let right = (-k * (1.0 - x)).exp();
        let denominator = 1.0 + (-k).exp();
        let c = (left + right) / denominator;
        let s = (right - left) / denominator;
        a += 4.0 / k.powi(3) * (1.0 - c) * (k * y).sin();
        dx -= 4.0 / k.powi(2) * s * (k * y).sin();
        dy += 4.0 / k.powi(2) * (1.0 - c) * (k * y).cos();
    }
    ([0.0, 0.0, mu_j * a], [mu_j * dy, -mu_j * dx, 0.0])
}

fn reference_energy(mu_j2: f64, maximum: usize) -> f64 {
    let pi = std::f64::consts::PI;
    let mut sum = 0.0;
    for m in (1..=maximum).step_by(2) {
        for n in (1..=maximum).step_by(2) {
            let mm = (m as f64).powi(2);
            let nn = (n as f64).powi(2);
            sum += 1.0 / (mm * nn * (mm + nn));
        }
    }
    32.0 * mu_j2 / pi.powi(6) * sum
}

// witness: FEA-EM-MAGNETOSTATIC-001/formulation @ magnetostatic/tet4|tet10/numerical_kernel;
//   boundary: Affine Tet4 and Tet10 extruded unit square, uniform axial current and grounded
//   boundary; independent Fourier-series potential, curl field and energy with tail bounds and
//   three meshes per order. Lowest-order edge basis on both geometry orders, so Tet10 matches
//   Tet4 to roundoff; no higher-order magnetic convergence or force evidence.
// witness: FEA-EM-QUALIFICATION-001/references @ magnetostatic/tet4|tet10/numerical_kernel;
//   coverage: partial;
//   check: square-conductor-magnetostatic; boundary: Magnetostatic reference: uniform-current square conductor against the Fourier
//   series with mesh convergence of potential, flux density and energy on Tet4 and Tet10
//   geometry; no magnetic material interface or gap reference.
#[test]
fn square_uniform_current_has_independent_potential_field_and_energy_convergence() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        square_uniform_current_converges(order);
    }
}

fn square_uniform_current_converges(order: ElementOrder) {
    let mut errors = Vec::new();
    let exact_energy = reference_energy(1.0, 255);
    // Energy-series tail <= 8/(3π⁴N³); verify a separate truncation agrees.
    assert!(
        (exact_energy - reference_energy(1.0, 511)).abs()
            < 8.0 / (3.0 * std::f64::consts::PI.powi(4) * 255.0f64.powi(3))
    );
    for cells in [1, 2, 4] {
        let (projection, data) = problem(order, cells, 1.0, 1.0);
        let result = solve_grounded_volume(
            &projection,
            &super::exterior_face_identities(&projection),
            &[],
            &data,
            1e-10,
        )
        .unwrap();
        let mut potential_error = 0.0;
        let mut field_error = 0.0;
        for sample in &result.samples(&projection).unwrap() {
            let [x, y, _] = sample.position_m;
            let (a, b) = reference(x, y, 1.0, 1023);
            for axis in 0..3 {
                potential_error += sample.volume_weight_m3
                    * (sample.vector_potential_wb_per_m.canonical_value()[axis] - a[axis]).powi(2);
                field_error += sample.volume_weight_m3
                    * (sample.flux_density_t.canonical_value()[axis] - b[axis]).powi(2);
            }
        }
        let error = (
            potential_error.sqrt(),
            field_error.sqrt(),
            (result.magnetic_energy_j.canonical_value() - exact_energy).abs() / exact_energy,
        );
        eprintln!(
            "magnetic {order:?} cells={cells} errors={error:?} iterations={}",
            result.iterations
        );
        assert!(result.relative_physical_residual < 8e-10);
        assert!(result.relative_gauge_residual < 8e-10);
        assert!(result.relative_energy_work_error < 1.6e-9);
        errors.push(error);
    }
    for pair in errors.windows(2) {
        assert!(pair[1].0 < pair[0].0 / 1.4);
        assert!(pair[1].1 < pair[0].1 / 1.4);
        assert!(pair[1].2 < pair[0].2 / 1.4);
    }
    assert!(errors[2].0 < 0.02);
    assert!(errors[2].1 < 0.10);
    assert!(errors[2].2 < 0.25);
}

// witness: FEA-EM-MAGNETOSTATIC-001/fields @ magnetostatic/tet4|tet10/numerical_kernel;
//   boundary: Affine Tet4 and Tet10 geometry with lowest-order edge basis: sign/permeability
//   scaling against a Tet4 solve; Tet10 geometry is not higher-order magnetic approximation.
#[test]
fn sign_permeability_and_affine_geometry_order_preserve_physical_scaling() {
    let (projection, data) = problem(ElementOrder::Tet4, 2, 1.0, 1.0);
    let base = solve_grounded_volume(
        &projection,
        &super::exterior_face_identities(&projection),
        &[],
        &data,
        1e-10,
    )
    .unwrap();
    let base_samples = base.samples(&projection).unwrap();
    for (order, mu, j) in [
        (ElementOrder::Tet4, 1.25663706212e-6, -2.0),
        (ElementOrder::Tet10, 1.0, 1.0),
    ] {
        let (projection, data) = problem(order, 2, mu, j);
        let result = solve_grounded_volume(
            &projection,
            &super::exterior_face_identities(&projection),
            &[],
            &data,
            1e-10,
        )
        .unwrap();
        for (sample, expected) in result
            .samples(&projection)
            .unwrap()
            .iter()
            .zip(&base_samples)
        {
            for axis in 0..3 {
                assert!(
                    (sample.flux_density_t.canonical_value()[axis] / (mu * j)
                        - expected.flux_density_t.canonical_value()[axis])
                        .abs()
                        < 1e-8
                );
                assert!(
                    (sample.vector_potential_wb_per_m.canonical_value()[axis] / (mu * j)
                        - expected.vector_potential_wb_per_m.canonical_value()[axis])
                        .abs()
                        < 1e-8
                );
                assert!(
                    (sample.magnetic_field_a_per_m.canonical_value()[axis] / j
                        - expected.magnetic_field_a_per_m.canonical_value()[axis])
                        .abs()
                        < 1e-8
                );
            }
        }
        assert!(
            (result.magnetic_energy_j.canonical_value() / (mu * j * j)
                - base.magnetic_energy_j.canonical_value())
            .abs()
                < 1e-9
        );
    }
}

#[test]
fn invalid_material_identity_current_and_gauge_source_compatibility_fail_closed() {
    let (projection, mut data) = problem(ElementOrder::Tet4, 2, 1.0, 1.0);
    data[0].permeability_h_per_m = runmat_types::physical::MagneticPermeability::new(0.0);
    assert!(solve_grounded_volume(
        &projection,
        &super::exterior_face_identities(&projection),
        &[],
        &data,
        1e-10
    )
    .is_err());
    data[0].permeability_h_per_m = runmat_types::physical::MagneticPermeability::new(1.0);
    assert!(solve_grounded_volume(
        &projection,
        &super::exterior_face_identities(&projection),
        &[],
        &data[..data.len() - 1],
        1e-10
    )
    .is_err());
    data.push(data[0]);
    assert!(solve_grounded_volume(
        &projection,
        &super::exterior_face_identities(&projection),
        &[],
        &data,
        1e-10
    )
    .is_err());
    data.pop();
    for element in &mut data {
        element.current_density_a_per_m2 = runmat_types::physical::Quantity::new([0.0; 3]);
    }
    // Interior element-owned discontinuous current with a nonzero weak divergence.
    let interior = projection
        .elements
        .iter()
        .position(|element| {
            element
                .node_indices
                .iter()
                .any(|&node| projection.nodes[node].coordinates_m.canonical_value() == [0.5; 3])
        })
        .unwrap();
    data[interior].current_density_a_per_m2 =
        runmat_types::physical::Quantity::new([1.0, 2.0, 3.0]);
    assert!(matches!(
        solve_grounded_volume(
            &projection,
            &super::exterior_face_identities(&projection),
            &[],
            &data,
            1e-10
        ),
        Err(MagnetostaticError::IncompatibleCurrent(_))
    ));
    data[interior].current_density_a_per_m2 = runmat_types::physical::Quantity::new([0.0; 3]);
    let zero = solve_grounded_volume(
        &projection,
        &super::exterior_face_identities(&projection),
        &[],
        &data,
        1e-10,
    )
    .unwrap();
    assert_eq!(zero.magnetic_energy_j.canonical_value(), 0.0);
    assert_eq!(zero.source_work_j.canonical_value(), 0.0);
    assert!(zero
        .edge_circulations_wb
        .canonical_values()
        .iter()
        .all(|&value| value == 0.0));
}

// witness: FEA-EM-INPUT-001/magnetic_constraints @ magnetostatic/tet4|tet10/numerical_kernel;
//   boundary: Affine Tet4/Tet10 volume-current, full grounded-boundary and gauge plan compared with
//   same kernel; rejects missing gauge, surface-current substitution and wrong domain. No
//   independent physical-reference or periodic claim.
#[test]
fn authored_material_current_ground_and_gauge_compile_into_physical_solve() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = match order {
            ElementOrder::Tet4 => slab_mesh(2),
            ElementOrder::Tet10 => quadratic_slab_mesh(2),
        };
        let input = authored_model(-2.0);
        let result = solve_canonical_static(&input, &mesh, "potential", 1e-10).unwrap();
        assert!(result.magnetic_energy_j.canonical_value() > 0.0);
        assert!(result.relative_energy_work_error < 1.6e-9);
        let (projection, data) = problem(
            order,
            2,
            VACUUM_PERMEABILITY_H_PER_M.canonical_value(),
            -2.0,
        );
        let reference = solve_grounded_volume(
            &projection,
            &super::exterior_face_identities(&projection),
            &[],
            &data,
            1e-10,
        )
        .unwrap();
        assert!(
            (result.magnetic_energy_j.canonical_value()
                / reference.magnetic_energy_j.canonical_value()
                - 1.0)
                .abs()
                < 1e-8
        );
        assert_eq!(
            result.edge_circulations_wb.canonical_values().len(),
            reference.edge_circulations_wb.canonical_values().len()
        );
        for (a, b) in result
            .edge_circulations_wb
            .canonical_values()
            .iter()
            .zip(reference.edge_circulations_wb.canonical_values())
        {
            assert!((a - b).abs() < 1e-14);
        }
        let mut missing = input.clone();
        missing.constraints.pop();
        assert!(solve_canonical_static(&missing, &mesh, "potential", 1e-10).is_err());
        let mut surface = input.clone();
        let excitation = surface.excitations[0].excitation().clone();
        surface.excitations[0] = ExcitationAction::CurrentDensity(excitation);
        assert!(solve_canonical_static(&surface, &mesh, "potential", 1e-10).is_err());
        let mut wrong_domain = input;
        wrong_domain.electromagnetic = Some(ElectromagneticDomain::Static {
            equation: StaticElectricalEquation::Electrostatics,
        });
        assert!(solve_canonical_static(&wrong_domain, &mesh, "potential", 1e-10).is_err());
    }
}

#[test]
fn exact_volume_superposition_zero_conductivity_and_missing_boundary_fail_closed() {
    let mesh = slab_mesh(2);
    let base = authored_model(2.0);
    let reference = solve_canonical_static(&base, &mesh, "potential", 1e-10).unwrap();
    let mut summed = authored_model(3.0);
    summed.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .conductivity_s_per_m = runmat_types::physical::ElectricalConductivity::new(0.0);
    let PhysicalSignal::ElectricCurrentDensity(mut signal) = summed.signals[0].clone() else {
        unreachable!()
    };
    signal.signal_id = SignalId("minus".into());
    signal.representation = SignalRepresentation::Constant(SignalValue::Vector3(
        [0.0, 0.0, -1.0].map(ElectricCurrentDensity::new),
    ));
    summed
        .signals
        .push(PhysicalSignal::ElectricCurrentDensity(signal));
    let mut action = summed.excitations[0].excitation().clone();
    action.excitation_id = ExcitationId("minus".into());
    action.signal_id = SignalId("minus".into());
    action.target = ExcitationTarget::Region(RegionTarget::volume(
        runmat_meshing_core::fixtures::fixture_solid("slab"),
    ));
    summed
        .excitations
        .push(ExcitationAction::MagneticVolumeCurrentDensity(action));
    summed.excitation_cases[0]
        .action_ids
        .push(ExcitationId("minus".into()));
    let result = solve_canonical_static(&summed, &mesh, "potential", 1e-10).unwrap();
    assert!(
        (result.magnetic_energy_j.canonical_value()
            / reference.magnetic_energy_j.canonical_value()
            - 1.0)
            .abs()
            < 1e-8
    );
    for (a, b) in result
        .edge_circulations_wb
        .canonical_values()
        .iter()
        .zip(reference.edge_circulations_wb.canonical_values())
    {
        assert!((a - b).abs() < 1e-14);
    }
    summed.constraints.retain(|c| {
        !matches!(c, AnalysisConstraint::Magnetic(m)
            if matches!(m.kind, MagneticConstraintKind::VectorPotentialGround { .. }))
    });
    assert!(solve_canonical_static(&summed, &mesh, "potential", 1e-10).is_err());
    let (mut projection, data) = problem(ElementOrder::Tet10, 2, 1.0, 1.0);
    let midside = projection.elements[0].node_indices[4];
    {
        let quantity_change = 0.01;
        let quantity_index = 0;
        let quantity_slot = &mut projection.nodes[midside].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert!(matches!(
        solve_grounded_volume(
            &projection,
            &super::exterior_face_identities(&projection),
            &[],
            &data,
            1e-10
        ),
        Err(MagnetostaticError::Unsupported(_))
    ));
    let (mut projection, data) = problem(ElementOrder::Tet4, 2, 1.0, 1.0);
    projection.boundary_faces.pop();
    assert!(matches!(
        solve_grounded_volume(
            &projection,
            &super::exterior_face_identities(&projection),
            &[],
            &data,
            1e-10
        ),
        Err(MagnetostaticError::Topology(_))
    ));
}

// witness: FEA-EM-MAGNETOSTATIC-001/fields @ magnetostatic/tet4/numerical_kernel;
//   boundary: Affine Tet4 shared-face normal curl continuity and global boundary flux closure; no
//   general magnetic force recovery.
#[test]
fn recovered_curl_has_shared_face_normal_flux_continuity_and_global_flux_closure() {
    let (projection, data) = problem(ElementOrder::Tet4, 3, 1.0, 1.0);
    let response = solve_grounded_volume(
        &projection,
        &super::exterior_face_identities(&projection),
        &[],
        &data,
        1e-10,
    )
    .unwrap();
    let field: std::collections::BTreeMap<_, _> = response
        .samples(&projection)
        .unwrap()
        .iter()
        .map(|sample| {
            (
                sample.element_identity,
                sample.flux_density_t.canonical_value(),
            )
        })
        .collect();
    let mut faces = std::collections::BTreeMap::<[usize; 3], Vec<[f64; 3]>>::new();
    for element in &projection.elements {
        for face in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            let mut nodes = face.map(|i| element.node_indices[i]);
            nodes.sort_unstable();
            faces
                .entry(nodes)
                .or_default()
                .push(field[&element.stable_identity]);
        }
    }
    for (nodes, fluxes) in faces.iter().filter(|(_, fields)| fields.len() == 2) {
        let a = projection.nodes[nodes[0]].coordinates_m.canonical_value();
        let b = projection.nodes[nodes[1]].coordinates_m.canonical_value();
        let c = projection.nodes[nodes[2]].coordinates_m.canonical_value();
        let u: [f64; 3] = std::array::from_fn(|axis| b[axis] - a[axis]);
        let v: [f64; 3] = std::array::from_fn(|axis| c[axis] - a[axis]);
        let normal = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let jump = (0..3)
            .map(|axis| (fluxes[0][axis] - fluxes[1][axis]) * normal[axis])
            .sum::<f64>();
        assert!(jump.abs() < 1e-13);
    }
    let mut net = 0.0;
    let mut absolute = 0.0;
    for face in &projection.boundary_faces {
        let owner = &projection.elements[face.adjacent_element_indices[0]];
        let flux = field[&owner.stable_identity]
            .iter()
            .zip(face.oriented_area_vector_m2)
            .map(|(b, n)| b * n)
            .sum::<f64>();
        net += flux;
        absolute += flux.abs();
    }
    assert!(net.abs() <= 1e-12 * absolute);
}

// witness: FEA-EM-INPUT-001/magnetic_constraints @ magnetostatic/tet4/numerical_kernel; coverage: partial;
//   boundary: A partially grounded boundary with insulated faces and one gauge is admitted and its
//   energy rises against the fully grounded case; a boundary without ground is rejected; Tet4 only.
#[test]
fn insulated_faces_and_one_gauge_admit_a_partially_grounded_field() {
    // The fixture current runs parallel to the slab's left face. Insulating that face
    // (n x H = 0) leaves one connected grounded surface, and the field is well posed.
    let mesh = crate::topology::test_meshes::slab_mesh(2);
    let full = solve_canonical_static(&authored_model(1.0), &mesh, "potential", 1e-10).unwrap();
    let mut model = authored_model(1.0);
    model
        .constraints
        .retain(|c| c.constraint_id() != "slab_left");
    let partial = solve_canonical_static(&model, &mesh, "potential", 1e-10).unwrap();
    assert!(
        partial.magnetic_energy_j.is_finite()
            && partial.magnetic_energy_j > runmat_types::physical::Quantity::new(0.0)
    );
    // Insulation removes constraints, so the stored energy for the same current can only rise.
    assert!(partial.magnetic_energy_j >= full.magnetic_energy_j * (1.0 - 1e-9));
    // Current may not leave through an insulated face: insulating the sides the current
    // crosses is rejected as an incompatible source, not solved.
    let mut crossing = authored_model(1.0);
    crossing
        .constraints
        .retain(|c| c.constraint_id() != "slab_insulated");
    assert!(solve_canonical_static(&crossing, &mesh, "potential", 1e-10).is_err());
    // Without any ground the gradient part of A is undetermined.
    let mut ungrounded = authored_model(1.0);
    ungrounded.constraints.retain(|c| {
        !matches!(c, AnalysisConstraint::Magnetic(m)
            if matches!(m.kind, MagneticConstraintKind::VectorPotentialGround { .. }))
    });
    assert!(solve_canonical_static(&ungrounded, &mesh, "potential", 1e-10).is_err());
}

// witness: FEA-EM-INPUT-001/magnetic_constraints @ magnetostatic/tet4/numerical_kernel; coverage:
//   partial;
//   boundary: A prescribed tangential vector potential on one face with grounds and insulated sides
//   drives the exact uniform flux density A = (0, a x/L, 0); gauge, periodic and phase-shifted
//   relations are not covered.
#[test]
fn prescribed_tangential_potential_drives_the_exact_uniform_field() {
    // Box [0,L]^3 with A = (0, a x / L, 0): B = (0, 0, a / L). A = 0 on x = 0 and on the
    // y faces (where A is normal), A = (0, a, 0) on x = L, z faces insulated (H is normal).
    let l = 2.0e-2;
    let a = 3.0e-4;
    let lines = crate::physics::maxwell::tests::layered::lines(0.0, l, 3);
    let mesh = crate::physics::maxwell::tests::layered::mesh(
        &lines,
        &lines,
        &lines,
        |_| "slab",
        move |centroid, _normal, _regions| {
            if centroid[0].abs() < 1e-12 {
                "x_low".into()
            } else if (centroid[0] - l).abs() < 1e-12 {
                "x_high".into()
            } else if centroid[1].abs() < 1e-12 || (centroid[1] - l).abs() < 1e-12 {
                "y_side".into()
            } else {
                "z_side".into()
            }
        },
    );
    let mut model = authored_model(0.0);
    model.constraints.retain(|c| {
        !matches!(c, AnalysisConstraint::Magnetic(m)
            if matches!(m.kind, MagneticConstraintKind::VectorPotentialGround { .. }))
    });
    for face in ["x_low", "y_side"] {
        model
            .constraints
            .push(AnalysisConstraint::Magnetic(MagneticConstraint {
                constraint_id: format!("ground_{face}"),
                kind: MagneticConstraintKind::VectorPotentialGround {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                    ),
                },
            }));
    }
    model
        .signals
        .push(PhysicalSignal::MagneticVectorPotential(Signal {
            signal_id: SignalId("A".into()),
            shape: SignalShape::Vector3,
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Constant(SignalValue::Vector3(
                [0.0, a, 0.0].map(MagneticVectorPotential::new),
            )),
            precision: SignalPrecision::Real64,
            provenance: SignalProvenance::Authored,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
        }));
    model
        .excitations
        .push(ExcitationAction::MagneticVectorPotential(
            AnalysisExcitation {
                excitation_id: ExcitationId("imposed".into()),
                target: ExcitationTarget::Region(RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("x_high"),
                )),
                frame: CoordinateFrame::Global,
                direction: None,
                distribution: ExcitationDistribution::IntensityPerArea,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: SignalId("A".into()),
                active_step_ids: vec!["potential".into()],
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: false,
                    preserve_moment: false,
                    relative_tolerance: 1e-12,
                },
            },
        ));
    if let Some(case) = model.excitation_cases.first_mut() {
        case.action_ids.push(ExcitationId("imposed".into()));
    }
    let response = solve_canonical_static(&model, &mesh, "potential", 1e-10).unwrap();
    let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    for sample in &response.samples(&projection).unwrap() {
        let b = sample.flux_density_t;
        assert!(
            (b.canonical_value()[2] - a / l).abs() <= 1e-9 * a / l,
            "{b:?}"
        );
        assert!(
            b.canonical_value()[0].abs() <= 1e-9 * a / l
                && b.canonical_value()[1].abs() <= 1e-9 * a / l,
            "{b:?}"
        );
    }
}
