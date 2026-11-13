//! Independent outgoing plane-wave references; no discrete impedance solve is the oracle.
use super::*;
use crate::topology::test_meshes::quadratic_slab_mesh;
mod diagnostics;
mod exact_operators;
mod asymptotic;

fn fixture(
    cells: usize,
    quadratic: bool,
    frequency: f64,
    amplitude: [f64; 2],
    rotated: bool,
) -> (AnalysisModel, SolverMeshArtifact) {
    let mut mesh = if quadratic {
        quadratic_slab_mesh(cells)
    } else {
        slab_mesh(cells)
    };
    if rotated {
        let a = std::f64::consts::FRAC_1_SQRT_2;
        for node in &mut mesh.topology.nodes {
            let [x, y, z] = node.coordinates_m.canonical_value();
            node.coordinates_m = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([a * (x - y), a * (x + y), z]);
        }
        mesh.seal_canonical_digest().unwrap();
        mesh.validate().unwrap();
    }
    let mut model = model();
    if rotated {
        model.materials[0].mass_density_kg_per_m3 =
            (Some(5.0)).map(runmat_types::physical::MassDensity::new);
        model.materials[0]
            .acoustic
            .as_mut()
            .unwrap()
            .speed_of_sound_m_per_s = runmat_types::physical::SpeedOfSound::new(4.0);
    }
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.sequence.steps[0].independent_domain = IndependentDomain::FrequencyHertz(vec![frequency]);
    model.signals.push(PhysicalSignal::Pressure(complex_signal(
        Pressure::new(amplitude[0]),
        Pressure::new(amplitude[1]),
    )));
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "pressure".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Signal(SignalId("drive".into())),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "outgoing".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::AcousticRadiation,
    });
    (model, mesh)
}

fn solve(
    cells: usize,
    quadratic: bool,
    frequency: f64,
    amplitude: [f64; 2],
    rotated: bool,
) -> CanonicalAcousticResponse {
    let (model, mesh) = fixture(cells, quadratic, frequency, amplitude, rotated);
    solve_canonical_harmonic(
        &model,
        &mesh,
        "acoustic",
        Frequency::new(frequency),
        AcousticHarmonicPolicy {
            relative_tolerance: 1e-11,
            ..Default::default()
        },
        1e-8,
    )
    .unwrap()
}

fn multiply(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

// Independent boundary quadrature of recovered fields, not the assembled B quadratic form.
// Element velocity is its advertised volume average, so this flux estimate must refine.
fn outward_recovered_power(result: &CanonicalAcousticResponse, target: &str) -> f64 {
    let mut power = 0.0;
    for face in result
        .projection
        .boundary_faces_for_target(&RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_boundary_selection(&(target)),
        ))
    {
        let velocity =
            &result.element_velocity_m_per_s.canonical_values()[face.adjacent_element_indices[0]];
        result
            .projection
            .visit_boundary_quadrature(face, |point| {
                let mut p = [0.0; 2];
                for (&index, &shape) in face.node_indices.iter().zip(point.shape_values()) {
                    p[0] += shape * result.pressure_pa.canonical_values()[index].real;
                    p[1] += shape * result.pressure_pa.canonical_values()[index].imaginary;
                }
                for axis in 0..3 {
                    power += 0.5
                        * (p[0] * velocity.real[axis] + p[1] * velocity.imaginary[axis])
                        * point.area_vector_weight_m2[axis];
                }
            })
            .unwrap();
    }
    power
}

// E[x^n] over a uniform tetrahedron is 6 h_n(x0,x1,x2,x3)/((n+1)(n+2)(n+3)).
// Integrate the analytic exponential, not a centroid approximation to mean velocity.
fn mean_wave(xs: [f64; 4], k: f64) -> [f64; 2] {
    let mut h = [0.0; 32];
    h[0] = 1.0;
    for x in xs {
        for n in 1..h.len() {
            h[n] += x * h[n - 1];
        }
    }
    let mut coefficient = [1.0, 0.0];
    let mut mean = [0.0, 0.0];
    for (n, hn) in h.into_iter().enumerate() {
        if n > 0 {
            coefficient = [
                coefficient[1] * k / n as f64,
                -coefficient[0] * k / n as f64,
            ];
        }
        let moment = 6.0 * hn / ((n + 1) * (n + 2) * (n + 3)) as f64;
        mean[0] += coefficient[0] * moment;
        mean[1] += coefficient[1] * moment;
    }
    mean
}

// witness: FEA-ACOUSTIC-BOUNDARY-001/radiation @ acoustic_harmonic/tet4|tet10/numerical_kernel;
//   check: local-radiation; boundary: Local radiation reference and original spatial gates; witness linkage is not a
//   passing verdict or nonlocal exterior-wave qualification.
#[test]
fn local_radiation_pressure_phase_mean_velocity_and_signed_power_refine() {
    let mut failures = Vec::<String>::new();
    let mut executed = std::collections::BTreeSet::new();
    // Numerical failures collect; solve/topology failures still fail closed immediately.
    macro_rules! gate {
        ($condition:expr, $context:expr) => {
            match $condition {
                true => {},
                false => failures.push(format!("{}: {}", $context, stringify!($condition))),
            }
        };
        ($condition:expr, $context:expr, $($message:tt)+) => {
            match $condition {
                true => {},
                false => failures.push(format!("{}: {}: {}", $context,
                    stringify!($condition), format_args!($($message)+))),
            }
        };
    }
    for quadratic in [false, true] {
        let frequencies: &[f64] = if quadratic { &[0.2, 0.5] } else { &[0.2] };
        // Tet10 adds cells 8 so both asserted velocity windows lie in the asymptotic range.
        let refinements: &[usize] = if quadratic { &[1, 2, 4, 8] } else { &[2, 4, 8] };
        for rotated in [false, true] {
            let (rho, c, direction) = if rotated {
                (
                    5.0,
                    4.0,
                    [
                        std::f64::consts::FRAC_1_SQRT_2,
                        std::f64::consts::FRAC_1_SQRT_2,
                        0.0,
                    ],
                )
            } else {
                (3.0, 2.0, [1.0, 0.0, 0.0])
            };
            let axial = |position: [f64; 3]| {
                position
                    .iter()
                    .zip(direction)
                    .map(|(x, d)| x * d)
                    .sum::<f64>()
            };
            let impedance = rho * c;
            for &base_frequency in frequencies {
                let frequency = base_frequency * c / 2.0;
                for (phase, amplitude) in [[1.0, 0.4], [-0.4, 1.0]].into_iter().enumerate() {
                    let k = std::f64::consts::TAU * frequency / c;
                    let norm = amplitude[0] * amplitude[0] + amplitude[1] * amplitude[1];
                    let mut pressure_errors = Vec::new();
                    let mut velocity_errors = Vec::new();
                    for &cells in refinements {
                        let result = solve(cells, quadratic, frequency, amplitude, rotated);
                        assert!(
                            executed.insert((
                                quadratic,
                                rotated,
                                base_frequency.to_bits(),
                                phase,
                                cells
                            )),
                            "duplicate radiation solve inventory entry"
                        );
                        let context = format!("quadratic={quadratic} rotated={rotated} frequency={frequency} amplitude={amplitude:?} cells={cells}");
                        let diagnostic = diagnostics::measure(
                            &result, cells, quadratic, frequency, amplitude, rotated, direction,
                        );
                        gate!(
                            diagnostic.recovery_max < 1e-11 * norm.sqrt() / impedance,
                            context,
                            "independent face recovery maximum={}",
                            diagnostic.recovery_max
                        );
                        eprintln!("radiation decomposition {context}: face_recovery_max={} galerkin_rms={} interpolation_rms={} cross_mean={} total_rms={} independent_total_rms={}",
                            diagnostic.recovery_max, diagnostic.galerkin_rms, diagnostic.interpolation_rms,
                            diagnostic.cross_mean, diagnostic.total_rms, diagnostic.independent_total_rms);
                        let pressure_error = result
                            .projection
                            .nodes
                            .iter()
                            .zip(result.pressure_pa.canonical_values())
                            .map(|(n, p)| {
                                let reference = multiply(
                                    amplitude,
                                    [
                                        (k * axial(n.coordinates_m.canonical_value())).cos(),
                                        -(k * axial(n.coordinates_m.canonical_value())).sin(),
                                    ],
                                );
                                (p.real - reference[0]).powi(2)
                                    + (p.imaginary - reference[1]).powi(2)
                            })
                            .sum::<f64>()
                            / result.pressure_pa.canonical_values().len() as f64;
                        let velocity_error = result
                            .projection
                            .elements
                            .iter()
                            .zip(result.element_velocity_m_per_s.canonical_values())
                            .map(|(e, v)| {
                                let xs = std::array::from_fn(|i| {
                                    axial(
                                        result.projection.nodes[e.node_indices[i]]
                                            .coordinates_m
                                            .canonical_value(),
                                    )
                                });
                                let p = multiply(amplitude, mean_wave(xs, k));
                                (0..3)
                                    .map(|axis| {
                                        (v.real[axis] - direction[axis] * p[0] / impedance).powi(2)
                                            + (v.imaginary[axis]
                                                - direction[axis] * p[1] / impedance)
                                                .powi(2)
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>()
                            / result.projection.elements.len() as f64;
                        pressure_errors.push(pressure_error.sqrt());
                        velocity_errors.push(velocity_error.sqrt());
                        let roundoff_scale = norm.sqrt() / impedance
                            + diagnostic.galerkin_rms
                            + diagnostic.interpolation_rms;
                        let roundoff = 256.0 * f64::EPSILON * roundoff_scale;
                        assert!(
                            (diagnostic.total_rms - velocity_error.sqrt()).abs() <= roundoff
                                && (diagnostic.independent_total_rms - velocity_error.sqrt()).abs()
                                    <= roundoff,
                            "mean-gradient total must match original velocity RMS ({context}): original={} decomposition={} independent={} roundoff={roundoff}",
                            velocity_error.sqrt(), diagnostic.total_rms, diagnostic.independent_total_rms
                        );
                        gate!(
                            result.relative_residual < 1e-10,
                            context,
                            "relative_residual={}",
                            result.relative_residual
                        );
                        gate!(
                            result.power_relative_error < 1e-8,
                            context,
                            "power_relative_error={}",
                            result.power_relative_error
                        );
                        // Outward radiation power is positive; inlet outward power is its negative.
                        gate!(
                            result.mean_input_power_w.canonical_value() > 0.0
                                && result.mean_impedance_loss_w.canonical_value() > 0.0,
                            context
                        );
                        if cells == refinements[2] {
                            let exact_power = norm / (2.0 * impedance);
                            let inlet = outward_recovered_power(&result, "slab_left");
                            let outlet = outward_recovered_power(&result, "slab_right");
                            let vectors =
                                ["slab_left", "slab_right"].map(|target| {
                                    let mut total = [0.0; 3];
                                    for face in result
                                        .projection
                                        .boundary_faces_for_target(&RegionTarget::boundary(
                                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                                            &(target),
                                        ),
                                    )) {
                                        for (value, component) in
                                            total.iter_mut().zip(face.oriented_area_vector_m2)
                                        {
                                            *value += component;
                                        }
                                    }
                                    total
                                });
                            eprintln!("radiation quadratic={quadratic} rotated={rotated} rho={rho} c={c} frequency={frequency} cells={cells} amplitude={amplitude:?} pressure={pressure_errors:?} velocity={velocity_errors:?} inlet={inlet} outlet={outlet} exact={exact_power} admittance_power={} face_vectors={vectors:?}", result.mean_impedance_loss_w.canonical_value());
                            gate!(
                                inlet < 0.0 && outlet > 0.0,
                                context,
                                "inlet={inlet}, outlet={outlet}"
                            );
                            gate!(
                                (inlet + exact_power).abs() < 0.005,
                                context,
                                "inlet={inlet}, exact={exact_power}"
                            );
                            gate!(
                                (outlet - exact_power).abs() < 0.005,
                                context,
                                "outlet={outlet}, exact={exact_power}"
                            );
                            gate!(
                                (result.mean_impedance_loss_w.canonical_value() - exact_power)
                                    .abs()
                                    < 0.005,
                                context,
                                "admittance_power={}, exact={exact_power}",
                                result.mean_impedance_loss_w.canonical_value()
                            );
                            gate!(
                                pressure_error.sqrt()
                                    < if quadratic {
                                        0.004 * norm.sqrt()
                                    } else {
                                        0.004
                                    },
                                context,
                                "pressure={pressure_errors:?}"
                            );
                            gate!(
                                velocity_error.sqrt() < 0.01,
                                context,
                                "velocity={velocity_errors:?}"
                            );
                        }
                    }
                    let context = format!("quadratic={quadratic} rotated={rotated} frequency={frequency} amplitude={amplitude:?} refinements={refinements:?}");
                    gate!(
                        pressure_errors
                            .windows(2)
                            .all(|e| e[0] / e[1] > if quadratic { 6.0 } else { 3.0 }),
                        context,
                        "pressure={pressure_errors:?}"
                    );
                    // Tet10 element-mean velocity converges at O(h^2): its error is an
                    // interpolation term (mean gradient of the P2 interpolant, face integral
                    // of a cubic, ratios 3.98, 4.00) plus a faster Galerkin term (ratios 4.7,
                    // 5.6). At cells 1 the two partly cancel (low frequency: interpolation
                    // 8.30e-4, Galerkin 7.34e-4, total 4.99e-4), so the 1 -> 2 ratio is
                    // pre-asymptotic, 2.954; then 3.658 and 3.775 (cells 8: 1.223e-5) toward
                    // 4. Element velocity is pinned to the face integral of the Galerkin
                    // pressure (recovery gate above), and that pressure is the exact Galerkin
                    // solution, reproduced to 7 digits by the independent script
                    // crates/runmat-analysis/fea/evidence/python/acoustic.py. The ratio gate
                    // therefore applies from cells 2; the 1 -> 2 window must still decrease.
                    gate!(
                        velocity_errors.windows(2).enumerate().all(|(window, e)| {
                            if !quadratic {
                                e[0] / e[1] > 1.5
                            } else if window == 0 {
                                e[1] < e[0]
                            } else {
                                e[0] / e[1] > 3.0
                            }
                        }),
                        context,
                        "velocity={velocity_errors:?}"
                    );
                }
            }
        }
    }
    let mut expected = std::collections::BTreeSet::new();
    // Independent literal inventory, so accidentally removed frequency/order/phase rows reject.
    for (quadratic, frequencies, cells) in [
        (false, vec![0.2_f64], vec![2, 4, 8]),
        (true, vec![0.2_f64, 0.5], vec![1, 2, 4, 8]),
    ] {
        for rotated in [false, true] {
            for &frequency in &frequencies {
                for phase in 0..2 {
                    for &cells in &cells {
                        expected.insert((quadratic, rotated, frequency.to_bits(), phase, cells));
                    }
                }
            }
        }
    }
    assert_eq!(executed.len(), 44);
    assert_eq!(executed, expected, "radiation solve inventory differs");
    eprintln!(
        "radiation completed {} mandatory solves; {} numerical gates failed\n{}",
        executed.len(),
        failures.len(),
        failures.join("\n")
    );
    assert!(
        failures.is_empty(),
        "radiation numerical gate failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn rotated_affine_pressure_gradients_reproduce_world_basis_for_both_orders() {
    for quadratic in [false, true] {
        let (_, mesh) = fixture(1, quadratic, 0.2, [1.0, 0.4], true);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        for element in &projection.elements {
            projection.visit_element_quadrature(element, |point| {
                for world_axis in 0..3 {
                    let gradient: [f64;3] = std::array::from_fn(|axis| {
                        element.node_indices.iter().enumerate().map(|(local,&node)| {
                            projection.nodes[node].coordinates_m.canonical_value()[world_axis]*point.physical_shape_gradients()[local][axis]
                        }).sum()
                    });
                    for (axis,&actual) in gradient.iter().enumerate() {
                        let expected = if axis == world_axis {1.0} else {0.0};
                        assert!((actual-expected).abs() < 1e-12, "quadratic={quadratic}, element={}, worldaxis={world_axis}, gradient={gradient:?}",element.element_id);
                    }
                }
            }).unwrap();
        }
    }
}

#[test]
fn local_radiation_operator_and_pressure_rotate_with_compiled_medium_scaling() {
    for quadratic in [false, true] {
        let assemblies = [false, true].map(|rotated| {
            let frequency = if rotated { 0.4 } else { 0.2 };
            let (model, mesh) = fixture(1, quadratic, frequency, [1.0, 0.4], rotated);
            let properties = crate::properties::compile_solid_property_map(&model, &mesh).unwrap();
            let projection = AnalysisProjection::derive(&mesh).unwrap();
            let step = &model.sequence.steps[0];
            let sample = ExcitationSamplePoint::frequency(frequency);
            let excitations = realize_step_excitations(&model, step, sample).unwrap();
            AcousticAssembly::assemble(
                projection,
                &properties,
                runmat_types::physical::AngularFrequency::new(std::f64::consts::TAU * frequency),
            )
            .unwrap()
            .with_boundaries(&model, step, &excitations, sample)
            .unwrap()
        });
        let n = assemblies[0].projection.nodes.len();
        let real = (0..n).map(|i| (i as f64 + 0.4).sin()).collect::<Vec<_>>();
        let imaginary = (0..n).map(|i| (i as f64 + 0.2).cos()).collect::<Vec<_>>();
        let original = assemblies[0].apply(&real, &imaginary);
        let rotated = assemblies[1].apply(&real, &imaginary);
        for (a, b) in original
            .0
            .iter()
            .chain(&original.1)
            .zip(rotated.0.iter().chain(&rotated.1))
        {
            assert!(
                (b - a * 3.0 / 5.0).abs() < 1e-12,
                "quadratic={quadratic}, original={a}, rotated={b}"
            );
        }
        let original = solve(1, quadratic, 0.2, [1.0, 0.4], false);
        let rotated = solve(1, quadratic, 0.4, [1.0, 0.4], true);
        for (index, (a, b)) in original
            .pressure_pa
            .canonical_values()
            .iter()
            .zip(rotated.pressure_pa.canonical_values())
            .enumerate()
        {
            assert!(
                (a.real - b.real).abs() < 1e-9 && (a.imaginary - b.imaginary).abs() < 1e-9,
                "quadratic={quadratic}, node={index}, original={a:?}, rotated={b:?}"
            );
        }
        let d = std::f64::consts::FRAC_1_SQRT_2;
        for (a, b) in original
            .element_velocity_m_per_s
            .canonical_values()
            .iter()
            .zip(rotated.element_velocity_m_per_s.canonical_values())
        {
            for (av, bv) in [(&a.real, &b.real), (&a.imaginary, &b.imaginary)] {
                let expected = [
                    d * (av[0] - av[1]) * 6.0 / 20.0,
                    d * (av[0] + av[1]) * 6.0 / 20.0,
                    av[2] * 6.0 / 20.0,
                ];
                for axis in 0..3 {
                    assert!((bv[axis]-expected[axis]).abs() < 1e-9, "quadratic={quadratic}, original={av:?}, rotated={bv:?}, expected={expected:?}");
                }
            }
        }
    }
}

#[test]
fn independent_mean_wave_reproduces_constant_coordinate() {
    let k = 0.7;
    let x = 0.4;
    let mean = mean_wave([x; 4], k);
    assert!((mean[0] - (k * x).cos()).abs() < 1e-14);
    assert!((mean[1] + (k * x).sin()).abs() < 1e-14);
}
