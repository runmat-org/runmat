use super::*;
use runmat_meshing_core::ElementOrder;
mod fixture;
mod flux;
pub(crate) mod layered;
mod terminal;

#[test]
fn longitudinal_impressed_current_is_not_projected_to_a_solenoidal_source() {
    let mut errors = Vec::new();
    for cells in [2, 4] {
        let (mesh, mut input) = fixture::problem(
            ElementOrder::Tet4,
            cells,
            ComplexPhasor {
                real: 1.0,
                imaginary: 0.0,
            },
        );
        for (source, sample) in input
            .source_current
            .iter_mut()
            .zip(input.sampling.integration_samples())
        {
            let electric = fixture::longitudinal(sample.position_m);
            // curl(E)=0, so J=-(sigma+i*omega*epsilon)E. No current cleaning.
            source.current_density_a_per_m2 = electric.map(|value| ComplexPhasor {
                real: -0.3 * value,
                imaginary: -value,
            });
        }
        let result = solve(&mesh, &input, 1e-10).unwrap();
        let mut defect = 0.0;
        let mut exact_norm = 0.0;
        for sample in &result.samples {
            let expected = fixture::longitudinal(sample.position_m);
            for (value, exact) in sample.electric_field_v_per_m.into_iter().zip(expected) {
                defect += sample.volume_weight_m3
                    * ((value.real - exact).powi(2) + value.imaginary.powi(2));
                exact_norm += sample.volume_weight_m3 * exact.powi(2);
            }
        }
        let relative = (defect / exact_norm).sqrt();
        eprintln!("Maxwell longitudinal cells={cells}, Erelative={relative}");
        assert!(result.relative_physical_residual < 8e-10);
        errors.push(relative);
    }
    assert!(
        errors[1] < errors[0] / 1.4,
        "longitudinal refinement {errors:?}"
    );
    assert!(errors[1] < 0.5, "longitudinal reference {errors:?}");
}

fn difference(a: ComplexPhasor, b: ComplexPhasor) -> f64 {
    (a.real - b.real).powi(2) + (a.imaginary - b.imaginary).powi(2)
}

#[test]
fn exact_sparse_plan_and_heterogeneous_element_laws_preserve_owned_assignment() {
    let (mesh, mut input) = fixture::problem(
        ElementOrder::Tet4,
        2,
        ComplexPhasor {
            real: 0.6,
            imaginary: 0.8,
        },
    );
    for (index, volume) in input.volumes.iter_mut().enumerate() {
        volume.conductivity_s_per_m = if index % 2 == 0 { 0.3 } else { 0.8 };
        volume.permittivity_f_per_m = if index % 2 == 0 { 1.0 } else { 2.0 };
    }
    let plan = MaxwellSystemPlan::derive(&mesh, &input.grounded_faces).unwrap();
    let identity = plan.identity().unwrap();
    let cold = MaxwellSystemPlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap();
    assert_eq!(identity, cold.identity().unwrap());
    cold.validate_for(&mesh).unwrap();
    let base = solve_with_plan(&mesh, &input, 1e-10, &cold).unwrap();
    input.volumes.reverse();
    let reordered = solve(&mesh, &input, 1e-10).unwrap();
    for (actual, expected) in reordered.samples.iter().zip(&base.samples) {
        for axis in 0..3 {
            assert!(
                difference(
                    actual.electric_field_v_per_m[axis],
                    expected.electric_field_v_per_m[axis]
                ) < 1e-15
            );
        }
    }
    assert!((base.ohmic_power_w - reordered.ohmic_power_w).abs() < 1e-12);
    let (foreign, _) = fixture::problem(
        ElementOrder::Tet4,
        1,
        ComplexPhasor {
            real: 1.0,
            imaginary: 0.0,
        },
    );
    assert!(cold.validate_for(&foreign).is_err());
    assert!(solve_with_plan(&foreign, &input, 1e-10, &cold).is_err());
}

#[test]
fn manufactured_complex_pec_mode_has_independent_field_and_power_convergence() {
    let phase = ComplexPhasor {
        real: 0.6,
        imaginary: 0.8,
    };
    // ∫sin²(pi*x)sin²(pi*z)dV=1/4; peak phasors contribute the factor1/2.
    let exact_loss = 0.3 / 8.0;
    let mut errors = Vec::new();
    for cells in [1, 2, 4, 6] {
        let (mesh, input) = fixture::problem(ElementOrder::Tet4, cells, phase);
        let result = solve(&mesh, &input, 1e-10).unwrap();
        let mut electric = 0.0;
        let mut flux = 0.0;
        for sample in &result.samples {
            let expected_e = fixture::electric(sample.position_m, phase);
            let expected_b = fixture::flux(sample.position_m, phase);
            for axis in 0..3 {
                electric += sample.volume_weight_m3
                    * difference(sample.electric_field_v_per_m[axis], expected_e[axis]);
                flux += sample.volume_weight_m3
                    * difference(sample.flux_density_t[axis], expected_b[axis]);
            }
        }
        let error = (electric.sqrt(), flux.sqrt());
        eprintln!(
            "Maxwell cells={cells}: electric={}, flux={}",
            error.0, error.1
        );
        errors.push(error);
        assert!(result.relative_physical_residual < 8e-10);
        assert!(result.relative_complex_power_error < 1.6e-9);
        if cells == 6 {
            assert!(error.0 < 0.25);
            assert!(error.1 < 0.60);
            assert!((result.ohmic_power_w - exact_loss).abs() / exact_loss < 0.25);
        }
    }
    for pair in errors.windows(2) {
        assert!(pair[1].0 < pair[0].0 / 1.4, "electric refinement {pair:?}");
        assert!(pair[1].1 < pair[0].1 / 1.4, "flux refinement {pair:?}");
    }
}

#[test]
fn phase_and_affine_geometry_order_preserve_signed_complex_fields() {
    let (mesh, input) = fixture::problem(
        ElementOrder::Tet4,
        2,
        ComplexPhasor {
            real: 1.0,
            imaginary: 0.0,
        },
    );
    let base = solve(&mesh, &input, 1e-10).unwrap();
    for (order, phase) in [
        (
            ElementOrder::Tet4,
            ComplexPhasor {
                real: -1.0,
                imaginary: 0.0,
            },
        ),
        (
            ElementOrder::Tet10,
            ComplexPhasor {
                real: 0.0,
                imaginary: 1.0,
            },
        ),
    ] {
        let (mesh, mut input) = fixture::problem(order, 2, phase);
        // Element ownership, not vector position, assigns coefficients.
        input.volumes.reverse();
        let result = solve(&mesh, &input, 1e-10).unwrap();
        assert_eq!(base.samples.len(), result.samples.len());
        for (actual, expected) in result.samples.iter().zip(&base.samples) {
            for axis in 0..3 {
                for (a, b) in [
                    (
                        actual.electric_field_v_per_m[axis],
                        expected.electric_field_v_per_m[axis],
                    ),
                    (actual.flux_density_t[axis], expected.flux_density_t[axis]),
                ] {
                    let b = ComplexPhasor {
                        real: b.real * phase.real - b.imaginary * phase.imaginary,
                        imaginary: b.real * phase.imaginary + b.imaginary * phase.real,
                    };
                    assert!(difference(a, b) < 1e-15);
                }
            }
        }
        assert!((result.ohmic_power_w - base.ohmic_power_w).abs() < 1e-10);
    }
}

#[test]
fn source_material_sampling_frequency_and_lossless_gates_fail_closed() {
    let (mesh, input) = fixture::problem(
        ElementOrder::Tet4,
        1,
        ComplexPhasor {
            real: 1.0,
            imaginary: 0.0,
        },
    );
    let reject = |changed: &MaxwellInput| {
        assert!(solve(&mesh, changed, 1e-10).is_err());
    };
    let mut changed = input.clone();
    changed.angular_frequency_rad_per_s = 0.0;
    reject(&changed);
    changed.angular_frequency_rad_per_s = f64::NAN;
    reject(&changed);
    let mut changed = input.clone();
    changed.volumes[0].permittivity_f_per_m = -1.0;
    reject(&changed);
    changed.volumes[0].permittivity_f_per_m = 1.0;
    changed.volumes[0].conductivity_s_per_m = -1.0;
    reject(&changed);
    let mut changed = input.clone();
    changed.volumes.push(changed.volumes[0]);
    reject(&changed);
    let mut changed = input.clone();
    changed.source_current.pop();
    reject(&changed);
    let mut changed = input.clone();
    changed.source_current[0] = changed.source_current[1].clone();
    reject(&changed);
    let mut changed = input.clone();
    changed.source_current[0].current_density_a_per_m2[0].real = f64::INFINITY;
    reject(&changed);
    let mut changed = input.clone();
    changed.angular_frequency_rad_per_s = 1e-200;
    reject(&changed);
    let mut changed = input.clone();
    for source in &mut changed.source_current {
        for value in &mut source.current_density_a_per_m2 {
            value.real *= 1e-200;
            value.imaginary *= 1e-200;
        }
    }
    // Nonzero forcing must never be mistaken for zero through squared-norm underflow.
    reject(&changed);
    let (other_mesh, _) = fixture::problem(
        ElementOrder::Tet4,
        2,
        ComplexPhasor {
            real: 1.0,
            imaginary: 0.0,
        },
    );
    assert!(solve(&other_mesh, &input, 1e-10).is_err());
    assert!(solve(&mesh, &input, 0.0).is_err());
    let mut zero = input.clone();
    for source in &mut zero.source_current {
        for value in &mut source.current_density_a_per_m2 {
            value.real = 0.0;
            value.imaginary = 0.0;
        }
    }
    let zero = solve(&mesh, &zero, 1e-10).unwrap();
    assert_eq!(zero.ohmic_power_w, 0.0);
    assert!(zero.samples.iter().all(|sample| sample
        .electric_field_v_per_m
        .iter()
        .all(|value| value.real == 0.0 && value.imaginary == 0.0)));
}
