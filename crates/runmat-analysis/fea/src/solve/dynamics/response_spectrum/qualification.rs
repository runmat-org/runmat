use runmat_analysis_core::{DirectionalCombination, ModalCombination};

use super::combine::{combine_directional_responses, combine_modal_responses};

const RELATIVE_TOLERANCE: f64 = 2.0e-13;

// witness: FEA-DYNAMICS-RESPONSE-001/spectrum_random_recovery @
//   linear_structural_dynamics/mesh_independent/numerical_kernel;
//   check: modal-combination-rules; boundary: Scalar signed synthetic modal vectors verify SRSS, absolute sum and CQC against
//   independent formulas. Mesh-independent arithmetic only; no missing-mass, directional or
//   random-vibration claim.
#[test]
fn modal_combination_matches_independent_quadratic_form_references() {
    let frequencies = [10.0, 11.0];
    let damping = 0.05;
    let signed = [3.0, -4.0];
    let vectors = signed.map(|value| vec![value]);

    assert_close(
        combine_modal_responses(&vectors, &frequencies, damping, ModalCombination::Srss).unwrap()
            [0],
        5.0,
    );
    assert_close(
        combine_modal_responses(
            &vectors,
            &frequencies,
            damping,
            ModalCombination::AbsoluteSum,
        )
        .unwrap()[0],
        7.0,
    );

    let correlation = independent_cqc_correlation(frequencies[0], frequencies[1], damping);
    let expected =
        (signed[0].powi(2) + signed[1].powi(2) + 2.0 * correlation * signed[0] * signed[1]).sqrt();
    assert_close(
        combine_modal_responses(
            &vectors,
            &frequencies,
            damping,
            ModalCombination::CompleteQuadraticCombination,
        )
        .unwrap()[0],
        expected,
    );

    let reinforcing = [vec![3.0], vec![4.0]];
    let expected = (25.0 + 24.0 * correlation).sqrt();
    assert_close(
        combine_modal_responses(
            &reinforcing,
            &frequencies,
            damping,
            ModalCombination::CompleteQuadraticCombination,
        )
        .unwrap()[0],
        expected,
    );
}

#[test]
fn ten_percent_boundary_and_directional_rules_match_hand_computed_values() {
    let cancellation = [vec![3.0], vec![-4.0]];
    assert_close(
        combine_modal_responses(
            &cancellation,
            &[10.0, 10.9],
            0.05,
            ModalCombination::TenPercent,
        )
        .unwrap()[0],
        1.0,
    );
    assert_close(
        combine_modal_responses(
            &cancellation,
            &[10.0, 11.1],
            0.05,
            ModalCombination::TenPercent,
        )
        .unwrap()[0],
        5.0,
    );

    let x = [3.0, 1.0];
    let y = [-4.0, 10.0];
    let z = [12.0, 1.0];
    let directions = [&x[..], &y[..], &z[..]];
    let srss = combine_directional_responses(&directions, DirectionalCombination::Srss).unwrap();
    assert_close(srss[0], 13.0);
    assert_close(srss[1], 102.0_f64.sqrt());
    let absolute =
        combine_directional_responses(&directions, DirectionalCombination::AbsoluteSum).unwrap();
    assert_close(absolute[0], 19.0);
    assert_close(absolute[1], 12.0);
    let percentage = combine_directional_responses(
        &directions,
        DirectionalCombination::PercentageRule {
            primary: 1.0,
            secondary: 0.4,
        },
    )
    .unwrap();
    assert_close(percentage[0], 14.8);
    assert_close(percentage[1], 10.8);
}

#[test]
fn cqc_is_scale_linear_and_symmetric_over_a_wide_physical_range() {
    let frequencies = [7.0, 7.4, 23.0];
    let values = [2.5, -1.25, 0.75];
    let oracle = independent_cqc_quadratic_form(&values, &frequencies, 0.03);
    for scale in [1.0e-12, 1.0, 1.0e12] {
        let vectors = values.map(|value| vec![scale * value]);
        let actual = combine_modal_responses(
            &vectors,
            &frequencies,
            0.03,
            ModalCombination::CompleteQuadraticCombination,
        )
        .unwrap()[0];
        assert_close(actual, scale * oracle);
    }
    for left in frequencies {
        for right in frequencies {
            assert_close(
                independent_cqc_correlation(left, right, 0.03),
                independent_cqc_correlation(right, left, 0.03),
            );
        }
    }
}

fn independent_cqc_quadratic_form(values: &[f64], frequencies: &[f64], damping: f64) -> f64 {
    let variance = values
        .iter()
        .enumerate()
        .flat_map(|(left, left_value)| {
            values.iter().enumerate().map(move |(right, right_value)| {
                left_value
                    * independent_cqc_correlation(frequencies[left], frequencies[right], damping)
                    * right_value
            })
        })
        .sum::<f64>();
    variance.max(0.0).sqrt()
}

fn independent_cqc_correlation(left_hz: f64, right_hz: f64, damping: f64) -> f64 {
    if left_hz == right_hz {
        return 1.0;
    }
    let ratio = right_hz / left_hz;
    8.0 * damping.powi(2) * (1.0 + ratio) * ratio.powf(1.5)
        / ((1.0 - ratio.powi(2)).powi(2) + 4.0 * damping.powi(2) * ratio * (1.0 + ratio).powi(2))
}

fn assert_close(actual: f64, expected: f64) {
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        (actual - expected).abs() <= RELATIVE_TOLERANCE * scale,
        "actual={actual:.17e}, expected={expected:.17e}"
    );
}
