use super::*;
use crate::topology::{
    quadrature::{
        tensor_tetrahedron_rule, visit_volume_rule, GAUSS_LEGENDRE_EIGHT_UNIT,
        GAUSS_LEGENDRE_SIXTEEN_UNIT, GAUSS_LEGENDRE_TWELVE_UNIT,
    },
    AnalysisProjection,
};
use std::f64::consts::PI;

#[derive(Debug)]
pub(super) struct Errors {
    pub temperature: f64,
    pub flux: f64,
    pub heat: f64,
    pub power: f64,
    pub temperature_contamination: f64,
    pub flux_contamination: f64,
    pub heat_contamination: f64,
    pub power_contamination: f64,
}

// Direct sine expansion of x(1-x), with k=rho=c=1 and A=q/2. No FE eigenproblem.
fn reference(x: f64, time: f64) -> (f64, f64, f64, f64) {
    let a = GENERATION / 2.0;
    let mut temperature = 0.0;
    let mut flux = 0.0;
    let mut heat = 0.0;
    let mut power = 0.0;
    for n in (1..64).step_by(2) {
        let n = n as f64;
        let decay = (-PI.powi(2) * n.powi(2) * time).exp();
        temperature += 8.0 * a / PI.powi(3) * (n * PI * x).sin() * decay / n.powi(3);
        flux -= 8.0 * a / PI.powi(2) * (n * PI * x).cos() * decay / n.powi(2);
        heat += 16.0 * a / PI.powi(4) * decay / n.powi(4);
        power += 16.0 * a / PI.powi(2) * decay / n.powi(2);
    }
    (temperature, flux, heat, power)
}

fn certify_series_tail(time: f64) {
    // For all integers n>=65, n^-p<=65^-p and successive exp(-c n²) ratios
    // are <= exp(-131c). Including even terms bounds the omitted odd series.
    // Work in logs so underflow is not mistaken for a mathematical zero remainder.
    let c = PI.powi(2) * time;
    let log_geometric = -c * 65.0_f64.powi(2) - (-(-131.0 * c).exp()).ln_1p();
    for (coefficient, p) in [
        (8.0 * (GENERATION / 2.0) / PI.powi(3), 3.0),
        (8.0 * (GENERATION / 2.0) / PI.powi(2), 2.0),
        (16.0 * (GENERATION / 2.0) / PI.powi(4), 4.0),
        (16.0 * (GENERATION / 2.0) / PI.powi(2), 2.0),
    ] {
        assert!(coefficient.ln() + log_geometric - p * 65.0_f64.ln() < (1e-12_f64).ln());
    }
}

struct Fields {
    temperature: Vec<f64>,
    flux: Vec<f64>,
    power: f64,
}

fn extrapolate(coarse: &FeaThermalRunResult, fine: &FeaThermalRunResult) -> Fields {
    let combine = |left: &[f64], right: &[f64]| {
        assert_eq!(left.len(), right.len());
        left.iter().zip(right).map(|(a, b)| 2.0 * b - a).collect()
    };
    Fields {
        temperature: combine(
            coarse
                .temperature_snapshots
                .last()
                .unwrap()
                .as_host_f64()
                .unwrap(),
            fine.temperature_snapshots
                .last()
                .unwrap()
                .as_host_f64()
                .unwrap(),
        ),
        flux: combine(
            coarse
                .heat_flux_snapshots
                .last()
                .unwrap()
                .as_host_f64()
                .unwrap(),
            fine.heat_flux_snapshots
                .last()
                .unwrap()
                .as_host_f64()
                .unwrap(),
        ),
        power: 2.0
            * fine
                .energy_balances
                .last()
                .unwrap()
                .prescribed_boundary_power_w
                .canonical_value()
            - coarse
                .energy_balances
                .last()
                .unwrap()
                .prescribed_boundary_power_w
                .canonical_value(),
    }
}

/// A second Richardson elimination of two first-order extrapolants with O(dt^2) remainders at dt
/// and dt/2: (4 fine - coarse) / 3 leaves O(dt^3).
fn eliminate_second(coarse: &Fields, fine: &Fields) -> Fields {
    let combine = |left: &[f64], right: &[f64]| {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right)
            .map(|(a, b)| (4.0 * b - a) / 3.0)
            .collect()
    };
    Fields {
        temperature: combine(&coarse.temperature, &fine.temperature),
        flux: combine(&coarse.flux, &fine.flux),
        power: (4.0 * fine.power - coarse.power) / 3.0,
    }
}

#[derive(Debug)]
struct Norms {
    temperature_error: f64,
    flux_error: f64,
    heat: f64,
    temperature_difference: f64,
    flux_difference: f64,
    heat_difference: f64,
}

fn integrate(
    mesh: &SolverMeshArtifact,
    field: &Fields,
    coarse: &Fields,
    rule: &[(f64, f64)],
) -> Norms {
    let projection = AnalysisProjection::derive(mesh).unwrap();
    assert_eq!(field.temperature.len(), projection.nodes.len());
    assert_eq!(field.flux.len(), projection.elements.len() * 3);
    let mut norms = Norms {
        temperature_error: 0.0,
        flux_error: 0.0,
        heat: 0.0,
        temperature_difference: 0.0,
        flux_difference: 0.0,
        heat_difference: 0.0,
    };
    let mut volume = 0.0;
    let mut weights = 0usize;
    for (index, element) in projection.elements.iter().enumerate() {
        let coordinates = element
            .node_indices
            .iter()
            .map(|&node| projection.nodes[node].coordinates_m.canonical_value())
            .collect::<Vec<_>>();
        let mut reference_flux_integral = 0.0;
        visit_volume_rule(
            element.order,
            &coordinates,
            tensor_tetrahedron_rule(rule),
            |point| {
                let interpolate = |values: &[f64]| {
                    element
                        .node_indices
                        .iter()
                        .zip(point.shape_values())
                        .map(|(&node, &shape)| shape * (values[node] - REFERENCE))
                        .sum::<f64>()
                };
                let temperature = interpolate(&field.temperature);
                let difference = temperature - interpolate(&coarse.temperature);
                let (exact, flux, _, _) = reference(point.position_m[0], FINAL_TIME);
                let weight = point.volume_weight_m3;
                norms.temperature_error += weight * (temperature - exact).powi(2);
                norms.temperature_difference += weight * difference.powi(2);
                norms.heat += weight * temperature;
                norms.heat_difference += weight * difference;
                reference_flux_integral += weight * flux;
                volume += weight;
                weights += 1;
            },
        )
        .unwrap();
        let mean_flux = reference_flux_integral / element.signed_volume_m3;
        for axis in 0..3 {
            let expected = if axis == 0 { mean_flux } else { 0.0 };
            norms.flux_error +=
                element.signed_volume_m3 * (field.flux[3 * index + axis] - expected).powi(2);
            norms.flux_difference += element.signed_volume_m3
                * (field.flux[3 * index + axis] - coarse.flux[3 * index + axis]).powi(2);
        }
    }
    // Summing n weights carries up to n·ε of rounding error.
    let tolerance = 1e-12_f64.max(weights as f64 * f64::EPSILON);
    assert!(
        (volume - 1.0).abs() < tolerance,
        "{volume} over {weights} weights"
    );
    norms
}

/// Three time levels: one Richardson elimination, with the O(dt^2) remainder measured from the two
/// extrapolants (they differ by 3 times the finer error). Four levels: a second elimination, with
/// the O(dt^3) remainder measured the same way (the two differ by 7 times the finer error).
pub(super) fn measure(mesh: &SolverMeshArtifact, results: &[FeaThermalRunResult]) -> Errors {
    assert!(results.len() == 3 || results.len() == 4);
    certify_series_tail(FINAL_TIME);
    for result in results {
        assert_eq!(
            result.ordered_node_ids,
            mesh.topology
                .nodes
                .iter()
                .map(|node| node.node_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            result.ordered_element_ids,
            mesh.topology
                .volume_elements
                .iter()
                .map(|element| element.element_id)
                .collect::<Vec<_>>()
        );
    }
    let first = results
        .windows(2)
        .map(|pair| extrapolate(&pair[0], &pair[1]))
        .collect::<Vec<_>>();
    let (coarse, fine, remainder_ratio) = if first.len() == 3 {
        (
            eliminate_second(&first[0], &first[1]),
            eliminate_second(&first[1], &first[2]),
            7.0,
        )
    } else {
        let mut first = first.into_iter();
        (first.next().unwrap(), first.next().unwrap(), 3.0)
    };
    let initial = integrate(mesh, &fine, &coarse, &GAUSS_LEGENDRE_EIGHT_UNIT);
    let refined = integrate(mesh, &fine, &coarse, &GAUSS_LEGENDRE_TWELVE_UNIT);
    let norms = integrate(mesh, &fine, &coarse, &GAUSS_LEGENDRE_SIXTEEN_UNIT);
    println!("cooling norm quadrature: eight={initial:?}; twelve={refined:?}; sixteen={norms:?}");
    // The smooth Fourier reference is not polynomial: five-point quadrature did not resolve
    // the coarse-cell squared error to the PREDECLARED integration tolerance. Refine the
    // quadrature, never that tolerance or any physics gate. All reported norms and temporal
    // contamination integrals must agree at the two highest independent positive rules.
    for (name, refined, finest) in [
        (
            "temperature_error",
            refined.temperature_error,
            norms.temperature_error,
        ),
        ("flux_error", refined.flux_error, norms.flux_error),
        ("heat", refined.heat, norms.heat),
        (
            "temperature_difference",
            refined.temperature_difference,
            norms.temperature_difference,
        ),
        (
            "flux_difference",
            refined.flux_difference,
            norms.flux_difference,
        ),
        (
            "heat_difference",
            refined.heat_difference,
            norms.heat_difference,
        ),
    ] {
        assert!(
            (refined - finest).abs() < 1e-8,
            "{name} quadrature unresolved: twelve={refined:e}, sixteen={finest:e}"
        );
    }
    let (_, _, exact_heat, exact_power) = reference(0.0, FINAL_TIME);
    let temperature_scale = GENERATION / 8.0;
    let flux_scale = GENERATION / 2.0;
    Errors {
        temperature: norms.temperature_error.sqrt() / temperature_scale,
        flux: norms.flux_error.sqrt() / flux_scale,
        heat: (norms.heat / exact_heat - 1.0).abs(),
        power: (fine.power / exact_power - 1.0).abs(),
        // For the residual O(dt^p) term, the two extrapolations at dt and dt/2 differ by
        // (2^p - 1) times the finer error. This is a measured contamination estimate, not an
        // exact bound; failure blocks rate qualification rather than silently increasing dt or
        // tolerance.
        temperature_contamination: norms.temperature_difference.sqrt()
            / (remainder_ratio * temperature_scale),
        flux_contamination: norms.flux_difference.sqrt() / (remainder_ratio * flux_scale),
        heat_contamination: norms.heat_difference.abs() / (remainder_ratio * exact_heat),
        power_contamination: (fine.power - coarse.power).abs() / (remainder_ratio * exact_power),
    }
}
