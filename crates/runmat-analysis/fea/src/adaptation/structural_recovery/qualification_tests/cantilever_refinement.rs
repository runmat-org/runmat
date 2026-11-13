//! Original cantilever refinement gates; diagnostic output keeps their verdicts separate.
use super::*;

/// Williams' clamped-free wedge exponent for a 90 degree edge in plane strain at nu = 0.3:
/// the smallest root of the clamped-free characteristic determinant (displacement field
/// `r^lambda`). The clamped root face of the unit cube meets its free faces along such edges.
const CLAMPED_FREE_EDGE_EXPONENT: f64 = 0.71117;

/// Observed convergence order of a displacement functional on nonuniform refinement, from
/// `(u3 - u2) / (u2 - u1) = (h2^p - h3^p) / (h1^p - h2^p)` with `h = 1 / divisions`.
fn observed_order(divisions: [f64; 3], values: [f64; 3]) -> f64 {
    let ratio = (values[2] - values[1]) / (values[1] - values[0]);
    let model = |order: f64| {
        let h = divisions.map(|division| division.recip().powf(order));
        (h[1] - h[2]) / (h[0] - h[1])
    };
    let (mut low, mut high) = (0.05, 8.0);
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        // The model ratio decreases monotonically as the order increases.
        if model(middle) > ratio {
            low = middle;
        } else {
            high = middle;
        }
    }
    0.5 * (low + high)
}

pub(super) fn qualify() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut errors = Vec::new();
    let mut peak_displacements = Vec::new();
    for divisions in [1, 2, 3, 4] {
        let mesh = structured_cantilever(divisions, ElementOrder::Tet10);
        let result = crate::run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                solver_mesh: Some(mesh.clone()),
                require_solver_mesh_for_solid: true,
                ..LinearStaticSolveOptions::default()
            },
        )
        .expect("the refined physical Tet10 cantilever must solve");
        let stress = result
            .field(FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
            .expect("the solved field must retain physical integration-point stress");
        peak_displacements.push(
            result
                .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
                .unwrap()
                .as_host_f64()
                .unwrap()
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max),
        );
        assert_eq!(
            stress.shape,
            vec![mesh.topology.volume_elements.len() * 4, 6]
        );
        let stress_values = stress.as_host_f64().unwrap();
        let stress_peak = stress_values.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let stress_range = std::array::from_fn::<_, 6, _>(|component| {
            stress_values
                .chunks_exact(6)
                .map(|v| v[component])
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), v| {
                    (low.min(v), high.max(v))
                })
        });
        eprintln!("original cantilever divisions={divisions} stress_peak={stress_peak} ranges={stress_range:?}");
        errors.push(
            estimate_structural_recovery_error(
                &mesh,
                "elements",
                stress,
                StructuralRecoveryEstimatorOptions::default(),
            )
            .expect("the solved point field must drive the estimator")
            .total_error,
        );
    }

    eprintln!(
        "original cantilever recovery errors={errors:?} peak_displacements={peak_displacements:?}"
    );
    assert!(
        errors.windows(2).all(|pair| pair[1] < 0.9 * pair[0]),
        "{errors:?}"
    );
    // A point functional converges like the energy error, h^(2 lambda), when the solution has an
    // r^lambda edge singularity; uniform refinement cannot exceed that rate whatever the element
    // order. Both consecutive triples must be monotone and converge at no less than 90% of it.
    let floor = 0.9 * 2.0 * CLAMPED_FREE_EDGE_EXPONENT;
    let orders = [0, 1].map(|start| {
        let values = [0, 1, 2].map(|offset| peak_displacements[start + offset]);
        assert!(
            (values[2] - values[1]).abs() < (values[1] - values[0]).abs()
                && (values[2] - values[1]) * (values[1] - values[0]) > 0.0,
            "{peak_displacements:?}"
        );
        observed_order([0, 1, 2].map(|offset| (start + offset + 1) as f64), values)
    });
    eprintln!("original cantilever observed displacement orders={orders:?} floor={floor}");
    assert!(
        orders.iter().all(|order| *order >= floor),
        "orders={orders:?} floor={floor} peaks={peak_displacements:?}"
    );
}
