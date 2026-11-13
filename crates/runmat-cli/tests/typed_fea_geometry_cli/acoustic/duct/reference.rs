use runmat_analysis_fea::topology::AnalysisProjection;

fn multiply(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}
fn wave(x: f64) -> [f64; 2] {
    let phase = std::f64::consts::TAU * 0.2 * x / 2.0;
    multiply([1.0, 0.4], [phase.cos(), -phase.sin()])
}
// Uniform affine tetrahedron moments, not centroid samples:
// E[x^n] = 6 h_n(x0,x1,x2,x3) / ((n+1)(n+2)(n+3)).
// With |kx| <= pi/5, the 32-term exponential remainder is below 1e-40.
fn mean_wave(xs: [f64; 4]) -> [f64; 2] {
    let k = std::f64::consts::TAU * 0.2 / 2.0;
    let mut h = [0.0; 32];
    h[0] = 1.0;
    for x in xs {
        for n in 1..32 {
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
        for component in 0..2 {
            mean[component] += coefficient[component] * moment;
        }
    }
    multiply([1.0, 0.4], mean)
}

pub(super) fn errors(
    projection: &AnalysisProjection,
    pressure: &[f64],
    velocity: &[f64],
    intensity: &[[f64; 3]],
) -> (f64, f64, f64) {
    let p_error = projection
        .nodes
        .iter()
        .zip(pressure.chunks_exact(2))
        .map(|(node, p)| {
            let exact = wave(node.coordinates_m.canonical_value()[0]);
            (p[0] - exact[0]).powi(2) + (p[1] - exact[1]).powi(2)
        })
        .sum::<f64>()
        / (1.16 * projection.nodes.len() as f64);
    let v_error = projection
        .elements
        .iter()
        .zip(velocity.chunks_exact(6))
        .map(|(element, v)| {
            let xs = std::array::from_fn(|i| {
                projection.nodes[element.node_indices[i]]
                    .coordinates_m
                    .canonical_value()[0]
            });
            // The physical mean formula requires affine geometry, including Tet10 edges.
            if element.node_indices.len() == 10 {
                for (edge, pair) in runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS
                    .iter()
                    .enumerate()
                {
                    let mid = projection.nodes[element.node_indices[4 + edge]]
                        .coordinates_m
                        .canonical_value();
                    let a = projection.nodes[element.node_indices[pair[0]]]
                        .coordinates_m
                        .canonical_value();
                    let b = projection.nodes[element.node_indices[pair[1]]]
                        .coordinates_m
                        .canonical_value();
                    for (axis, coordinate) in mid.iter().enumerate() {
                        assert!((*coordinate - 0.5 * (a[axis] + b[axis])).abs() < 1e-12);
                    }
                }
            }
            let exact = mean_wave(xs);
            (v[0] - exact[0] / 6.0).powi(2)
                + (v[1] - exact[1] / 6.0).powi(2)
                + v[2..].iter().map(|value| value.powi(2)).sum::<f64>()
        })
        .sum::<f64>()
        / projection.elements.len() as f64;
    assert_eq!(intensity.len(), projection.elements.len());
    let i_error = intensity
        .iter()
        .map(|i| (i[0] - 1.16 / 12.0).powi(2) + i[1].powi(2) + i[2].powi(2))
        .sum::<f64>()
        / intensity.len() as f64;
    (p_error.sqrt(), v_error.sqrt(), i_error.sqrt())
}
