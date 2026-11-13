//! Fixed three-dimensional sizing decomposition, with a reconstruction residual.
//! This is a metric construction aid; independent mesh acceptance remains mandatory.
use super::super::{MetricContractError, MetricTensor3};

pub(super) struct Spectrum {
    pub values: [f64; 3],
    pub vectors: [[f64; 3]; 3],
    pub error: f64,
}
pub(super) fn difference(
    b: MetricTensor3,
    a: MetricTensor3,
) -> Result<Spectrum, MetricContractError> {
    let matrix = |m: MetricTensor3| [[m.xx, m.xy, m.xz], [m.xy, m.yy, m.yz], [m.xz, m.yz, m.zz]];
    let aa = matrix(a);
    let bb = matrix(b);
    let original: [[f64; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| bb[i][j] - aa[i][j]));
    let scale = original
        .iter()
        .flatten()
        .copied()
        .map(f64::abs)
        .fold(0., f64::max);
    let mut q = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    if scale == 0. {
        return Ok(Spectrum {
            values: [0.; 3],
            vectors: q,
            error: 0.,
        });
    }
    if !scale.is_finite() {
        return Err(MetricContractError::invalid(
            "metric difference",
            "overflow",
        ));
    }
    let mut d = original.map(|row| row.map(|v| v / scale));
    // Largest-pivot Jacobi: constant storage and bounded work for a 3x3 tensor.
    for _ in 0..48 {
        let (p, r) = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .max_by(|&(i, j), &(k, l)| d[i][j].abs().total_cmp(&d[k][l].abs()))
            .unwrap();
        if d[p][r].abs() <= 8. * f64::EPSILON {
            break;
        }
        let tau = (d[r][r] - d[p][p]) / (2. * d[p][r]);
        let t = if tau >= 0. {
            1. / (tau + tau.hypot(1.))
        } else {
            -1. / (-tau + tau.hypot(1.))
        };
        let c = 1. / t.hypot(1.);
        let s = t * c;
        let off = d[p][r];
        d[p][p] -= t * off;
        d[r][r] += t * off;
        d[p][r] = 0.;
        d[r][p] = 0.;
        for k in 0..3 {
            if k != p && k != r {
                let x = d[k][p];
                let y = d[k][r];
                d[k][p] = c * x - s * y;
                d[p][k] = d[k][p];
                d[k][r] = s * x + c * y;
                d[r][k] = d[k][r];
            }
            let x = q[k][p];
            let y = q[k][r];
            q[k][p] = c * x - s * y;
            q[k][r] = s * x + c * y;
        }
    }
    let values = std::array::from_fn(|i| d[i][i] * scale);
    let vectors = std::array::from_fn(|i| std::array::from_fn(|j| q[j][i]));
    let residual = (0..3)
        .map(|i| {
            (0..3)
                .map(|j| {
                    let reconstructed = (0..3).map(|k| q[i][k] * values[k] * q[j][k]).sum::<f64>();
                    (original[i][j] - reconstructed).abs()
                })
                .sum::<f64>()
        })
        .fold(0., f64::max);
    let input_scale = a.maximum_row_density().max(b.maximum_row_density());
    let error = residual * 4. + input_scale * (512. * f64::EPSILON);
    if values.iter().any(|v| !v.is_finite()) || !error.is_finite() {
        return Err(MetricContractError::invalid(
            "metric spectrum",
            "nonfinite reconstruction",
        ));
    }
    Ok(Spectrum {
        values,
        vectors,
        error,
    })
}
