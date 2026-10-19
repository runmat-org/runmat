use runmat_meshing_size::metric::MetricTensor3;

pub(super) fn metric_circumcenter(
    points: [[f64; 3]; 4],
    metric: MetricTensor3,
) -> Option<[f64; 3]> {
    if is_isotropic(metric) {
        return euclidean_circumcenter(points);
    }

    let l00 = metric.xx.sqrt();
    let l10 = metric.xy / l00;
    let l20 = metric.xz / l00;
    let l11 = (metric.yy - l10 * l10).sqrt();
    let l21 = (metric.yz - l20 * l10) / l11;
    let l22 = (metric.zz - l20 * l20 - l21 * l21).sqrt();
    if ![l00, l10, l20, l11, l21, l22]
        .into_iter()
        .all(f64::is_finite)
    {
        return None;
    }
    let transformed = points.map(|point| {
        let relative = subtract(point, points[0]);
        [
            l00 * relative[0] + l10 * relative[1] + l20 * relative[2],
            l11 * relative[1] + l21 * relative[2],
            l22 * relative[2],
        ]
    });
    let transformed_center = euclidean_circumcenter_from_origin(transformed)?;
    let z = transformed_center[2] / l22;
    let y = (transformed_center[1] - l21 * z) / l11;
    let x = (transformed_center[0] - l10 * y - l20 * z) / l00;
    let offset = [x, y, z];
    offset
        .into_iter()
        .all(f64::is_finite)
        .then(|| std::array::from_fn(|axis| points[0][axis] + offset[axis]))
}

fn is_isotropic(metric: MetricTensor3) -> bool {
    metric.xy == 0.0
        && metric.xz == 0.0
        && metric.yz == 0.0
        && metric.xx == metric.yy
        && metric.yy == metric.zz
}

fn euclidean_circumcenter(points: [[f64; 3]; 4]) -> Option<[f64; 3]> {
    let relative = points.map(|point| subtract(point, points[0]));
    let offset = euclidean_circumcenter_from_origin(relative)?;
    Some(std::array::from_fn(|axis| points[0][axis] + offset[axis]))
}

fn euclidean_circumcenter_from_origin(points: [[f64; 3]; 4]) -> Option<[f64; 3]> {
    let mut matrix = [[0.0; 3]; 3];
    let mut right = [0.0; 3];
    for row in 0..3 {
        matrix[row] = points[row + 1];
        right[row] = 0.5
            * points[row + 1]
                .into_iter()
                .map(|value| value * value)
                .sum::<f64>();
    }
    solve(matrix, right)
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn solve(mut matrix: [[f64; 3]; 3], mut right: [f64; 3]) -> Option<[f64; 3]> {
    for pivot in 0..3 {
        let pivot_row = (pivot..3).max_by(|left, right_row| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right_row][pivot].abs())
        })?;
        if matrix[pivot_row][pivot] == 0.0 || !matrix[pivot_row][pivot].is_finite() {
            return None;
        }
        matrix.swap(pivot, pivot_row);
        right.swap(pivot, pivot_row);
        for row in (pivot + 1)..3 {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for column in pivot..3 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            right[row] -= factor * right[pivot];
        }
    }
    let mut result = [0.0; 3];
    for row in (0..3).rev() {
        let remainder = ((row + 1)..3)
            .map(|column| matrix[row][column] * result[column])
            .sum::<f64>();
        result[row] = (right[row] - remainder) / matrix[row][row];
    }
    result.into_iter().all(f64::is_finite).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_scale_cannot_change_the_circumcenter() {
        let points = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 6.0],
        ];
        let first =
            metric_circumcenter(points, MetricTensor3::isotropic_length_m(0.25).unwrap()).unwrap();
        let second =
            metric_circumcenter(points, MetricTensor3::isotropic_length_m(4.0).unwrap()).unwrap();
        assert_eq!(first, [1.0, 2.0, 3.0]);
        assert_eq!(first, second);
    }
}
