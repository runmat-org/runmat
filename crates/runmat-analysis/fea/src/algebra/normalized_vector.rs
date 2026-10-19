//! Scale-aware unit vector and analytic first/mixed directional derivatives.
use thiserror::Error;
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("nonfinite, zero or unresolvable normalized vector differential")]
pub struct NormalizedVectorError;
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn scaled(vector: [f64; 3]) -> Result<([f64; 3], f64, f64), NormalizedVectorError> {
    if vector.iter().any(|x| !x.is_finite()) {
        return Err(NormalizedVectorError);
    }
    let scale = vector.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(NormalizedVectorError);
    }
    let v = vector.map(|x| x / scale);
    let length = v[0].hypot(v[1]).hypot(v[2]);
    Ok((v.map(|x| x / length), scale, length))
}
pub fn normalized_vector(vector: [f64; 3]) -> Result<[f64; 3], NormalizedVectorError> {
    Ok(scaled(vector)?.0)
}
#[derive(Clone, Copy, Debug)]
pub struct NormalizedVectorDifferential {
    pub unit: [f64; 3],
    pub first: [[f64; 3]; 2],
    pub mixed: [f64; 3],
}
impl NormalizedVectorDifferential {
    pub fn derive(
        vector: [f64; 3],
        a: [f64; 3],
        b: [f64; 3],
        ab: [f64; 3],
    ) -> Result<Self, NormalizedVectorError> {
        if a.iter()
            .chain(b.iter())
            .chain(ab.iter())
            .any(|x| !x.is_finite())
        {
            return Err(NormalizedVectorError);
        }
        let (u, scale, length) = scaled(vector)?;
        let a = a.map(|x| x / scale);
        let b = b.map(|x| x / scale);
        let ab = ab.map(|x| x / scale);
        let first = [a, b].map(|v| std::array::from_fn(|i| (v[i] - u[i] * dot(u, v)) / length));
        let mixed = std::array::from_fn(|i| {
            (ab[i] - first[1][i] * dot(u, a) - u[i] * (dot(first[1], a) + dot(u, ab))) / length
                - first[0][i] * dot(u, b) / length
        });
        let result = Self {
            unit: u,
            first,
            mixed,
        };
        if result
            .first
            .iter()
            .flatten()
            .chain(result.mixed.iter())
            .any(|x| !x.is_finite())
        {
            return Err(NormalizedVectorError);
        }
        Ok(result)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn closed_orthogonal_unit_derivatives_and_scale_invariance() {
        for scale in [1e-300, 1., 1e300] {
            let v = NormalizedVectorDifferential::derive(
                [scale, 0., 0.],
                [0., scale, 0.],
                [0., 0., scale],
                [0.; 3],
            )
            .unwrap();
            assert_eq!(v.unit, [1., 0., 0.]);
            assert_eq!(v.first, [[0., 1., 0.], [0., 0., 1.]]);
            assert_eq!(v.mixed, [0.; 3]);
            let v = NormalizedVectorDifferential::derive(
                [scale, 0., 0.],
                [0., scale, 0.],
                [0., scale, 0.],
                [0.; 3],
            )
            .unwrap();
            assert_eq!(v.mixed, [-1., 0., 0.]);
        }
    }
    #[test]
    fn mixed_derivative_matches_two_step_sizes_without_using_its_formula() {
        let v = [1.2, -0.4, 0.7];
        let a = [0.2, 0.5, -0.3];
        let b = [-0.4, 0.1, 0.2];
        let ab = [0.3, -0.1, 0.4];
        let result = NormalizedVectorDifferential::derive(v, a, b, ab).unwrap();
        for h in [1e-3, 3e-4] {
            let eval = |s: f64, t: f64| {
                normalized_vector(std::array::from_fn(|i| {
                    v[i] + s * a[i] + t * b[i] + s * t * ab[i]
                }))
                .unwrap()
            };
            let pp = eval(h, h);
            let pm = eval(h, -h);
            let mp = eval(-h, h);
            let mm = eval(-h, -h);
            for i in 0..3 {
                assert!(
                    (result.mixed[i] - (pp[i] - pm[i] - mp[i] + mm[i]) / (4. * h * h)).abs() < 1e-6
                );
            }
        }
    }
}
