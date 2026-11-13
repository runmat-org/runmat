use super::linear_algebra::LinearAlgebraBackend;
use crate::algebra::dot_canonical;

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuReferenceBackend;

impl LinearAlgebraBackend for CpuReferenceBackend {
    fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        dot_canonical(a, b)
    }

    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }

    fn vec_sub(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::dot_scoped,
        planning::resource::{LocalExecutionPlan, LocalThreadCount},
    };

    #[test]
    fn reference_and_scoped_dot_use_the_same_canonical_reduction() {
        let left = (0..1_031)
            .map(|index| if index % 2 == 0 { 1.0e12 } else { -1.0e12 })
            .collect::<Vec<_>>();
        let right = (0..left.len())
            .map(|index| 1.0 + index as f64 * 1.0e-9)
            .collect::<Vec<_>>();
        let expected = CpuReferenceBackend.dot(&left, &right);
        for workers in [2, 3, 4, 7] {
            let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(workers).unwrap()).unwrap();
            assert_eq!(dot_scoped(&left, &right, plan).unwrap(), expected);
        }
    }
}
