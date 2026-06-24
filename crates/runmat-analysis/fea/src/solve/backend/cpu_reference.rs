use super::linear_algebra::LinearAlgebraBackend;

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuReferenceBackend;

impl LinearAlgebraBackend for CpuReferenceBackend {
    fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
