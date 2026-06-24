pub trait LinearAlgebraBackend {
    fn dot(&self, a: &[f64], b: &[f64]) -> f64;
    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]);
    fn vec_sub(&self, a: &[f64], b: &[f64]) -> Vec<f64>;
}
