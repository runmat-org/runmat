//! Host helper for the Monte Carlo evolution loop when GPU acceleration is
//! unavailable.

use crate::builtins::common::random;
use runmat_builtins::Tensor;

pub fn stochastic_evolution_host(
    tensor: &mut Tensor,
    drift: f64,
    scale: f64,
    steps: u32,
) -> Result<(), String> {
    if tensor.data.is_empty() || steps == 0 {
        return Ok(());
    }

    let len = tensor.data.len();
    for _ in 0..steps {
        let samples = random::generate_normal(len, "stochastic_evolution_host")?;
        for (value, noise) in tensor.data.iter_mut().zip(samples.iter()) {
            let term = drift + scale * noise;
            *value *= term.exp();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::random;

    #[test]
    fn cpu_fallback_handles_zero_scale() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let mut tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor");
        stochastic_evolution_host(&mut tensor, 0.1, 0.0, 3).expect("evolve");
        let expected = (0..2)
            .map(|i| (i as f64 + 1.0) * (0.1f64 * 3.0).exp())
            .collect::<Vec<_>>();
        assert_eq!(tensor.shape, vec![2, 1]);
        for (got, exp) in tensor.data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12, "got {got} expected {exp}");
        }
    }
}
