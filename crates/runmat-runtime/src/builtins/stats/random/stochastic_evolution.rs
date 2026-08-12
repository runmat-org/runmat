//! Host helper for the Monte Carlo evolution loop when GPU acceleration is
//! unavailable.

use crate::builtins::common::random;
use crate::BuiltinResult;
use runmat_builtins::{NumericDType, NumericScalar, Tensor};

const NAME: &str = "stochastic_evolution";

pub fn stochastic_evolution_host(
    tensor: &mut Tensor,
    drift: f64,
    scale: f64,
    steps: u32,
) -> BuiltinResult<()> {
    if tensor.is_empty() || steps == 0 {
        return Ok(());
    }

    let len = tensor.len();
    match tensor.numeric_dtype() {
        NumericDType::F64 => {
            for _ in 0..steps {
                let samples = random::generate_normal(len, NAME)?;
                for (index, noise) in samples.into_iter().enumerate() {
                    let NumericScalar::F64(value) = tensor
                        .numeric_value_at(index)
                        .expect("double tensor index is in bounds")
                    else {
                        unreachable!("double tensor must expose double scalar storage");
                    };
                    let term = drift + scale * noise;
                    tensor
                        .set_numeric_assignment_at(index, NumericScalar::F64(value * term.exp()))?;
                }
            }
        }
        NumericDType::F32 => {
            let drift = drift as f32;
            let scale = scale as f32;
            for _ in 0..steps {
                let samples = random::generate_normal(len, NAME)?;
                for (index, noise) in samples.into_iter().enumerate() {
                    let NumericScalar::F32(value) = tensor
                        .numeric_value_at(index)
                        .expect("single tensor index is in bounds")
                    else {
                        unreachable!("single tensor must expose single scalar storage");
                    };
                    let term = drift + scale * noise as f32;
                    tensor
                        .set_numeric_assignment_at(index, NumericScalar::F32(value * term.exp()))?;
                }
            }
        }
        dtype => {
            return Err(format!(
                "{NAME}: optimized evolution requires single or double state, got {}",
                dtype.class_name()
            )
            .into());
        }
    }

    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::random;
    use runmat_builtins::IntegerStorage;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cpu_fallback_handles_zero_scale() {
        let _guard = random::test_guard();
        random::reset_rng();
        let mut tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor");
        stochastic_evolution_host(&mut tensor, 0.1, 0.0, 3).expect("evolve");
        let expected = (0..2)
            .map(|i| (i as f64 + 1.0) * (0.1f64 * 3.0).exp())
            .collect::<Vec<_>>();
        assert_eq!(tensor.shape, vec![2, 1]);
        for (got, exp) in tensor.materialize_f64().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12, "got {got} expected {exp}");
        }
    }

    #[test]
    fn cpu_fallback_preserves_native_single_rounding() {
        let _guard = random::test_guard();
        random::reset_rng();
        let mut tensor = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).expect("single tensor");
        stochastic_evolution_host(&mut tensor, 0.1, 0.0, 3).expect("evolve");
        let factor = (0.1f32).exp();
        let expected = vec![factor * factor * factor, 2.0 * factor * factor * factor];
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        let actual = (0..tensor.len())
            .map(|index| match tensor.numeric_value_at(index) {
                Some(NumericScalar::F32(value)) => value,
                other => panic!("expected native single value, got {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn cpu_fallback_rejects_nonempty_integer_state_without_mutating_it() {
        let mut tensor =
            Tensor::new_integer(IntegerStorage::I64(vec![3]), vec![1, 1]).expect("integer state");
        let err = stochastic_evolution_host(&mut tensor, 0.1, 0.0, 1)
            .expect_err("integer state must not enter floating optimizer");
        assert!(err.to_string().contains("requires single or double state"));
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::I64(vec![3]))
        );
    }

    #[test]
    fn zero_steps_and_empty_state_are_semantic_noops_for_any_class() {
        let mut integer =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("integer");
        stochastic_evolution_host(&mut integer, 0.1, 0.2, 0).expect("zero steps");
        assert_eq!(
            integer.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX]))
        );

        let mut empty =
            Tensor::new_integer(IntegerStorage::I8(Vec::new()), vec![0, 1]).expect("empty");
        stochastic_evolution_host(&mut empty, 0.1, 0.2, 3).expect("empty state");
        assert!(empty.is_empty());
    }
}
