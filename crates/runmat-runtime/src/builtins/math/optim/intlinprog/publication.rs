//! One binary64 publication policy shared by outputs and enabled observers.
use super::*;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

pub(super) fn number(value: &BigRational) -> f64 {
    value.to_f64().unwrap_or_else(|| {
        if value.is_negative() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }
    })
}
pub(in crate::builtins::math::optim) fn nonnegative(value: &BigRational) -> f64 {
    let result = number(value);
    if result == 0.0 && !value.is_zero() {
        f64::from_bits(1)
    } else {
        result
    }
}
pub(super) struct BinaryPoint {
    pub values: Vec<f64>,
    domains_admitted: bool,
}
impl BinaryPoint {
    pub async fn prepare(
        witness: &[BigRational],
        integers: &[usize],
        semis: &[domains::Semi],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let Some(values) = proofs::output_point(witness, cooperation).await? else {
            return Ok(None);
        };
        let mut domains_admitted = true;
        for &column in integers {
            let actual = BigRational::from_float(values[column]).expect("finite public coordinate");
            domains_admitted &= actual.is_integer() && actual == witness[column];
            cooperation.completed(1).await?;
        }
        for domain in semis {
            let actual =
                BigRational::from_float(values[domain.column]).expect("finite public coordinate");
            domains_admitted &= domain.contains(&actual);
            cooperation.completed(1).await?;
        }
        Ok(Some(Self {
            values,
            domains_admitted,
        }))
    }
    /// Outputs may reuse the consumed witness coordinates. Observers supply
    /// temporary binary-coordinate scratch, never mutate the retained proof.
    pub async fn evaluate<V: compiled::Validation>(
        &self,
        scratch: &mut [BigRational],
        node: &compiled::Node<'_>,
        controls: &controls::Controls,
        validation: &mut V,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Evaluation> {
        if scratch.len() != self.values.len() {
            return Err(numerical("public-point geometry changed"));
        }
        for (actual, slot) in self.values.iter().zip(scratch.iter_mut()) {
            *slot = BigRational::from_float(*actual).expect("finite public coordinate");
            cooperation.completed(1).await?;
        }
        validation
            .publication(
                compiled::Publication {
                    node,
                    values: &self.values,
                    exact_values: scratch,
                    domains_admitted: self.domains_admitted,
                    tolerance: controls.constraint_tolerance,
                    cutoff: controls.cutoff.as_ref(),
                },
                cooperation,
            )
            .await
    }
}
pub(in crate::builtins::math::optim) struct Evaluation {
    pub objective: f64,
    pub violation: f64,
    pub admitted: bool,
}
