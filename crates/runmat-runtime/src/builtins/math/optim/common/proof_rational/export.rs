//! Return an exact proof result to an existing library-rational consumer.
//! No rounded scalar conversion or second arithmetic implementation.
use super::*;
impl Rational {
    pub async fn export(
        &self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<num_rational::BigRational> {
        c.completed(0).await?;
        let parts = self.materialize(policy, c).await?;
        let numerator = magnitude::copy(&parts.numerator, policy, c).await?;
        let denominator = magnitude::copy(&parts.denominator, policy, c).await?;
        // Existing library ownership ingress; limb copying above cooperates.
        // Its constructors are not claimed to be hard-preemptible kernels.
        let numerator = num_bigint::BigInt::new(
            if self.is_zero() {
                num_bigint::Sign::NoSign
            } else if self.is_negative() {
                num_bigint::Sign::Minus
            } else {
                num_bigint::Sign::Plus
            },
            numerator,
        );
        let denominator = num_bigint::BigInt::new(num_bigint::Sign::Plus, denominator);
        // materialize() has a positive denominator and all arithmetic results
        // are already reduced. Do not repeat an opaque library GCD here.
        let value = num_rational::BigRational::new_raw(numerator, denominator);
        c.completed(0).await?;
        Ok(value)
    }
}
