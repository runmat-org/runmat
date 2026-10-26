//! Cross-cancelled exact rational operations. No unbounded synchronous operator.
use super::*;
impl Rational {
    pub async fn add(
        &self,
        other: &Self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        c.completed(0).await?;
        if self.is_zero() {
            return Ok(other.clone());
        }
        if other.is_zero() {
            return Ok(self.clone());
        }
        let a = self.materialize(policy, c).await?;
        let b = other.materialize(policy, c).await?;
        let gcd = magnitude::gcd(&a.denominator, &b.denominator, policy, c).await?;
        let ad = magnitude::quotient(&a.denominator, &gcd, policy, c).await?;
        let bd = magnitude::quotient(&b.denominator, &gcd, policy, c).await?;
        let left = magnitude::multiply(&a.numerator, &bd, policy, c).await?;
        let right = magnitude::multiply(&b.numerator, &ad, policy, c).await?;
        let (negative, numerator) = if self.negative == other.negative {
            (
                self.negative,
                magnitude::add(&left, &right, policy, c).await?,
            )
        } else {
            match magnitude::compare(&left, &right, c).await? {
                Ordering::Equal => return Ok(Self::zero()),
                Ordering::Greater => (
                    self.negative,
                    magnitude::subtract(&left, &right, policy, c).await?,
                ),
                Ordering::Less => (
                    other.negative,
                    magnitude::subtract(&right, &left, policy, c).await?,
                ),
            }
        };
        let denominator = magnitude::multiply(&ad, &b.denominator, policy, c).await?;
        Self::normalized(negative, numerator, denominator, policy, c).await
    }
    pub async fn sub(
        &self,
        other: &Self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        self.add(&other.negated(), policy, c).await
    }
    pub async fn mul(
        &self,
        other: &Self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        c.completed(0).await?;
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero());
        }
        let a = self.materialize(policy, c).await?;
        let b = other.materialize(policy, c).await?;
        let g = magnitude::gcd(&a.numerator, &b.denominator, policy, c).await?;
        let h = magnitude::gcd(&b.numerator, &a.denominator, policy, c).await?;
        let an = magnitude::quotient(&a.numerator, &g, policy, c).await?;
        let bd = magnitude::quotient(&b.denominator, &g, policy, c).await?;
        let bn = magnitude::quotient(&b.numerator, &h, policy, c).await?;
        let ad = magnitude::quotient(&a.denominator, &h, policy, c).await?;
        Ok(Self::parts(
            self.negative != other.negative,
            magnitude::multiply(&an, &bn, policy, c).await?,
            magnitude::multiply(&ad, &bd, policy, c).await?,
        ))
    }
    pub async fn div(
        &self,
        other: &Self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        c.completed(0).await?;
        if other.is_zero() {
            return Err(policy.invalid("exact proof divisor is zero"));
        }
        let b = other.materialize(policy, c).await?;
        let reciprocal = Self {
            negative: other.negative,
            storage: Storage::Parts(Rc::new(Parts {
                numerator: magnitude::copy(&b.denominator, policy, c).await?,
                denominator: magnitude::copy(&b.numerator, policy, c).await?,
            })),
        };
        self.mul(&reciprocal, policy, c).await
    }
}
