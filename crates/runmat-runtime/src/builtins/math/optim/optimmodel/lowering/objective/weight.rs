//! Scalar objective weights share the existing full-range cooperative proof
//! owner. The original graph is not expanded into a second expression tree.
use super::*;
impl Weight {
    pub(super) fn one() -> Self {
        Self {
            proposal: 1.,
            exact: Rational::from_usize(1),
        }
    }
    pub(super) async fn sum(
        &self,
        other: &Self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            proposal: self.proposal + other.proposal,
            exact: self.exact.add(&other.exact, policy, c).await?,
        })
    }
    pub(super) async fn scale(
        &self,
        factor: f64,
        operation: Scale,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let exact = Rational::from_float(factor)
            .ok_or_else(|| policy.invalid("objective weight must be finite"))?;
        Ok(match operation {
            Scale::Divide => Self {
                proposal: self.proposal / factor,
                exact: self.exact.div(&exact, policy, c).await?,
            },
            Scale::Multiply => Self {
                proposal: self.proposal * factor,
                exact: self.exact.mul(&exact, policy, c).await?,
            },
        })
    }
}
