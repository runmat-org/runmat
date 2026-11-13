//! Sparse degree-two coefficient algebra. Exact entries decide cancellation
//! and convexity; binary64 values are proposals for the numerical QP only.
use super::*;
use std::collections::BTreeMap;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Monomial {
    Constant,
    Linear(usize),
    Quadratic(usize, usize),
}
impl Monomial {
    fn multiply(self, other: Self) -> Option<Self> {
        Some(match (self, other) {
            (Self::Constant, term) | (term, Self::Constant) => term,
            (Self::Linear(a), Self::Linear(b)) => Self::Quadratic(a.min(b), a.max(b)),
            _ => return None,
        })
    }
}
#[derive(Clone)]
pub(super) struct Weight {
    pub exact: Rational,
    pub proposal: f64,
}
impl Weight {
    pub fn constant(value: f64, policy: Policy) -> BuiltinResult<Self> {
        Ok(Self {
            exact: Rational::from_float(value)
                .ok_or_else(|| policy.invalid("quadratic constant must be finite"))?,
            proposal: value,
        })
    }
    async fn add(
        &self,
        other: &Self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            exact: self.exact.add(&other.exact, policy, c).await?,
            proposal: self.proposal + other.proposal,
        })
    }
    pub(super) async fn mul(
        &self,
        other: &Self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            exact: self.exact.mul(&other.exact, policy, c).await?,
            proposal: self.proposal * other.proposal,
        })
    }
    pub async fn divided(
        &self,
        divisor: &Self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            exact: self.exact.div(&divisor.exact, policy, c).await?,
            proposal: self.proposal / divisor.proposal,
        })
    }
    pub fn negated(&self) -> Self {
        Self {
            exact: -&self.exact,
            proposal: -self.proposal,
        }
    }
}
#[derive(Clone, Default)]
pub(super) struct Polynomial {
    // Immutable row sharing avoids copying exact coefficient tables at source
    // DAG joins or source index/reshape edges.
    entries: std::rc::Rc<BTreeMap<Monomial, Weight>>,
}
impl Polynomial {
    pub fn zero() -> Self {
        Self::default()
    }
    pub fn scalar(value: f64, policy: Policy) -> BuiltinResult<Self> {
        let mut entries = BTreeMap::new();
        if value != 0. {
            entries.insert(Monomial::Constant, Weight::constant(value, policy)?);
        }
        Ok(Self {
            entries: std::rc::Rc::new(entries),
        })
    }
    pub fn variable(column: usize) -> Self {
        let mut entries = BTreeMap::new();
        entries.insert(
            Monomial::Linear(column),
            Weight {
                exact: Rational::from_usize(1),
                proposal: 1.,
            },
        );
        Self {
            entries: std::rc::Rc::new(entries),
        }
    }
    pub fn entries(&self) -> impl Iterator<Item = (Monomial, &Weight)> {
        self.entries.iter().map(|(&key, value)| (key, value))
    }
    pub fn degree(&self) -> usize {
        self.entries.keys().next_back().map_or(0, |key| match key {
            Monomial::Constant => 0,
            Monomial::Linear(_) => 1,
            Monomial::Quadratic(..) => 2,
        })
    }
    pub fn constant(&self) -> Option<Weight> {
        if self.degree() != 0 {
            return None;
        }
        Some(
            self.entries
                .get(&Monomial::Constant)
                .cloned()
                .unwrap_or(Weight {
                    exact: Rational::zero(),
                    proposal: 0.,
                }),
        )
    }
    pub async fn add(
        self,
        other: Self,
        subtract: bool,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut entries = match std::rc::Rc::try_unwrap(self.entries) {
            Ok(entries) => entries,
            Err(shared) => {
                let mut entries = BTreeMap::new();
                for (&key, value) in shared.iter() {
                    entries.insert(key, value.clone());
                    c.completed(1).await?;
                }
                entries
            }
        };
        for (key, value) in other.entries() {
            accumulate(
                &mut entries,
                key,
                if subtract {
                    value.negated()
                } else {
                    value.clone()
                },
                policy,
                c,
            )
            .await?;
        }
        Ok(Self {
            entries: std::rc::Rc::new(entries),
        })
    }
    pub async fn scaled(
        &self,
        weight: &Weight,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut entries = BTreeMap::new();
        if !weight.exact.is_zero() {
            for (key, value) in self.entries() {
                let value = value.mul(weight, policy, c).await?;
                if !value.exact.is_zero() {
                    entries.insert(key, value);
                }
                c.completed(1).await?;
            }
        }
        Ok(Self {
            entries: std::rc::Rc::new(entries),
        })
    }
    pub async fn product(
        &self,
        other: &Self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        let mut entries = BTreeMap::new();
        for (a, left) in self.entries() {
            for (b, right) in other.entries() {
                let Some(key) = a.multiply(b) else {
                    return Ok(Err(Stop::NonAffine));
                };
                let value = left.mul(right, policy, c).await?;
                accumulate(&mut entries, key, value, policy, c).await?;
            }
            c.completed(1).await?;
        }
        Ok(Ok(Self {
            entries: std::rc::Rc::new(entries),
        }))
    }
}
async fn accumulate(
    entries: &mut BTreeMap<Monomial, Weight>,
    key: Monomial,
    value: Weight,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let value = match entries.remove(&key) {
        Some(previous) => previous.add(&value, policy, c).await?,
        None => value,
    };
    if !value.exact.is_zero() {
        entries.insert(key, value);
    }
    c.completed(1).await
}
