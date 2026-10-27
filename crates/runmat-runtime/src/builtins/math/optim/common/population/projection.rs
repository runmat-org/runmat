//! Population argument projection only; dispatch, worker waves, and counting
//! remain in Invocation. A custom projection retains newly returned GC roots.
use super::{Population, ScalarCooperation};
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use futures::FutureExt;
use runmat_value::{Tensor, Value};

/// An exact prepared projection, not an execution or allocation grant. The
/// borrowed generation keeps the original receiver rooted until execution.
pub(crate) enum Prepared<'a> {
    Ready(Argument),
    Subscript {
        receiver: &'a Value,
        path: crate::object::indexing::ObjectSubscriptPath,
        builtin: &'static str,
    },
}
impl<'a> Prepared<'a> {
    pub(crate) fn execute(
        self,
        runtime: Option<crate::context::RuntimeContext>,
    ) -> impl std::future::Future<Output = BuiltinResult<Argument>> + 'a {
        use futures::future::Either;
        match self {
            Self::Ready(argument) => Either::Left(futures::future::ready(Ok(argument))),
            Self::Subscript {
                receiver,
                path,
                builtin,
            } => Either::Right(
                crate::object::protocol::read_subscript_path_with_runtime(
                    runtime,
                    receiver.clone(),
                    path,
                    Some(builtin),
                )
                .map(|returned| returned.map(Argument::Rooted)),
            ),
        }
    }
}

pub(crate) enum Argument {
    /// Metadata/new numeric backing or an unchanged edge protected by the
    /// caller's retained generation. No new GC identity is introduced.
    Source(Value),
    /// A semantic projection may introduce GC identities. Its existing lease
    /// must survive until the consuming callback returns a rooted successor.
    Rooted(RootedValue),
}
impl Argument {
    pub(crate) fn into_parts(self) -> (Value, Option<RootedValueLease>) {
        match self {
            Self::Source(value) => (value, None),
            Self::Rooted(value) => {
                let (value, lease) = value.into_value_and_lease();
                (value, Some(lease))
            }
        }
    }
}
/// Borrowed exact coordinates for the initial physical-observation receipt.
/// This is not a replacement for semantic projection of a custom object.
#[derive(Clone, Copy)]
pub(crate) struct Coordinates<'a> {
    pub values: &'a [f64],
    pub members: usize,
    pub variables: usize,
}
impl Coordinates<'_> {
    pub fn get(self, row: usize, column: usize) -> f64 {
        self.values[row + self.members * column]
    }
}
/// Statically dispatched argument construction, not an execution provider.
/// rows are the ordered actual members for one callback. Implementations must
/// retain their generation and respect semantic indexing for custom objects.
pub(crate) trait Arguments {
    fn members(&self) -> usize;
    /// Borrow the actual admitted native representation, if one exists. This
    /// supplies no execution/allocation permission and never bypasses custom
    /// semantic projection.
    fn numeric_rows(&self) -> Option<&Population> {
        None
    }
    fn known_coordinates(&self) -> Option<Coordinates<'_>> {
        let population = self.numeric_rows()?;
        Some(Coordinates {
            values: population.positions.as_f64_slice()?,
            members: population.members,
            variables: population.variables,
        })
    }
    async fn prepare_projection<'a>(
        &'a self,
        rows: &[usize],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Prepared<'a>>;
    async fn project(
        &self,
        rows: &[usize],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Argument> {
        let prepared = self.prepare_projection(rows, cooperation).await?;
        prepared.execute(cooperation.runtime().cloned()).await
    }
}
impl Arguments for Population {
    fn numeric_rows(&self) -> Option<&Population> {
        Some(self)
    }
    fn members(&self) -> usize {
        self.members
    }
    async fn prepare_projection<'a>(
        &'a self,
        rows: &[usize],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Prepared<'a>> {
        if rows.is_empty() {
            return Err(self
                .policy
                .invalid("callback row is outside the population"));
        }
        let mut original = rows.len() == self.members;
        for (index, &row) in rows.iter().enumerate() {
            if row >= self.members {
                return Err(self
                    .policy
                    .invalid("callback row is outside the population"));
            }
            original &= index == row;
            cooperation.completed(1).await?;
        }
        if original {
            return Ok(Prepared::Ready(Argument::Source(self.value())));
        }
        if rows.len() == 1 {
            return self
                .row(rows[0], cooperation)
                .await
                .map(|value| Prepared::Ready(Argument::Source(value)));
        }
        let count = rows
            .len()
            .checked_mul(self.variables)
            .ok_or_else(|| self.policy.invalid("vectorized population extent overflow"))?;
        let mut data = self.policy.capacity(count, cooperation.runtime())?;
        for column in 0..self.variables {
            for &row in rows {
                data.push(self.get(row, column));
                cooperation.completed(1).await?;
            }
        }
        Ok(Prepared::Ready(Argument::Source(Value::Tensor(
            Tensor::new(data, vec![rows.len(), self.variables])
                .map_err(|error| self.policy.invalid(error))?,
        ))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::BuiltinCatalogIdentity;
    #[test]
    fn numeric_projection_preserves_full_backing_and_ordered_rows() {
        futures::executor::block_on(async {
            let policy = super::super::Policy {
                identity: BuiltinCatalogIdentity {
                    name: "particleswarm",
                },
                error: &runmat_builtins::PARTICLESWARM_ERROR_INVALID_ARGUMENT,
            };
            let population = Population::new(
                policy,
                Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]).unwrap(),
                3,
                2,
            )
            .unwrap();
            let mut cooperation = ScalarCooperation::for_builtin(policy.identity, None);
            let (full, lease) = population
                .project(&[0, 1, 2], &mut cooperation)
                .await
                .unwrap()
                .into_parts();
            let Value::Tensor(full) = full else {
                panic!("numeric matrix")
            };
            assert!(lease.is_none());
            assert_eq!(
                full.as_f64_slice().unwrap().as_ptr(),
                population.positions.as_f64_slice().unwrap().as_ptr()
            );
            let (subset, _) = population
                .project(&[2, 0], &mut cooperation)
                .await
                .unwrap()
                .into_parts();
            let Value::Tensor(subset) = subset else {
                panic!("numeric subset")
            };
            assert_eq!(subset.shape(), [2, 2]);
            assert_eq!(subset.as_f64_slice().unwrap(), [3., 1., 6., 4.]);
            assert!(population.project(&[3], &mut cooperation).await.is_err());
        });
    }
}
