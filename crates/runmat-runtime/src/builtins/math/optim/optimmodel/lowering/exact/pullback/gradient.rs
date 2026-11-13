//! Exact column accumulation; sparse mode never initializes an N-vector per row.
use super::*;
pub(super) enum Gradient {
    Dense(Vec<Rational>),
    Sparse(Weights),
}
impl Gradient {
    pub(super) async fn add(
        &mut self,
        column: usize,
        value: Rational,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        match self {
            Self::Dense(values) => {
                values[column] = values[column].add(&value, policy, c).await?;
            }
            Self::Sparse(values) => {
                if value.is_zero() {
                    return Ok(());
                }
                values.try_reserve(1).map_err(|e| policy.reserve(e))?;
                match values.entry(column) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(value);
                    }
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        *entry.get_mut() = entry.get().add(&value, policy, c).await?;
                        if entry.get().is_zero() {
                            entry.remove();
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
