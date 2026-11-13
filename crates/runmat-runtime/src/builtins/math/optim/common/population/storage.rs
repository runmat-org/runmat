//! Authoritative numeric population generations. Observer/vectorized arguments
//! share Tensor backing; only an actually retained predecessor triggers COW.
use super::{Policy, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};
pub(crate) struct Population {
    pub policy: Policy,
    pub positions: Tensor,
    pub members: usize,
    pub variables: usize,
}
impl Population {
    pub fn new(
        policy: Policy,
        positions: Tensor,
        members: usize,
        variables: usize,
    ) -> BuiltinResult<Self> {
        if positions.shape() != [members, variables] || positions.as_f64_slice().is_none() {
            return Err(policy.invalid("population must be a full double rows-by-variables matrix"));
        }
        Ok(Self {
            policy,
            positions,
            members,
            variables,
        })
    }
    pub fn get(&self, member: usize, variable: usize) -> f64 {
        self.positions
            .as_f64_slice()
            .expect("admitted double population")[member + self.members * variable]
    }
    pub fn value(&self) -> Value {
        Value::Tensor(self.positions.clone())
    }
    pub async fn row(
        &self,
        member: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Value> {
        let mut values = self
            .policy
            .capacity(self.variables, cooperation.runtime())?;
        for column in 0..self.variables {
            values.push(self.get(member, column));
            cooperation.completed(1).await?;
        }
        Ok(Value::Tensor(
            Tensor::new(values, vec![1, self.variables])
                .map_err(|error| self.policy.invalid(error))?,
        ))
    }
    pub async fn writable(&mut self, cooperation: &mut ScalarCooperation) -> BuiltinResult<()> {
        if self.positions.try_as_f64_slice_mut_unique().is_some() {
            return Ok(());
        }
        let mut values = self
            .policy
            .capacity(self.positions.len(), cooperation.runtime())?;
        for chunk in self
            .positions
            .as_f64_slice()
            .expect("admitted population")
            .chunks(1024)
        {
            values.extend_from_slice(chunk);
            cooperation.completed(chunk.len() as u64).await?;
        }
        self.positions = Tensor::new(values, vec![self.members, self.variables])
            .map_err(|error| self.policy.invalid(error))?;
        Ok(())
    }
}
