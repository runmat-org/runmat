//! Supplied measurements carry the original coordinate header through a real
//! missing-lane invocation. A repaired/different point cannot reuse them.
use super::super::observation::Known;
use super::*;
pub(in super::super) struct Seed {
    pub point: Point,
    pub observation: Observation,
    pub supplied: Known,
}
impl Seed {
    pub(in super::super) async fn same_point(
        &self,
        point: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if self.point.variables() != point.len() {
            return Ok(false);
        }
        for (column, value) in point.iter().enumerate() {
            if self.point.coordinate(column).to_bits() != value.to_bits() {
                return Ok(false);
            }
            c.completed(1).await?;
        }
        Ok(true)
    }
}
impl Completed {
    pub(in super::super) async fn with_seed(
        mut self,
        seed: Option<Seed>,
        tolerance: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if let Some(seed) = seed {
            if self.point.variables() != seed.point.variables() {
                return Err(invalid("initial measurement changed original extent"));
            }
            for column in 0..self.point.variables() {
                if self.point.coordinate(column).to_bits()
                    != seed.point.coordinate(column).to_bits()
                {
                    return Err(invalid(
                        "initial measurement cannot be assigned to a different actual point",
                    ));
                }
                c.completed(1).await?;
            }
            self.observation = self
                .observation
                .with_known(&seed.observation, seed.supplied, tolerance, c)
                .await?;
        }
        Ok(self)
    }
}
