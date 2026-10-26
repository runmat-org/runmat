//! Subpopulation ranges are metadata over the one authoritative generation.
use super::super::{invalid, ScalarCooperation, POLICY};
use crate::BuiltinResult;
pub(in super::super) struct Layout {
    offsets: Vec<usize>,
}
impl Layout {
    pub async fn new(counts: &[usize], cooperation: &mut ScalarCooperation) -> BuiltinResult<Self> {
        if counts.is_empty() {
            return Err(invalid(
                "PopulationSize must have at least one positive count",
            ));
        }
        let mut offsets = POLICY.capacity(
            counts
                .len()
                .checked_add(1)
                .ok_or_else(|| invalid("subpopulation count overflow"))?,
            cooperation.runtime(),
        )?;
        offsets.push(0);
        let mut total = 0usize;
        for &count in counts {
            if count == 0 {
                return Err(invalid("every PopulationSize entry must be positive"));
            }
            total = total
                .checked_add(count)
                .ok_or_else(|| invalid("total population size overflow"))?;
            offsets.push(total);
            cooperation.completed(1).await?;
        }
        Ok(Self { offsets })
    }
    pub fn groups(&self) -> usize {
        self.offsets.len() - 1
    }
    pub fn total(&self) -> usize {
        *self.offsets.last().expect("admitted layout")
    }
    pub fn range(&self, group: usize) -> std::ops::Range<usize> {
        self.offsets[group]..self.offsets[group + 1]
    }
    pub fn group_for(&self, member: usize) -> Option<usize> {
        (member < self.total())
            .then(|| self.offsets.partition_point(|&offset| offset <= member) - 1)
    }
}
