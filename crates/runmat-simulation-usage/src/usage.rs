use crate::{CountOverflow, OperationCount, OperationTotal};
use serde::{Deserialize, Serialize};

/// Transport categories; the measurement schedule owns individual kernel counts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum UsageCategory {
    Meshing,
    Assembly,
    Factorization,
    Solve,
    Other,
}

/// Cumulative completed work. Reservations and estimates are stored separately.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompletedUsage {
    pub meshing: OperationCount,
    pub assembly: OperationCount,
    pub factorization: OperationCount,
    pub solve: OperationCount,
    pub other: OperationCount,
}

impl CompletedUsage {
    /// Aggregate transport can represent sums wider than one admitted grant.
    pub fn wide_total(self) -> Result<OperationTotal, CountOverflow> {
        [
            self.meshing,
            self.assembly,
            self.factorization,
            self.solve,
            self.other,
        ]
        .into_iter()
        .map(OperationTotal::from)
        .try_fold(OperationTotal::ZERO, OperationTotal::checked_add)
    }

    pub fn total(self) -> Result<OperationCount, CountOverflow> {
        [
            self.meshing,
            self.assembly,
            self.factorization,
            self.solve,
            self.other,
        ]
        .into_iter()
        .try_fold(OperationCount::ZERO, OperationCount::checked_add)
    }

    pub(crate) fn with_completed(
        mut self,
        category: UsageCategory,
        completed: OperationCount,
    ) -> Result<Self, CountOverflow> {
        let destination = match category {
            UsageCategory::Meshing => &mut self.meshing,
            UsageCategory::Assembly => &mut self.assembly,
            UsageCategory::Factorization => &mut self.factorization,
            UsageCategory::Solve => &mut self.solve,
            UsageCategory::Other => &mut self.other,
        };
        *destination = destination.checked_add(completed)?;
        self.total()?;
        Ok(self)
    }
}
