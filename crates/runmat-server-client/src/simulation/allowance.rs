use chrono::{DateTime, Utc};
use runmat_simulation_usage::{MeasurementVersion, OperationCount, OperationTotal};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SimulationRenewal {
    Week,
    Month,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationPeriod {
    pub starts_at: DateTime<Utc>,
    pub refills_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationAllowance {
    pub measurement_version: Option<MeasurementVersion>,
    pub runmat_version: Option<String>,
    pub renewal: Option<SimulationRenewal>,
    pub period: Option<SimulationPeriod>,
    pub allowance_ops: Option<OperationTotal>,
    /// Eligible unspent, unreserved pack balance, supplied by Server.
    pub pack_ops: OperationTotal,
    pub used_ops: OperationTotal,
    pub reserved_ops: OperationTotal,
    pub waived: bool,
    pub published: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SimulationBudgetError {
    #[error("the simulation allowance is not published")]
    AllowanceUnpublished,
    #[error("the simulation allowance is used up")]
    AllowanceUsed,
    #[error("the simulation spending cap must be positive")]
    ZeroOverride,
    #[error("the simulation allowance summary is inconsistent or exceeds its integer range")]
    InvalidSummary,
}

impl SimulationAllowance {
    /// Preflight the available allowance and packs for display or a lower cap.
    /// Default admission omits budgetOps so Server chooses its fixed cap from
    /// the admission-time balance, rather than this earlier summary.
    /// A later start response may grant less if concurrent work reserves funds.
    /// An academic waiver uses the representable per-run maximum unless lowered.
    pub fn spending_cap(
        &self,
        lower_cap: Option<OperationCount>,
    ) -> Result<OperationCount, SimulationBudgetError> {
        if lower_cap.is_some_and(|cap| cap.operations() == 0) {
            return Err(SimulationBudgetError::ZeroOverride);
        }
        let available = if self.waived {
            u128::from(u64::MAX)
        } else {
            let allowance = match (self.published, self.allowance_ops) {
                (true, Some(allowance)) => allowance.operations(),
                (false, None) if self.pack_ops.operations() > 0 => 0,
                (false, None) => return Err(SimulationBudgetError::AllowanceUnpublished),
                _ => return Err(SimulationBudgetError::InvalidSummary),
            };
            allowance
                .checked_sub(self.used_ops.operations())
                .and_then(|remaining| remaining.checked_sub(self.reserved_ops.operations()))
                .and_then(|remaining| remaining.checked_add(self.pack_ops.operations()))
                .ok_or(SimulationBudgetError::InvalidSummary)?
        };
        let available = available.min(u128::from(u64::MAX)) as u64;
        let cap = lower_cap.map_or(available, |cap| cap.operations().min(available));
        if cap == 0 {
            return Err(SimulationBudgetError::AllowanceUsed);
        }
        Ok(OperationCount::from_operations(cap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_availability_is_capped_exactly_with_explicit_waiver_and_unpublished_states() {
        let mut allowance = SimulationAllowance {
            measurement_version: None,
            runmat_version: None,
            renewal: None,
            period: None,
            allowance_ops: Some(OperationTotal::from_operations(100)),
            pack_ops: OperationTotal::from_operations(20),
            used_ops: OperationTotal::from_operations(30),
            reserved_ops: OperationTotal::from_operations(10),
            waived: false,
            published: true,
        };
        assert_eq!(allowance.spending_cap(None).unwrap().operations(), 80);
        assert_eq!(
            allowance
                .spending_cap(Some(OperationCount::from_operations(50)))
                .unwrap()
                .operations(),
            50
        );
        assert_eq!(
            allowance.spending_cap(Some(OperationCount::ZERO)),
            Err(SimulationBudgetError::ZeroOverride)
        );
        allowance.allowance_ops = Some(OperationTotal::from_operations(u128::from(u64::MAX) + 100));
        assert_eq!(allowance.spending_cap(None).unwrap().operations(), u64::MAX);
        allowance.allowance_ops = None;
        allowance.published = false;
        allowance.used_ops = OperationTotal::ZERO;
        allowance.reserved_ops = OperationTotal::ZERO;
        assert_eq!(allowance.spending_cap(None).unwrap().operations(), 20);
        allowance.pack_ops = OperationTotal::ZERO;
        assert_eq!(
            allowance.spending_cap(None),
            Err(SimulationBudgetError::AllowanceUnpublished)
        );
        allowance.waived = true;
        assert_eq!(allowance.spending_cap(None).unwrap().operations(), u64::MAX);
        assert_eq!(
            allowance
                .spending_cap(Some(OperationCount::from_operations(7)))
                .unwrap()
                .operations(),
            7
        );
        allowance.waived = false;
        allowance.published = true;
        allowance.allowance_ops = Some(OperationTotal::ZERO);
        assert_eq!(
            allowance.spending_cap(None),
            Err(SimulationBudgetError::AllowanceUsed)
        );
        allowance.used_ops = OperationTotal::from_operations(1);
        assert_eq!(
            allowance.spending_cap(None),
            Err(SimulationBudgetError::InvalidSummary)
        );
    }
}
