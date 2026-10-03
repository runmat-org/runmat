use runmat_execution::{ResourceBoundUnavailable, ResourceByteBound};

/// Retained allocations can outlive one graph node. Counts preserve which
/// unavailable bounds remain live when a predecessor's last consumer exits.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ResourceAccount {
    pub(super) known_bytes: u64,
    pub(super) unavailable_bounds: UnavailableCounts,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct UnavailableCounts {
    peak_memory: u32,
    scratch: u32,
    retained_value: u32,
}

impl UnavailableCounts {
    fn count_mut(&mut self, reason: ResourceBoundUnavailable) -> &mut u32 {
        match reason {
            ResourceBoundUnavailable::NoPeakMemoryBound => &mut self.peak_memory,
            ResourceBoundUnavailable::NoScratchBound => &mut self.scratch,
            ResourceBoundUnavailable::NoRetainedValueBound => &mut self.retained_value,
        }
    }

    pub(super) fn is_empty(self) -> bool {
        self == Self::default()
    }

    fn first(self) -> Option<ResourceBoundUnavailable> {
        if self.peak_memory != 0 {
            Some(ResourceBoundUnavailable::NoPeakMemoryBound)
        } else if self.scratch != 0 {
            Some(ResourceBoundUnavailable::NoScratchBound)
        } else if self.retained_value != 0 {
            Some(ResourceBoundUnavailable::NoRetainedValueBound)
        } else {
            None
        }
    }

    fn combine(self, other: Self) -> Option<Self> {
        Some(Self {
            peak_memory: self.peak_memory.checked_add(other.peak_memory)?,
            scratch: self.scratch.checked_add(other.scratch)?,
            retained_value: self.retained_value.checked_add(other.retained_value)?,
        })
    }
}

impl ResourceAccount {
    /// Alternative stages do not overlap: readback completes before output
    /// upload. Keep the larger proven peak, or an unavailable bound when
    /// either stage cannot prove its peak under a finite provider quota.
    pub(super) fn peak(self, other: Self) -> Self {
        if !self.unavailable_bounds.is_empty() {
            self
        } else if !other.unavailable_bounds.is_empty() {
            other
        } else if self.known_bytes >= other.known_bytes {
            self
        } else {
            other
        }
    }

    pub(super) fn combine(self, other: Self) -> Option<Self> {
        Some(Self {
            known_bytes: self.known_bytes.checked_add(other.known_bytes)?,
            unavailable_bounds: self.unavailable_bounds.combine(other.unavailable_bounds)?,
        })
    }

    pub(super) fn add(self, bound: ResourceByteBound) -> Option<Self> {
        Some(match bound {
            ResourceByteBound::Known(bytes) => Self {
                known_bytes: self.known_bytes.checked_add(bytes)?,
                ..self
            },
            ResourceByteBound::Unavailable(reason) => {
                let mut next = self;
                let count = next.unavailable_bounds.count_mut(reason);
                *count = count.checked_add(1)?;
                next
            }
        })
    }

    pub(super) fn subtract(self, bound: ResourceByteBound) -> Option<Self> {
        Some(match bound {
            ResourceByteBound::Known(bytes) => Self {
                known_bytes: self.known_bytes.checked_sub(bytes)?,
                ..self
            },
            ResourceByteBound::Unavailable(reason) => {
                let mut next = self;
                let count = next.unavailable_bounds.count_mut(reason);
                *count = count.checked_sub(1)?;
                next
            }
        })
    }

    pub(super) fn within(
        self,
        budget: Option<u64>,
        budget_kind: ResourceBudgetKind,
    ) -> Result<(), ResourceAdmissionError> {
        let Some(budget) = budget else {
            return Ok(());
        };
        if let Some(reason) = self.unavailable_bounds.first() {
            return Err(ResourceAdmissionError::InsufficientEvidence {
                reason,
                budget: budget_kind,
            });
        }
        if self.known_bytes > budget {
            return Err(ResourceAdmissionError::OverBudget);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::placement) enum ResourceBudgetKind {
    HostMemory,
    ProviderScratch(u32),
    ProviderMemory(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::placement) enum ResourceAdmissionError {
    InsufficientEvidence {
        reason: ResourceBoundUnavailable,
        budget: ResourceBudgetKind,
    },
    OverBudget,
    Overflow,
    UnavailableProvider,
    TransactionalResultsRequired,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_retained_bound_is_released_without_losing_other_live_facts() {
        let unknown =
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound);
        let live = ResourceAccount::default()
            .add(unknown)
            .unwrap()
            .add(ResourceByteBound::Known(12))
            .unwrap();
        assert!(matches!(
            live.within(Some(20), ResourceBudgetKind::HostMemory),
            Err(ResourceAdmissionError::InsufficientEvidence { .. })
        ));
        let released = live.subtract(unknown).unwrap();
        assert_eq!(released.known_bytes, 12);
        assert_eq!(
            released.within(Some(20), ResourceBudgetKind::HostMemory),
            Ok(())
        );
    }

    #[test]
    fn known_sum_overflow_never_becomes_unavailable_or_zero() {
        assert!(ResourceAccount::default()
            .add(ResourceByteBound::Known(u64::MAX))
            .unwrap()
            .add(ResourceByteBound::Known(1))
            .is_none());
    }
}
