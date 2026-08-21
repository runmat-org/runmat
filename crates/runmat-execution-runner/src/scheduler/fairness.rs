use serde::{Deserialize, Serialize};

/// Bounded strict priority: FIFO is preserved within a priority, and after a
/// configured burst the oldest lower-priority task receives one scheduling
/// opportunity. Equivalent candidates are always ordered by stable IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FairnessPolicy {
    pub max_priority_burst: u16,
}

impl Default for FairnessPolicy {
    fn default() -> Self {
        Self {
            max_priority_burst: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct FairnessState {
    pub last_priority: Option<i16>,
    pub consecutive: u16,
}

impl FairnessState {
    pub fn select_priority(
        &mut self,
        ordered_priorities: impl IntoIterator<Item = i16>,
        policy: FairnessPolicy,
    ) -> Option<i16> {
        let mut priorities = ordered_priorities.into_iter();
        let highest = priorities.next()?;
        let next_distinct = priorities.find(|priority| *priority != highest);
        let selected = if self.last_priority == Some(highest)
            && self.consecutive >= policy.max_priority_burst
        {
            next_distinct.unwrap_or(highest)
        } else {
            highest
        };
        self.record(selected);
        Some(selected)
    }

    pub fn record(&mut self, selected: i16) {
        if self.last_priority == Some(selected) {
            self.consecutive = self.consecutive.saturating_add(1);
        } else {
            self.last_priority = Some(selected);
            self.consecutive = 1;
        }
    }
}
