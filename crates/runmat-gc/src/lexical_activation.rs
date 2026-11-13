//! Private GC activation backing for shared lexical bindings.
//!
//! The activation is not a language Value. Its sorted table is the only
//! mutable binding authority; each assigned slot owns one immutable Value
//! generation that an external reader may retain across replacement.

use runmat_types::BindingId;
pub use runmat_value::SharedValueGeneration as LexicalValueOwner;
use runmat_value::{
    exact_capacity_vec, ExactHostAllocationClaim, ExactHostAllocationPermit,
    ExactHostNumericAllocationError, Value,
};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LexicalSlotState {
    Uninitialized,
    Assigned,
    Vacant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum LexicalBindingMutationError {
    #[error("lexical activation does not own this binding")]
    BindingNotOwned,
    #[error("lexical activation revision exhausted")]
    RevisionExhausted,
}

#[derive(Clone, Debug)]
struct LexicalEntry {
    binding: BindingId,
    state: LexicalSlotState,
    value: Option<LexicalValueOwner>,
}

/// A preadmitted replacement table. Moving entries into it is allocation-free.
pub struct PreparedLexicalTable {
    entries: Vec<LexicalEntry>,
    _claim: Option<ExactHostAllocationClaim>,
    expected_revision: u64,
}

/// Ownership returned from a GC mutation so its backing and exact claim
/// retire after the exclusive GC borrow has ended.
pub struct RetiredLexicalTable {
    _entries: Vec<LexicalEntry>,
    _claim: Option<ExactHostAllocationClaim>,
}

pub enum LexicalReservationOutcome {
    Existing(Option<PreparedLexicalTable>),
    Applied(Option<RetiredLexicalTable>),
    Stale(Option<PreparedLexicalTable>),
}

impl PreparedLexicalTable {
    pub fn requested_bytes(capacity: usize) -> Result<usize, ExactHostNumericAllocationError> {
        std::alloc::Layout::array::<LexicalEntry>(capacity)
            .map(|layout| layout.size())
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub fn prepare(
        capacity: usize,
        expected_revision: u64,
        permit: Option<&dyn ExactHostAllocationPermit>,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let bytes = Self::requested_bytes(capacity)?;
        let claim = permit
            .map(|permit| permit.reserve_exact_claim(bytes))
            .transpose()?;
        let entries = exact_capacity_vec(capacity)?;
        Ok(Self {
            entries,
            _claim: claim,
            expected_revision,
        })
    }

    /// Copy immutable generation handles into already allocated backing.
    /// The source activation remains authoritative until the paired commit.
    pub fn populate_from(&mut self, source: &LexicalActivation) -> bool {
        if self.expected_revision != source.revision
            || self.entries.capacity() < source.entries.len()
            || !self.entries.is_empty()
        {
            return false;
        }
        self.entries.extend(source.entries.iter().cloned());
        true
    }

    /// Apply one final action to the off-state replacement table. No Value
    /// payload is cloned and capacity was checked before insertion.
    pub fn apply(
        &mut self,
        binding: BindingId,
        value: Option<LexicalValueOwner>,
    ) -> Result<(), Option<LexicalValueOwner>> {
        match self
            .entries
            .binary_search_by_key(&binding, |entry| entry.binding)
        {
            Ok(index) => {
                let entry = &mut self.entries[index];
                entry.state = if value.is_some() {
                    LexicalSlotState::Assigned
                } else {
                    LexicalSlotState::Vacant
                };
                entry.value = value;
            }
            Err(index) => {
                if self.entries.len() == self.entries.capacity() {
                    return Err(value);
                }
                self.entries.insert(
                    index,
                    LexicalEntry {
                        binding,
                        state: if value.is_some() {
                            LexicalSlotState::Assigned
                        } else {
                            LexicalSlotState::Vacant
                        },
                        value,
                    },
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct LexicalActivation {
    entries: Vec<LexicalEntry>,
    // The physical Vec backing retires before its reservation.
    _entries_claim: Option<ExactHostAllocationClaim>,
    revision: u64,
}

impl LexicalActivation {
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            _entries_claim: None,
            revision: 0,
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn owned_bindings(&self) -> impl Iterator<Item = BindingId> + '_ {
        self.entries.iter().map(|entry| entry.binding)
    }

    pub fn slot_state(&self, binding: BindingId) -> Option<LexicalSlotState> {
        self.find(binding)
            .ok()
            .map(|index| self.entries[index].state)
    }

    pub fn read(&self, binding: BindingId) -> Option<LexicalValueOwner> {
        self.find(binding)
            .ok()
            .and_then(|index| self.entries[index].value.clone())
    }

    pub fn owns_binding(&self, binding: BindingId) -> bool {
        self.find(binding).is_ok()
    }

    pub fn for_each_value(&self, mut visit: impl FnMut(&Value)) {
        for entry in &self.entries {
            if let Some(owner) = &entry.value {
                visit(owner.value());
            }
        }
    }

    pub fn growth_plan(&self) -> Result<Option<(usize, u64)>, ExactHostNumericAllocationError> {
        if self.entries.len() < self.entries.capacity() {
            return Ok(None);
        }
        let capacity = self
            .entries
            .len()
            .checked_mul(2)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?
            .max(1);
        Ok(Some((capacity, self.revision)))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    /// Infallible after a prepared batch has checked this revision and the
    /// exclusive lexical guard excludes all competing mutation.
    pub fn install_prepared_batch(
        &mut self,
        mut prepared: PreparedLexicalTable,
        next_revision: u64,
    ) -> RetiredLexicalTable {
        debug_assert_eq!(self.revision, prepared.expected_revision);
        let entries = std::mem::replace(&mut self.entries, std::mem::take(&mut prepared.entries));
        let claim = std::mem::replace(&mut self._entries_claim, prepared._claim.take());
        self.revision = next_revision;
        RetiredLexicalTable {
            _entries: entries,
            _claim: claim,
        }
    }

    /// Swap one previously validated existing binding without allocation.
    pub fn replace_existing_batch(
        &mut self,
        binding: BindingId,
        value: Option<LexicalValueOwner>,
    ) -> Option<LexicalValueOwner> {
        let index = self.find(binding).expect("prevalidated lexical binding");
        let entry = &mut self.entries[index];
        entry.state = if value.is_some() {
            LexicalSlotState::Assigned
        } else {
            LexicalSlotState::Vacant
        };
        std::mem::replace(&mut entry.value, value)
    }

    /// Insert or replace after the batch has proved enough spare capacity for
    /// every new binding in this activation.
    pub fn upsert_prevalidated_batch(
        &mut self,
        binding: BindingId,
        value: Option<LexicalValueOwner>,
    ) -> Option<LexicalValueOwner> {
        match self.find(binding) {
            Ok(_) => self.replace_existing_batch(binding, value),
            Err(index) => {
                debug_assert!(self.entries.len() < self.entries.capacity());
                self.entries.insert(
                    index,
                    LexicalEntry {
                        binding,
                        state: if value.is_some() {
                            LexicalSlotState::Assigned
                        } else {
                            LexicalSlotState::Vacant
                        },
                        value,
                    },
                );
                None
            }
        }
    }

    pub fn set_prevalidated_revision(&mut self, revision: u64) {
        self.revision = revision;
    }

    pub fn reserve_owned(
        &mut self,
        binding: BindingId,
        prepared: Option<PreparedLexicalTable>,
    ) -> LexicalReservationOutcome {
        let index = match self.find(binding) {
            Ok(_) => return LexicalReservationOutcome::Existing(prepared),
            Err(index) => index,
        };
        let Some(next_revision) = self.revision.checked_add(1) else {
            return LexicalReservationOutcome::Stale(prepared);
        };
        let mut retired = None;
        if let Some(mut prepared) = prepared {
            if prepared.expected_revision != self.revision
                || prepared.entries.capacity() <= self.entries.len()
            {
                return LexicalReservationOutcome::Stale(Some(prepared));
            }
            prepared.entries.append(&mut self.entries);
            let retired_entries = std::mem::replace(&mut self.entries, prepared.entries);
            let retired_claim = std::mem::replace(&mut self._entries_claim, prepared._claim.take());
            retired = Some(RetiredLexicalTable {
                _entries: retired_entries,
                _claim: retired_claim,
            });
        }
        if self.entries.len() == self.entries.capacity() {
            return LexicalReservationOutcome::Stale(None);
        }
        self.entries.insert(
            index,
            LexicalEntry {
                binding,
                state: LexicalSlotState::Uninitialized,
                value: None,
            },
        );
        self.revision = next_revision;
        LexicalReservationOutcome::Applied(retired)
    }

    pub fn assign(
        &mut self,
        binding: BindingId,
        value: LexicalValueOwner,
    ) -> Result<Option<LexicalValueOwner>, LexicalValueOwner> {
        let index = match self.find(binding) {
            Ok(index) => index,
            Err(_) => return Err(value),
        };
        let Some(next_revision) = self.revision.checked_add(1) else {
            return Err(value);
        };
        let entry = &mut self.entries[index];
        entry.state = LexicalSlotState::Assigned;
        let retired = entry.value.replace(value);
        self.revision = next_revision;
        Ok(retired)
    }

    pub fn remove(
        &mut self,
        binding: BindingId,
    ) -> Result<Option<LexicalValueOwner>, LexicalBindingMutationError> {
        let index = self
            .find(binding)
            .map_err(|_| LexicalBindingMutationError::BindingNotOwned)?;
        let entry = &mut self.entries[index];
        if entry.state != LexicalSlotState::Assigned {
            return Ok(None);
        }
        let next_revision = self
            .revision
            .checked_add(1)
            .ok_or(LexicalBindingMutationError::RevisionExhausted)?;
        entry.state = LexicalSlotState::Vacant;
        let retired = entry.value.take();
        self.revision = next_revision;
        Ok(retired)
    }

    fn find(&self, binding: BindingId) -> Result<usize, usize> {
        self.entries
            .binary_search_by_key(&binding, |entry| entry.binding)
    }
}
