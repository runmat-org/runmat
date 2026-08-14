use runmat_runtime::native::NativeValueRef;
use runmat_value::Value;

use crate::{JitError, JitResult};

#[derive(Debug)]
struct Entry {
    generation: u64,
    references: usize,
    value: Option<Value>,
}

/// Invocation-owned opaque value table.
///
/// Generated code sees only generation-checked `NativeValueRef` tokens. The
/// current Rust `Value` representation never crosses the native ABI.
#[derive(Debug, Default)]
pub struct ValueArena {
    entries: Vec<Entry>,
    free: Vec<usize>,
}

impl ValueArena {
    pub fn insert(&mut self, value: Value) -> NativeValueRef {
        let index = self.free.pop().unwrap_or_else(|| {
            self.entries.push(Entry {
                generation: 1,
                references: 0,
                value: None,
            });
            self.entries.len() - 1
        });
        let entry = &mut self.entries[index];
        entry.references = 1;
        entry.value = Some(value);
        NativeValueRef {
            handle: index as u64 + 1,
            generation: entry.generation,
        }
    }

    pub fn get(&self, reference: NativeValueRef) -> JitResult<&Value> {
        let entry = self.entry(reference)?;
        entry.value.as_ref().ok_or(JitError::StaleValue)
    }

    pub fn get_mut(&mut self, reference: NativeValueRef) -> JitResult<&mut Value> {
        let entry = self.entry_mut(reference)?;
        entry.value.as_mut().ok_or(JitError::StaleValue)
    }

    pub fn retain(&mut self, reference: NativeValueRef) -> JitResult<()> {
        if reference.is_null() {
            return Ok(());
        }
        let entry = self.entry_mut(reference)?;
        entry.references = entry
            .references
            .checked_add(1)
            .ok_or(JitError::StaleValue)?;
        Ok(())
    }

    pub fn release(&mut self, reference: NativeValueRef) -> JitResult<()> {
        if reference.is_null() {
            return Ok(());
        }
        let index = checked_index(reference)?;
        let entry = self.entries.get_mut(index).ok_or(JitError::StaleValue)?;
        if entry.generation != reference.generation
            || entry.value.is_none()
            || entry.references == 0
        {
            return Err(JitError::StaleValue);
        }
        entry.references -= 1;
        if entry.references == 0 {
            entry.value = None;
            // A wrapped generation could make a prehistoric reference valid
            // again. Retire the slot permanently instead of permitting ABA.
            if let Some(next_generation) = entry.generation.checked_add(1) {
                entry.generation = next_generation;
                self.free.push(index);
            }
        }
        Ok(())
    }

    fn entry(&self, reference: NativeValueRef) -> JitResult<&Entry> {
        if reference.is_null() {
            return Err(JitError::StaleValue);
        }
        let entry = self
            .entries
            .get(checked_index(reference)?)
            .ok_or(JitError::StaleValue)?;
        if entry.generation != reference.generation || entry.value.is_none() {
            return Err(JitError::StaleValue);
        }
        Ok(entry)
    }

    fn entry_mut(&mut self, reference: NativeValueRef) -> JitResult<&mut Entry> {
        if reference.is_null() {
            return Err(JitError::StaleValue);
        }
        let entry = self
            .entries
            .get_mut(checked_index(reference)?)
            .ok_or(JitError::StaleValue)?;
        if entry.generation != reference.generation || entry.value.is_none() {
            return Err(JitError::StaleValue);
        }
        Ok(entry)
    }
}

fn checked_index(reference: NativeValueRef) -> JitResult<usize> {
    let handle = reference
        .handle
        .checked_sub(1)
        .ok_or(JitError::StaleValue)?;
    usize::try_from(handle).map_err(|_| JitError::StaleValue)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_are_generation_checked_and_reuse_slots_without_aliasing() {
        let mut arena = ValueArena::default();
        let first = arena.insert(Value::Num(1.0));
        arena.retain(first).unwrap();
        arena.release(first).unwrap();
        assert_eq!(arena.get(first).unwrap(), &Value::Num(1.0));
        arena.release(first).unwrap();
        assert!(matches!(arena.get(first), Err(JitError::StaleValue)));

        let second = arena.insert(Value::Num(2.0));
        assert_eq!(second.handle, first.handle);
        assert_ne!(second.generation, first.generation);
        assert_eq!(arena.get(second).unwrap(), &Value::Num(2.0));
    }

    #[test]
    fn null_is_a_noop_for_lifetime_but_never_materializes_a_value() {
        let mut arena = ValueArena::default();
        arena.retain(NativeValueRef::NULL).unwrap();
        arena.release(NativeValueRef::NULL).unwrap();
        assert!(matches!(
            arena.get(NativeValueRef::NULL),
            Err(JitError::StaleValue)
        ));
    }
}
