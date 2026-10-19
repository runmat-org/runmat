use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct JavaObjectHandle {
    pub handle: u64,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JavaObjectMetadata {
    pub class_name: String,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ObjectRegistryError {
    #[error("Java object handle {handle} belongs to stale generation {found}; current generation is {current}")]
    Stale {
        handle: u64,
        found: u64,
        current: u64,
    },
    #[error("Java object handle {0} is unknown")]
    Unknown(u64),
    #[error("Java object registry exhausted its handle space")]
    Exhausted,
}

#[derive(Debug)]
struct ObjectEntry<T> {
    metadata: JavaObjectMetadata,
    value: T,
}

#[derive(Debug)]
pub struct JavaObjectRegistry<T> {
    generation: u64,
    next_handle: u64,
    entries: BTreeMap<u64, ObjectEntry<T>>,
}

impl<T> Default for JavaObjectRegistry<T> {
    fn default() -> Self {
        Self {
            generation: 1,
            next_handle: 1,
            entries: BTreeMap::new(),
        }
    }
}

impl<T> JavaObjectRegistry<T> {
    pub fn insert(
        &mut self,
        metadata: JavaObjectMetadata,
        value: T,
    ) -> Result<JavaObjectHandle, ObjectRegistryError> {
        let handle = self.next_handle;
        self.next_handle = self
            .next_handle
            .checked_add(1)
            .ok_or(ObjectRegistryError::Exhausted)?;
        self.entries.insert(handle, ObjectEntry { metadata, value });
        Ok(JavaObjectHandle {
            handle,
            generation: self.generation,
        })
    }

    pub fn get(
        &self,
        handle: JavaObjectHandle,
    ) -> Result<(&JavaObjectMetadata, &T), ObjectRegistryError> {
        self.validate_generation(handle)?;
        let entry = self
            .entries
            .get(&handle.handle)
            .ok_or(ObjectRegistryError::Unknown(handle.handle))?;
        Ok((&entry.metadata, &entry.value))
    }

    pub fn remove(&mut self, handle: JavaObjectHandle) -> Result<T, ObjectRegistryError> {
        self.validate_generation(handle)?;
        self.entries
            .remove(&handle.handle)
            .map(|entry| entry.value)
            .ok_or(ObjectRegistryError::Unknown(handle.handle))
    }

    pub fn restart(&mut self) {
        self.entries.clear();
        self.generation = self.generation.saturating_add(1);
        self.next_handle = 1;
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (JavaObjectHandle, &JavaObjectMetadata, &T)> {
        let generation = self.generation;
        self.entries.iter().map(move |(handle, entry)| {
            (
                JavaObjectHandle {
                    handle: *handle,
                    generation,
                },
                &entry.metadata,
                &entry.value,
            )
        })
    }

    fn validate_generation(&self, handle: JavaObjectHandle) -> Result<(), ObjectRegistryError> {
        if handle.generation == self.generation {
            Ok(())
        } else {
            Err(ObjectRegistryError::Stale {
                handle: handle.handle,
                found: handle.generation,
                current: self.generation,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_fences_stale_handles() {
        let mut registry = JavaObjectRegistry::default();
        let handle = registry
            .insert(
                JavaObjectMetadata {
                    class_name: "java.lang.Object".into(),
                },
                7,
            )
            .unwrap();
        registry.restart();
        assert!(matches!(
            registry.get(handle),
            Err(ObjectRegistryError::Stale { .. })
        ));
    }

    #[test]
    fn cloned_identity_is_not_implicit() {
        let mut registry = JavaObjectRegistry::default();
        let first = registry
            .insert(
                JavaObjectMetadata {
                    class_name: "java.lang.String".into(),
                },
                "fixture".to_string(),
            )
            .unwrap();
        assert_eq!(registry.get(first).unwrap().1, "fixture");
        assert_eq!(registry.remove(first).unwrap(), "fixture");
        assert!(registry.is_empty());
    }

    #[test]
    fn iteration_retains_generation_and_metadata() {
        let mut registry = JavaObjectRegistry::default();
        let handle = registry
            .insert(
                JavaObjectMetadata {
                    class_name: "java.lang.Object".into(),
                },
                11,
            )
            .unwrap();
        let entries = registry.iter().collect::<Vec<_>>();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, handle);
        assert_eq!(entries[0].1.class_name, "java.lang.Object");
        assert_eq!(*entries[0].2, 11);
    }
}
