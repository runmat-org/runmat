//! Explicitly installed, isolated preference authority for deterministic hosts.
//! This adapter is not durable and is never a default session fallback.

use super::document::PreferenceDocument;
use super::{PreferenceGroups, PreferenceServiceError, PreferenceValues, RuntimePreferenceService};
use crate::context::{RuntimeContext, RuntimeServiceFuture};
use crate::execution::value_codec::{decode_durable_value, encode_durable_value};
use runmat_execution::value::ValuePayload;
use runmat_value::Value;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

type EncodedValues = BTreeMap<String, ValuePayload>;

/// Clones share one isolated transaction owner; unrelated instances do not.
#[derive(Clone)]
pub struct IsolatedMemoryPreferenceService {
    groups: Rc<RefCell<PreferenceDocument>>,
}

impl Default for IsolatedMemoryPreferenceService {
    fn default() -> Self {
        Self {
            groups: Rc::new(RefCell::new(PreferenceDocument::empty())),
        }
    }
}

impl IsolatedMemoryPreferenceService {
    pub fn new() -> Self {
        Self::default()
    }

    /// Restores an explicitly versioned isolated fixture, rejecting legacy or
    /// unknown documents rather than silently resetting user values.
    pub fn from_document_bytes(bytes: &[u8]) -> Result<Self, PreferenceServiceError> {
        Ok(Self {
            groups: Rc::new(RefCell::new(PreferenceDocument::decode(bytes)?)),
        })
    }

    pub fn document_bytes(&self) -> Result<Vec<u8>, PreferenceServiceError> {
        self.groups.borrow().encode()
    }
}

impl RuntimePreferenceService for IsolatedMemoryPreferenceService {
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        // The store owns only versioned payloads, never retained runtime Values.
        Some(())
    }

    fn all_groups(
        &self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<PreferenceGroups, PreferenceServiceError>> {
        let groups = self.groups.clone();
        Box::pin(async move {
            checkpoint(&runtime).await?;
            groups
                .borrow()
                .groups
                .iter()
                .map(|(name, values)| Ok((name.clone(), decode_group(values)?)))
                .collect()
        })
    }

    fn group_values(
        &self,
        runtime: RuntimeContext,
        group: String,
    ) -> RuntimeServiceFuture<Result<Option<PreferenceValues>, PreferenceServiceError>> {
        let groups = self.groups.clone();
        Box::pin(async move {
            checkpoint(&runtime).await?;
            groups
                .borrow()
                .groups
                .get(&group)
                .map(decode_group)
                .transpose()
        })
    }

    fn contains_names(
        &self,
        runtime: RuntimeContext,
        group: String,
        names: Vec<String>,
    ) -> RuntimeServiceFuture<Result<Vec<bool>, PreferenceServiceError>> {
        let groups = self.groups.clone();
        Box::pin(async move {
            checkpoint(&runtime).await?;
            let groups = groups.borrow();
            let values = groups.groups.get(&group);
            Ok(names
                .iter()
                .map(|name| values.is_some_and(|values| values.contains_key(name)))
                .collect())
        })
    }

    fn read_or_create_many(
        &self,
        runtime: RuntimeContext,
        group: String,
        names: Vec<String>,
        defaults: Option<Vec<Value>>,
    ) -> RuntimeServiceFuture<Result<Vec<Value>, PreferenceServiceError>> {
        let groups = self.groups.clone();
        Box::pin(async move {
            checkpoint(&runtime).await?;
            if defaults
                .as_ref()
                .is_some_and(|defaults| defaults.len() != names.len())
            {
                return Err(PreferenceServiceError::InvalidTransaction(
                    "default and preference-name cardinalities differ".into(),
                ));
            }
            // Encode every supplied default, even when its name is already
            // present. No late unsupported value may follow an insertion.
            let defaults = defaults
                .map(|defaults| {
                    defaults
                        .iter()
                        .map(encode_durable_value)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?;
            checkpoint(&runtime).await?;
            let mut state = groups.borrow_mut();
            let mut candidate = state.clone();
            let mut output = Vec::with_capacity(names.len());
            let mut changed = false;
            for (index, name) in names.iter().enumerate() {
                let payload = if let Some(value) = candidate
                    .groups
                    .get(&group)
                    .and_then(|values| values.get(name))
                {
                    value.clone()
                } else if let Some(defaults) = defaults.as_ref() {
                    let value = defaults[index].clone();
                    candidate
                        .groups
                        .entry(group.clone())
                        .or_default()
                        .insert(name.clone(), value.clone());
                    changed = true;
                    value
                } else {
                    return Err(PreferenceServiceError::Missing {
                        group,
                        name: name.clone(),
                    });
                };
                output.push(decode_durable_value(&payload)?);
            }
            if changed {
                *state = candidate;
            }
            Ok(output)
        })
    }

    fn set_many(
        &self,
        runtime: RuntimeContext,
        group: String,
        entries: Vec<(String, Value)>,
    ) -> RuntimeServiceFuture<Result<(), PreferenceServiceError>> {
        let groups = self.groups.clone();
        Box::pin(async move {
            checkpoint(&runtime).await?;
            let encoded = entries
                .iter()
                .map(|(name, value)| Ok((name.clone(), encode_durable_value(value)?)))
                .collect::<Result<Vec<_>, PreferenceServiceError>>()?;
            checkpoint(&runtime).await?;
            if encoded.is_empty() {
                return Ok(());
            }
            let mut state = groups.borrow_mut();
            let values = state.groups.entry(group).or_default();
            for (name, value) in encoded {
                values.insert(name, value);
            }
            Ok(())
        })
    }
}

async fn checkpoint(runtime: &RuntimeContext) -> Result<(), PreferenceServiceError> {
    let mut budget = runtime.cpu_cooperation();
    budget
        .checkpoint(0)
        .await
        .map_err(|source| PreferenceServiceError::Storage(source.into_runtime_error()))
}

fn decode_group(values: &EncodedValues) -> Result<PreferenceValues, PreferenceServiceError> {
    values
        .iter()
        .map(|(name, payload)| Ok((name.clone(), decode_durable_value(payload)?)))
        .collect()
}

#[cfg(test)]
#[path = "memory/tests.rs"]
mod tests;
