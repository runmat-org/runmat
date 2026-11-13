//! Explicit isolated preference authority for deterministic hosts and tests.
//! This is not durable storage and is never installed as a session fallback.

use super::document::{missing_default_indices, PreferenceDocument};
use super::input_roots::PreferenceInputRoots;
use super::{PreferenceGroups, PreferenceServiceError, PreferenceValues, RuntimePreferenceService};
use crate::context::{RuntimeContext, RuntimeServiceFuture};
use crate::context::{
    RuntimeUserPathService, UserPathMutation, UserPathServiceError, UserPathSnapshot,
};
use crate::execution::CpuCooperationBudget;
use futures::lock::Mutex;
use runmat_execution::value::ValuePayload;
use runmat_value::Value;
use std::collections::BTreeMap;
use std::rc::Rc;

type EncodedValues = BTreeMap<String, ValuePayload>;

#[derive(Clone)]
pub struct IsolatedMemoryPreferenceService {
    document: Rc<Mutex<PreferenceDocument>>,
}

impl Default for IsolatedMemoryPreferenceService {
    fn default() -> Self {
        Self {
            document: Rc::new(Mutex::new(PreferenceDocument::empty())),
        }
    }
}

impl IsolatedMemoryPreferenceService {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_document_bytes(bytes: &[u8]) -> Result<Self, PreferenceServiceError> {
        Ok(Self {
            document: Rc::new(Mutex::new(PreferenceDocument::decode(bytes)?)),
        })
    }

    pub async fn document_bytes(&self) -> Result<Vec<u8>, PreferenceServiceError> {
        Ok(self.document.lock().await.encode()?)
    }
}

impl RuntimePreferenceService for IsolatedMemoryPreferenceService {
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        Some(()) // Only encoded payloads are retained, never runtime Value owners.
    }

    fn all_groups(
        &self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<PreferenceGroups, PreferenceServiceError>> {
        let document = self.document.clone();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let document = budget
                .await_cancellable(document.lock())
                .await
                .map_err(storage)?;
            let mut result = PreferenceGroups::new();
            for (group, values) in &document.groups {
                let mut decoded = PreferenceValues::new();
                for (name, payload) in values {
                    decoded.insert(
                        name.clone(),
                        super::decode_selected_preference(payload, &mut budget, &runtime).await?,
                    );
                    checkpoint(&mut budget, 1).await?;
                }
                result.insert(group.clone(), decoded);
                checkpoint(&mut budget, 1).await?;
            }
            Ok(result)
        })
    }

    fn group_values(
        &self,
        runtime: RuntimeContext,
        group: String,
    ) -> RuntimeServiceFuture<Result<Option<PreferenceValues>, PreferenceServiceError>> {
        let document = self.document.clone();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let document = budget
                .await_cancellable(document.lock())
                .await
                .map_err(storage)?;
            let Some(values) = document.groups.get(&group) else {
                return Ok(None);
            };
            let mut decoded = PreferenceValues::new();
            for (name, payload) in values {
                decoded.insert(
                    name.clone(),
                    super::decode_selected_preference(payload, &mut budget, &runtime).await?,
                );
                checkpoint(&mut budget, 1).await?;
            }
            Ok(Some(decoded))
        })
    }

    fn contains_names(
        &self,
        runtime: RuntimeContext,
        group: String,
        names: Vec<String>,
    ) -> RuntimeServiceFuture<Result<Vec<bool>, PreferenceServiceError>> {
        let document = self.document.clone();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let document = budget
                .await_cancellable(document.lock())
                .await
                .map_err(storage)?;
            let values = document.groups.get(&group);
            let mut result = Vec::with_capacity(names.len());
            for name in &names {
                result.push(values.is_some_and(|values| values.contains_key(name)));
                checkpoint(&mut budget, 1).await?;
            }
            Ok(result)
        })
    }

    fn read_or_create_many(
        &self,
        runtime: RuntimeContext,
        group: String,
        names: Vec<String>,
        defaults: Option<Vec<Value>>,
    ) -> RuntimeServiceFuture<Result<Vec<Value>, PreferenceServiceError>> {
        let document = self.document.clone();
        let roots = defaults
            .as_ref()
            .map(|defaults| PreferenceInputRoots::for_values(defaults))
            .transpose();
        Box::pin(async move {
            let _roots = roots?;
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            if defaults
                .as_ref()
                .is_some_and(|defaults| defaults.len() != names.len())
            {
                return Err(PreferenceServiceError::InvalidTransaction(
                    "default and preference-name cardinalities differ".into(),
                ));
            }
            let mut document = budget
                .await_cancellable(document.lock())
                .await
                .map_err(storage)?;
            let mut encoded_defaults = vec![None; names.len()];
            if let Some(defaults) = &defaults {
                for index in missing_default_indices(&document, &group, &names) {
                    encoded_defaults[index] = Some(
                        super::encode_selected_preference(&defaults[index], &mut budget, &runtime)
                            .await?,
                    );
                    checkpoint(&mut budget, 1).await?;
                }
            }
            let mut candidate = EncodedValues::new();
            if let Some(existing) = document.groups.get(&group) {
                for (name, payload) in existing {
                    candidate.insert(name.clone(), payload.clone());
                    checkpoint(&mut budget, 1).await?;
                }
            }
            let mut output = Vec::with_capacity(names.len());
            let mut changed = false;
            for (index, name) in names.iter().enumerate() {
                if !candidate.contains_key(name) {
                    if let Some(default) = encoded_defaults.get(index).and_then(Option::as_ref) {
                        candidate.insert(name.clone(), default.clone());
                        changed = true;
                    } else {
                        return Err(PreferenceServiceError::Missing {
                            group,
                            name: name.clone(),
                        });
                    }
                }
                let payload = candidate
                    .get(name)
                    .expect("existing or defaulted preference");
                output
                    .push(super::decode_selected_preference(payload, &mut budget, &runtime).await?);
                checkpoint(&mut budget, 1).await?;
            }
            checkpoint(&mut budget, 0).await?;
            if changed {
                document.groups.insert(group, candidate);
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
        let document = self.document.clone();
        let roots = PreferenceInputRoots::for_entries(&entries);
        Box::pin(async move {
            let _roots = roots?;
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let mut encoded = Vec::with_capacity(entries.len());
            for (name, value) in &entries {
                encoded.push((
                    name.clone(),
                    super::encode_selected_preference(value, &mut budget, &runtime).await?,
                ));
                checkpoint(&mut budget, 1).await?;
            }
            if encoded.is_empty() {
                return Ok(());
            }
            let mut document = budget
                .await_cancellable(document.lock())
                .await
                .map_err(storage)?;
            let mut candidate = EncodedValues::new();
            if let Some(existing) = document.groups.get(&group) {
                for (name, payload) in existing {
                    candidate.insert(name.clone(), payload.clone());
                    checkpoint(&mut budget, 1).await?;
                }
            }
            for (name, payload) in encoded {
                candidate.insert(name, payload);
                checkpoint(&mut budget, 1).await?;
            }
            checkpoint(&mut budget, 0).await?;
            document.groups.insert(group, candidate);
            Ok(())
        })
    }
}

impl RuntimeUserPathService for IsolatedMemoryPreferenceService {
    fn read(
        &self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>> {
        let document = self.document.clone();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let document = budget
                .await_cancellable(document.lock())
                .await
                .map_err(storage)?;
            checkpoint(&mut budget, 1).await?;
            Ok(UserPathSnapshot {
                setting: document.user_path.clone(),
                revision: document.user_path_revision,
            })
        })
    }

    fn commit(
        &self,
        runtime: RuntimeContext,
        mutation: UserPathMutation,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>> {
        let document = self.document.clone();
        Box::pin(async move {
            let setting = mutation.into_setting()?;
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let mut document = budget
                .await_cancellable(document.lock())
                .await
                .map_err(storage)?;
            let revision = document.user_path_revision.checked_add(1).ok_or_else(|| {
                PreferenceServiceError::InvalidTransaction(
                    "application user-path revision is exhausted".into(),
                )
            })?;
            checkpoint(&mut budget, 1).await?;
            document.user_path = setting.clone();
            document.user_path_revision = revision;
            Ok(UserPathSnapshot { setting, revision })
        })
    }
}

async fn checkpoint(
    budget: &mut CpuCooperationBudget,
    work: u64,
) -> Result<(), PreferenceServiceError> {
    budget.checkpoint(work).await.map_err(storage)
}

fn storage(source: crate::execution::ExecutionServiceError) -> PreferenceServiceError {
    PreferenceServiceError::Execution(source)
}

#[cfg(test)]
#[path = "memory/tests.rs"]
mod tests;
