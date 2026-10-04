//! Browser application-owned settings authority (Window and Worker IndexedDB).

#[path = "browser/connection.rs"]
mod connection;
#[path = "browser/store.rs"]
mod store;
#[path = "browser/transaction.rs"]
mod transaction;

use super::document::{DocumentMutation, DocumentMutationError, PreferenceDocumentError};
use super::{PreferenceGroups, PreferenceServiceError, PreferenceValues, RuntimePreferenceService};
use crate::context::{
    RuntimeContext, RuntimeServiceFuture, RuntimeUserPathService, UserPathMutation,
    UserPathServiceError, UserPathSnapshot,
};
use crate::execution::value_codec::{decode_durable_value, encode_durable_value};
use crate::execution::CpuCooperationBudget;
use futures::lock::Mutex;
use runmat_value::Value;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;
use wasm_bindgen::JsValue;

#[derive(Clone)]
pub struct BrowserPreferenceService {
    namespace: String,
    gate: Arc<Mutex<()>>,
}

impl BrowserPreferenceService {
    /// The host supplies an application/user namespace; browser origin remains
    /// an additional IndexedDB isolation boundary. Construction has no effect.
    pub fn new(namespace: String) -> Result<Self, PreferenceServiceError> {
        if namespace.is_empty() {
            return Err(PreferenceServiceError::InvalidTransaction(
                "browser application settings namespace cannot be empty".into(),
            ));
        }
        Ok(Self {
            namespace: format!("runmat.application.settings.v1/{namespace}"),
            gate: Arc::new(Mutex::new(())),
        })
    }

    pub fn anonymous_origin_local() -> Self {
        Self::new("anonymous".into()).expect("fixed nonempty namespace")
    }

    async fn read_document(
        &self,
        budget: &mut CpuCooperationBudget,
    ) -> Result<super::document::PreferenceDocument, PreferenceServiceError> {
        checkpoint(budget, 0).await?;
        let database = budget
            .await_cancellable(connection::open(&self.namespace))
            .await
            .map_err(PreferenceServiceError::Execution)?
            .map_err(read_error)?;
        budget
            .await_cancellable(store::read_document(&database))
            .await
            .map_err(PreferenceServiceError::Execution)?
            .map_err(read_error)
    }

    async fn mutate(
        &self,
        budget: &mut CpuCooperationBudget,
        mutation: DocumentMutation,
    ) -> Result<transaction::BrowserPrepared, PreferenceServiceError> {
        let gate = budget
            .await_cancellable(self.gate.clone().lock_owned())
            .await
            .map_err(PreferenceServiceError::Execution)?;
        let database = budget
            .await_cancellable(connection::open(&self.namespace))
            .await
            .map_err(PreferenceServiceError::Execution)?
            .map_err(read_error)?;
        // The readwrite transaction starts only after the final cancellable
        // authorization checkpoint; its observer survives a dropped caller.
        checkpoint(budget, 0).await?;
        transaction::mutate(database, mutation, gate)
            .await
            .map_err(commit_error)
    }
}

impl RuntimePreferenceService for BrowserPreferenceService {
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        Some(())
    }

    fn all_groups(
        &self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<PreferenceGroups, PreferenceServiceError>> {
        let service = self.clone();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            let document = service.read_document(&mut budget).await?;
            let mut output = PreferenceGroups::new();
            for (group, values) in document.groups {
                let mut decoded = PreferenceValues::new();
                for (name, payload) in values {
                    decoded.insert(name, decode_durable_value(&payload)?);
                    checkpoint(&mut budget, 1).await?;
                }
                output.insert(group, decoded);
                checkpoint(&mut budget, 1).await?;
            }
            Ok(output)
        })
    }

    fn group_values(
        &self,
        runtime: RuntimeContext,
        group: String,
    ) -> RuntimeServiceFuture<Result<Option<PreferenceValues>, PreferenceServiceError>> {
        let service = self.clone();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            let document = service.read_document(&mut budget).await?;
            let Some(values) = document.groups.get(&group) else {
                return Ok(None);
            };
            let mut decoded = PreferenceValues::new();
            for (name, payload) in values {
                decoded.insert(name.clone(), decode_durable_value(payload)?);
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
        let service = self.clone();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            let document = service.read_document(&mut budget).await?;
            let values = document.groups.get(&group);
            let mut output = Vec::with_capacity(names.len());
            for name in names {
                output.push(values.is_some_and(|values| values.contains_key(&name)));
                checkpoint(&mut budget, 1).await?;
            }
            Ok(output)
        })
    }

    fn read_or_create_many(
        &self,
        runtime: RuntimeContext,
        group: String,
        names: Vec<String>,
        defaults: Option<Vec<Value>>,
    ) -> RuntimeServiceFuture<Result<Vec<Value>, PreferenceServiceError>> {
        let service = self.clone();
        Box::pin(async move {
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
            let mut encoded = None;
            if let Some(defaults) = defaults {
                let mut values = Vec::with_capacity(defaults.len());
                for value in defaults {
                    values.push(encode_durable_value(&value)?);
                    checkpoint(&mut budget, 1).await?;
                }
                encoded = Some(values);
            }
            if names.is_empty() {
                return Ok(Vec::new());
            }
            if encoded.is_none() {
                let document = service.read_document(&mut budget).await?;
                let mut output = Vec::with_capacity(names.len());
                for name in names {
                    let payload = document
                        .groups
                        .get(&group)
                        .and_then(|values| values.get(&name))
                        .ok_or_else(|| PreferenceServiceError::Missing {
                            group: group.clone(),
                            name: name.clone(),
                        })?;
                    output.push(decode_durable_value(payload)?);
                    checkpoint(&mut budget, 1).await?;
                }
                return Ok(output);
            }
            let prepared = service
                .mutate(
                    &mut budget,
                    DocumentMutation::ReadOrCreate {
                        group,
                        names,
                        defaults: encoded,
                    },
                )
                .await?;
            prepared
                .values
                .iter()
                .map(decode_durable_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(PreferenceServiceError::ValueEncoding)
        })
    }

    fn set_many(
        &self,
        runtime: RuntimeContext,
        group: String,
        entries: Vec<(String, Value)>,
    ) -> RuntimeServiceFuture<Result<(), PreferenceServiceError>> {
        let service = self.clone();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let mut encoded = Vec::with_capacity(entries.len());
            for (name, value) in entries {
                encoded.push((name, encode_durable_value(&value)?));
                checkpoint(&mut budget, 1).await?;
            }
            if encoded.is_empty() {
                return Ok(());
            }
            service
                .mutate(
                    &mut budget,
                    DocumentMutation::Set {
                        group,
                        entries: encoded,
                    },
                )
                .await?;
            Ok(())
        })
    }
}

impl RuntimeUserPathService for BrowserPreferenceService {
    fn read(
        &self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>> {
        let service = self.clone();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            let document = service.read_document(&mut budget).await?;
            checkpoint(&mut budget, 1).await?;
            Ok(UserPathSnapshot {
                setting: document.user_path,
                revision: document.user_path_revision,
            })
        })
    }

    fn commit(
        &self,
        runtime: RuntimeContext,
        mutation: UserPathMutation,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>> {
        let service = self.clone();
        Box::pin(async move {
            let setting = mutation.into_setting()?;
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let prepared = service
                .mutate(&mut budget, DocumentMutation::UserPath(setting))
                .await?;
            Ok(prepared
                .user_path
                .expect("user-path mutation prepares a setting snapshot"))
        })
    }
}

#[derive(Debug, thiserror::Error)]
enum BrowserPreferenceError {
    #[error("IndexedDB is unavailable in this browser scope")]
    Unavailable,
    #[error("application settings outcome could not be observed")]
    LostObservation,
    #[error("invalid stored application settings value")]
    InvalidStoredValue,
    #[error("browser storage failed: {0}")]
    Browser(#[source] BrowserJsError),
    #[error(transparent)]
    Document(#[from] PreferenceDocumentError),
    #[error(transparent)]
    Mutation(#[from] DocumentMutationError),
}

#[derive(Debug)]
struct BrowserJsError(JsValue);

impl fmt::Display for BrowserJsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", self.0)
    }
}

impl std::error::Error for BrowserJsError {}

fn browser_error(source: JsValue) -> BrowserPreferenceError {
    BrowserPreferenceError::Browser(BrowserJsError(source))
}

fn read_error(error: BrowserPreferenceError) -> PreferenceServiceError {
    match error {
        BrowserPreferenceError::Document(source) => source.into(),
        BrowserPreferenceError::Mutation(DocumentMutationError::Missing { group, name }) => {
            PreferenceServiceError::Missing { group, name }
        }
        BrowserPreferenceError::Mutation(DocumentMutationError::RevisionExhausted) => {
            PreferenceServiceError::InvalidTransaction(
                "application user-path revision is exhausted".into(),
            )
        }
        other => PreferenceServiceError::Storage(
            crate::build_runtime_error("browser application settings read failed")
                .with_source(other)
                .build(),
        ),
    }
}

fn commit_error(error: BrowserPreferenceError) -> PreferenceServiceError {
    if matches!(error, BrowserPreferenceError::LostObservation) {
        return PreferenceServiceError::CommitIndeterminate(
            crate::build_runtime_error("browser application settings commit was not observed")
                .with_source(error)
                .build(),
        );
    }
    read_error(error)
}

async fn checkpoint(
    budget: &mut CpuCooperationBudget,
    work: u64,
) -> Result<(), PreferenceServiceError> {
    budget
        .checkpoint(work)
        .await
        .map_err(PreferenceServiceError::Execution)
}

#[cfg(test)]
#[path = "browser/tests.rs"]
mod tests;
