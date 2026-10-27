//! Browser application-owned settings authority (Window and Worker IndexedDB).

#[path = "browser/connection.rs"]
mod connection;
#[path = "browser/retirement.rs"]
mod retirement;
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
use crate::execution::CpuCooperationBudget;
use futures::lock::Mutex;
use runmat_execution::profiling::CommandProfilingReceipt;
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
        profiling: Option<&CommandProfilingReceipt>,
    ) -> Result<super::document::PreferenceDocument, PreferenceServiceError> {
        checkpoint(budget, 0).await?;
        let database = budget
            .await_cancellable(connection::open(&self.namespace, profiling))
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
        profiling: Option<&CommandProfilingReceipt>,
    ) -> Result<transaction::BrowserCheckedSource, PreferenceServiceError> {
        let result = self
            .mutate_request(
                budget,
                transaction::BrowserMutation::Prepared(mutation),
                profiling,
            )
            .await?;
        match result {
            completed @ transaction::BrowserCheckedSource::Completed { .. } => Ok(completed),
            transaction::BrowserCheckedSource::NeedsPreparation(_) => {
                unreachable!("prepared mutation owns no caller defaults")
            }
        }
    }

    async fn mutate_request(
        &self,
        budget: &mut CpuCooperationBudget,
        mutation: transaction::BrowserMutation,
        profiling: Option<&CommandProfilingReceipt>,
    ) -> Result<transaction::BrowserCheckedSource, PreferenceServiceError> {
        let gate = budget
            .await_cancellable(self.gate.clone().lock_owned())
            .await
            .map_err(PreferenceServiceError::Execution)?;
        let database = budget
            .await_cancellable(connection::open(&self.namespace, profiling))
            .await
            .map_err(PreferenceServiceError::Execution)?
            .map_err(read_error)?;
        // The readwrite transaction starts only after the final cancellable
        // authorization checkpoint; its observer survives a dropped caller.
        checkpoint(budget, 0).await?;
        transaction::mutate(database, mutation, gate, profiling)
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
        let profiling = runtime.command_profiling().cloned();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            let document = service
                .read_document(&mut budget, profiling.as_ref())
                .await?;
            let mut output = PreferenceGroups::new();
            for (group, values) in document.groups {
                let mut decoded = PreferenceValues::new();
                for (name, payload) in values {
                    decoded.insert(
                        name,
                        super::decode_selected_preference(&payload, &mut budget, &runtime).await?,
                    );
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
        let profiling = runtime.command_profiling().cloned();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            let document = service
                .read_document(&mut budget, profiling.as_ref())
                .await?;
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
        let service = self.clone();
        let profiling = runtime.command_profiling().cloned();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            let document = service
                .read_document(&mut budget, profiling.as_ref())
                .await?;
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
        let profiling = runtime.command_profiling().cloned();
        let rooted_defaults = defaults
            .map(|values| {
                super::input_roots::PreferenceInputRoots::for_values(&values)
                    .map(|roots| (values, roots))
            })
            .transpose();
        Box::pin(async move {
            let rooted_defaults = rooted_defaults?;
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            if rooted_defaults
                .as_ref()
                .is_some_and(|(defaults, _)| defaults.len() != names.len())
            {
                return Err(PreferenceServiceError::InvalidTransaction(
                    "default and preference-name cardinalities differ".into(),
                ));
            }
            if names.is_empty() {
                return Ok(Vec::new());
            }
            let Some((defaults, roots)) = rooted_defaults else {
                let document = service
                    .read_document(&mut budget, profiling.as_ref())
                    .await?;
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
                    output.push(
                        super::decode_selected_preference(payload, &mut budget, &runtime).await?,
                    );
                    checkpoint(&mut budget, 1).await?;
                }
                return Ok(output);
            };
            let shared = Rc::new(transaction::SharedDefaults::new(defaults, roots));
            loop {
                checkpoint(&mut budget, 0).await?;
                let prepared = service
                    .mutate_request(
                        &mut budget,
                        transaction::BrowserMutation::ReadOrCreate {
                            group: group.clone(),
                            names: names.clone(),
                            defaults: shared.clone(),
                        },
                        profiling.as_ref(),
                    )
                    .await?;
                match prepared {
                    transaction::BrowserCheckedSource::NeedsPreparation(indices) => {
                        if indices.is_empty() {
                            return Err(PreferenceServiceError::InvalidTransaction(
                                "browser preference default selection made no progress".into(),
                            ));
                        }
                        for index in indices {
                            let payload = super::encode_selected_preference(
                                shared.value(index),
                                &mut budget,
                                &runtime,
                            )
                            .await?;
                            shared.mark_prepared(index, payload);
                        }
                    }
                    transaction::BrowserCheckedSource::Completed { values, .. } => {
                        let mut output = Vec::with_capacity(values.len());
                        for payload in &values {
                            output.push(
                                super::decode_selected_preference(payload, &mut budget, &runtime)
                                    .await?,
                            );
                        }
                        return Ok(output);
                    }
                }
            }
        })
    }

    fn set_many(
        &self,
        runtime: RuntimeContext,
        group: String,
        entries: Vec<(String, Value)>,
    ) -> RuntimeServiceFuture<Result<(), PreferenceServiceError>> {
        let service = self.clone();
        let profiling = runtime.command_profiling().cloned();
        let roots = super::input_roots::PreferenceInputRoots::for_entries(&entries);
        Box::pin(async move {
            let _roots = roots?;
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let mut encoded = Vec::with_capacity(entries.len());
            for (name, value) in entries {
                encoded.push((
                    name,
                    super::encode_selected_preference(&value, &mut budget, &runtime).await?,
                ));
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
                    profiling.as_ref(),
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
        let profiling = runtime.command_profiling().cloned();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            let document = service
                .read_document(&mut budget, profiling.as_ref())
                .await?;
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
        let profiling = runtime.command_profiling().cloned();
        Box::pin(async move {
            let setting = mutation.into_setting()?;
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let prepared = service
                .mutate(
                    &mut budget,
                    DocumentMutation::UserPath(setting),
                    profiling.as_ref(),
                )
                .await?;
            let transaction::BrowserCheckedSource::Completed { user_path, .. } = prepared else {
                unreachable!("user-path mutation owns no caller defaults")
            };
            Ok(user_path.expect("user-path mutation prepares a setting snapshot"))
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
