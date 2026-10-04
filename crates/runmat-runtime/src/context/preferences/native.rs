//! Native app-owned preference service. One active write per service; an OS
//! file lock coordinates separate processes and independently created hosts.

use super::document::PreferenceDocument;
use super::{PreferenceGroups, PreferenceServiceError, PreferenceValues, RuntimePreferenceService};
use crate::context::{RuntimeContext, RuntimeServiceFuture};
use crate::execution::value_codec::{decode_durable_value, encode_durable_value};
use crate::execution::CpuCooperationBudget;
use runmat_execution::value::ValuePayload;
use runmat_value::Value;
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use tokio::sync::{oneshot, Semaphore};

#[path = "native/store.rs"]
mod store;
use store::{NativeStore, NativeStoreError};

#[derive(Clone)]
pub struct NativePreferenceService {
    store: NativeStore,
    driver: tokio::runtime::Handle,
    gate: Arc<Semaphore>,
}

impl NativePreferenceService {
    /// `path` is an absolute application/user settings file supplied by the
    /// host; construction performs no filesystem effect.
    pub fn new(
        path: PathBuf,
        driver: tokio::runtime::Handle,
    ) -> Result<Self, PreferenceServiceError> {
        Ok(Self {
            store: NativeStore::new(path).map_err(PreferenceServiceError::Io)?,
            driver,
            gate: Arc::new(Semaphore::new(1)),
        })
    }

    async fn read(
        &self,
        budget: &mut CpuCooperationBudget,
    ) -> Result<PreferenceDocument, PreferenceServiceError> {
        let store = self.store.clone();
        let task = self.driver.spawn_blocking(move || store.read());
        budget
            .await_cancellable(task)
            .await
            .map_err(cancelled)?
            .map_err(join_failed)?
            .map_err(store_error)
    }

    async fn mutate(
        &self,
        runtime: RuntimeContext,
        mut budget: CpuCooperationBudget,
        mutation: Mutation,
    ) -> Result<Vec<Value>, PreferenceServiceError> {
        checkpoint(&mut budget, 0).await?;
        let permit = budget
            .await_cancellable(self.gate.clone().acquire_owned())
            .await
            .map_err(cancelled)?
            .map_err(|source| operational("preference write gate closed", source))?;
        let (prepared_tx, prepared_rx) = oneshot::channel();
        let (authorized_tx, authorized_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = oneshot::channel();
        let store = self.store.clone();
        // The blocking transaction owns its permit, lock, stage and completion
        // after submission. The caller owns only the authorization sender.
        self.driver.spawn_blocking(move || {
            let _permit = permit;
            worker(store, mutation, prepared_tx, authorized_rx, finished_tx);
        });
        let prepared = budget
            .await_cancellable(prepared_rx)
            .await
            .map_err(cancelled)?
            .map_err(|source| operational("preference preparation ended", source))?
            .map_err(transaction_error)?;
        let mut values = Vec::with_capacity(prepared.values.len());
        for payload in &prepared.values {
            values.push(decode_durable_value(payload)?);
            checkpoint(&mut budget, 1).await?;
        }
        if prepared.commit_required {
            checkpoint(&mut budget, 0).await?;
            authorized_tx.send(()).map_err(|_| {
                operational(
                    "preference authorization owner ended",
                    std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "preference worker ended before authorization",
                    ),
                )
            })?;
            // Cancellation after authorization is not proof of rollback.
            finished_rx
                .await
                .map_err(|source| {
                    indeterminate("preference commit outcome was not observed", source)
                })?
                .map_err(store_error)?;
        }
        Ok(values)
    }
}

impl RuntimePreferenceService for NativePreferenceService {
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        Some(()) // The host retains encoded documents, never runtime Value owners.
    }

    fn all_groups(
        &self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<PreferenceGroups, PreferenceServiceError>> {
        let service = self.clone();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let document = service.read(&mut budget).await?;
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
            checkpoint(&mut budget, 0).await?;
            let document = service.read(&mut budget).await?;
            let Some(values) = document.groups.get(&group) else {
                return Ok(None);
            };
            let mut output = PreferenceValues::new();
            for (name, payload) in values {
                output.insert(name.clone(), decode_durable_value(payload)?);
                checkpoint(&mut budget, 1).await?;
            }
            Ok(Some(output))
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
            checkpoint(&mut budget, 0).await?;
            let document = service.read(&mut budget).await?;
            let values = document.groups.get(&group);
            let mut output = Vec::with_capacity(names.len());
            for name in &names {
                output.push(values.is_some_and(|values| values.contains_key(name)));
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
                for value in &defaults {
                    values.push(encode_durable_value(value)?);
                    checkpoint(&mut budget, 1).await?;
                }
                encoded = Some(values);
            }
            if names.is_empty() {
                return Ok(Vec::new());
            }
            if encoded.is_none() {
                let document = service.read(&mut budget).await?;
                let mut output = Vec::with_capacity(names.len());
                for name in &names {
                    let payload = document
                        .groups
                        .get(&group)
                        .and_then(|values| values.get(name))
                        .ok_or_else(|| PreferenceServiceError::Missing {
                            group: group.clone(),
                            name: name.clone(),
                        })?;
                    output.push(decode_durable_value(payload)?);
                    checkpoint(&mut budget, 1).await?;
                }
                return Ok(output);
            }
            service
                .mutate(
                    runtime,
                    budget,
                    Mutation::ReadOrCreate {
                        group,
                        names,
                        defaults: encoded,
                    },
                )
                .await
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
            for (name, value) in &entries {
                encoded.push((name.clone(), encode_durable_value(value)?));
                checkpoint(&mut budget, 1).await?;
            }
            if encoded.is_empty() {
                return Ok(());
            }
            service
                .mutate(
                    runtime,
                    budget,
                    Mutation::Set {
                        group,
                        entries: encoded,
                    },
                )
                .await?;
            Ok(())
        })
    }
}

enum Mutation {
    Set {
        group: String,
        entries: Vec<(String, ValuePayload)>,
    },
    ReadOrCreate {
        group: String,
        names: Vec<String>,
        defaults: Option<Vec<ValuePayload>>,
    },
}

struct Prepared {
    values: Vec<ValuePayload>,
    commit_required: bool,
}

#[derive(Debug, thiserror::Error)]
enum TransactionError {
    #[error(transparent)]
    Store(#[from] NativeStoreError),
    #[error("preference '{name}' does not exist in group '{group}'")]
    Missing { group: String, name: String },
}

fn worker(
    store: NativeStore,
    mutation: Mutation,
    prepared: oneshot::Sender<Result<Prepared, TransactionError>>,
    authorized: mpsc::Receiver<()>,
    finished: oneshot::Sender<Result<(), NativeStoreError>>,
) {
    let preparation = (|| {
        let mut write = store.begin_write()?;
        let mut values = Vec::new();
        let mut changed = false;
        match mutation {
            Mutation::Set { group, entries } => {
                let target = write.document.groups.entry(group).or_default();
                for (name, payload) in entries {
                    target.insert(name, payload);
                }
                changed = true;
            }
            Mutation::ReadOrCreate {
                group,
                names,
                defaults,
            } => {
                for (index, name) in names.iter().enumerate() {
                    let target = write.document.groups.entry(group.clone()).or_default();
                    if !target.contains_key(name) {
                        let Some(default) = defaults.as_ref().and_then(|values| values.get(index))
                        else {
                            return Err(TransactionError::Missing {
                                group,
                                name: name.clone(),
                            });
                        };
                        target.insert(name.clone(), default.clone());
                        changed = true;
                    }
                    values.push(
                        target
                            .get(name)
                            .expect("existing or inserted preference")
                            .clone(),
                    );
                }
            }
        }
        let stage = if changed { Some(write.stage()?) } else { None };
        Ok((
            Prepared {
                values,
                commit_required: changed,
            },
            stage,
        ))
    })();
    match preparation {
        Err(error) => {
            let _ = prepared.send(Err(error));
        }
        Ok((receipt, stage)) => {
            let needs_commit = receipt.commit_required;
            if prepared.send(Ok(receipt)).is_err() {
                return;
            }
            if !needs_commit {
                return;
            }
            if authorized.recv().is_err() {
                return;
            }
            let result = stage.expect("changed document has a stage").commit();
            let _ = finished.send(result);
        }
    }
}

async fn checkpoint(
    budget: &mut CpuCooperationBudget,
    work: u64,
) -> Result<(), PreferenceServiceError> {
    budget.checkpoint(work).await.map_err(cancelled)
}

fn cancelled(source: crate::execution::ExecutionServiceError) -> PreferenceServiceError {
    PreferenceServiceError::Storage(source.into_runtime_error())
}

fn transaction_error(source: TransactionError) -> PreferenceServiceError {
    match source {
        TransactionError::Store(source) => store_error(source),
        TransactionError::Missing { group, name } => {
            PreferenceServiceError::Missing { group, name }
        }
    }
}

fn store_error(source: NativeStoreError) -> PreferenceServiceError {
    match source {
        NativeStoreError::Io(source) => PreferenceServiceError::Io(source),
        NativeStoreError::Document(source) => source.into(),
        NativeStoreError::Indeterminate(source) => indeterminate(
            "preference replacement occurred but durability is unknown",
            source,
        ),
    }
}

fn operational(
    message: &'static str,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> PreferenceServiceError {
    PreferenceServiceError::Storage(
        crate::build_runtime_error(message)
            .with_source(source)
            .build(),
    )
}

fn indeterminate(
    message: &'static str,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> PreferenceServiceError {
    PreferenceServiceError::CommitIndeterminate(
        crate::build_runtime_error(message)
            .with_source(source)
            .build(),
    )
}

fn join_failed(source: tokio::task::JoinError) -> PreferenceServiceError {
    operational("preference I/O worker ended", source)
}

#[cfg(test)]
#[path = "native/tests.rs"]
mod tests;
