//! Native app-owned preference service. One active write per service; an OS
//! file lock coordinates separate processes and independently created hosts.

use super::document::PreferenceDocument;
use super::{PreferenceGroups, PreferenceServiceError, PreferenceValues, RuntimePreferenceService};
use crate::context::{
    RuntimeContext, RuntimeServiceFuture, RuntimeUserPathService, UserPathMutation,
    UserPathServiceError, UserPathSnapshot,
};
use crate::execution::value_codec::{decode_durable_value, encode_durable_value};
use crate::execution::CpuCooperationBudget;
use runmat_value::Value;
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use tokio::sync::{oneshot, Semaphore};

#[path = "native/store.rs"]
mod store;
use store::{NativeStore, NativeStoreError, NativeWrite};
#[path = "native/transaction.rs"]
mod transaction;
use transaction::{worker, Mutation, TransactionError};

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
        let permit = budget
            .await_cancellable(self.gate.clone().acquire_owned())
            .await
            .map_err(cancelled)?
            .map_err(|source| operational("preference read gate closed", source))?;
        let store = self.store.clone();
        let task = self.driver.spawn_blocking(move || {
            let _permit = permit;
            store.read()
        });
        budget
            .await_cancellable(task)
            .await
            .map_err(cancelled)?
            .map_err(join_failed)?
            .map_err(store_error)
    }

    async fn mutate(
        &self,
        mut budget: CpuCooperationBudget,
        mutation: Mutation,
    ) -> Result<(Vec<Value>, Option<UserPathSnapshot>), PreferenceServiceError> {
        checkpoint(&mut budget, 0).await?;
        let permit = budget
            .await_cancellable(self.gate.clone().acquire_owned())
            .await
            .map_err(cancelled)?
            .map_err(|source| operational("preference write gate closed", source))?;
        let (prepared_tx, prepared_rx) = oneshot::channel();
        let (authorized_tx, authorized_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = oneshot::channel();
        let write = self.acquire_write(&mut budget).await?;
        // The blocking transaction owns its permit, lock, stage and completion
        // after submission. The caller owns only the authorization sender.
        self.driver.spawn_blocking(move || {
            let _permit = permit;
            worker(write, mutation, prepared_tx, authorized_rx, finished_tx);
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
        Ok((values, prepared.user_path))
    }

    async fn acquire_write(
        &self,
        budget: &mut CpuCooperationBudget,
    ) -> Result<NativeWrite, PreferenceServiceError> {
        loop {
            checkpoint(budget, 0).await?;
            let store = self.store.clone();
            let attempt = self.driver.spawn_blocking(move || store.try_begin_write());
            let maybe_write = budget
                .await_cancellable(attempt)
                .await
                .map_err(cancelled)?
                .map_err(join_failed)?
                .map_err(store_error)?;
            if let Some(write) = maybe_write {
                return Ok(write);
            }
            // The host driver's timer yields while another process owns the
            // advisory lock. No blocking worker waits for that lock, and an
            // abandoned invocation schedules no further attempts.
            let timer = {
                let _entered = self.driver.enter();
                tokio::time::sleep(std::time::Duration::from_millis(10))
            };
            budget.await_cancellable(timer).await.map_err(cancelled)?;
        }
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
                    budget,
                    Mutation::ReadOrCreate {
                        group,
                        names,
                        defaults: encoded,
                    },
                )
                .await
                .map(|(values, _)| values)
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

impl RuntimeUserPathService for NativePreferenceService {
    fn read(
        &self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>> {
        let service = self.clone();
        Box::pin(async move {
            let mut budget = runtime.cpu_cooperation();
            checkpoint(&mut budget, 0).await?;
            let document = service.read(&mut budget).await?;
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
            let (_, snapshot) = service.mutate(budget, Mutation::UserPath(setting)).await?;
            Ok(snapshot.expect("user-path mutation prepares a setting snapshot"))
        })
    }
}

async fn checkpoint(
    budget: &mut CpuCooperationBudget,
    work: u64,
) -> Result<(), PreferenceServiceError> {
    budget.checkpoint(work).await.map_err(cancelled)
}

fn cancelled(source: crate::execution::ExecutionServiceError) -> PreferenceServiceError {
    PreferenceServiceError::Execution(source)
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
        NativeStoreError::RevisionExhausted => PreferenceServiceError::InvalidTransaction(
            "application user-path revision is exhausted".into(),
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
