//! Parent-local retention of original child scheduler console evidence.
use super::*;
use runmat_runtime::context::{
    RuntimeConsoleAdmission, RuntimeConsoleScopeIdentity, RuntimeConsoleScopeRef, RuntimeContext,
    RuntimeRetainedConsoleSource,
};

pub(super) enum LocalConsoleAuthority {
    /// Trusted standalone Rust submissions have their own original task grant;
    /// they make no claim about retention by an outer captured invocation.
    StandaloneOriginalTask,
    Parent(Arc<dyn RuntimeConsoleAdmission>),
}

struct ChildDriverConsoleSource {
    driver: Arc<LocalDriver>,
}

impl RuntimeRetainedConsoleSource for ChildDriverConsoleSource {
    fn source_identity(&self) -> Option<(ExecutionScopeId, PoolId)> {
        Some((self.driver.scope_id, self.driver.pool_id))
    }

    fn visit(
        &self,
        visitor: &mut dyn FnMut(RuntimeConsoleScopeRef<'_>) -> Result<(), String>,
    ) -> Result<(), String> {
        let driver = self
            .driver
            .driver
            .lock()
            .map_err(|_| "child console driver poisoned")?;
        let owned = self
            .driver
            .parent_console_attempts
            .lock()
            .map_err(|_| "child console ownership poisoned")?;
        // This borrow is the original scheduler inventory. No snapshot(),
        // TaskExecutionEvidence clone, or transcript-sized staging copy.
        for attempt in driver.snapshot_ref().attempts.values() {
            if !owned.contains(&attempt.request.id) {
                continue;
            }
            let console = attempt
                .terminal
                .as_ref()
                .and_then(|terminal| terminal.console.as_deref())
                .or_else(|| {
                    attempt
                        .unverified_retirement
                        .as_ref()
                        .and_then(|retirement| retirement.console.as_deref())
                });
            if let Some(transcript) = console {
                visitor(RuntimeConsoleScopeRef {
                    identity: self.driver.console_identity(&attempt.request),
                    transcript,
                })?;
            }
            let scopes = attempt
                .terminal
                .as_ref()
                .map(|terminal| terminal.console_scopes.as_slice())
                .or_else(|| {
                    attempt
                        .unverified_retirement
                        .as_ref()
                        .map(|retirement| retirement.console_scopes.as_slice())
                })
                .unwrap_or(&[]);
            for scope in scopes {
                visitor(RuntimeConsoleScopeRef {
                    identity: RuntimeConsoleScopeIdentity {
                        scope_id: scope.scope_id,
                        pool_id: scope.pool_id,
                        task_id: scope.task_id,
                        attempt_id: scope.attempt_id,
                    },
                    transcript: &scope.transcript,
                })?;
            }
        }
        Ok(())
    }
}

impl LocalDriver {
    pub(crate) fn bind_console_parent(
        self: &Arc<Self>,
        runtime: &RuntimeContext,
        requested: bool,
    ) -> Result<(), runmat_runtime::execution::ExecutionServiceError> {
        if !requested {
            return Ok(());
        }
        let host = runtime
            .service_ports()
            .host()
            .filter(|host| host.console_limits().is_some())
            .ok_or_else(|| {
                runmat_runtime::execution::ExecutionServiceError::Infrastructure(
                    "requested child console capture has no retaining parent host".into(),
                )
            })?;
        let owner = host.console_admission().ok_or_else(|| {
            runmat_runtime::execution::ExecutionServiceError::Infrastructure(
                "requested child console capture has no authenticated root admission owner".into(),
            )
        })?;
        self.install_console_parent(owner)?;
        host.retain_console_source(self.retained_console_source())
            .map_err(Into::into)
    }

    /// Install an explicitly inherited admission authority for an owned child
    /// driver. The caller retains the returned source through terminal encode.
    pub(crate) fn install_console_parent(
        &self,
        owner: Arc<dyn RuntimeConsoleAdmission>,
    ) -> Result<(), runmat_runtime::execution::ExecutionServiceError> {
        {
            let mut saved = self
                .console_admission
                .lock()
                .expect("local console admission poisoned");
            if let LocalConsoleAuthority::Parent(original) = &*saved {
                if !Arc::ptr_eq(original, &owner) {
                    return Err(
                        runmat_runtime::execution::ExecutionServiceError::Infrastructure(
                            "local child driver cannot replace its root console admission owner"
                                .into(),
                        ),
                    );
                }
            } else {
                *saved = LocalConsoleAuthority::Parent(owner);
            }
        }
        Ok(())
    }

    pub(crate) fn retained_console_source(
        self: &Arc<Self>,
    ) -> Arc<dyn RuntimeRetainedConsoleSource> {
        Arc::new(ChildDriverConsoleSource {
            driver: Arc::clone(self),
        })
    }

    /// Identify the physical Graph root without substituting another scope or
    /// an earlier failed retry for the scheduler's committed success.
    pub(crate) fn delegated_console_root(
        &self,
        task_id: TaskId,
        require_committed: bool,
    ) -> NativeExecutionResult<Option<runmat_execution::console::ConsoleSourceIdentity>> {
        let driver = self.driver.lock().expect("local driver poisoned");
        let task = driver
            .task_record(task_id)
            .map_err(NativeExecutionError::Driver)?;
        let attempt = if require_committed {
            let committed = task.committed.as_ref().ok_or_else(|| {
                NativeExecutionError::Protocol("captured Graph success has no result commit".into())
            })?;
            if !matches!(&committed.result, AttemptSuccess::Graph { .. }) {
                return Err(NativeExecutionError::Protocol(
                    "captured Graph commit contains a non-Graph result".into(),
                ));
            }
            driver.snapshot_ref().attempts.get(&committed.attempt_id)
        } else {
            driver
                .snapshot_ref()
                .attempts
                .values()
                .filter(|attempt| {
                    attempt.request.task_id == task_id
                        && attempt
                            .terminal
                            .as_ref()
                            .is_some_and(|terminal| terminal.console.is_some())
                })
                .max_by_key(|attempt| attempt.request.ordinal)
        };
        let Some(attempt) = attempt else {
            if require_committed {
                return Err(NativeExecutionError::Protocol(
                    "captured Graph commit has no original physical attempt".into(),
                ));
            }
            return Ok(None);
        };
        let console = attempt
            .terminal
            .as_ref()
            .and_then(|terminal| terminal.console.as_deref());
        let owned = self
            .parent_console_attempts
            .lock()
            .expect("local console ownership poisoned");
        if attempt.request.task_id != task_id || !owned.contains(&attempt.request.id) {
            return Err(NativeExecutionError::Protocol(
                "delegated Graph root is not an originally admitted captured task attempt".into(),
            ));
        }
        if require_committed
            && !console.is_some_and(|console| {
                console.completion() == runmat_execution::console::ConsoleCompletion::Complete
            })
        {
            return Err(NativeExecutionError::Protocol(
                "captured Graph commit lacks its complete original root transcript".into(),
            ));
        }
        Ok(Some(self.console_identity(&attempt.request)))
    }

    pub(crate) fn console_identity(&self, attempt: &AttemptRequest) -> RuntimeConsoleScopeIdentity {
        RuntimeConsoleScopeIdentity {
            scope_id: attempt.scope_id,
            pool_id: self.pool_id,
            task_id: attempt.task_id,
            attempt_id: attempt.id,
        }
    }

    pub(crate) fn console_admission_owner(&self) -> Option<Arc<dyn RuntimeConsoleAdmission>> {
        match &*self
            .console_admission
            .lock()
            .expect("local console admission poisoned")
        {
            LocalConsoleAuthority::StandaloneOriginalTask => None,
            LocalConsoleAuthority::Parent(owner) => Some(Arc::clone(owner)),
        }
    }

    pub(crate) fn console_subgrant(
        &self,
        request: &AttemptRequest,
    ) -> Result<Option<runmat_execution::console::ConsoleBudget>, TransferFailure> {
        let Some(controls) = request.task.console else {
            return Ok(None);
        };
        let quoted = self
            .quote_console_subgrant(Some(controls), request.task.resources.memory_bytes)
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        let floor = runmat_execution::console::invocation_memory_bytes(
            &request.task.environment,
            Some(controls),
        )
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?
        .checked_add(runmat_process_host::ipc::WORKER_PARENT_MONITOR_STACK_BYTES)
        .ok_or_else(|| {
            TransferFailure::Infrastructure("console invocation floor overflow".into())
        })?;
        let memory = crate::console_retention::allocation_floor(floor, Some(controls), quoted)
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        let encoded = crate::console_retention::artifact_floor(0, Some(controls), quoted)
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        if memory > request.task.resources.memory_bytes
            || encoded > request.task.resources.max_artifact_bytes
        {
            return Err(TransferFailure::Infrastructure(
                "console subtree exceeds the original child task reservation".into(),
            ));
        }
        match &*self
            .console_admission
            .lock()
            .expect("local console admission poisoned")
        {
            LocalConsoleAuthority::StandaloneOriginalTask => {
                crate::console_retention::ConsoleRetentionOwner::source_subgrant(
                    request.task.resources.memory_bytes,
                    controls,
                    self.max_workers(),
                )
                .map(Some)
                .map_err(|error| TransferFailure::Infrastructure(error.to_string()))
            }
            LocalConsoleAuthority::Parent(owner) => {
                let grant = owner
                    .delegate(self.console_identity(request), controls, self.max_workers())
                    .map_err(TransferFailure::Infrastructure)?;
                self.parent_console_attempts
                    .lock()
                    .expect("local console ownership poisoned")
                    .insert(request.id);
                Ok(Some(grant))
            }
        }
    }

    pub(crate) fn quote_console_subgrant(
        &self,
        controls: Option<runmat_execution::console::ConsoleLimits>,
        original_memory: u64,
    ) -> NativeExecutionResult<Option<runmat_execution::console::ConsoleBudget>> {
        let Some(controls) = controls else {
            return Ok(None);
        };
        match &*self
            .console_admission
            .lock()
            .expect("local console admission poisoned")
        {
            LocalConsoleAuthority::StandaloneOriginalTask => {
                crate::console_retention::ConsoleRetentionOwner::source_subgrant(
                    original_memory,
                    controls,
                    self.max_workers(),
                )
                .map(Some)
            }
            LocalConsoleAuthority::Parent(owner) => owner
                .quote_delegate(controls, self.max_workers())
                .map(Some)
                .map_err(NativeExecutionError::Protocol),
        }
    }
}
