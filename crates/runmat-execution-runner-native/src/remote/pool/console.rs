//! Remote driver leases complete subtree grants before dispatch. Worker/local
//! record admission uses the same canonical component and explicit subgrants.
use super::*;
use crate::console_retention::ConsoleRetentionOwner;
use runmat_execution::console::ScopedConsoleTranscript;
use runmat_execution_artifact::ProgramExecutionConsoleBudget;
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;
use runmat_runtime::context::RuntimeConsoleScopeIdentity as Identity;

fn identity(request: &AttemptRequest) -> Identity {
    Identity {
        scope_id: request.scope_id,
        pool_id: request.task.pool_id,
        task_id: request.task_id,
        attempt_id: request.id,
    }
}

impl RemotePoolDriver {
    pub(crate) fn configure_console_retention(
        &self,
        resources: &AllocationResources,
    ) -> NativeExecutionResult<()> {
        let driver = self.driver.lock().expect("remote driver poisoned");
        let mut configured = self
            .console_retention
            .lock()
            .expect("console retention poisoned");
        let expected = ProgramExecutionConsoleBudget::for_driver_allocation(resources.memory_bytes);
        if let Some(owner) = configured.as_ref() {
            if owner.limits() != expected {
                return Err(NativeExecutionError::Protocol(
                    "console driver allocation changed".into(),
                ));
            }
            return Ok(());
        }
        let owner = ConsoleRetentionOwner::new(resources.memory_bytes);
        for attempt in driver.snapshot_ref().attempts.values() {
            let Some(controls) = attempt.request.task.console else {
                continue;
            };
            let terminal = attempt.terminal.as_ref();
            let unverified = attempt.unverified_retirement.as_ref();
            let (root, scopes) = if let Some(record) =
                terminal.filter(|r| r.console.is_some() || !r.console_scopes.is_empty())
            {
                (record.console.as_deref(), record.console_scopes.as_slice())
            } else if let Some(record) = unverified {
                (record.console.as_deref(), record.console_scopes.as_slice())
            } else {
                (None, &[] as &[ScopedConsoleTranscript])
            };
            owner.restore_subtree(
                identity(&attempt.request),
                controls,
                root,
                scopes,
                self.console_max_in_flight,
            )?;
        }
        *configured = Some(owner);
        Ok(())
    }

    pub(crate) fn console_retention_limits(&self) -> Option<ProgramExecutionConsoleBudget> {
        self.console_retention
            .lock()
            .expect("console retention poisoned")
            .as_ref()
            .map(|owner| owner.limits())
    }

    pub(crate) fn reserve_console_attempt(
        &self,
        request: &AttemptRequest,
    ) -> NativeExecutionResult<Option<ProgramExecutionConsoleBudget>> {
        let Some(controls) = request.task.console else {
            return Ok(None);
        };
        let owner = self
            .console_retention
            .lock()
            .expect("console retention poisoned")
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                NativeExecutionError::Protocol(
                    "console capture has no authenticated driver component".into(),
                )
            })?;
        owner
            .delegate_scope(identity(request), controls, self.console_max_in_flight)
            .map(Some)
    }

    pub(super) fn validate_console_report_reservation(
        &self,
        report: &BackendReport,
    ) -> NativeExecutionResult<()> {
        if report.console.is_none() && report.console_scopes.is_empty() {
            return Ok(());
        }
        let owner = self
            .console_retention
            .lock()
            .expect("console retention poisoned")
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                NativeExecutionError::Protocol(
                    "console reply has no admitted driver component".into(),
                )
            })?;
        owner.validate_subtree(
            Identity {
                scope_id: self.scope_id,
                pool_id: self.pool_id,
                task_id: report.task_id,
                attempt_id: report.attempt_id,
            },
            report.console.as_deref(),
            &report.console_scopes,
        )
    }

    pub(super) fn settle_console_attempt(
        &self,
        request: &AttemptRequest,
        root: Option<&runmat_execution::console::ConsoleTranscript>,
        scopes: &[ScopedConsoleTranscript],
    ) -> NativeExecutionResult<()> {
        if root.is_none() && scopes.is_empty() {
            return Ok(());
        }
        let owner = self
            .console_retention
            .lock()
            .expect("console retention poisoned")
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                NativeExecutionError::Protocol(
                    "accepted console has no admitted driver component".into(),
                )
            })?;
        owner.settle_subtree_scope(identity(request), root, scopes)
    }
}
