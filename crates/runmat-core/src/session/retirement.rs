use runmat_runtime::RuntimeError;

use super::RunMatSession;

/// The lifecycle outcome of the canonical session close operation.
///
/// A callback error after all hosts retire is an ordinary runtime error.
/// `Retained` instead means that at least one host may still own callbacks or
/// private bytes, so a one-shot worker must keep its session alive until its
/// enclosing process tree has retired.
pub enum SessionRetirement {
    Retained(RuntimeError),
    Retired(Result<(), RuntimeError>),
}

impl SessionRetirement {
    pub fn into_result(self) -> Result<(), RuntimeError> {
        match self {
            Self::Retained(error) => Err(error),
            Self::Retired(result) => result,
        }
    }
}

/// Keep the first callback failure and its source when a later host still
/// cannot be proved retired. The variant remains `Retained` regardless of
/// which failure supplies the primary diagnostic.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn retained_after(
    prior: Result<(), RuntimeError>,
    retained: RuntimeError,
) -> SessionRetirement {
    let error = runmat_runtime::runtime_error::complete_with_cleanup(prior, Err(retained))
        .expect_err("retained host always supplies an error");
    SessionRetirement::Retained(error)
}

#[cfg(not(target_arch = "wasm32"))]
pub(in crate::session) fn combine_failure(
    first: Option<RuntimeError>,
    next: RuntimeError,
) -> RuntimeError {
    match first {
        Some(first) => {
            runmat_runtime::runtime_error::complete_with_cleanup::<()>(Err(first), Err(next))
                .expect_err("both failures are errors")
        }
        None => next,
    }
}

fn append(
    first: Result<(), RuntimeError>,
    later: Result<(), RuntimeError>,
) -> Result<(), RuntimeError> {
    runmat_runtime::runtime_error::complete_with_cleanup(first, later)
}

impl RunMatSession {
    /// Gracefully retire session-owned workspace values and admitted cleanup
    /// callbacks. Externally owned Active handles are not waited on.
    pub async fn close_gracefully(&self) -> Result<(), RuntimeError> {
        self.close_gracefully_retirement().await.into_result()
    }

    /// A `Retained` result keeps the exact foreign owner live; a `Retired`
    /// error is a completed cleanup callback failure, not a resource leak.
    pub async fn close_gracefully_retirement(&self) -> SessionRetirement {
        self.retire_graphics_host();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.close_gracefully_native().await
        }
        #[cfg(target_arch = "wasm32")]
        {
            let runtime = self.runtime_context.clone();
            let mut result = self
                .lifecycle_host
                .retire_workspace_and_drain(&runtime)
                .await;
            result = append(
                result,
                self.base_workspace
                    .close()
                    .map(|_| ())
                    .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime),
            );
            self.lifecycle_host.force_close();
            SessionRetirement::Retired(result)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn drain_retired_host_cleanup(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), RuntimeError> {
        // A retired foreign host can release its last explicit callback or
        // engine-context owner. Observe eligibility and collect its callback
        // receipt before retiring another host or forcing lifecycle closure.
        self.local_tasks
            .participate_foreground(self.lifecycle_host.retire_workspace_and_drain(runtime))
            .await
            .and_then(|result| result)
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn close_gracefully_native(&self) -> SessionRetirement {
        let runtime = self.runtime_context.clone();
        let mut result = self
            .local_tasks
            .participate_foreground(self.lifecycle_host.retire_workspace_and_drain(&runtime))
            .await
            .and_then(|result| result);

        match self
            .source_evaluator
            .retire_selected_foreign_hosts_retirement(&runtime, &self.local_tasks)
            .await
        {
            SessionRetirement::Retained(error) => return retained_after(result, error),
            SessionRetirement::Retired(selected) => result = append(result, selected),
        }
        result = append(result, self.drain_retired_host_cleanup(&runtime).await);

        let mex = match self
            .local_tasks
            .participate_foreground(
                self.mex_runtime
                    .shutdown_gracefully_retirement(runtime.clone()),
            )
            .await
        {
            Ok(retirement) => retirement,
            Err(error) => return retained_after(result, error),
        };
        match mex {
            runmat_runtime::foreign::MexSessionRetirement::Retained(error) => {
                return retained_after(result, error);
            }
            runmat_runtime::foreign::MexSessionRetirement::Retired(mex) => {
                result = append(result, mex);
            }
        }
        result = append(result, self.drain_retired_host_cleanup(&runtime).await);

        match self
            .native_ffi_adapter
            .shutdown_retirement(Some(&runtime))
            .await
        {
            runmat_runtime::foreign::NativeFfiRetirement::Retained(error) => {
                return retained_after(result, error);
            }
            runmat_runtime::foreign::NativeFfiRetirement::Retired {
                scope,
                result: native,
            } => {
                result = append(result, native);
                if scope == Some(runmat_process_host::ContainedChildRetirement::JoinedMember) {
                    let mismatch = runmat_runtime::build_runtime_error(
                        "base native host returned only joined-member retirement",
                    )
                    .with_identifier("RunMat:NativeFFI:RetirementScope")
                    .build();
                    return retained_after(result, mismatch);
                }
            }
        }
        result = append(result, self.drain_retired_host_cleanup(&runtime).await);

        match self
            .python_adapter
            .shutdown_selected_retirement(Some(&runtime))
            .await
        {
            runmat_runtime::foreign::PythonAdapterRetirement::Retained(error) => {
                return retained_after(result, error);
            }
            runmat_runtime::foreign::PythonAdapterRetirement::Retired {
                scope,
                result: python,
            } => {
                result = append(result, python);
                if scope == Some(runmat_process_host::ContainedChildRetirement::JoinedMember) {
                    let mismatch = runmat_runtime::build_runtime_error(
                        "base Python host returned only joined-member retirement",
                    )
                    .with_identifier("RunMat:Python:RetirementScope")
                    .build();
                    return retained_after(result, mismatch);
                }
            }
        }
        result = append(result, self.drain_retired_host_cleanup(&runtime).await);

        if let Err(error) = self.java_adapter.shutdown_selected_retirement() {
            return retained_after(result, error);
        }
        result = append(result, self.drain_retired_host_cleanup(&runtime).await);

        result = append(
            result,
            self.base_workspace
                .close()
                .map(|_| ())
                .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime),
        );
        self.lifecycle_host.force_close();
        SessionRetirement::Retired(result)
    }
}
