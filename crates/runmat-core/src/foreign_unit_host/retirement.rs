//! Ordered retirement of one selected foreign unit. Each later adapter is
//! visited only after the preceding owner has actually retired.

use runmat_process_host::{ContainedChildRetirement, ProcessGroupPlacement};
use runmat_runtime::context::RuntimeLocalTaskSupervisor;
use runmat_runtime::foreign::{MexSessionRetirement, NativeFfiRetirement, PythonAdapterRetirement};

use super::{ForeignUnitHost, ForeignUnitRetirement, RuntimeContext, RuntimeError};

type Cleanup = Result<(), RuntimeError>;

fn append(prior: Cleanup, later: Cleanup) -> Cleanup {
    runmat_runtime::runtime_error::complete_with_cleanup(prior, later)
}

fn retained(prior: Cleanup, error: RuntimeError) -> ForeignUnitRetirement {
    ForeignUnitRetirement::Retained(
        append(prior, Err(error)).expect_err("retained host supplies an error"),
    )
}

fn selected_scope_matches(
    placement: ProcessGroupPlacement,
    scope: Option<ContainedChildRetirement>,
) -> bool {
    matches!(
        (placement, scope),
        (_, None)
            | (
                ProcessGroupPlacement::OwnGroup,
                Some(ContainedChildRetirement::OwnedTree)
            )
            | (
                ProcessGroupPlacement::JoinAncestorGroup,
                Some(ContainedChildRetirement::JoinedMember)
            )
    )
}

impl ForeignUnitHost {
    /// MEX callbacks run under the exact selected local-task owner. Native,
    /// Python, then Java retirement follow only after each predecessor's
    /// terminal proof; any Retained result keeps the whole selected host and
    /// its private bytes with the caller.
    pub async fn shutdown_retirement(
        &self,
        runtime: RuntimeContext,
        local_tasks: &RuntimeLocalTaskSupervisor,
    ) -> ForeignUnitRetirement {
        let mex = match local_tasks
            .participate_foreground(self.mex.shutdown_gracefully_retirement(runtime.clone()))
            .await
        {
            Ok(mex) => mex,
            Err(error) => return ForeignUnitRetirement::Retained(error),
        };
        let prior = match mex {
            MexSessionRetirement::Retained(error) => {
                return ForeignUnitRetirement::Retained(error);
            }
            MexSessionRetirement::Retired(prior) => prior,
        };
        self.retire_native(prior, &runtime).await
    }

    async fn retire_native(
        &self,
        prior: Cleanup,
        runtime: &RuntimeContext,
    ) -> ForeignUnitRetirement {
        let (scope, result) = match self.native_ffi.shutdown_retirement(Some(runtime)).await {
            NativeFfiRetirement::Retained(error) => return retained(prior, error),
            NativeFfiRetirement::Retired { scope, result } => (scope, result),
        };
        let prior = append(prior, result);
        if !selected_scope_matches(self.process_group_placement, scope) {
            let mismatch = runmat_runtime::build_runtime_error(
                "native child retirement scope did not match its selected host containment",
            )
            .with_identifier("RunMat:NativeFFI:RetirementScope")
            .build();
            return retained(prior, mismatch);
        }
        self.retire_python(prior, runtime).await
    }

    async fn retire_python(
        &self,
        prior: Cleanup,
        runtime: &RuntimeContext,
    ) -> ForeignUnitRetirement {
        match self
            .python
            .shutdown_selected_retirement(Some(runtime))
            .await
        {
            PythonAdapterRetirement::Retained(error) => retained(prior, error),
            PythonAdapterRetirement::Retired { scope, result } => {
                let prior = append(prior, result);
                if !selected_scope_matches(self.process_group_placement, scope) {
                    let mismatch = runmat_runtime::build_runtime_error(
                        "Python child retirement scope did not match its selected host containment",
                    )
                    .with_identifier("RunMat:Python:RetirementScope")
                    .build();
                    return retained(prior, mismatch);
                }
                self.retire_java(prior)
            }
        }
    }

    fn retire_java(&self, prior: Cleanup) -> ForeignUnitRetirement {
        match self.java.shutdown_selected_retirement() {
            Ok(()) => ForeignUnitRetirement::Retired(prior),
            Err(error) => retained(prior, error),
        }
    }
}
