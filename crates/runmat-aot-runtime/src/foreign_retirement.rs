//! Ordered teardown for a standalone executable's foreign owners.

use runmat_process_host::ContainedChildRetirement;
use runmat_runtime::foreign::{
    JavaAdapter, MexSessionRetirement, NativeFfiAdapter, NativeFfiRetirement, PythonAdapter,
    PythonAdapterRetirement,
};
use runmat_runtime::RuntimeError;

use crate::mex::MexRuntimeGuard;

// Teardown errors are cold. Keep the normal successful path allocation-free,
// while a retained/failed owner carries its full diagnostic on the heap.
type Cleanup = Result<(), Box<RuntimeError>>;

pub(crate) enum AotForeignRetirement {
    Retained(Box<RuntimeError>),
    Retired(Cleanup),
}

impl AotForeignRetirement {
    pub(crate) fn error(&self) -> Option<&RuntimeError> {
        match self {
            Self::Retained(error) | Self::Retired(Err(error)) => Some(error.as_ref()),
            Self::Retired(Ok(())) => None,
        }
    }

    pub(crate) fn into_result(self) -> Cleanup {
        match self {
            Self::Retained(error) => Err(error),
            Self::Retired(result) => result,
        }
    }
}

fn append(prior: Cleanup, later: Cleanup) -> Cleanup {
    runmat_runtime::runtime_error::complete_with_cleanup(
        prior.map_err(|error| *error),
        later.map_err(|error| *error),
    )
    .map_err(Box::new)
}

fn retained(prior: Cleanup, error: RuntimeError) -> AotForeignRetirement {
    AotForeignRetirement::Retained(
        append(prior, Err(Box::new(error))).expect_err("retained owner supplies an error"),
    )
}

fn full_tree(scope: Option<ContainedChildRetirement>) -> bool {
    scope != Some(ContainedChildRetirement::JoinedMember)
}

fn scope_error(owner: &str) -> RuntimeError {
    runmat_runtime::build_runtime_error(format!(
        "standalone {owner} child returned only joined-member retirement"
    ))
    .with_identifier("RunMat:AOT:RetirementScope")
    .build()
}

pub(crate) async fn retire(
    runtime: &runmat_runtime::context::RuntimeContext,
    mex: &mut MexRuntimeGuard,
    native: &NativeFfiAdapter,
    python: &PythonAdapter,
    java: &JavaAdapter,
) -> AotForeignRetirement {
    let mut result = match mex.shutdown_gracefully_retirement().await {
        MexSessionRetirement::Retained(error) => {
            return AotForeignRetirement::Retained(Box::new(error))
        }
        MexSessionRetirement::Retired(result) => result.map_err(Box::new),
    };

    match native.shutdown_retirement(Some(runtime)).await {
        NativeFfiRetirement::Retained(error) => return retained(result, error),
        NativeFfiRetirement::Retired {
            scope,
            result: native,
        } => {
            result = append(result, native.map_err(Box::new));
            if !full_tree(scope) {
                return retained(result, scope_error("native"));
            }
        }
    }

    match python.shutdown_selected_retirement(Some(runtime)).await {
        PythonAdapterRetirement::Retained(error) => return retained(result, error),
        PythonAdapterRetirement::Retired {
            scope,
            result: python,
        } => {
            result = append(result, python.map_err(Box::new));
            if !full_tree(scope) {
                return retained(result, scope_error("Python"));
            }
        }
    }

    if let Err(error) = java.shutdown_selected_retirement() {
        return retained(result, error);
    }
    AotForeignRetirement::Retired(result)
}
