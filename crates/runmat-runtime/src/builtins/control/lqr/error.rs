//! Caller-specific diagnostics for the one standard Riccati request.

use runmat_builtins::{
    LQR_ERROR_INTERNAL, LQR_ERROR_INVALID_DIMENSIONS, LQR_ERROR_SOLVER_FAILED,
    LQR_ERROR_UNSUPPORTED_INPUT,
};

use super::lqr_error;
use crate::builtins::control::tf_model::control_error;
use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use crate::RuntimeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::control) enum RiccatiApi {
    Lqr,
    Care,
    Dlqr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RiccatiIssue {
    InvalidDimensions,
    UnsupportedInput,
    SolverFailed,
    Internal,
}

impl RiccatiApi {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Lqr => "lqr",
            Self::Care => "care",
            Self::Dlqr => "dlqr",
        }
    }

    pub(super) fn admission(
        self,
        runtime: Option<&RuntimeContext>,
        error: HostAllocationError,
    ) -> RuntimeError {
        let cancellation = runtime.map(RuntimeContext::cancellation_context);
        crate::builtin::builtin_control_error(cancellation.as_ref(), self.name(), error)
    }

    pub(super) fn error(self, issue: RiccatiIssue, message: impl Into<String>) -> RuntimeError {
        let message = message.into();
        match self {
            Self::Lqr => lqr_error(
                match issue {
                    RiccatiIssue::InvalidDimensions => &LQR_ERROR_INVALID_DIMENSIONS,
                    RiccatiIssue::UnsupportedInput => &LQR_ERROR_UNSUPPORTED_INPUT,
                    RiccatiIssue::SolverFailed => &LQR_ERROR_SOLVER_FAILED,
                    RiccatiIssue::Internal => &LQR_ERROR_INTERNAL,
                },
                message,
            ),
            Self::Care => control_error(
                "care",
                match issue {
                    RiccatiIssue::InvalidDimensions => "RunMat:care:InvalidDimensions",
                    RiccatiIssue::UnsupportedInput => "RunMat:care:UnsupportedInput",
                    RiccatiIssue::SolverFailed => "RunMat:care:SolverFailed",
                    RiccatiIssue::Internal => "RunMat:care:Internal",
                },
                message,
            ),
            Self::Dlqr => control_error(
                "dlqr",
                match issue {
                    RiccatiIssue::InvalidDimensions => "RunMat:dlqr:InvalidDimensions",
                    RiccatiIssue::UnsupportedInput => "RunMat:dlqr:UnsupportedInput",
                    RiccatiIssue::SolverFailed => "RunMat:dlqr:SolverFailed",
                    RiccatiIssue::Internal => "RunMat:dlqr:Internal",
                },
                message,
            ),
        }
    }
}
