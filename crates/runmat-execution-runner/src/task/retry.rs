use runmat_execution::task::RetryPolicy;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryCause {
    Lost,
    Infrastructure,
    Execution,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryDecision {
    Retry,
    Fail,
    Indeterminate,
}

pub fn retry_decision(
    policy: RetryPolicy,
    completed_attempts: u16,
    cause: RetryCause,
) -> RetryDecision {
    let retry = match policy {
        RetryPolicy::Never => false,
        RetryPolicy::IdempotentInfrastructure => {
            completed_attempts < 3 && matches!(cause, RetryCause::Lost | RetryCause::Infrastructure)
        }
        RetryPolicy::ExplicitlyIdempotent { max_attempts } => {
            completed_attempts < max_attempts
                && matches!(cause, RetryCause::Lost | RetryCause::Infrastructure)
        }
        RetryPolicy::TestPolicy { max_attempts } => {
            completed_attempts < max_attempts && cause != RetryCause::Rejected
        }
    };
    if retry {
        RetryDecision::Retry
    } else if cause == RetryCause::Lost && policy == RetryPolicy::Never {
        RetryDecision::Indeterminate
    } else {
        RetryDecision::Fail
    }
}
