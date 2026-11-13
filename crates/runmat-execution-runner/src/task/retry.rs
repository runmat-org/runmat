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
    let retry = completed_attempts < maximum_attempts(policy)
        && match policy {
            RetryPolicy::Never => false,
            RetryPolicy::IdempotentInfrastructure | RetryPolicy::ExplicitlyIdempotent { .. } => {
                matches!(cause, RetryCause::Lost | RetryCause::Infrastructure)
            }
            RetryPolicy::TestPolicy { .. } => cause != RetryCause::Rejected,
        };
    if retry {
        RetryDecision::Retry
    } else if cause == RetryCause::Lost && policy == RetryPolicy::Never {
        RetryDecision::Indeterminate
    } else {
        RetryDecision::Fail
    }
}

/// One authority for admission and retry eligibility; zero is rejected at admission.
pub fn maximum_attempts(policy: RetryPolicy) -> u16 {
    match policy {
        RetryPolicy::Never => 1,
        RetryPolicy::IdempotentInfrastructure => 3,
        RetryPolicy::ExplicitlyIdempotent { max_attempts }
        | RetryPolicy::TestPolicy { max_attempts } => max_attempts,
    }
}
