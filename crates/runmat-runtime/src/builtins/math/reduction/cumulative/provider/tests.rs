use super::*;
use futures::executor::block_on;
#[test]
fn all_scan_identities_share_no_attempt_success_and_terminal_error_rules() {
    for kind in [
        CumulativeKind::Sum,
        CumulativeKind::Product,
        CumulativeKind::Minimum,
        CumulativeKind::Maximum,
    ] {
        assert_eq!(
            block_on(attempt(
                Box::pin(async {
                    Ok::<_, runmat_accelerate_api::ProviderExecutionError>(None::<usize>)
                }),
                kind
            ))
            .unwrap(),
            None
        );
        assert_eq!(
            block_on(attempt(
                Box::pin(async {
                    Ok::<_, runmat_accelerate_api::ProviderExecutionError>(Some(7usize))
                }),
                kind
            ))
            .unwrap(),
            Some(7)
        );
        let error = block_on(attempt::<usize>(
            Box::pin(async {
                Err(runmat_accelerate_api::ProviderExecutionError::from_source(
                    std::io::Error::other("scan execution failure"),
                ))
            }),
            kind,
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        let mut cause: &dyn std::error::Error = &error;
        let mut found = false;
        loop {
            found |= cause.downcast_ref::<std::io::Error>().is_some();
            let Some(source) = cause.source() else { break };
            cause = source;
        }
        assert!(
            found,
            "terminal provider cause must survive the shared boundary"
        );
    }
}
