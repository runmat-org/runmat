use super::transition_ns;
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, PlacementGraphEdge, ResourceByteBound,
};

#[test]
fn unresolved_conditional_output_charges_both_legal_publication_branches() {
    let from = CandidateOutputResidency::ConditionalHostOrProvider {
        device_id: 7,
        host_bytes: ResourceByteBound::Known(8),
    };
    let mut edge = PlacementGraphEdge {
        from: 0,
        to: 1,
        bytes: 8,
        host_to_provider_ns: 17,
        provider_to_host_ns: 23,
        cross_provider_ns: 31,
    };
    assert_eq!(
        transition_ns(from, CandidateExecutionLocation::Host, &edge),
        23
    );
    assert_eq!(
        transition_ns(
            from,
            CandidateExecutionLocation::Provider { device_id: 7 },
            &edge
        ),
        17
    );
    assert_eq!(
        transition_ns(
            from,
            CandidateExecutionLocation::Provider { device_id: 8 },
            &edge
        ),
        31
    );
    // A host upload may be more expensive than a cross-provider transfer.
    edge.host_to_provider_ns = 41;
    assert_eq!(
        transition_ns(
            from,
            CandidateExecutionLocation::Provider { device_id: 8 },
            &edge
        ),
        41
    );
    // Resolving the output to the execution provider removes the uncertainty.
    assert_eq!(
        transition_ns(
            CandidateOutputResidency::Provider { device_id: 7 },
            CandidateExecutionLocation::Provider { device_id: 7 },
            &edge,
        ),
        0
    );
}
