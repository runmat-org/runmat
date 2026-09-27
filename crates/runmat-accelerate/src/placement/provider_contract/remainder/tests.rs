use super::*;

fn query(kind: FloatingRemainderKind) -> ProviderFeasibilityQuery {
    let input = ProviderRepresentation {
        element_type: ProviderElementType::F64,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![2, 1],
        residency: ProviderResidency::Host,
    };
    let mut right = input.clone();
    right.shape = vec![1, 3];
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(identity(kind)),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![input, right],
        outputs: vec![],
        workload: runmat_accelerate_api::ProviderWorkload {
            numerical_domain: NumericalDomainEvidence::FloatingRemainder(kind),
            ..Default::default()
        },
    }
}

#[test]
fn actual_provider_capabilities_require_exact_floating_strategy() {
    let provider = crate::simple_provider::InProcessProvider::new();
    for kind in [
        FloatingRemainderKind::Modulus,
        FloatingRemainderKind::Remainder,
    ] {
        let mut query = query(kind);
        let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&query) else {
            panic!("actual InProcess remainder capability");
        };
        assert_eq!(estimate.output_bytes, Some(6 * 8));
        query.workload.numerical_domain = NumericalDomainEvidence::Unknown;
        assert!(!provider.query_feasibility(&query).is_supported());
        query.workload.numerical_domain = NumericalDomainEvidence::FloatingRemainder(kind);
        query.inputs[1].element_type = ProviderElementType::F32;
        assert!(!provider.query_feasibility(&query).is_supported());
        query.inputs[1].element_type = ProviderElementType::F64;
        query.inputs[1].shape = vec![3, 2];
        assert!(!provider.query_feasibility(&query).is_supported());
        query.inputs[1].shape = vec![1, 3];
        query.inputs[0].storage = ProviderStorage::DenseComplexInterleaved;
        assert!(!provider.query_feasibility(&query).is_supported());
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn remainder_work_includes_full_exponent_gap_and_actual_fused_geometry() {
    let mut query = query(FloatingRemainderKind::Modulus);
    let work = wgpu_work(ProviderPrecision::F64, &query).unwrap();
    assert_eq!(work.dispatches, 1);
    assert_eq!(work.queue_submissions, 2); // warmup and execution
    assert_eq!(work.allocations, 2);
    assert_eq!(
        work.scratch_bytes,
        std::mem::size_of::<crate::backend::wgpu::params::BinaryBroadcastParams>() as u64
    );
    assert_eq!(work.extra_upload_bytes, work.scratch_bytes * 2);
    assert_eq!(work.execution_units, 6 * (64 + (2097 + 7 * 52) * 8));
    query.inputs[0].shape = vec![0, 1];
    assert_eq!(
        wgpu_work(ProviderPrecision::F64, &query)
            .unwrap()
            .queue_submissions,
        0
    );
    query.inputs[0].shape = vec![u64::MAX, 1];
    assert!(wgpu_work(ProviderPrecision::F64, &query).is_none());
}
