use super::*;
use crate::simple_provider::InProcessProvider;
use runmat_accelerate_api::ProviderWorkload;

fn representation(class: ProviderElementType, shape: &[u64]) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type: class,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.to_vec(),
        residency: ProviderResidency::Device,
    }
}

fn query(branch: ProviderElementType) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(RESIDENT_SELECT_OPERATION_IDENTITY),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![
            representation(ProviderElementType::Logical, &[2, 1]),
            representation(branch, &[1, 3]),
            representation(branch, &[1, 3]),
        ],
        outputs: vec![representation(branch, &[2, 3])],
        workload: ProviderWorkload {
            elements: Some(6),
            ..ProviderWorkload::default()
        },
    }
}

#[test]
fn in_process_selection_admits_only_native_resident_typed_route() {
    let provider = InProcessProvider::new();
    assert!(provider.capability_snapshot().supports_operation(
        &ProviderOperationIdentity::new(RESIDENT_SELECT_OPERATION_IDENTITY),
        ProviderOperationFamily::Elementwise,
    ));
    for branch in [ProviderElementType::F32, ProviderElementType::F64] {
        let request = query(branch);
        let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&request)
        else {
            panic!("matching native floating branches should be eligible");
        };
        assert_eq!(estimate.output_bytes, Some(6 * branch.byte_width()));
        assert_eq!(estimate.dispatches, Some(1));
        assert_eq!(estimate.transient_bytes, None);
    }
    let mut request = query(ProviderElementType::F32);
    request.inputs[0].element_type = ProviderElementType::F32;
    assert!(!provider.query_feasibility(&request).is_supported());
    request = query(ProviderElementType::F32);
    request.inputs[2].element_type = ProviderElementType::F64;
    assert!(!provider.query_feasibility(&request).is_supported());
    request = query(ProviderElementType::F32);
    request.inputs[0].residency = ProviderResidency::Host;
    assert!(!provider.query_feasibility(&request).is_supported());
    request = query(ProviderElementType::F32);
    request.outputs[0].shape = vec![3, 2];
    assert!(!provider.query_feasibility(&request).is_supported());
    request = query(ProviderElementType::F32);
    request.inputs[0].shape = vec![3];
    assert!(!provider.query_feasibility(&request).is_supported());
}

#[cfg(feature = "wgpu")]
#[test]
fn geometry_fixture_bounds_wgpu_selection_without_claiming_execution() {
    let provider = super::super::fusion_elementwise::fixture::GeometryProvider::new();
    let request = query(ProviderElementType::F64);
    let work = physical_work(&provider, &request).expect("resolved native select geometry");
    assert_eq!(work.estimate.output_bytes, Some(48));
    assert_eq!(work.estimate.dispatches, Some(1));
    assert_eq!(work.queue_submissions, 2);
    assert_eq!(work.scalar_visits, 6);
    assert_eq!(
        work.uniform_bytes,
        crate::backend::wgpu_geometry::fusion_broadcast_uniform_bytes(
            3,
            crate::backend::wgpu::params::BCAST_MAX_RANK,
        )
        .unwrap()
    );
    let mut wrong_precision = query(ProviderElementType::F32);
    assert!(!validate(&provider, true, &wrong_precision)
        .unwrap()
        .is_supported());
    wrong_precision = request;
    wrong_precision.inputs[0].layout = ProviderLayout::Strided;
    assert!(validate(&provider, true, &wrong_precision)
        .unwrap()
        .is_supported());
    wrong_precision.inputs[0].shape = vec![1, 1, 1];
    assert!(!validate(&provider, true, &wrong_precision)
        .unwrap()
        .is_supported());
}
