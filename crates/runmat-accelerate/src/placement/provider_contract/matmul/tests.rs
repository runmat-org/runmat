use super::*;
use runmat_accelerate_api::{MatmulAffine, ProviderWorkload};

fn query(diagonal: bool, residency: ProviderResidency) -> ProviderFeasibilityQuery {
    let repr = |shape: Vec<u64>, residency| ProviderRepresentation {
        element_type: ProviderElementType::F64,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape,
        residency,
    };
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(crate::placement::fusion_operation_token(
            &crate::fusion::FusionKind::MatmulEpilogue,
        )),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![repr(vec![2, 3], residency), repr(vec![3, 4], residency)],
        outputs: vec![repr(
            if diagonal { vec![2, 1] } else { vec![2, 4] },
            ProviderResidency::Device,
        )],
        workload: ProviderWorkload {
            matmul: Some(MatmulWorkload {
                left: 0,
                right: 1,
                rows: 2,
                inner: 3,
                columns: 4,
                affine: MatmulAffine::default(),
                row: None,
                column: None,
                clamp_min: None,
                clamp_max: None,
                exponent: None,
                diagonal,
            }),
            ..Default::default()
        },
    }
}

#[test]
fn retained_diagonal_is_not_full_product_and_borrowed_inputs_are_not_allocations() {
    let provider = crate::simple_provider::InProcessProvider::new();
    for (diagonal, output, temporary, units, dispatches) in
        [(false, 64, 64, 56, 1), (true, 16, 128, 58, 2)]
    {
        let resident = work(
            &provider,
            false,
            &query(diagonal, ProviderResidency::Device),
        )
        .unwrap();
        let host = work(&provider, false, &query(diagonal, ProviderResidency::Host)).unwrap();
        assert_eq!(resident.estimate.output_bytes, Some(output));
        assert_eq!(resident.scratch_bytes, temporary);
        assert_eq!(host.scratch_bytes, temporary + 144);
        assert_eq!(resident.scalar_units, units);
        assert_eq!(resident.estimate.dispatches, Some(dispatches));
        assert_eq!(resident.queue_submissions, dispatches);
        assert_eq!(resident.allocations, 1 + u32::from(diagonal));
        assert_eq!(resident.uniform_bytes, 0);
    }
}

#[test]
fn geometry_class_domain_and_overflow_are_validated_not_trusted() {
    let provider = crate::simple_provider::InProcessProvider::new();
    let original = query(false, ProviderResidency::Host);
    let mut invalid = original.clone();
    invalid.outputs[0].shape = vec![1, 1];
    assert!(work(&provider, false, &invalid).is_none());
    invalid = original.clone();
    invalid.inputs[0].element_type = ProviderElementType::F32;
    assert!(work(&provider, false, &invalid).is_none());
    invalid = original.clone();
    invalid.workload.matmul.as_mut().unwrap().exponent = Some(0.5f64.to_bits());
    assert!(work(&provider, false, &invalid).is_none());
    invalid.workload.matmul.as_mut().unwrap().clamp_min = Some(0.0f64.to_bits());
    assert!(work(&provider, false, &invalid).is_some());
    invalid = original.clone();
    invalid.workload.matmul.as_mut().unwrap().rows = u64::MAX;
    invalid.inputs[0].shape[0] = u64::MAX;
    invalid.outputs[0].shape[0] = u64::MAX;
    assert!(work(&provider, false, &invalid).is_none());
    invalid = original;
    invalid.workload.matmul = None;
    assert!(matches!(
        validate(&provider, false, &invalid),
        Some(ProviderFeasibility::Rejected { .. })
    ));
}

#[test]
fn workload_serialization_preserves_signed_control_bits_and_old_unknown() {
    let mut request = query(false, ProviderResidency::Host);
    request.workload.matmul.as_mut().unwrap().affine.scale = Some((-0.0f64).to_bits());
    let encoded = serde_json::to_string(&request).unwrap();
    assert_eq!(
        serde_json::from_str::<ProviderFeasibilityQuery>(&encoded).unwrap(),
        request
    );
    let old: ProviderWorkload =
        serde_json::from_str(r#"{"elements":4,"flops":null,"batch":null}"#).unwrap();
    assert_eq!(old.matmul, None);
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_geometry_accounts_diagonal_warmup_uniforms_and_empty_allocation() {
    let provider = crate::simple_provider::InProcessProvider::new();
    let q = query(true, ProviderResidency::Device);
    let physical = work(&provider, true, &q).unwrap();
    use crate::backend::wgpu::params;
    let uniform = (std::mem::size_of::<params::MatmulParams>()
        + std::mem::size_of::<params::MatmulEpilogueParamsF64>()
        + std::mem::size_of::<params::DiagExtractParams>()) as u64;
    assert_eq!(physical.estimate.output_bytes, Some(16));
    assert_eq!(physical.estimate.dispatches, Some(2));
    assert_eq!(physical.queue_submissions, 3);
    assert_eq!(physical.uniform_bytes, uniform);
    assert_eq!(physical.scratch_bytes, 64 + 8 + uniform);
    assert_eq!(physical.allocations, 6);
    let mut empty = query(false, ProviderResidency::Device);
    empty.inputs[0].shape[0] = 0;
    empty.outputs[0].shape[0] = 0;
    empty.workload.matmul.as_mut().unwrap().rows = 0;
    let physical = work(&provider, true, &empty).unwrap();
    assert_eq!(physical.estimate.output_bytes, Some(8));
    assert_eq!(physical.estimate.dispatches, Some(0));
    assert_eq!(physical.uniform_bytes, 0);
    assert_eq!(physical.queue_submissions, 0);
}
