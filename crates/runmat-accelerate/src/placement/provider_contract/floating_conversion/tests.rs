use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{
    GpuTensorStorage, HostNumericDataView, HostNumericTensorView, ProviderWorkload,
};

fn query(name: &str, element_type: ProviderElementType, shape: &[u64]) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(name),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![ProviderRepresentation {
            element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: shape.to_vec(),
            residency: ProviderResidency::Device,
        }],
        outputs: vec![],
        workload: ProviderWorkload::default(),
    }
}

#[test]
fn typed_conversion_feasibility_matches_real_inprocess_output_and_bytes() {
    let provider = InProcessProvider::new();
    for name in [
        DOUBLE_CATALOG_ENTRY.identity.name,
        SINGLE_CATALOG_ENTRY.identity.name,
    ] {
        let request = query(name, ProviderElementType::F64, &[1, 1, 2]);
        let ProviderFeasibility::Supported { estimate } =
            validate(&provider, false, &request).unwrap()
        else {
            panic!("typed conversion supported");
        };
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&[1.25, 16_777_217.0]),
                shape: &[1, 1, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let output = if name == DOUBLE_CATALOG_ENTRY.identity.name {
            block_on(provider.unary_double(&input))
        } else {
            block_on(provider.unary_single(&input))
        }
        .unwrap();
        let actual = block_on(provider.download_numeric(&output)).unwrap();
        assert_eq!(actual.shape, vec![1, 1, 2]);
        assert_eq!(
            estimate.output_bytes,
            Some(if name == DOUBLE_CATALOG_ENTRY.identity.name {
                16
            } else {
                8
            })
        );
        assert_eq!(estimate.dispatches, Some(0));
        provider.free(&input).unwrap();
        provider.free(&output).unwrap();
    }
    assert!(validate(
        &provider,
        false,
        &query(
            SINGLE_CATALOG_ENTRY.identity.name,
            ProviderElementType::F32,
            &[0, 2, 3]
        )
    )
    .unwrap()
    .is_supported());
}

#[test]
fn unsupported_classes_layouts_and_declared_outputs_are_not_advertised() {
    let provider = InProcessProvider::new();
    for class in [
        ProviderElementType::I64,
        ProviderElementType::Logical,
        ProviderElementType::ComplexF64,
    ] {
        assert!(!validate(
            &provider,
            false,
            &query(SINGLE_CATALOG_ENTRY.identity.name, class, &[1, 2])
        )
        .unwrap()
        .is_supported());
    }
    let mut request = query(
        SINGLE_CATALOG_ENTRY.identity.name,
        ProviderElementType::F64,
        &[1, 2],
    );
    request.outputs = request.inputs.clone();
    assert!(!validate(&provider, false, &request).unwrap().is_supported());
    request.outputs.clear();
    request.inputs[0].storage = ProviderStorage::Sparse;
    assert!(!validate(&provider, false, &request).unwrap().is_supported());
}

#[test]
fn cast_physical_work_tracks_frozen_geometry_and_requested_storage() {
    use super::super::fusion_elementwise::fixture::GeometryProvider;
    for lanes in [64, 256, 512] {
        let provider = GeometryProvider(lanes);
        let capacity =
            u64::from(crate::backend::wgpu_geometry::elementwise_chunk_capacity(lanes).unwrap());
        for elements in [capacity - 1, capacity, capacity + 1] {
            let mut request = query(
                SINGLE_CATALOG_ENTRY.identity.name,
                ProviderElementType::F64,
                &[1, elements],
            );
            request.inputs[0].residency = ProviderResidency::Host;
            let work = work::build(&provider, &request, ProviderElementType::F32).unwrap();
            let dispatches = elements.div_ceil(capacity) as u32;
            assert_eq!(work.source, ProviderPrecision::F64);
            assert_eq!(work.target, ProviderPrecision::F32);
            assert_eq!(work.workgroup, lanes);
            assert_eq!(work.estimate.dispatches, Some(dispatches));
            assert_eq!(work.estimate.output_bytes, Some(elements * 4));
            assert_eq!(work.uniform_bytes, u64::from(dispatches) * 16);
            assert_eq!(
                work.estimate.transient_bytes,
                Some(elements * 12 + work.uniform_bytes)
            );
            assert_eq!(work.scratch_bytes, elements * 8 + work.uniform_bytes);
            assert_eq!(work.allocations, dispatches + 2);
            assert_eq!(work.scalar_visits, elements);
        }
    }
}

#[test]
fn same_type_alias_and_empty_cast_have_no_shader_or_submission_work() {
    use super::super::fusion_elementwise::fixture::GeometryProvider;
    let provider = GeometryProvider(256);
    for elements in [0, 8] {
        let request = query(
            SINGLE_CATALOG_ENTRY.identity.name,
            ProviderElementType::F32,
            &[1, elements],
        );
        let work = work::build(&provider, &request, ProviderElementType::F32).unwrap();
        assert_eq!(work.estimate.dispatches, Some(0));
        assert_eq!(work.allocations, 0);
        assert_eq!(work.uniform_bytes, 0);
        assert_eq!(work.scratch_bytes, 0);
        assert_eq!(work.scalar_visits, 0);
        assert_eq!(
            work.estimate.transient_bytes,
            (elements != 0).then_some(elements * 4)
        );
    }
    let request = query(
        SINGLE_CATALOG_ENTRY.identity.name,
        ProviderElementType::F64,
        &[0, 2, 3],
    );
    let work = work::build(&provider, &request, ProviderElementType::F32).unwrap();
    assert_eq!(work.allocations, 1);
    assert_eq!(
        work.estimate.output_bytes,
        Some(4),
        "raw allocator's minimum request, not empty payload length"
    );
    assert_eq!(
        work.estimate.transient_bytes, None,
        "borrowed backing size is not recoverable from shape"
    );
    assert_eq!(work.estimate.dispatches, Some(0));
    assert_eq!(work.scalar_visits, 0);
}
