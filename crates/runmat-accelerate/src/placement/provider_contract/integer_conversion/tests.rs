use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{
    HostIntegerDataView, HostIntegerTensorView, HostTensorView, ProviderWorkload,
};

mod accounting;

fn query(
    entry: &BuiltinCatalogEntry,
    input_type: ProviderElementType,
    shape: &[u64],
) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(entry.identity.name),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![ProviderRepresentation {
            element_type: input_type,
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
fn exact_eight_canonical_capabilities_match_real_typed_hooks() {
    let provider = InProcessProvider::new();
    let source = provider
        .upload(&HostTensorView {
            data: &[-1.0, 3.5, f64::NAN, 256.0],
            shape: &[1, 2, 2],
        })
        .unwrap();
    for (entry, target) in OPERATIONS {
        let query = query(entry, ProviderElementType::F64, &[1, 2, 2]);
        assert!(matches!(
            provider.query_feasibility(&query),
            ProviderFeasibility::Supported { .. }
        ));
        let output = block_on(provider.cast_to_integer(&source, *target)).unwrap();
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(*target)
        );
        assert_eq!(output.shape(), source.shape());
        let values = block_on(provider.download_integer(&output)).unwrap();
        assert_eq!(values.shape, vec![1, 2, 2]);
        assert_eq!(values.data.element_type(), *target);
        output.release().unwrap();
    }
    source.release().unwrap();
}

#[test]
fn wide_integer_and_empty_queries_match_resident_execution() {
    let provider = InProcessProvider::new();
    for data in [vec![9_007_199_254_740_993, u64::MAX], vec![]] {
        let shape = [1, data.len()];
        let source = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&data),
                shape: &shape,
            })
            .unwrap();
        let query = query(
            &UINT64_CATALOG_ENTRY,
            ProviderElementType::U64,
            &[1, data.len() as u64],
        );
        assert!(matches!(
            provider.query_feasibility(&query),
            ProviderFeasibility::Supported { .. }
        ));
        let output = block_on(provider.cast_to_integer(&source, IntegerElementType::U64)).unwrap();
        assert_eq!(
            block_on(provider.download_integer(&output)).unwrap().data,
            runmat_accelerate_api::HostIntegerDataOwned::U64(data)
        );
        output.release().unwrap();
        source.release().unwrap();
    }
}

#[test]
fn unsupported_physical_classes_storage_and_output_shape_are_rejected() {
    let provider = InProcessProvider::new();
    for ty in [
        ProviderElementType::F32,
        ProviderElementType::Logical,
        ProviderElementType::ComplexF64,
    ] {
        assert!(matches!(
            provider.query_feasibility(&query(&INT8_CATALOG_ENTRY, ty, &[1, 2])),
            ProviderFeasibility::Rejected { .. }
        ));
    }
    let mut request = query(&INT8_CATALOG_ENTRY, ProviderElementType::F64, &[1, 2]);
    request.inputs[0].storage = ProviderStorage::DenseComplexInterleaved;
    assert!(matches!(
        provider.query_feasibility(&request),
        ProviderFeasibility::Rejected { .. }
    ));
    request.inputs[0].storage = ProviderStorage::DenseReal;
    request.outputs = vec![request.inputs[0].clone()];
    assert!(matches!(
        provider.query_feasibility(&request),
        ProviderFeasibility::Rejected { .. }
    ));
}
