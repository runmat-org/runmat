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
