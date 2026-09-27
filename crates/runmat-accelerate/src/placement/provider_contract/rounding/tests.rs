use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{HostTensorView, ProviderWorkload};

#[test]
fn declared_rounding_hooks_execute_the_admitted_representations() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload(&HostTensorView {
            data: &[0.5, -0.5, 1.75, -1.75],
            shape: &[2, 1, 2],
        })
        .unwrap();
    for (index, entry) in ENTRIES.iter().enumerate() {
        let mut query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(entry.identity.name),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![ProviderRepresentation {
                element_type: ProviderElementType::F64,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: vec![2, 1, 2],
                residency: ProviderResidency::Host,
            }],
            outputs: vec![],
            workload: ProviderWorkload::default(),
        };
        assert!(
            provider.query_feasibility(&query).is_supported(),
            "{}",
            entry.identity.name
        );
        let output = block_on(match index {
            0 => provider.unary_ceil(&input),
            1 => provider.unary_fix(&input),
            2 => provider.unary_floor(&input),
            _ => provider.unary_round(&input),
        })
        .unwrap();
        let expected = match index {
            0 => vec![1.0, 0.0, 2.0, -1.0],
            1 => vec![0.0, 0.0, 1.0, -1.0],
            2 => vec![0.0, -1.0, 1.0, -2.0],
            _ => vec![1.0, -1.0, 2.0, -2.0],
        };
        assert_eq!(block_on(provider.download(&output)).unwrap().data, expected);
        assert_eq!(output.shape, input.shape);
        assert_ne!(output.buffer_id, input.buffer_id);
        provider.free(&output).unwrap();
        query.inputs[0].element_type = ProviderElementType::F32;
        assert!(!provider.query_feasibility(&query).is_supported());
        query.inputs[0].element_type = ProviderElementType::F64;
        query.inputs[0].storage = ProviderStorage::Sparse;
        assert!(!provider.query_feasibility(&query).is_supported());
    }
    provider.free(&input).unwrap();
}
