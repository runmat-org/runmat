use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    ProviderWorkload,
};

fn query(
    entry: &BuiltinCatalogEntry,
    class: ProviderElementType,
    shape: &[u64],
) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(entry.identity.name),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![ProviderRepresentation {
            element_type: class,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: shape.to_vec(),
            residency: ProviderResidency::Device,
        }],
        outputs: Vec::new(),
        workload: ProviderWorkload::default(),
    }
}

#[test]
fn native_exponential_contract_matches_both_actual_classes_and_full_shapes() {
    let provider = InProcessProvider::new();
    for class in [ProviderElementType::F32, ProviderElementType::F64] {
        for shape in [vec![], vec![4], vec![8], vec![1, 2, 1, 2], vec![0, 3]] {
            let count = if shape.contains(&0) {
                0
            } else {
                shape.iter().product()
            };
            let values = [
                -0.0_f64,
                1e-8,
                -0.125,
                0.25,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                1.0,
            ][..count]
                .to_vec();
            let single = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
            let input = provider
                .upload_numeric(&HostNumericTensorView {
                    data: if class == ProviderElementType::F32 {
                        HostNumericDataView::F32(&single)
                    } else {
                        HostNumericDataView::F64(&values)
                    },
                    shape: &shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            for &(entry, operation) in OPERATIONS {
                let request = query(
                    entry,
                    class,
                    &shape.iter().map(|&n| n as u64).collect::<Vec<_>>(),
                );
                assert_eq!(
                    runmat_builtins::builtin_id_by_name(entry.identity.name)
                        .unwrap()
                        .0,
                    request.operation.0
                );
                assert!(provider
                    .capability_snapshot()
                    .supports_operation(&request.operation, request.family));
                let ProviderFeasibility::Supported { estimate } =
                    provider.query_feasibility(&request)
                else {
                    panic!("exact native exponential must be admitted");
                };
                let bytes = count as u64
                    * if class == ProviderElementType::F32 {
                        4
                    } else {
                        8
                    };
                assert_eq!(estimate.output_bytes, Some(bytes));
                assert_eq!(estimate.transient_bytes, Some(bytes * 2));
                assert_eq!(
                    estimate.dispatches,
                    Some(0),
                    "native maps submit no device queue work"
                );
                let output = block_on(match operation {
                    Operation::Exp => provider.unary_exp(&input),
                    Operation::Expm1 => provider.unary_expm1(&input),
                })
                .unwrap();
                assert_eq!(output.shape(), shape);
                assert!(!output.shares_allocation_with(&input));
                let actual = block_on(provider.download_numeric(&output)).unwrap();
                assert_eq!(actual.shape, shape);
                match (class, actual.data) {
                    (ProviderElementType::F32, HostNumericDataOwned::F32(actual)) => {
                        let expected = single
                            .iter()
                            .map(|&x| match operation {
                                Operation::Exp => x.exp(),
                                Operation::Expm1 => x.exp_m1(),
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(
                            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        );
                    }
                    (ProviderElementType::F64, HostNumericDataOwned::F64(actual)) => {
                        let expected = values
                            .iter()
                            .map(|&x| match operation {
                                Operation::Exp => x.exp(),
                                Operation::Expm1 => x.exp_m1(),
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(
                            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        );
                    }
                    _ => panic!("native class changed"),
                }
                output.release().unwrap();
                input.allocation_lease().validate_live().unwrap();
            }
            input.release().unwrap();
        }
    }
}

#[test]
fn exponential_queries_reject_other_domains_and_missing_physical_owners() {
    let provider = InProcessProvider::new();
    for &(entry, operation) in OPERATIONS {
        let valid = query(entry, ProviderElementType::F64, &[1, 2, 1, 2]);
        let mut cases = Vec::new();
        for class in [
            ProviderElementType::I64,
            ProviderElementType::Logical,
            ProviderElementType::ComplexF64,
        ] {
            let mut request = valid.clone();
            request.inputs[0].element_type = class;
            cases.push(request);
        }
        for storage in [
            ProviderStorage::Sparse,
            ProviderStorage::DenseComplexInterleaved,
        ] {
            let mut request = valid.clone();
            request.inputs[0].storage = storage;
            cases.push(request);
        }
        for layout in [ProviderLayout::Strided, ProviderLayout::Opaque] {
            let mut request = valid.clone();
            request.inputs[0].layout = layout;
            cases.push(request);
        }
        for residency in [ProviderResidency::Mirrored, ProviderResidency::Unknown] {
            let mut request = valid.clone();
            request.inputs[0].residency = residency;
            cases.push(request);
        }
        let mut request = valid.clone();
        request.inputs.clear();
        cases.push(request);
        let mut request = valid.clone();
        request.inputs.push(request.inputs[0].clone());
        cases.push(request);
        let mut request = valid.clone();
        request.family = ProviderOperationFamily::Reduction;
        cases.push(request);
        let mut request = valid.clone();
        request.workload.elements = Some(5);
        cases.push(request);
        let mut request = valid.clone();
        request.inputs[0].shape = vec![u64::MAX, 2];
        cases.push(request);
        for mutate in 0..4 {
            let mut request = valid.clone();
            let mut output = request.inputs[0].clone();
            match mutate {
                0 => output.shape = vec![4],
                1 => output.element_type = ProviderElementType::F32,
                2 => output.residency = ProviderResidency::Host,
                _ => output.layout = ProviderLayout::Strided,
            }
            request.outputs.push(output);
            cases.push(request);
        }
        for request in cases {
            assert!(matches!(
                provider.query_feasibility(&request),
                ProviderFeasibility::Rejected { .. }
            ));
        }
        let snapshot = super::super::in_process_capability_snapshot(&provider);
        assert!(
            matches!(validate(&provider, true, &snapshot, &valid),
            Some(ProviderFeasibility::Rejected { rejection })
                if rejection.code == ProviderRejectionCode::UnknownResourceEstimate),
            "a generic geometry hint cannot replace an actual WGPU physical owner"
        );
        let mut too_long = valid.clone();
        too_long.inputs[0].shape = vec![u64::from(u32::MAX) + 1];
        assert!(matches!(validate(&provider, true, &snapshot, &too_long),
            Some(ProviderFeasibility::Rejected { rejection })
                if rejection.code == ProviderRejectionCode::ResourceLimit));
        let mut empty = valid;
        empty.inputs[0].shape = vec![0, 3];
        assert!(matches!(
            validate(&provider, true, &snapshot, &empty),
            Some(ProviderFeasibility::Rejected { .. })
        ));
        let complex = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&[1.0, 2.0]),
                shape: &[1, 1],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .unwrap();
        assert!(block_on(match operation {
            Operation::Exp => provider.unary_exp(&complex),
            Operation::Expm1 => provider.unary_expm1(&complex),
        })
        .is_err());
        complex.allocation_lease().validate_live().unwrap();
        complex.release().unwrap();
    }
}
