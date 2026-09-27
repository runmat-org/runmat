use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{
    GpuTensorStorage, HostNumericDataView, HostNumericTensorView, ProviderWorkload,
};

fn query(operation: DirectUnary) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(operation.entry().identity.name),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 2, 1, 2],
            residency: ProviderResidency::Host,
        }],
        outputs: Vec::new(),
        workload: ProviderWorkload::default(),
    }
}

#[test]
fn exact_trigonometric_capabilities_match_real_inprocess_hooks() {
    let provider = InProcessProvider::new();
    let values: [f64; 4] = [-0.5, 0.0, 0.25, 1.0];
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&values),
            shape: &[1, 2, 1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    for &operation in OPERATIONS {
        let query = query(operation);
        assert_eq!(
            builtin_id_by_name(operation.entry().identity.name)
                .unwrap()
                .0,
            query.operation.0
        );
        assert!(provider
            .capability_snapshot()
            .supports_operation(&query.operation, query.family));
        assert!(matches!(
            provider.query_feasibility(&query),
            ProviderFeasibility::Supported { .. }
        ));
        let output = block_on(match operation {
            DirectUnary::Sin => provider.unary_sin(&input),
            DirectUnary::Cos => provider.unary_cos(&input),
            DirectUnary::Tan => provider.unary_tan(&input),
            DirectUnary::Sinh => provider.unary_sinh(&input),
            DirectUnary::Cosh => provider.unary_cosh(&input),
            DirectUnary::Asinh => provider.unary_asinh(&input),
            DirectUnary::Atan => provider.unary_atan(&input),
            DirectUnary::Tanh => provider.unary_tanh(&input),
        })
        .unwrap();
        let expected: Vec<_> = values
            .iter()
            .map(|&v| match operation {
                DirectUnary::Sin => v.sin(),
                DirectUnary::Cos => v.cos(),
                DirectUnary::Tan => v.tan(),
                DirectUnary::Sinh => v.sinh(),
                DirectUnary::Cosh => v.cosh(),
                DirectUnary::Asinh => v.asinh(),
                DirectUnary::Atan => v.atan(),
                DirectUnary::Tanh => v.tanh(),
            })
            .collect();
        let actual = block_on(provider.download(&output)).unwrap();
        assert_eq!(actual.shape, vec![1, 2, 1, 2]);
        for (actual, expected) in actual.data.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "{}",
                operation.entry().identity.name
            );
        }
        provider.free(&output).unwrap();
    }
    provider.free(&input).unwrap();
}

#[test]
fn trigonometric_queries_do_not_admit_wrong_physical_class_storage_or_outputs() {
    let provider = InProcessProvider::new();
    for &operation in OPERATIONS {
        for element in [
            ProviderElementType::F32,
            ProviderElementType::I64,
            ProviderElementType::Logical,
        ] {
            let mut request = query(operation);
            request.inputs[0].element_type = element;
            assert!(matches!(
                validate(&provider, false, &request),
                Some(ProviderFeasibility::Rejected { .. })
            ));
        }
        let mut request = query(operation);
        request.inputs[0].storage = ProviderStorage::Sparse;
        assert!(matches!(
            validate(&provider, false, &request),
            Some(ProviderFeasibility::Rejected { .. })
        ));
        request = query(operation);
        request.inputs[0].storage = ProviderStorage::DenseComplexInterleaved;
        request.inputs[0].element_type = ProviderElementType::ComplexF64;
        assert_eq!(
            matches!(
                validate(&provider, false, &request),
                Some(ProviderFeasibility::Supported { .. })
            ),
            operation.complex()
        );
        request = query(operation);
        request.outputs = vec![request.inputs[0].clone()];
        request.outputs[0].residency = ProviderResidency::Device;
        request.outputs[0].shape = vec![4, 1];
        assert!(matches!(
            validate(&provider, false, &request),
            Some(ProviderFeasibility::Rejected { .. })
        ));
        request = query(operation);
        request.inputs[0].layout = ProviderLayout::Strided;
        assert!(matches!(
            validate(&provider, false, &request),
            Some(ProviderFeasibility::Rejected { .. })
        ));
    }
}

#[test]
fn direct_complex_trigonometric_queries_match_interleaved_provider_values() {
    let provider = InProcessProvider::new();
    let data = [0.25, 0.5, -0.75, -0.25];
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&data),
            shape: &[1, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    for operation in [DirectUnary::Sin, DirectUnary::Cos, DirectUnary::Tan] {
        let mut request = query(operation);
        request.inputs[0].shape = vec![1, 2];
        request.inputs[0].storage = ProviderStorage::DenseComplexInterleaved;
        request.inputs[0].element_type = ProviderElementType::ComplexF64;
        assert!(matches!(
            provider.query_feasibility(&request),
            ProviderFeasibility::Supported { .. }
        ));
        let output = block_on(match operation {
            DirectUnary::Sin => provider.unary_sin(&input),
            DirectUnary::Cos => provider.unary_cos(&input),
            DirectUnary::Tan => provider.unary_tan(&input),
            _ => unreachable!(),
        })
        .unwrap();
        let actual = block_on(provider.download(&output)).unwrap();
        assert_eq!(actual.shape, vec![1, 2]);
        for (actual, source) in actual.data.chunks_exact(2).zip(data.chunks_exact(2)) {
            let value = num_complex::Complex64::new(source[0], source[1]);
            let expected = match operation {
                DirectUnary::Sin => value.sin(),
                DirectUnary::Cos => value.cos(),
                DirectUnary::Tan => value.tan(),
                _ => unreachable!(),
            };
            assert!((actual[0] - expected.re).abs() < 1e-12);
            assert!((actual[1] - expected.im).abs() < 1e-12);
        }
        provider.free(&output).unwrap();
    }
    provider.free(&input).unwrap();
}

#[test]
fn trigonometric_query_does_not_claim_range_dependent_or_compound_kernels() {
    let provider = InProcessProvider::new();
    for name in [
        "asin", "acos", "acosh", "atanh", "sind", "cosd", "tand", "sinpi", "cospi", "deg2rad",
        "rad2deg", "pol2cart",
    ] {
        let mut request = query(DirectUnary::Sin);
        request.operation = ProviderOperationIdentity::new(name);
        assert!(validate(&provider, false, &request).is_none(), "{name}");
    }
}
