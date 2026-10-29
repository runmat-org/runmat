//! Actual native exponential recipes, shared by existing F32/F64 owner witnesses.
use super::*;

pub(super) async fn qualify(provider: &'static WgpuProvider, double: bool) {
    use runmat_accelerate_api::{
        ProviderFeasibility, ProviderFeasibilityQuery, ProviderOperationFamily,
        ProviderOperationIdentity, ProviderWorkload,
    };
    let values = [-0.125_f64, -0.0, 1e-8, 0.25];
    let single = values.map(|value| value as f32);
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: if double {
                HostNumericDataView::F64(&values)
            } else {
                HostNumericDataView::F32(&single)
            },
            shape: &[1, 2, 1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    for (entry, operation) in [
        (
            &runmat_builtins::EXP_CATALOG_ENTRY,
            Operation::UnaryMath(runmat_types::ScalarUnaryMath::Exp),
        ),
        (&runmat_builtins::EXPM1_CATALOG_ENTRY, Operation::Expm1),
    ] {
        let request = ProviderRealElementwiseRequest {
            operation,
            source: source_representation(&input).unwrap(),
            source_buffer_bytes: input.physical().requested_bytes(),
        };
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(entry.identity.name),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![request.source.clone()],
            outputs: Vec::new(),
            workload: ProviderWorkload::default(),
        };
        let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&query) else {
            panic!("actual exponential hook must have exact typed capability");
        };
        let recipe = ProviderRealElementwiseRecipe::plan(provider, request)
            .unwrap()
            .unwrap();
        assert_eq!(
            estimate.output_bytes,
            Some(recipe.physical().output_buffer_bytes)
        );
        assert_eq!(
            estimate.transient_bytes,
            Some(
                input.physical().requested_bytes()
                    + recipe.physical().output_buffer_bytes
                    + recipe.physical().uniform_buffer_bytes
            )
        );
        assert_eq!(estimate.dispatches, Some(recipe.physical().dispatches));
        let cold = recipe.execute(&input).await.unwrap();
        let warm = recipe.execute(&input).await.unwrap();
        assert_eq!(cold.observed, warm.observed);
        assert_eq!(cold.observed.uniform_allocations, 1);
        assert_eq!(
            cold.observed.queue_submissions,
            cold.observed.dispatches + 1
        );
        let ordinary = match operation {
            Operation::Expm1 => provider.unary_expm1(&input).await.unwrap(),
            _ => provider.unary_exp(&input).await.unwrap(),
        };
        for output in [&cold.output, &warm.output, &ordinary] {
            assert!(!output.shares_allocation_with(&input));
            assert_eq!(output.shape(), input.shape());
            assert_eq!(
                output.physical().element_type(),
                input.physical().element_type()
            );
            let downloaded = provider.download_numeric(output).await.unwrap();
            let actual = match downloaded.data {
                HostNumericDataOwned::F64(values) if double => values,
                HostNumericDataOwned::F32(values) if !double => {
                    values.into_iter().map(f64::from).collect()
                }
                _ => panic!("native exponential class changed"),
            };
            for (index, actual) in actual.into_iter().enumerate() {
                let expected = if double {
                    if operation == Operation::Expm1 {
                        values[index].exp_m1()
                    } else {
                        values[index].exp()
                    }
                } else {
                    f64::from(if operation == Operation::Expm1 {
                        single[index].exp_m1()
                    } else {
                        single[index].exp()
                    })
                };
                let tolerance = if double { 2e-12 } else { 3e-6 };
                assert!(actual.is_finite(), "finite exponential oracle");
                if expected == 0.0 {
                    assert_eq!(actual.to_bits(), expected.to_bits(), "exact zero sign");
                } else {
                    // A tiny expm1 result must retain relative accuracy: zero
                    // cannot pass through an unrelated unit-scale allowance.
                    assert!((actual - expected).abs() <= tolerance * expected.abs());
                }
            }
        }
        input.allocation_lease().validate_live().unwrap();
    }
}
