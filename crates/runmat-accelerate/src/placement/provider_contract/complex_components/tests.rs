use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{
    GpuTensorStorage, HostNumericDataView, HostNumericTensorView, ProviderWorkload,
};

fn input(shape: &[u64], complex: bool) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type: if complex {
            ProviderElementType::ComplexF64
        } else {
            ProviderElementType::F64
        },
        storage: if complex {
            ProviderStorage::DenseComplexInterleaved
        } else {
            ProviderStorage::DenseReal
        },
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.to_vec(),
        residency: ProviderResidency::Host,
    }
}
fn query(operation: Operation, inputs: Vec<ProviderRepresentation>) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(operation.identity()),
        family: ProviderOperationFamily::Elementwise,
        inputs,
        outputs: vec![],
        workload: ProviderWorkload::default(),
    }
}
fn snapshot(provider: &InProcessProvider) -> ProviderCapabilitySnapshot {
    let mut snapshot = provider.capability_snapshot();
    if !snapshot.supports_operation(
        &ProviderOperationIdentity::new(Operation::Construct.identity()),
        ProviderOperationFamily::Elementwise,
    ) {
        append(&mut snapshot);
    }
    snapshot
}

#[test]
fn real_and_complex_component_declarations_match_actual_provider_results() {
    let provider = InProcessProvider::new();
    let snapshot = snapshot(&provider);
    for complex in [false, true] {
        let values = if complex {
            vec![2.0, 3.0, -4.0, 5.0]
        } else {
            vec![2.0, -4.0]
        };
        let handle = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&values),
                shape: &[1, 2],
                storage: if complex {
                    GpuTensorStorage::ComplexInterleaved
                } else {
                    GpuTensorStorage::Real
                },
            })
            .unwrap();
        for rule in [
            NumericComponentRule::Conjugate,
            NumericComponentRule::RealPart,
            NumericComponentRule::ImaginaryPart,
        ] {
            let operation = Operation::Component(rule);
            assert!(feasibility(
                HookProfile::InProcess,
                provider.precision(),
                &snapshot,
                &query(operation, vec![input(&[1, 2], complex)])
            )
            .unwrap()
            .is_supported());
            let output = if matches!(rule, NumericComponentRule::Conjugate) {
                block_on(provider.unary_conj(&handle))
                    .unwrap()
                    .expect("admitted conjugation")
            } else {
                block_on(match rule {
                    NumericComponentRule::Conjugate => unreachable!("typed hook handled above"),
                    NumericComponentRule::RealPart => provider.unary_real(&handle),
                    NumericComponentRule::ImaginaryPart => provider.unary_imag(&handle),
                })
                .unwrap()
            };
            let expected = match (rule, complex) {
                (NumericComponentRule::Conjugate, true) => vec![2.0, -3.0, -4.0, -5.0],
                (NumericComponentRule::ImaginaryPart, true) => vec![3.0, 5.0],
                (NumericComponentRule::ImaginaryPart, false) => vec![0.0, 0.0],
                _ => vec![2.0, -4.0],
            };
            assert_eq!(block_on(provider.download(&output)).unwrap().data, expected);
            output.release().unwrap();
        }
        handle.release().unwrap();
    }
}

#[test]
fn construction_cost_includes_native_complex_output_and_scalar_expansion() {
    let provider = InProcessProvider::new();
    let snapshot = snapshot(&provider);
    for shapes in [
        vec![input(&[2, 3], false)],
        vec![input(&[2, 3], false), input(&[1, 1], false)],
    ] {
        let request = query(Operation::Construct, shapes);
        let ProviderFeasibility::Supported { estimate } = feasibility(
            HookProfile::InProcess,
            provider.precision(),
            &snapshot,
            &request,
        )
        .unwrap() else {
            panic!("constructor")
        };
        assert_eq!(estimate.output_bytes, Some(96));
        assert_eq!(
            estimate.transient_bytes,
            Some(if request.inputs.len() == 1 { 144 } else { 152 })
        );
    }
    for inputs in [
        vec![input(&[2, 1], false), input(&[1, 3], false)],
        vec![input(&[1, 2], true)],
        vec![input(&[u64::MAX, 2], false)],
    ] {
        assert!(!feasibility(
            HookProfile::InProcess,
            provider.precision(),
            &snapshot,
            &query(Operation::Construct, inputs)
        )
        .unwrap()
        .is_supported());
    }
}

#[test]
fn constructor_hooks_preserve_shape_and_interleave_real_and_imaginary_components() {
    let provider = InProcessProvider::new();
    let real = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[2.0, -4.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let imaginary = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[3.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    for (output, expected) in [
        (
            block_on(provider.complex_from_real(&real)).unwrap(),
            vec![2.0, 0.0, -4.0, 0.0],
        ),
        (
            block_on(provider.complex_from_real_imag(&real, &imaginary)).unwrap(),
            vec![2.0, 3.0, -4.0, 3.0],
        ),
    ] {
        assert_eq!(output.shape(), vec![1, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(block_on(provider.download(&output)).unwrap().data, expected);
        output.release().unwrap();
    }
    real.release().unwrap();
    imaginary.release().unwrap();
}
