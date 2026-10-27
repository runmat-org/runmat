use super::*;
use futures::executor::block_on;

#[test]
fn native_single_and_double_predicates_preserve_logical_outputs() {
    let provider = InProcessProvider::new();
    for input in [
        HostNumericDataView::F32(&[f32::NAN, 0.0, f32::INFINITY, -1.0]),
        HostNumericDataView::F64(&[f64::NAN, 0.0, f64::INFINITY, -1.0]),
    ] {
        let data = provider
            .upload_numeric(&HostNumericTensorView {
                data: input,
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let zero = provider
            .zeros_like(&data, &[2, 2])
            .unwrap()
            .expect("admitted optional constructor");
        let finite = provider.logical_isfinite(&data).unwrap();
        let nonzero = block_on(provider.elem_ne(&data, &zero)).unwrap();
        let selected = provider.logical_and(&finite, &nonzero).unwrap();
        assert_eq!(
            block_on(provider.download(&selected)).unwrap().data,
            [0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            block_on(provider.download(&provider.logical_or(&finite, &nonzero).unwrap()))
                .unwrap()
                .data,
            [1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            block_on(provider.download(&provider.logical_xor(&finite, &nonzero).unwrap()))
                .unwrap()
                .data,
            [1.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            block_on(provider.download(&provider.logical_not(&data).unwrap()))
                .unwrap()
                .data,
            [0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            block_on(provider.download(&provider.logical_isnan(&data).unwrap()))
                .unwrap()
                .data,
            [1.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            block_on(provider.download(&provider.logical_isinf(&data).unwrap()))
                .unwrap()
                .data,
            [0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            block_on(provider.download(&block_on(provider.elem_eq(&data, &zero)).unwrap()))
                .unwrap()
                .data,
            [0.0, 1.0, 0.0, 0.0]
        );
        assert!(selected.view().is_logical());
        assert!(!selected.shares_allocation_with(&data));
    }
}

#[test]
fn native_relational_predicates_respect_nan_infinity_signed_zero_and_broadcast() {
    let provider = InProcessProvider::new();
    for (left, right) in [
        (
            HostNumericDataView::F32(&[f32::NAN, f32::INFINITY, -0.0, -2.0]),
            HostNumericDataView::F32(&[0.0, 0.0, 0.0, -1.0]),
        ),
        (
            HostNumericDataView::F64(&[f64::NAN, f64::INFINITY, -0.0, -2.0]),
            HostNumericDataView::F64(&[0.0, 0.0, 0.0, -1.0]),
        ),
    ] {
        let left = provider
            .upload_numeric(&HostNumericTensorView {
                data: left,
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let right = provider
            .upload_numeric(&HostNumericTensorView {
                data: right,
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        for (predicate, expected) in [
            (
                BinaryPredicate::Compare(RelationalOperator::LessThan),
                [0.0, 0.0, 0.0, 1.0],
            ),
            (
                BinaryPredicate::Compare(RelationalOperator::LessThanOrEqual),
                [0.0, 0.0, 1.0, 1.0],
            ),
            (
                BinaryPredicate::Compare(RelationalOperator::GreaterThan),
                [0.0, 1.0, 0.0, 0.0],
            ),
            (
                BinaryPredicate::Compare(RelationalOperator::GreaterThanOrEqual),
                [0.0, 1.0, 1.0, 0.0],
            ),
        ] {
            let output = provider
                .binary_predicate_impl(&left, &right, predicate)
                .unwrap();
            assert_eq!(block_on(provider.download(&output)).unwrap().data, expected);
            assert!(output.view().is_logical());
            assert!(!output.shares_allocation_with(&left));
        }
    }
    let left = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[-1.0, 2.0]),
            shape: &[2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let right = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[0.0, 3.0, -2.0]),
            shape: &[1, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let output = block_on(provider.elem_lt(&left, &right)).unwrap();
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(
        block_on(provider.download(&output)).unwrap().data,
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    );
}

#[test]
fn integer_complex_and_foreign_inputs_do_not_enter_floating_predicates() {
    let provider = InProcessProvider::new();
    let integer = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::I64(&[i64::MAX]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.logical_isfinite(&integer).unwrap_err()
    ));
    // Exercise the public asynchronous boundary, not just its synchronous
    // preparation helper: unsupported class admission must remain pre-effect.
    for result in [
        block_on(provider.elem_eq(&integer, &integer)),
        block_on(provider.elem_ne(&integer, &integer)),
        block_on(provider.elem_lt(&integer, &integer)),
        block_on(provider.elem_le(&integer, &integer)),
        block_on(provider.elem_gt(&integer, &integer)),
        block_on(provider.elem_ge(&integer, &integer)),
    ] {
        assert!(result.unwrap_err().is_unsupported());
    }
    let complex = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0, 0.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.logical_isfinite(&complex).unwrap_err()
    ));
    let foreign = InProcessProvider::new()
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.logical_and(&integer, &foreign).unwrap_err()
    ));
    assert!(!block_on(provider.elem_ne(&integer, &foreign))
        .unwrap_err()
        .is_unsupported());
}
