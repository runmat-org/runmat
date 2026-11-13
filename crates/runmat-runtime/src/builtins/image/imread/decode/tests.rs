use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::num::NonZeroU64;
use std::rc::Rc;

#[test]
fn decoded_geometry_rejects_overflow_and_truncated_pixel_storage() {
    let error =
        checked_interleaved_geometry(usize::MAX, 2, 4, 3, 0).expect_err("geometry overflow");
    assert_eq!(error.identifier(), IMREAD_ERROR_SHAPE.identifier);
    let error = checked_interleaved_geometry(2, 3, 4, 3, 23).expect_err("truncated storage");
    assert_eq!(error.identifier(), IMREAD_ERROR_SHAPE.identifier);
    assert_eq!(
        checked_interleaved_geometry(2, 3, 4, 3, 24).unwrap(),
        (6, 24, 18)
    );
}

#[test]
fn decoded_geometry_rejects_zero_or_excess_channels_before_allocation() {
    for (input, output) in [(0, 1), (1, 0), (1, 2)] {
        let error = checked_interleaved_geometry(0, 0, input, output, 0)
            .expect_err("invalid channel contract");
        assert_eq!(error.identifier(), IMREAD_ERROR_SHAPE.identifier);
    }
    let mut cooperation = ImageCooperation::new(None, "imread");
    let error = futures::executor::block_on(tensor_from_interleaved(
        &[1_u8],
        1,
        1,
        1,
        2,
        NumericDType::U8,
        &mut cooperation,
    ))
    .expect_err("output channel exceeds decoded source");
    assert_eq!(error.identifier(), IMREAD_ERROR_SHAPE.identifier);
}

#[test]
fn native_byte_capacity_is_checked_before_reservation() {
    let elements = (isize::MAX as usize / std::mem::size_of::<u16>()) + 1;
    let error = native_buffer::<u16>(elements).expect_err("addressable capacity exceeded");
    assert_eq!(error.identifier(), IMREAD_ERROR_SHAPE.identifier);
    let reserved = native_buffer::<u16>(3).unwrap();
    assert!(reserved.is_empty());
    assert!(reserved.capacity() >= 3);
}

#[test]
fn valid_interleaved_alpha_keeps_native_class_and_column_major_order() {
    let raw = [10_u8, 20, 30, 40, 50, 60, 70, 80];
    let mut cooperation = ImageCooperation::new(None, "imread");
    let image = futures::executor::block_on(tensor_from_interleaved(
        &raw,
        2,
        1,
        4,
        3,
        NumericDType::U8,
        &mut cooperation,
    ))
    .unwrap();
    let alpha = futures::executor::block_on(alpha_from_interleaved(
        &raw,
        2,
        1,
        4,
        3,
        NumericDType::U8,
        &mut cooperation,
    ))
    .unwrap();
    assert_eq!(image.numeric_dtype(), NumericDType::U8);
    assert_eq!(image.shape(), &[1, 2, 3]);
    assert_eq!(
        image.materialize_f64(),
        vec![10.0, 50.0, 20.0, 60.0, 30.0, 70.0]
    );
    assert_eq!(alpha.numeric_dtype(), NumericDType::U8);
    assert_eq!(alpha.shape(), &[1, 2]);
    assert_eq!(alpha.materialize_f64(), vec![40.0, 80.0]);
}

#[test]
fn nonsquare_rgba_preserves_column_major_order_and_native_classes() {
    fn assert_layout<T: DecodedSample + From<u8>>() {
        let raw = (1_u8..=24).map(T::from).collect::<Vec<_>>();
        let mut cooperation = ImageCooperation::new(None, "imread");
        let image = futures::executor::block_on(tensor_from_interleaved(
            &raw,
            3,
            2,
            4,
            3,
            T::DTYPE,
            &mut cooperation,
        ))
        .unwrap();
        let alpha = futures::executor::block_on(alpha_from_interleaved(
            &raw,
            3,
            2,
            4,
            3,
            T::DTYPE,
            &mut cooperation,
        ))
        .unwrap();
        assert_eq!(image.numeric_dtype(), T::DTYPE);
        assert_eq!(image.shape(), &[2, 3, 3]);
        assert_eq!(
            image.materialize_f64(),
            vec![
                1.0, 13.0, 5.0, 17.0, 9.0, 21.0, 2.0, 14.0, 6.0, 18.0, 10.0, 22.0, 3.0, 15.0, 7.0,
                19.0, 11.0, 23.0,
            ]
        );
        assert_eq!(alpha.numeric_dtype(), T::DTYPE);
        assert_eq!(alpha.shape(), &[2, 3]);
        assert_eq!(
            alpha.materialize_f64(),
            vec![4.0, 16.0, 8.0, 20.0, 12.0, 24.0]
        );
    }

    assert_layout::<u8>();
    assert_layout::<u16>();
    assert_layout::<f32>();
}

#[test]
fn native_decode_yields_and_cancellation_prevents_partial_publication() {
    let raw = vec![7_u8; 64 * 64 * 4];
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = ImageCooperation::with_budget(budget, "imread");
    let tensor = futures::executor::block_on(tensor_from_interleaved(
        &raw,
        64,
        64,
        4,
        3,
        NumericDType::U8,
        &mut cooperation,
    ))
    .unwrap();
    assert_eq!(tensor.shape(), &[64, 64, 3]);
    assert!(
        turns.get() >= 3,
        "fill and conversion both charge real work"
    );

    let source = Rc::new(CancellationSource::new());
    let cancel_on_turn = source.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            cancel_on_turn.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = ImageCooperation::with_budget(budget, "imread");
    let error = futures::executor::block_on(tensor_from_interleaved(
        &raw,
        64,
        64,
        4,
        3,
        NumericDType::U8,
        &mut cooperation,
    ))
    .expect_err("cancelled fill must not publish a partial tensor");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
}

#[test]
fn unpolled_native_decode_does_not_start_work() {
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = ImageCooperation::with_budget(budget, "imread");
    let raw = vec![1_u8; 4];
    let future = tensor_from_interleaved(&raw, 1, 1, 4, 3, NumericDType::U8, &mut cooperation);
    drop(future);
    assert_eq!(turns.get(), 0);
}
