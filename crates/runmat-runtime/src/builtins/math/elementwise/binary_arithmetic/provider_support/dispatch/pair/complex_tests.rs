use super::*;
use futures::FutureExt;
use runmat_accelerate_api::{GpuHandleProvenance, GpuTensorStorage, NumericElementType};
use runmat_builtins::PLUS_CATALOG_ENTRY;
use std::sync::atomic::Ordering;

mod fixture;
use fixture::{inputs, with_case, ResultMode};

#[test]
fn mixed_complex_singleton_expansion_preserves_storage_precision_and_explicit_intent() {
    with_case(ResultMode::Valid, |provider| {
        let (left, right) = inputs(provider);
        for operation in [
            ArithmeticProviderOperation::Add,
            ArithmeticProviderOperation::Subtract,
            ArithmeticProviderOperation::Multiply,
        ] {
            assert!(matches!(
                operation.pair_route(&left, &right),
                Some(ArithmeticPairRoute::Complex(
                    runmat_accelerate_api::ProviderPrecision::F32
                ))
            ));
        }
        let result = futures::executor::block_on(try_pair(
            ArithmeticProviderOperation::Add,
            PLUS_CATALOG_ENTRY.identity,
            &left,
            &right,
        ))
        .unwrap()
        .expect("admitted provider result");
        let Value::GpuTensor(output) = result else {
            panic!("complex arithmetic must remain resident")
        };
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(
            output.physical().storage(),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&output),
            Some(GpuHandleProvenance::Explicit)
        );
        assert!(left.allocation_lease().validate_live().is_ok());
        assert!(right.allocation_lease().validate_live().is_ok());
        drop(output);
        assert_eq!(
            provider
                .output
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .retirements(),
            1
        );
    });
}

#[test]
fn malformed_complex_outputs_are_retired_without_releasing_borrowed_inputs() {
    for mode in [
        ResultMode::WrongStorage,
        ResultMode::WrongPrecision,
        ResultMode::WrongShape,
        ResultMode::ForeignOwner,
        ResultMode::BorrowedAlias,
    ] {
        with_case(mode, |provider| {
            let (left, right) = inputs(provider);
            let result = futures::executor::block_on(try_pair(
                ArithmeticProviderOperation::Add,
                PLUS_CATALOG_ENTRY.identity,
                &left,
                &right,
            ));
            let error = result.expect_err("invalid provider result must not publish");
            assert!(
                error
                    .message()
                    .contains("invalid complex arithmetic output"),
                "{mode:?}: {error}"
            );
            if !matches!(mode, ResultMode::BorrowedAlias) {
                assert_eq!(
                    provider
                        .output
                        .lock()
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .retirements(),
                    1
                );
            }
            assert!(left.allocation_lease().validate_live().is_ok());
            assert!(right.allocation_lease().validate_live().is_ok());
        });
    }
}

#[test]
fn abandoned_complex_operation_keeps_borrowed_inputs_live() {
    with_case(ResultMode::Pending, |provider| {
        let (left, right) = inputs(provider);
        let mut operation = Box::pin(try_pair(
            ArithmeticProviderOperation::Add,
            PLUS_CATALOG_ENTRY.identity,
            &left,
            &right,
        ));
        assert!(operation.as_mut().now_or_never().is_none());
        drop(operation);
        assert!(left.allocation_lease().validate_live().is_ok());
        assert!(right.allocation_lease().validate_live().is_ok());
        assert!(provider.output.lock().unwrap().is_none());
    });
}

#[test]
fn complex_admission_requires_matching_floating_precision_before_provider_effects() {
    with_case(ResultMode::Valid, |provider| {
        let (left, _) = inputs(provider);
        let right = provider
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap()
            .handle;
        let result = futures::executor::block_on(try_pair(
            ArithmeticProviderOperation::Add,
            PLUS_CATALOG_ENTRY.identity,
            &left,
            &right,
        ))
        .unwrap();
        assert!(result.is_none());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        assert!(left.allocation_lease().validate_live().is_ok());
        assert!(right.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn unsupported_complex_hook_remains_a_typed_host_fallback() {
    with_case(ResultMode::Unsupported, |provider| {
        let (left, right) = inputs(provider);
        let result = futures::executor::block_on(try_pair(
            ArithmeticProviderOperation::Add,
            PLUS_CATALOG_ENTRY.identity,
            &left,
            &right,
        ))
        .unwrap();
        assert!(result.is_none());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(left.allocation_lease().validate_live().is_ok());
        assert!(right.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn failed_complex_hook_is_terminal_without_host_replay() {
    with_case(ResultMode::Failed, |provider| {
        let (left, right) = inputs(provider);
        let error = futures::executor::block_on(try_pair(
            ArithmeticProviderOperation::Add,
            PLUS_CATALOG_ENTRY.identity,
            &left,
            &right,
        ))
        .expect_err("device failure must remain terminal");
        assert!(error.message().contains("device lost"));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(provider.output.lock().unwrap().is_none());
        assert!(left.allocation_lease().validate_live().is_ok());
        assert!(right.allocation_lease().validate_live().is_ok());
    });
}
