use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_accelerate_api::IntegerElementType;
use runmat_builtins::{MTIMES_DESCRIPTOR, MTIMES_ERROR_INVALID_INPUT};
use runmat_value::{IntValue, IntegerStorage, LogicalArray, Tensor};

fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
    err
}

#[test]
fn operational_owner_errors_keep_retry_causes_and_public_identity() {
    use crate::builtins::common::gpu_helpers::ResidentComputeDecline;
    for (decline, retry) in [
        (
            ResidentComputeDecline::PolicyDenied,
            crate::GpuGatherRetry::Requested,
        ),
        (
            ResidentComputeDecline::InvalidOwner,
            crate::GpuGatherRetry::Never,
        ),
    ] {
        let original = crate::build_runtime_error("shared product owner failure")
            .with_source(std::io::Error::other("retained owner cause"))
            .build();
        let mapped = errors::map_control_flow(decline.with_domain_error(original));
        assert_eq!(
            mapped.identifier(),
            runmat_builtins::MTIMES_ERROR_INTERNAL.identifier
        );
        assert_eq!(mapped.context.builtin.as_deref(), Some("mtimes"));
        assert_eq!(mapped.gpu_gather_retry, retry);
        let mut cause = std::error::Error::source(&mapped);
        let mut retained = false;
        while let Some(error) = cause {
            retained |= error.downcast_ref::<std::io::Error>().is_some();
            cause = error.source();
        }
        assert!(retained);
    }
    let cancellation = crate::build_runtime_error("cancelled")
        .with_identifier("RunMat:Cancelled")
        .with_failure_origin(
            crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation,
        )
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build();
    let mapped = errors::map_control_flow(cancellation);
    assert_eq!(mapped.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(
        mapped.failure_origin(),
        crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation
    );
    assert_eq!(mapped.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let specific = crate::build_runtime_error("provider integrity failure")
        .with_identifier("RunMat:Provider:Integrity")
        .build();
    assert_eq!(
        errors::map_control_flow(specific).identifier(),
        Some("RunMat:Provider:Integrity")
    );
}

fn integer_scalar_mtimes_cases() -> Vec<(IntegerStorage, IntValue, IntegerStorage)> {
    vec![
        (
            IntegerStorage::I8(vec![i8::MAX, i8::MIN, 2]),
            IntValue::I8(2),
            IntegerStorage::I8(vec![i8::MAX, i8::MIN, 4]),
        ),
        (
            IntegerStorage::I16(vec![i16::MAX, i16::MIN, 2]),
            IntValue::I16(2),
            IntegerStorage::I16(vec![i16::MAX, i16::MIN, 4]),
        ),
        (
            IntegerStorage::I32(vec![i32::MAX, i32::MIN, 2]),
            IntValue::I32(2),
            IntegerStorage::I32(vec![i32::MAX, i32::MIN, 4]),
        ),
        (
            IntegerStorage::I64(vec![i64::MAX, i64::MIN, 2]),
            IntValue::I64(2),
            IntegerStorage::I64(vec![i64::MAX, i64::MIN, 4]),
        ),
        (
            IntegerStorage::U8(vec![u8::MAX, 2, 0]),
            IntValue::U8(2),
            IntegerStorage::U8(vec![u8::MAX, 4, 0]),
        ),
        (
            IntegerStorage::U16(vec![u16::MAX, 2, 0]),
            IntValue::U16(2),
            IntegerStorage::U16(vec![u16::MAX, 4, 0]),
        ),
        (
            IntegerStorage::U32(vec![u32::MAX, 2, 0]),
            IntValue::U32(2),
            IntegerStorage::U32(vec![u32::MAX, 4, 0]),
        ),
        (
            IntegerStorage::U64(vec![u64::MAX, (1_u64 << 53) + 1, 0]),
            IntValue::U64(2),
            IntegerStorage::U64(vec![u64::MAX, (1_u64 << 54) + 2, 0]),
        ),
    ]
}

fn integer_element_type(storage: &IntegerStorage) -> IntegerElementType {
    match storage {
        IntegerStorage::I8(_) => IntegerElementType::I8,
        IntegerStorage::I16(_) => IntegerElementType::I16,
        IntegerStorage::I32(_) => IntegerElementType::I32,
        IntegerStorage::I64(_) => IntegerElementType::I64,
        IntegerStorage::U8(_) => IntegerElementType::U8,
        IntegerStorage::U16(_) => IntegerElementType::U16,
        IntegerStorage::U32(_) => IntegerElementType::U32,
        IntegerStorage::U64(_) => IntegerElementType::U64,
    }
}

mod host;
mod integer;
mod provider;
mod wgpu;

fn mtimes_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(super::mtimes_builtin(lhs, rhs))
}
