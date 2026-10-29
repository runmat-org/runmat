//! Bound only the admission branch used by the existing certified typed
//! producer: real Single base and original Double literal 0, 1, or 2. This is
//! not a quote for general POWER scans, scalar readback, or resident probes.
use super::{admit_inner, FusionAdmissionContext, FusionAdmissionWork};
use crate::builtins::common::spec::{
    FusionAdmissionPreparationQuery, FusionAdmissionResources, FusionProofStage,
};
use runmat_accelerate_api::{ExactSingleBasePower, PowerScalarControl};
use runmat_value::{NumericDType, NumericScalar, Value};

pub(super) fn resources(
    query: FusionAdmissionPreparationQuery<'_>,
) -> Option<FusionAdmissionResources> {
    let [_, Some(exponent)] = query.operands else {
        return None;
    };
    let bits = match exponent {
        Value::Num(value) => value.to_bits(),
        Value::Tensor(tensor) if tensor.len() == 1 => match tensor.numeric_value_at(0)? {
            NumericScalar::F64(value) => value.to_bits(),
            _ => return None,
        },
        _ => return None,
    };
    // Same implementation authority as ScalarProgram validation and typed
    // WGSL lowering; never widen Single bits to invent a Double control.
    ExactSingleBasePower::for_control(PowerScalarControl::DoubleBits(bits))?;
    if query.max_rank < 2 {
        return None;
    }
    let mut fact_bytes = 0u64;
    for operand in query.operands {
        let (rank, origin) = match operand {
            Some(Value::Num(_)) => (0, 0),
            Some(Value::Tensor(tensor))
                if matches!(
                    tensor.numeric_dtype(),
                    NumericDType::F32 | NumericDType::F64
                ) =>
            {
                (tensor.shape().len(), 0)
            }
            Some(Value::GpuTensor(handle)) => (handle.shape().len(), 17),
            // A source-interior result is either absent or the existing
            // request-local computed scalar. Its fact cannot exceed the
            // admitted source rank/origin payload bound.
            None => (query.max_rank, 17),
            _ => return None,
        };
        if rank > query.max_rank {
            return None;
        }
        let shape = rank.checked_mul(std::mem::size_of::<runmat_types::DimensionFact>())?;
        fact_bytes = fact_bytes
            .checked_add(u64::try_from(shape).ok()?)?
            .checked_add(origin)?;
    }
    // The ORIGINAL async future is constructed on the stack but never polled:
    // its concrete compiler-owned layout is exactly the Box request. Empty
    // operands confer no numerical authority and perform no admission work.
    let mut work = FusionAdmissionWork::None;
    let context = FusionAdmissionContext::new(&[], &[], &[], &[], &mut work)?;
    let frame = admit_inner(context);
    let frame_bytes = u64::try_from(std::mem::size_of_val(&frame)).ok()?;
    // The universal scalar branch records exactly one RangePropagation stage.
    // AdmissionObservation::completed clones that one-stage backing before
    // its original work drops; the clone remains in the resolved source.
    let stage = u64::try_from(std::mem::size_of::<FusionProofStage>()).ok()?;
    Some(FusionAdmissionResources {
        scratch_bytes: frame_bytes.checked_add(fact_bytes)?.checked_add(stage)?,
        retained_bytes: stage,
    })
}
