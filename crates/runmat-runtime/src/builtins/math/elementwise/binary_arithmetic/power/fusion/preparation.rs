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

#[cfg(test)]
pub(super) fn assert_selected_finite_quote() {
    use super::tests::{computed_base, power_receipt};
    use super::{FusionScalarControl, SEMANTICS};
    use runmat_value::Tensor;

    let admission = SEMANTICS.admit.expect("selected POWER admission owner");
    assert_eq!(admission.preparation_arity(), Some(2));
    assert!(
        SEMANTICS.finite_preparation_class().is_none(),
        "domain admission is not pure"
    );
    for exponent in [0.0_f64, -0.0, 1.0, 2.0] {
        let value = Value::Num(exponent);
        let operands = [None, Some(&value)];
        let resources = admission
            .preparation_resources(FusionAdmissionPreparationQuery {
                operands: &operands,
                max_rank: 4,
            })
            .expect("existing certified typed producer's Double control");
        let mut empty_work = FusionAdmissionWork::None;
        let empty = FusionAdmissionContext::new(&[], &[], &[], &[], &mut empty_work).unwrap();
        let actual_frame = admit_inner(empty);
        let stage = std::mem::size_of::<FusionProofStage>() as u64;
        assert_eq!(
            resources.scratch_bytes,
            std::mem::size_of_val(&actual_frame) as u64
                + (4 * std::mem::size_of::<runmat_types::DimensionFact>()) as u64
                + 17
                + stage
        );
        assert_eq!(resources.retained_bytes, stage);
        let receipt = power_receipt(computed_base(Value::Num(exponent)).unwrap());
        assert_eq!(
            receipt.exponent_scalar,
            Some(FusionScalarControl::DoubleBits(exponent.to_bits()))
        );
        let tensor = Value::Tensor(Tensor::new(vec![exponent], vec![1, 1]).unwrap());
        assert!(admission
            .preparation_resources(FusionAdmissionPreparationQuery {
                operands: &[None, Some(&tensor)],
                max_rank: 2,
            })
            .is_some());
    }
    let single = Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap());
    assert!(
        admission
            .preparation_resources(FusionAdmissionPreparationQuery {
                operands: &[None, Some(&single)],
                max_rank: 2,
            })
            .is_none(),
        "Single bits are never widened into certified Double authority"
    );
    let other = Value::Num(3.0);
    assert!(
        admission
            .preparation_resources(FusionAdmissionPreparationQuery {
                operands: &[None, Some(&other)],
                max_rank: 2,
            })
            .is_none(),
        "broader POWER is not part of the existing typed product"
    );
    assert!(
        computed_base(other).is_some(),
        "original general admission remains unchanged"
    );
}
