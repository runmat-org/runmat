use super::*;
use crate::value_fact::value_fact;
use runmat_value::Tensor;

fn proof(left: &Value, right: &Value) -> Option<FusionAdmissionEvidence> {
    let facts = [value_fact(left), value_fact(right)];
    let operands = [None, Some(right)];
    let mut work = FusionAdmissionWork::None;
    futures::executor::block_on(admit(
        FusionAdmissionContext::new(&[0, 1], &operands, &facts, &[None, None], &mut work)
            .expect("aligned pairwise operands"),
    ))
    .unwrap()
}

#[test]
fn exact_positive_zero_pairwise_proves_nonnegative_or_nan() {
    let left = Value::Tensor(Tensor::from_f32(vec![-3.0, f32::NAN], vec![1, 2]).unwrap());
    let zero = Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap());
    let admitted = proof(&left, &zero).expect("selected Single pairwise maximum");
    assert_eq!(
        admitted.numerical_domain,
        NumericalDomainEvidence::NonnegativeOrNaN
    );
    assert_eq!(
        admitted.output_kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
    let negative_zero = Value::Tensor(Tensor::from_f32(vec![-0.0], vec![1, 1]).unwrap());
    assert!(proof(&left, &negative_zero).is_none());
    let changed_zero =
        Value::Tensor(Tensor::from_f32(vec![f32::MIN_POSITIVE], vec![1, 1]).unwrap());
    assert!(proof(&left, &changed_zero).is_none());
    assert!(proof(&left, &Value::Num(0.0)).is_none());
}
