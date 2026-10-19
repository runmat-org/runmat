//! Metadata-only admission shared by causal and zero-phase IIR forms.
//! Numerical conversion and provider lease authentication remain at their
//! owning execution boundaries; this pass never downloads an operand.

use runmat_value::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IirForm {
    Causal,
    ZeroPhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IirOperand {
    Numerator,
    Denominator,
    Signal,
    InitialState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IirFaultKind {
    UnsupportedClass,
    Empty,
    NonVector,
    LeadingDenominatorZero,
    SignalTooShort,
    GeometryOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct IirAdmissionFault {
    pub operand: IirOperand,
    pub kind: IirFaultKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct IirAdmittedGeometry {
    pub coefficient_len: usize,
}

pub(crate) fn preflight(
    form: IirForm,
    numerator: &Value,
    denominator: &Value,
    signal: &Value,
    initial_state: Option<&Value>,
) -> Result<IirAdmittedGeometry, IirAdmissionFault> {
    let b_len = check_operand(IirOperand::Numerator, numerator, true, true)?;
    let a_len = check_operand(IirOperand::Denominator, denominator, true, true)?;
    let x_len = check_operand(
        IirOperand::Signal,
        signal,
        matches!(form, IirForm::ZeroPhase),
        matches!(form, IirForm::ZeroPhase),
    )?;
    if let Some(initial_state) = initial_state {
        check_operand(IirOperand::InitialState, initial_state, false, false)?;
    }
    if leading_denominator_is_zero(denominator) {
        return Err(IirAdmissionFault {
            operand: IirOperand::Denominator,
            kind: IirFaultKind::LeadingDenominatorZero,
        });
    }
    if matches!(form, IirForm::ZeroPhase) {
        let state_len = b_len.max(a_len).saturating_sub(1);
        let reflection = state_len.checked_mul(3).ok_or(IirAdmissionFault {
            operand: IirOperand::Signal,
            kind: IirFaultKind::GeometryOverflow,
        })?;
        if x_len <= reflection && state_len > 0 {
            return Err(IirAdmissionFault {
                operand: IirOperand::Signal,
                kind: IirFaultKind::SignalTooShort,
            });
        }
    }
    Ok(IirAdmittedGeometry {
        coefficient_len: b_len.max(a_len),
    })
}

fn check_operand(
    operand: IirOperand,
    value: &Value,
    require_vector: bool,
    require_nonempty: bool,
) -> Result<usize, IirAdmissionFault> {
    let shape = match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::ComplexTensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(logical) => Some(logical.shape.as_slice()),
        Value::GpuTensor(handle) => Some(handle.shape()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => None,
        _ => return Err(fault(operand, IirFaultKind::UnsupportedClass)),
    };
    let Some(shape) = shape else {
        return Ok(1);
    };
    if require_vector && shape.iter().filter(|&&extent| extent > 1).count() > 1 {
        return Err(fault(operand, IirFaultKind::NonVector));
    }
    let length = shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or(fault(operand, IirFaultKind::GeometryOverflow))?;
    if require_nonempty && length == 0 {
        return Err(fault(operand, IirFaultKind::Empty));
    }
    Ok(length)
}

fn fault(operand: IirOperand, kind: IirFaultKind) -> IirAdmissionFault {
    IirAdmissionFault { operand, kind }
}

fn leading_denominator_is_zero(value: &Value) -> bool {
    match value {
        Value::Num(value) => *value == 0.0,
        Value::Int(value) => value.is_zero(),
        Value::Bool(value) => !*value,
        Value::Complex(re, im) => *re == 0.0 && *im == 0.0,
        Value::Tensor(tensor) => tensor
            .numeric_value_at(0)
            .is_some_and(|value| value.is_zero()),
        Value::ComplexTensor(tensor) => tensor
            .numeric_value_at(0)
            .is_some_and(|(re, im)| re.is_zero() && im.is_zero()),
        Value::LogicalArray(logical) => logical.data.first() == Some(&0),
        Value::GpuTensor(_) => false,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{LogicalArray, Tensor};

    #[test]
    fn form_specific_signal_geometry_does_not_constrain_causal_arrays() {
        let matrix = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
        assert!(preflight(
            IirForm::Causal,
            &Value::Num(1.0),
            &Value::Num(1.0),
            &matrix,
            None
        )
        .is_ok());
        assert_eq!(
            preflight(
                IirForm::ZeroPhase,
                &Value::Num(1.0),
                &Value::Num(1.0),
                &matrix,
                None
            ),
            Err(fault(IirOperand::Signal, IirFaultKind::NonVector))
        );
    }

    #[test]
    fn denominator_zero_and_short_reflection_are_known_without_payload_download() {
        let logical_zero = Value::LogicalArray(LogicalArray::new(vec![0], vec![1, 1]).unwrap());
        assert_eq!(
            preflight(
                IirForm::Causal,
                &Value::Num(1.0),
                &logical_zero,
                &Value::Num(1.0),
                None
            ),
            Err(fault(
                IirOperand::Denominator,
                IirFaultKind::LeadingDenominatorZero
            ))
        );
        assert_eq!(
            preflight(
                IirForm::Causal,
                &Value::Num(1.0),
                &Value::Num(0.0),
                &Value::Num(1.0),
                None
            ),
            Err(fault(
                IirOperand::Denominator,
                IirFaultKind::LeadingDenominatorZero
            ))
        );
        let b = Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap());
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
        assert_eq!(
            preflight(IirForm::ZeroPhase, &b, &Value::Num(1.0), &x, None),
            Err(fault(IirOperand::Signal, IirFaultKind::SignalTooShort))
        );
    }
}
