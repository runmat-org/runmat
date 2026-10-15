//! Typed Add sign propagation for the selected resolver. Source inference
//! remains the only authority for broadcast shape and result class.

use super::*;
use runmat_runtime::builtins::common::spec::{
    scan_represented_sign, FusionAdmissionWork, FusionSignEvidence,
};
use runmat_types::OperatorKind;
use runmat_types::{real_floating_add_mode, ValueKindFact};

pub(super) fn applies(operation: &FusionOp, snapshot: Option<SemanticBinding>) -> bool {
    matches!(
        operation,
        FusionOp::Primitive {
            op: PrimitiveOp::Add,
            ..
        }
    ) || snapshot
        .and_then(|binding| binding.declaration.semantics)
        .and_then(|semantics| semantics.scalar_operation)
        == Some(runmat_types::ScalarOperation::Operator(OperatorKind::Add))
}

pub(super) fn add_sign(
    program: &ResolvedProgram,
    operation: &FusionOp,
    inferred: &assembly::InferredOperation<'_>,
    values: &[Option<&Value>],
    snapshot: Option<SemanticBinding>,
    work: &mut FusionAdmissionWork,
) -> Result<Option<(FusionSignEvidence, ValueKindFact)>> {
    if !applies(operation, snapshot) {
        return Ok(None);
    }
    let [left, right] = inferred.arguments.as_slice() else {
        return Ok(None);
    };
    let [left_id, right_id] = inferred.input_ids else {
        return Ok(None);
    };
    let [left_value, right_value] = values else {
        return Ok(None);
    };
    let (Some(left_numeric), Some(right_numeric)) = (left.numeric(), right.numeric()) else {
        return Ok(None);
    };
    if [left_numeric, right_numeric].iter().any(|numeric| {
        numeric.domain != NumericDomain::Real
            || !matches!(numeric.class, NumericClass::Single | NumericClass::Double)
    }) || inferred.fact.shape.element_count().is_none()
    {
        return Ok(None);
    }
    *work = FusionAdmissionWork::SignProof {
        operation: OperatorKind::Add,
        host_elements: 0,
        propagated_operands: 0,
    };
    let mut scanned = 0_u64;
    let mut propagated = 0_u64;
    for (id, fact, value) in [
        (*left_id, left, *left_value),
        (*right_id, right, *right_value),
    ] {
        let class = fact.numeric().expect("checked floating fact").class;
        if program.sign.get(&id).copied() == Some(FusionSignEvidence::NonnegativeOrNaN { class }) {
            propagated += 1;
            *work = FusionAdmissionWork::SignProof {
                operation: OperatorKind::Add,
                host_elements: scanned,
                propagated_operands: propagated,
            };
            continue;
        }
        let Some(value) = value else {
            return Ok(None);
        };
        let actual = runmat_runtime::value_fact::value_fact(value);
        if actual.numeric() != fact.numeric() || actual.shape != fact.shape {
            return Ok(None);
        }
        let Some((proven, elements)) = scan_represented_sign(value, class) else {
            return Ok(None);
        };
        scanned = scanned
            .checked_add(elements)
            .ok_or_else(|| anyhow!("Add sign scan element count overflows"))?;
        *work = FusionAdmissionWork::SignProof {
            operation: OperatorKind::Add,
            host_elements: scanned,
            propagated_operands: propagated,
        };
        if !proven {
            return Ok(None);
        }
    }
    let Some(mode) = real_floating_add_mode(left, right) else {
        return Ok(None);
    };
    let projected = infer_binary(OperatorKind::Add, left, right);
    if !projected.diagnostics.is_empty() || projected.fact.shape != inferred.fact.shape {
        return Ok(None);
    }
    let Some(output) = projected.fact.numeric() else {
        return Ok(None);
    };
    if output.domain != NumericDomain::Real
        || output.class != mode.result_class()
        || inferred
            .fact
            .numeric()
            .is_some_and(|numeric| numeric != output)
    {
        return Ok(None);
    }
    Ok(Some((
        FusionSignEvidence::NonnegativeOrNaN {
            class: output.class,
        },
        ValueKindFact::Numeric(output),
    )))
}
