//! Fixed matrix construction is valid only for primitive scalar stack values.
//! Shape-one tensors are not interchangeable with `Num`, `Int`, `Bool`, or
//! `Complex`: the fixed constructor converts each stack value as a scalar.

use runmat_hir::OperatorKind;
use runmat_mir::{
    MirAggregateElement, MirBody, MirConstant, MirLocalId, MirLocalKind, MirOperand,
    MirOutputTarget, MirPlace, MirRvalue, MirStmtKind,
};
use runmat_value::Value;

pub(super) fn needs_dynamic_concat(body: &MirBody, elements: &[MirAggregateElement]) -> bool {
    elements.iter().any(|element| {
        element
            .operand()
            .is_none_or(|operand| !primitive_scalar(body, operand, &mut Vec::new()))
    })
}

fn primitive_scalar(body: &MirBody, operand: &MirOperand, visiting: &mut Vec<MirLocalId>) -> bool {
    match operand {
        MirOperand::Constant(
            MirConstant::Number(_) | MirConstant::IntegerLiteral(_) | MirConstant::Bool(_),
        ) => true,
        MirOperand::Constant(MirConstant::Symbol(name)) => matches!(
            runmat_runtime::builtin::builtin_constant_value(&name.0),
            Some(Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _))
        ),
        MirOperand::Constant(MirConstant::String(_) | MirConstant::EmptyArray)
        | MirOperand::FunctionHandle(_) => false,
        MirOperand::Local(local) => primitive_temporary(body, *local, visiting),
    }
}

fn primitive_temporary(body: &MirBody, local: MirLocalId, visiting: &mut Vec<MirLocalId>) -> bool {
    if visiting.len() >= body.locals.len()
        || visiting.contains(&local)
        || !body
            .locals
            .get(local.0)
            .is_some_and(|slot| slot.id == local && matches!(slot.kind, MirLocalKind::Temporary))
    {
        return false;
    }
    let mut definition = None;
    for statement in body.blocks.iter().flat_map(|block| &block.statements) {
        match &statement.kind {
            MirStmtKind::Assign { place, value } if place_root_local(place) == Some(local) => {
                if !matches!(place, MirPlace::Local(_)) || definition.replace(value).is_some() {
                    return false;
                }
            }
            MirStmtKind::MultiAssign { targets, .. }
                if targets.targets.iter().any(|target| match target {
                    MirOutputTarget::Place(place) => place_root_local(place) == Some(local),
                    MirOutputTarget::Sequence(target) => {
                        place_root_local(target.base()) == Some(local)
                    }
                    MirOutputTarget::Discard => false,
                }) =>
            {
                return false
            }
            MirStmtKind::SequenceAssign { target, .. }
                if place_root_local(target.base()) == Some(local) =>
            {
                return false
            }
            MirStmtKind::PlaceMutation(mutation)
                if place_root_local(&mutation.place) == Some(local) =>
            {
                return false
            }
            MirStmtKind::WorkspaceEffect { bindings, .. } if bindings.contains(&local) => {
                return false
            }
            _ => {}
        }
    }
    let Some(value) = definition else {
        return false;
    };
    visiting.push(local);
    let proven = match value {
        MirRvalue::Use(operand) => primitive_scalar(body, operand, visiting),
        MirRvalue::Unary(operator, operand)
            if matches!(
                operator,
                OperatorKind::UnaryPlus | OperatorKind::UnaryMinus | OperatorKind::Not
            ) =>
        {
            primitive_scalar(body, operand, visiting)
        }
        MirRvalue::Binary(left, operator, right) if scalar_binary(*operator) => {
            primitive_scalar(body, left, visiting) && primitive_scalar(body, right, visiting)
        }
        _ => false,
    };
    visiting.pop();
    proven
}

fn scalar_binary(operator: OperatorKind) -> bool {
    matches!(
        operator,
        OperatorKind::Add
            | OperatorKind::Subtract
            | OperatorKind::MatrixMultiply
            | OperatorKind::ElementwiseMultiply
            | OperatorKind::MatrixPower
            | OperatorKind::ElementwisePower
            | OperatorKind::Mldivide
            | OperatorKind::Mrdivide
            | OperatorKind::ElementwiseDivide
            | OperatorKind::ElementwiseLeftDivide
            | OperatorKind::Equal
            | OperatorKind::NotEqual
            | OperatorKind::Less
            | OperatorKind::LessEqual
            | OperatorKind::Greater
            | OperatorKind::GreaterEqual
            | OperatorKind::ElementwiseAnd
            | OperatorKind::ElementwiseOr
    )
}

fn place_root_local(place: &MirPlace) -> Option<MirLocalId> {
    match place {
        MirPlace::Local(local) => Some(*local),
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            place_root_local(base)
        }
        MirPlace::Binding(_) => None,
    }
}
