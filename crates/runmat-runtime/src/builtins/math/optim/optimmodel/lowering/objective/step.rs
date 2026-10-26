//! Typed recognition over canonical operator and selected builtin identities.
use super::*;
use runmat_types::OperatorKind as Op;
pub(super) enum Step {
    Constant,
    Add(GcHandle, GcHandle),
    Subtract(GcHandle, GcHandle),
    Forward(GcHandle),
    Scale {
        next: GcHandle,
        factor: f64,
        operation: Scale,
    },
    Sum {
        next: GcHandle,
    },
    Square {
        residual: GcHandle,
    },
    Other,
}
fn node(source: &evaluation::Plan, root: GcHandle, policy: Policy) -> BuiltinResult<&Node> {
    source
        .nodes
        .get(&root)
        .map(|node| node.node())
        .ok_or_else(|| policy.invalid("objective lost original node"))
}
pub(in crate::builtins::math::optim::optimmodel::lowering) fn scalar(
    source: &evaluation::Plan,
    root: GcHandle,
    policy: Policy,
) -> BuiltinResult<Option<f64>> {
    let node = node(source, root, policy)?;
    if evaluation::element_count(&node.shape, policy)? != 1 {
        return Ok(None);
    }
    let Kind::Constant(value) = &node.kind else {
        return Ok(None);
    };
    Ok(match value {
        Value::Num(value) => Some(*value),
        Value::Tensor(value) if value.numeric_dtype() == runmat_value::NumericDType::F64 => {
            match value.numeric_value_at(0) {
                Some(runmat_value::NumericScalar::F64(value)) => Some(value),
                _ => None,
            }
        }
        Value::SparseTensor(value)
            if !value.is_complex()
                && value.numeric_dtype() == Some(runmat_value::NumericDType::F64) =>
        {
            value.get(0, 0)
        }
        _ => None,
    })
}
fn is(name: &str, identity: runmat_builtins::BuiltinCatalogIdentity) -> bool {
    runmat_builtins::builtin_catalog_entry_by_name(name)
        .is_some_and(|entry| entry.identity == identity)
}
pub(super) async fn select(
    source: &evaluation::Plan,
    root: GcHandle,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Step> {
    c.completed(0).await?;
    let current = node(source, root, policy)?;
    Ok(match &current.kind {
        Kind::Constant(_) => Step::Constant,
        Kind::Labels { source }
        | Kind::Unary {
            operator: Op::UnaryPlus,
            operand: source,
        } => Step::Forward(*source),
        Kind::Unary {
            operator: Op::UnaryMinus,
            operand,
        } => Step::Scale {
            next: *operand,
            factor: -1.,
            operation: Scale::Multiply,
        },
        Kind::Binary {
            operator: Op::Add,
            left,
            right,
        } if evaluation::element_count(&node(source, *left, policy)?.shape, policy)? == 1
            && evaluation::element_count(&node(source, *right, policy)?.shape, policy)? == 1 =>
        {
            Step::Add(*left, *right)
        }
        Kind::Binary {
            operator: Op::Subtract,
            left,
            right,
        } if evaluation::element_count(&node(source, *left, policy)?.shape, policy)? == 1
            && evaluation::element_count(&node(source, *right, policy)?.shape, policy)? == 1 =>
        {
            Step::Subtract(*left, *right)
        }
        Kind::Binary {
            operator: Op::MatrixMultiply | Op::ElementwiseMultiply,
            left,
            right,
        } => {
            match (
                scalar(source, *left, policy)?,
                scalar(source, *right, policy)?,
            ) {
                (Some(factor), _) if factor.is_finite() => Step::Scale {
                    next: *right,
                    factor,
                    operation: Scale::Multiply,
                },
                (_, Some(factor)) if factor.is_finite() => Step::Scale {
                    next: *left,
                    factor,
                    operation: Scale::Multiply,
                },
                _ => Step::Other,
            }
        }
        Kind::Binary {
            operator: Op::Mrdivide | Op::ElementwiseDivide,
            left,
            right,
        } => match scalar(source, *right, policy)? {
            Some(factor) if factor != 0. && factor.is_finite() => Step::Scale {
                next: *left,
                factor,
                operation: Scale::Divide,
            },
            _ => Step::Other,
        },
        Kind::Binary {
            operator: Op::Mldivide | Op::ElementwiseLeftDivide,
            left,
            right,
        } => match scalar(source, *left, policy)? {
            Some(factor) if factor != 0. && factor.is_finite() => Step::Scale {
                next: *right,
                factor,
                operation: Scale::Divide,
            },
            _ => Step::Other,
        },
        Kind::Binary {
            operator,
            left,
            right,
        } if matches!(operator, Op::MatrixPower | Op::ElementwisePower)
            && scalar(source, *right, policy)? == Some(2.) =>
        {
            let input = node(source, *left, policy)?;
            if let Some(residual) = super::super::norm_square::operand(source, *left, policy)? {
                return Ok(Step::Square { residual });
            }
            if matches!(&input.kind, Kind::Builtin { function, .. }
                if is(function.0.as_str(), runmat_builtins::NORM_CATALOG_ENTRY.identity))
            {
                return Ok(Step::Other);
            }
            if *operator == Op::ElementwisePower
                || evaluation::element_count(&input.shape, policy)? == 1
            {
                Step::Square { residual: *left }
            } else {
                Step::Other
            }
        }
        Kind::Builtin {
            function, operands, ..
        } if is(
            function.0.as_str(),
            runmat_builtins::SUM_CATALOG_ENTRY.identity,
        ) && operands.len() == 1 =>
        {
            Step::Sum { next: operands[0] }
        }
        _ => Step::Other,
    })
}
