//! Cheap typed route selection avoids a quadratic coefficient pass for ordinary
//! affine objectives. This is not convexity or polynomial-degree evidence.
use super::super::operations::Operation;
use super::*;
use runmat_types::OperatorKind as Op;
pub(super) async fn candidate(
    source: &evaluation::Plan,
    root: GcHandle,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<selection::Selection>> {
    let mut roots = runmat_value::exact_capacity_vec(1).map_err(|e| policy.storage(e))?;
    roots.push(root);
    let selected = selection::Selection::prepare(source, roots, policy, c).await?;
    let mut dependent = std::collections::HashMap::new();
    dependent
        .try_reserve(selected.nodes.len())
        .map_err(|e| policy.reserve(e))?;
    let mut required = false;
    for &handle in &source.order {
        if !selected.nodes.contains(&handle) {
            c.completed(1).await?;
            continue;
        }
        let source_node = &source.nodes[&handle];
        let node = source_node.node();
        let mut any = false;
        let mut edge = 0;
        while let Some(child) = source_node.dependency_at(edge) {
            any |= dependent.get(&child).copied().ok_or_else(|| {
                policy.invalid("quadratic route has a non-topological source edge")
            })?;
            edge += 1;
            c.completed(1).await?;
        }
        let active = matches!(node.kind, Kind::Variable(_)) || any;
        match &node.kind {
            Kind::Binary {
                operator,
                left,
                right,
            } => {
                let a = dependent[left];
                let b = dependent[right];
                required |= match operator {
                    Op::MatrixMultiply | Op::ElementwiseMultiply => a && b,
                    Op::MatrixPower | Op::ElementwisePower => {
                        a || b
                            || (objective::step::scalar(source, *right, policy)? == Some(2.)
                                && super::super::norm_square::operand(source, *left, policy)?
                                    .is_some())
                    }
                    Op::Mrdivide | Op::ElementwiseDivide => b,
                    Op::Mldivide | Op::ElementwiseLeftDivide => a,
                    _ => false,
                };
            }
            Kind::Builtin {
                function, operands, ..
            } => {
                required |= match Operation::selected(function.0.as_str()) {
                    Some(Operation::Product | Operation::Cumprod) => any,
                    Some(Operation::Dot) => {
                        operands.iter().filter(|root| dependent[*root]).count() > 1
                    }
                    None => any,
                    _ => false,
                };
            }
            _ => {}
        }
        dependent.insert(handle, active);
        c.completed(1).await?;
    }
    Ok(required.then_some(selected))
}
