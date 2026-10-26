//! Atomic original scalar/vector Euclidean norm-square pairs. This is handle
//! metadata over the original Plan, not a rewritten source or an irrational value.
use super::*;
use runmat_gc::GcHandle;
use runmat_types::OperatorKind as Op;
use std::collections::{HashMap, HashSet};

fn euclidean(value: &Value) -> bool {
    match value {
        Value::Num(value) => *value == 2.,
        Value::Int(value) => value.to_exact_f64() == Some(2.),
        Value::Tensor(value) if value.len() == 1 => match value.numeric_value_at(0) {
            Some(runmat_value::NumericScalar::F64(value)) => value == 2.,
            Some(runmat_value::NumericScalar::F32(value)) => value == 2.,
            Some(value) => value
                .into_int_value()
                .is_some_and(|value| value.to_exact_f64() == Some(2.)),
            _ => false,
        },
        _ => false,
    }
}
pub(super) fn operand(
    source: &evaluation::Plan,
    norm: GcHandle,
    policy: Policy,
) -> BuiltinResult<Option<GcHandle>> {
    let node = source
        .nodes
        .get(&norm)
        .ok_or_else(|| policy.invalid("norm-square lost original norm node"))?
        .node();
    let Kind::Builtin {
        function,
        operands,
        arguments,
    } = &node.kind
    else {
        return Ok(None);
    };
    if runmat_builtins::builtin_catalog_entry_by_name(function.0.as_str())
        .is_none_or(|entry| entry.identity != runmat_builtins::NORM_CATALOG_ENTRY.identity)
        || operands.len() != 1
        || !(arguments.is_empty() || arguments.len() == 1 && euclidean(&arguments[0]))
        || evaluation::element_count(&node.shape, policy)? != 1
    {
        return Ok(None);
    }
    let input = source
        .nodes
        .get(&operands[0])
        .ok_or_else(|| policy.invalid("norm-square lost original residual"))?
        .node();
    use crate::builtins::math::linalg::solve::norm::{classify_tensor, TensorKind};
    Ok(matches!(classify_tensor(&input.shape), Ok(TensorKind::Vector)).then_some(operands[0]))
}

#[derive(Default)]
pub(super) struct Plan {
    squares: HashMap<GcHandle, (GcHandle, GcHandle)>,
    norms: HashSet<GcHandle>,
}
impl Plan {
    pub async fn prepare(
        source: &evaluation::Plan,
        selected: &selection::Selection,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        let mut plan = Self::default();
        for &root in &source.order {
            if !selected.nodes.contains(&root) {
                c.completed(1).await?;
                continue;
            }
            let node = source.nodes[&root].node();
            if let Kind::Binary {
                operator: Op::MatrixPower | Op::ElementwisePower,
                left,
                right,
            } = &node.kind
            {
                if objective::step::scalar(source, *right, policy)? == Some(2.) {
                    if let Some(residual) = operand(source, *left, policy)? {
                        plan.squares.try_reserve(1).map_err(|e| policy.reserve(e))?;
                        plan.norms.try_reserve(1).map_err(|e| policy.reserve(e))?;
                        plan.squares.insert(root, (*left, residual));
                        plan.norms.insert(*left);
                    }
                }
            }
            c.completed(1).await?;
        }
        if plan.squares.is_empty() {
            return Ok(Ok(plan));
        }
        // A norm value cannot be substituted by its square or by a rational
        // approximation. Every selected use must be this exact admitted pair.
        for &root in &selected.roots {
            if plan.norms.contains(&root) {
                return Ok(Err(Stop::UnsupportedFunction));
            }
            c.completed(1).await?;
        }
        for &root in &selected.nodes {
            let node = &source.nodes[&root];
            let mut edge = 0;
            while let Some(child) = node.dependency_at(edge) {
                if plan.norms.contains(&child)
                    && !plan
                        .squares
                        .get(&root)
                        .is_some_and(|&(norm, _)| norm == child)
                {
                    return Ok(Err(Stop::UnsupportedFunction));
                }
                edge += 1;
                c.completed(1).await?;
            }
        }
        Ok(Ok(plan))
    }
    pub fn residual(&self, square: GcHandle) -> Option<GcHandle> {
        self.squares.get(&square).map(|&(_, residual)| residual)
    }
    pub fn is_norm(&self, node: GcHandle) -> bool {
        self.norms.contains(&node)
    }
    pub fn redirected(&self, parent: GcHandle, child: GcHandle) -> GcHandle {
        self.squares
            .get(&parent)
            .filter(|&&(norm, _)| norm == child)
            .map_or(child, |&(_, residual)| residual)
    }
    pub async fn redirect_uses(
        &self,
        source: &evaluation::Plan,
        remaining: &mut HashMap<GcHandle, usize>,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        for &norm in &self.norms {
            let Some(residual) = operand(source, norm, policy)? else {
                return Err(policy.invalid("norm-square original geometry changed"));
            };
            let original = remaining
                .get_mut(&residual)
                .ok_or_else(|| policy.invalid("norm-square lost original input edge"))?;
            *original = original
                .checked_sub(1)
                .ok_or_else(|| policy.invalid("norm-square input-use underflow"))?;
            remaining.remove(&norm);
            c.completed(1).await?;
        }
        for &(_, residual) in self.squares.values() {
            let uses = remaining
                .get_mut(&residual)
                .ok_or_else(|| policy.invalid("norm-square input-use slot is absent"))?;
            *uses = uses
                .checked_add(1)
                .ok_or_else(|| policy.invalid("norm-square input-use overflow"))?;
            c.completed(1).await?;
        }
        Ok(())
    }
}
