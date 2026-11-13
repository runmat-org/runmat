//! Exact original exponent evidence precedes integer/identity power admission.
use super::array::{Array, Degree, Offset};
use super::*;
#[derive(Clone)]
pub(super) struct Exponents {
    shape: SharedShape,
    values: Offset,
    strides: std::rc::Rc<Vec<usize>>,
}
impl Exponents {
    pub(super) fn count(&self, policy: Policy) -> BuiltinResult<usize> {
        evaluation::element_count(&self.shape, policy)
    }
    pub(super) fn at(&self, row: usize, policy: Policy) -> BuiltinResult<f64> {
        self.values.at(row, policy)
    }
    pub(super) fn broadcast(
        &self,
        row: usize,
        shape: &[usize],
        policy: Policy,
    ) -> BuiltinResult<f64> {
        use crate::builtins::common::broadcast::broadcast_index;
        self.at(
            broadcast_index(row, shape, &self.shape, &self.strides),
            policy,
        )
    }
}
pub(super) async fn evidence(
    root: runmat_gc::GcHandle,
    candidate: &Array,
    source: &evaluation::Plan,
    layout: &layout::Layout,
    facts: &std::collections::HashMap<runmat_gc::GcHandle, NodeFacts>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Exponents> {
    if candidate.degree != Degree::Constant {
        return Ok(Err(Stop::NonAffine));
    }
    if !candidate.coefficients.is_zero() {
        return Err(policy.invalid("constant exponent proposal contains decision coefficients"));
    }
    let mut selected = std::collections::HashSet::new();
    let mut pending = Vec::new();
    pending.try_reserve(1).map_err(|e| policy.reserve(e))?;
    pending.push(root);
    while let Some(handle) = pending.pop() {
        selected.try_reserve(1).map_err(|e| policy.reserve(e))?;
        if !selected.insert(handle) {
            c.completed(1).await?;
            continue;
        }
        let node = source
            .nodes
            .get(&handle)
            .ok_or_else(|| policy.invalid("power exponent lost original node"))?;
        let mut edge = 0;
        while let Some(child) = node.dependency_at(edge) {
            pending.try_reserve(1).map_err(|e| policy.reserve(e))?;
            pending.push(child);
            edge += 1;
            c.completed(1).await?;
        }
        c.completed(1).await?;
    }
    let original = exact::evaluate_selected(
        exact::Source {
            source,
            layout,
            facts,
        },
        Some(exact::Subset {
            nodes: &selected,
            roots: std::slice::from_ref(&root),
        }),
        exact::Coordinates::Origin,
        exact::Retain::Roots,
        policy,
        c,
    )
    .await?;
    for row in 0..candidate.count(policy)? {
        if !candidate.assigned.at(row, policy)? {
            return Ok(Err(Stop::Unassigned));
        }
        let value = candidate.offset.at(row, policy)?;
        if !value.is_finite() || value.fract() != 0. {
            return Ok(Err(Stop::UnsupportedFunction));
        }
        let expected = super::super::super::common::proof_rational::Rational::from_float(value)
            .ok_or_else(|| policy.invalid("integer exponent must be finite"))?;
        let Some(actual) = original.value(root, row, policy)? else {
            return Ok(Err(Stop::UnsupportedFunction));
        };
        if !actual.compare(&expected, policy, c).await?.is_eq() {
            return Ok(Err(Stop::UnsupportedFunction));
        }
        c.completed(1).await?;
    }
    Ok(Ok(Exponents {
        shape: candidate.shape.clone(),
        values: candidate.offset.clone(),
        strides: std::rc::Rc::new(crate::builtins::common::broadcast::compute_strides(
            &candidate.shape,
        )),
    }))
}
mod proposal;
pub(super) use proposal::identity;
pub(super) use proposal::lower as proposal;
