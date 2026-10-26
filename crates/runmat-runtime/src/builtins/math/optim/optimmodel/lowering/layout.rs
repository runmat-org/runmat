//! Original definitions/domains survive coefficient cancellation. One captured
//! immutable definition generation supplies shape and bounds for this solve.
use super::*;
use runmat_gc::GcHandle;
use runmat_value::optimization_variable::OptimizationVariableDomain;
use std::collections::HashMap;
mod bounds;
mod initial_point;
pub(super) use bounds::{continuous_bound_values, continuous_bounds};
pub(super) use initial_point::initial_point;

pub(super) struct Variable {
    pub name: Utf8Text,
    pub shape: SharedShape,
    pub start: usize,
    pub count: usize,
    pub domain: OptimizationVariableDomain,
    pub view: super::super::super::optimvar::View,
}
pub(super) struct Layout {
    pub variables: Vec<Variable>,
    pub order: Vec<usize>,
    pub columns: usize,
    by_definition: HashMap<GcHandle, usize>,
}
impl Layout {
    pub(super) async fn prepare(
        plan: &evaluation::Plan,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        evaluation::validate_variable_names(plan, policy, cooperation).await?;
        let mut variables = Vec::new();
        let mut by_definition = HashMap::new();
        for node in plan.nodes.values() {
            cooperation.completed(1).await?;
            let Kind::Variable(Value::Object(object)) = &node.node().kind else {
                continue;
            };
            let view = super::super::super::optimvar::View::new(object, cooperation).await?;
            let definition = view.selection().definition();
            if by_definition.contains_key(&definition) {
                continue;
            }
            let count = view.definition().element_count();
            let name = view.definition().name_owner().clone();
            let shape = view.definition().shape().clone();
            let domain = view.definition().domain();
            variables.try_reserve(1).map_err(|e| policy.reserve(e))?;
            by_definition
                .try_reserve(1)
                .map_err(|e| policy.reserve(e))?;
            by_definition.insert(definition, variables.len());
            variables.push(Variable {
                name,
                shape,
                start: 0,
                count,
                domain,
                view,
            });
        }
        let mut order =
            runmat_value::exact_capacity_vec(variables.len()).map_err(|e| policy.storage(e))?;
        for index in 0..variables.len() {
            order.push(index);
            cooperation.completed(1).await?;
        }
        use crate::builtins::common::cooperative_sort::{SortProgress, StableSortWork};
        let mut sort = StableSortWork::new(&mut order, |&left, &right| {
            variables[left].name.cmp(&variables[right].name)
        });
        loop {
            let progress = sort.advance().map_err(|source| policy.reserve(source.0))?;
            match progress {
                SortProgress::Pending(work) => cooperation.completed(work).await?,
                SortProgress::Complete(work) => {
                    cooperation.completed(work).await?;
                    break;
                }
            }
        }
        drop(sort);
        let mut columns = 0usize;
        for &index in &order {
            let variable = &mut variables[index];
            variable.start = columns;
            columns = columns
                .checked_add(variable.count)
                .ok_or_else(|| policy.invalid("decision-coordinate count overflows usize"))?;
            cooperation.completed(1).await?;
        }
        Ok(Self {
            variables,
            order,
            columns,
            by_definition,
        })
    }
    pub(super) fn coordinate(&self, column: usize) -> Option<(&Variable, usize)> {
        if column >= self.columns {
            return None;
        }
        let index = self.order.partition_point(|&index| {
            let variable = &self.variables[index];
            variable.start + variable.count <= column
        });
        let variable = &self.variables[*self.order.get(index)?];
        Some((variable, column - variable.start))
    }
    pub(super) fn variable(
        &self,
        definition: GcHandle,
        policy: Policy,
    ) -> BuiltinResult<&Variable> {
        self.by_definition
            .get(&definition)
            .and_then(|&index| self.variables.get(index))
            .ok_or_else(|| {
                policy.invalid("expression variable is outside the original compiled layout")
            })
    }
}
