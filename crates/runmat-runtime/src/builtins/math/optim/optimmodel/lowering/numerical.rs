//! Original floating-DAG point evaluation and eventual solution storage share
//! one rooted point owner. Shape/name metadata is original, not a compiler row.
use super::*;
use runmat_gc::GcHandle;
use runmat_value::{NumericScalar, NumericStorage, Tensor};
use std::collections::HashMap;
pub(super) struct Point {
    pub solution: RootedValue,
    values: HashMap<GcHandle, RootedValue>,
}
impl Point {
    pub(super) async fn evaluate(
        program: &Program,
        roots: &[GcHandle],
        candidate: &[f64],
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        Self::evaluate_layout(
            &program.source,
            &program.layout,
            None,
            roots,
            candidate,
            policy,
            cooperation,
        )
        .await
    }
    pub(super) async fn evaluate_layout(
        source: &evaluation::Plan,
        layout: &layout::Layout,
        selected: Option<&selection::Selection>,
        roots: &[GcHandle],
        candidate: &[f64],
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let Some(solution) = solution(layout, candidate, policy, cooperation).await? else {
            return Ok(None);
        };
        let mut evaluated = match evaluation::evaluate_trial(
            source,
            selected,
            roots,
            solution.value(),
            policy,
            cooperation,
        )
        .await?
        {
            Ok(value) => value,
            Err(evaluation::NumericUndefined::Nonreal) => return Ok(None),
        };
        let mut values = HashMap::new();
        values
            .try_reserve(roots.len())
            .map_err(|e| policy.reserve(e))?;
        for &root in roots {
            if let std::collections::hash_map::Entry::Vacant(entry) = values.entry(root) {
                entry.insert(evaluated.take(root, policy, cooperation).await?);
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(Self { solution, values }))
    }
    pub(super) fn at(&self, root: GcHandle, row: usize, policy: Policy) -> BuiltinResult<f64> {
        let value = self
            .values
            .get(&root)
            .ok_or_else(|| policy.invalid("original numerical root is unavailable"))?
            .value();
        number(value, row, policy)
    }
    pub(super) fn matches<'a>(
        &'a self,
        program: &'a Program,
        candidate: &'a [f64],
        policy: Policy,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<bool>> + 'a {
        self.matches_layout(&program.layout, candidate, policy, cooperation)
    }
    pub(super) fn matches_layout<'a>(
        &'a self,
        layout: &'a layout::Layout,
        candidate: &'a [f64],
        policy: Policy,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<bool>> + 'a {
        coordinates_match(layout, &self.solution, candidate, policy, cooperation)
    }

    pub(super) fn root(&self, root: GcHandle, policy: Policy) -> BuiltinResult<&RootedValue> {
        self.values
            .get(&root)
            .ok_or_else(|| policy.invalid("original numerical root is unavailable"))
    }
}

/// Decode the same canonical original numerical carrier for source roots and x0.
pub(super) fn number(value: &Value, row: usize, policy: Policy) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) if row == 0 => Ok(*value),
        Value::Tensor(array) => match array.numeric_value_at(row) {
            Some(NumericScalar::F64(value)) => Ok(value),
            _ => Err(policy.invalid("original numerical root lost its double storage")),
        },
        Value::SparseTensor(array) => {
            let count = array
                .rows
                .checked_mul(array.cols)
                .ok_or_else(|| policy.invalid("original sparse root shape overflow"))?;
            if row >= count {
                return Err(policy.invalid("original sparse root coordinate exceeds its shape"));
            }
            array
                .get(row % array.rows, row / array.rows)
                .ok_or_else(|| policy.invalid("original sparse root lost its double class"))
        }
        _ => Err(policy.invalid("original numerical root is not real double")),
    }
}

pub(super) async fn solution(
    layout: &layout::Layout,
    candidate: &[f64],
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<RootedValue>> {
    if candidate.len() != layout.columns {
        return Ok(None);
    }
    for &value in candidate {
        if !value.is_finite() {
            return Ok(None);
        }
        cooperation.completed(1).await?;
    }
    let mut variables =
        runmat_value::exact_capacity_vec(layout.variables.len()).map_err(|e| policy.storage(e))?;
    for &index in &layout.order {
        let variable = &layout.variables[index];
        let mut values =
            runmat_value::exact_capacity_vec(variable.count).map_err(|e| policy.storage(e))?;
        for &value in &candidate[variable.start..variable.start + variable.count] {
            values.push(value);
            cooperation.completed(1).await?;
        }
        let value = Value::Tensor(
            Tensor::from_numeric_storage(NumericStorage::F64(values), variable.shape.clone())
                .map_err(|e| policy.invalid(e))?,
        );
        variables.push((
            variable.name.as_str(),
            RootedValue::in_optional_context(value, cooperation.runtime())?,
        ));
    }
    let solution = crate::execution::structure::construct_struct_from_fields_in_optional_context(
        cooperation.runtime(),
        variables.iter().map(|(name, value)| (*name, value.value())),
    )?;

    Ok(Some(solution))
}

/// Exact represented-coordinate authentication without claiming feasibility.
pub(super) async fn coordinates_match(
    layout: &layout::Layout,
    solution: &RootedValue,
    candidate: &[f64],
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if candidate.len() != layout.columns {
        return Ok(false);
    }
    let Value::Struct(fields) = solution.value() else {
        return Err(policy.invalid("validated solution lost its original structure"));
    };
    for variable in &layout.variables {
        let Some(Value::Tensor(array)) = fields.fields.get(variable.name.as_str()) else {
            return Err(policy.invalid("validated solution lost an original variable"));
        };
        for row in 0..variable.count {
            let Some(NumericScalar::F64(value)) = array.numeric_value_at(row) else {
                return Err(policy.invalid("validated solution lost its numerical backing"));
            };
            if value.to_bits() != candidate[variable.start + row].to_bits() {
                return Ok(false);
            }
            cooperation.completed(1).await?;
        }
    }
    Ok(true)
}
