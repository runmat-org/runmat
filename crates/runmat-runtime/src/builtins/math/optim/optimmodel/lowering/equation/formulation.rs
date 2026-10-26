//! Equation roots remain residual data, never optimization hard constraints.
use super::*;
use runmat_value::optimization_model::EquationProblemData;
use runmat_value::optimization_model::OptimizationConstraintRelation;

pub(super) struct Entry {
    pub name: Option<Utf8Text>,
    pub left: GcHandle,
    pub right: GcHandle,
    pub shape: SharedShape,
}
pub(super) struct Formulation {
    pub entries: Vec<Entry>,
    pub roots: Vec<GcHandle>,
}
impl Formulation {
    pub(super) async fn prepare(
        problem: &EquationProblemData,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut out = Self {
            entries: Vec::new(),
            roots: Vec::new(),
        };
        match &problem.equations {
            Value::Struct(fields) => {
                for (name, value) in &fields.fields {
                    c.completed(0).await?;
                    let name = crate::call::argument_text::copy_str(name, c.runtime())?;
                    out.push(value, Some(name), policy)?;
                    c.completed(1).await?;
                }
            }
            value => {
                if !constraints::collection::empty(value, policy)? {
                    out.push(value, None, policy)?;
                }
            }
        }
        if out.entries.is_empty() {
            return Err(policy.invalid("solve requires at least one equation"));
        }
        Ok(out)
    }
    fn push(&mut self, value: &Value, name: Option<Utf8Text>, policy: Policy) -> BuiltinResult<()> {
        let original = constraints::reference(value, policy)?;
        if original.relation != OptimizationConstraintRelation::Equal {
            return Err(policy.invalid("an equation problem requires equality expressions"));
        }
        self.entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
        self.roots.try_reserve(2).map_err(|e| policy.reserve(e))?;
        self.roots.extend([original.left, original.right]);
        self.entries.push(Entry {
            name,
            left: original.left,
            right: original.right,
            shape: original.shape.clone(),
        });
        Ok(())
    }
}

pub(in crate::builtins::math::optim::optimmodel::lowering) struct Evidence {
    pub(super) formulation: Formulation,
    pub geometries: Vec<constraint_geometry::Geometry>,
    pub rows: usize,
}
impl Evidence {
    pub(super) async fn prepare(
        formulation: Formulation,
        source: &evaluation::Plan,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut geometries = runmat_value::exact_capacity_vec(formulation.entries.len())
            .map_err(|e| policy.storage(e))?;
        let mut rows = 0usize;
        for entry in &formulation.entries {
            let geometry = constraint_geometry::Geometry::from_roots(
                source,
                entry.left,
                entry.right,
                &entry.shape,
                policy,
            )?;
            rows = rows
                .checked_add(geometry.plan.len())
                .ok_or_else(|| policy.invalid("equation residual count overflows usize"))?;
            geometries.push(geometry);
            c.completed(1).await?;
        }
        if rows == 0 {
            return Err(policy.invalid("solve requires a nonempty equation residual"));
        }
        Ok(Self {
            formulation,
            geometries,
            rows,
        })
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn residual(
        &self,
        point: &numerical::Point,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<f64>> {
        let mut values =
            runmat_value::exact_capacity_vec(self.rows).map_err(|e| policy.storage(e))?;
        for (entry, geometry) in self.formulation.entries.iter().zip(&self.geometries) {
            for row in 0..geometry.plan.len() {
                let (left, right) = geometry
                    .indices(row)
                    .ok_or_else(|| policy.invalid("equation residual geometry changed"))?;
                let value =
                    point.at(entry.left, left, policy)? - point.at(entry.right, right, policy)?;
                if !value.is_finite() {
                    return Err(policy.invalid("equation residual must be finite and real"));
                }
                values.push(value);
                c.completed(1).await?;
            }
        }
        Ok(values)
    }
}
