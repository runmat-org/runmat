//! Original scalar objective and comparison roles. Presentation names/shapes
//! are kept independently of sparse row assembly and coefficient cancellation.
use super::*;
use runmat_gc::GcHandle;
use runmat_value::optimization_model::OptimizationConstraintRelation as Relation;
use runmat_value::optimization_model::OptimizationObjectiveSense as Sense;

use runmat_value::optimization_model::{
    OptimizationObjectiveSenses as Senses, OptimizationProblemData,
};
pub(super) struct Objective {
    pub root: GcHandle,
    pub sense: Sense,
}
pub(super) struct Constraint {
    pub name: Option<Utf8Text>,
    pub left: GcHandle,
    pub right: GcHandle,
    pub relation: Relation,
    pub shape: SharedShape,
}
pub(super) struct Formulation {
    pub objective: Option<Objective>,
    pub constraints: Vec<Constraint>,
    pub roots: Vec<GcHandle>,
}
impl Formulation {
    pub(super) async fn prepare(
        model: &OptimizationProblemData,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut out = Self {
            objective: None,
            constraints: Vec::new(),
            roots: Vec::new(),
        };
        match &model.objective {
            Value::Struct(fields) => {
                if fields.fields.len() > 1 {
                    return Err(policy.invalid(
                        "multiple objectives are not supported by these scalar model solvers",
                    ));
                }
                if let Some((name, value)) = fields.fields.iter().next() {
                    let sense = match &model.sense {
                        Senses::Uniform(sense) => *sense,
                        Senses::Named(senses) => senses
                            .iter()
                            .find(|(key, _)| key.as_str() == name)
                            .map(|(_, sense)| *sense)
                            .ok_or_else(|| {
                                policy.invalid("named objective has no original sense")
                            })?,
                    };
                    out.objective(value, sense, policy)?;
                }
            }
            value => {
                let Senses::Uniform(sense) = model.sense else {
                    return Err(
                        policy.invalid("unnamed objective requires a uniform objective sense")
                    );
                };
                out.objective(value, sense, policy)?;
            }
        }
        match &model.constraints {
            Value::Struct(fields) => {
                out.constraints
                    .try_reserve(fields.fields.len())
                    .map_err(|e| policy.reserve(e))?;
                for (name, value) in &fields.fields {
                    out.constraint(value, Some(Utf8Text::new(name.to_owned())), policy)?;
                    cooperation.completed(1).await?;
                }
            }
            Value::StructArray(array) if array.is_empty() && array.field_names().len() == 0 => {}
            value => out.constraint(value, None, policy)?,
        }
        Ok(out)
    }
    fn objective(&mut self, value: &Value, sense: Sense, policy: Policy) -> BuiltinResult<()> {
        let root = owner::reference(value, policy)?;
        let node = owner::OwnedNode::read(root, policy)?;
        let count = evaluation::element_count(&node.node().shape, policy)?;
        self.roots.try_reserve(1).map_err(|e| policy.reserve(e))?;
        self.roots.push(root);
        if count == 0 && matches!(node.node().kind, Kind::EmptyInitializer) {
            return Ok(());
        }
        if count != 1 {
            return Err(policy.invalid("scalar model solve requires a scalar objective"));
        }
        self.objective = Some(Objective { root, sense });
        Ok(())
    }
    fn constraint(
        &mut self,
        value: &Value,
        name: Option<Utf8Text>,
        policy: Policy,
    ) -> BuiltinResult<()> {
        let reference = constraints::reference(value, policy)?;
        self.roots.try_reserve(2).map_err(|e| policy.reserve(e))?;
        self.constraints
            .try_reserve(1)
            .map_err(|e| policy.reserve(e))?;
        self.roots.extend([reference.left, reference.right]);
        self.constraints.push(Constraint {
            name,
            left: reference.left,
            right: reference.right,
            relation: reference.relation,
            shape: reference.shape.clone(),
        });
        Ok(())
    }
}
