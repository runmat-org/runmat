//! Publication into one immutable product's portable identity domain.
//! Whole-assembly flow may include inherited bodies. They contribute to
//! inference, but their original revision's facts remain in that product,
//! rather than aliasing a current function with the same portable ordinal.
use super::AnalysisStore;
use runmat_types::{FunctionId, ProgramFunctionId};
use std::collections::{BTreeMap, BTreeSet, HashMap};

mod facts;
mod parallel;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
enum References {
    EvalLocal,
    ClosedUnit,
}

pub(super) struct Projection<'a> {
    functions: BTreeMap<ProgramFunctionId, ProgramFunctionId>,
    locals: Option<(&'a HashMap<FunctionId, FunctionId>, References)>,
}

impl Projection<'_> {
    fn contains(&self, function: ProgramFunctionId) -> bool {
        self.functions.contains_key(&function)
    }
    fn function(&self, function: &mut ProgramFunctionId) -> Result<(), String> {
        *function = *self.functions.get(function).ok_or_else(|| {
            format!(
                "analysis owner {} is outside the selected immutable product",
                function.0
            )
        })?;
        Ok(())
    }
    fn point(&self, point: &mut runmat_types::ProgramPointId) -> Result<(), String> {
        self.function(&mut point.function)
    }
    fn value(&self, value: &mut runmat_types::RegionValueId) -> Result<(), String> {
        self.function(&mut value.function)
    }
    fn local(&self, function: &mut FunctionId) {
        if let Some((locals, _)) = self.locals {
            if let Some(destination) = locals.get(function) {
                *function = *destination;
            }
        }
    }
}

impl AnalysisStore {
    /// Apply the same local reference domain as MIR's eval-local rename.
    /// Original portable function/point/region identities do not change.
    pub fn rebind_eval_local_functions(
        &self,
        ids: &HashMap<FunctionId, FunctionId>,
    ) -> Result<Self, String> {
        self.rebind_locals(ids, References::EvalLocal)
    }

    /// A proved closed product also owns explicitly external-encoded local
    /// references within its renamed group, matching the MIR closed rename.
    pub fn rebind_closed_functions(
        &self,
        ids: &HashMap<FunctionId, FunctionId>,
    ) -> Result<Self, String> {
        self.rebind_locals(ids, References::ClosedUnit)
    }

    fn rebind_locals(
        &self,
        ids: &HashMap<FunctionId, FunctionId>,
        references: References,
    ) -> Result<Self, String> {
        if ids.values().copied().collect::<BTreeSet<_>>().len() != ids.len() {
            return Err("analysis local function rename is not injective".into());
        }
        let functions: BTreeMap<_, _> = self
            .functions
            .iter()
            .map(|facts| (facts.function, facts.function))
            .collect();
        if functions.len() != self.functions.len() {
            return Err("analysis portable source owners collide".into());
        }
        self.project(&Projection {
            functions,
            locals: Some((ids, references)),
        })
    }

    /// Publish only the proved current product owners. The input map is
    /// compiler-local -> original portable identity, supplied by the product
    /// owner after checking the body's immutable revision. Local callable
    /// bindings stay typed FunctionId; dependency invalidations are retained.
    pub fn project_product_functions(
        &self,
        owners: &BTreeMap<FunctionId, ProgramFunctionId>,
    ) -> Result<Self, String> {
        let mut identities = BTreeMap::new();
        let mut destinations = BTreeSet::new();
        let mut locals = BTreeSet::new();
        for (local, portable) in owners {
            let function = self.function_for_local(*local).ok_or_else(|| {
                format!("product function {local:?} has no unique canonical analysis self binding")
            })?;
            if identities.insert(function.function, *portable).is_some()
                || !destinations.insert(*portable)
                || !locals.insert(*local)
            {
                return Err(
                    "current product analysis function identities are not injective".into(),
                );
            }
        }
        self.project(&Projection {
            functions: identities,
            locals: None,
        })
    }

    fn project(&self, projection: &Projection<'_>) -> Result<Self, String> {
        let mut result = self.clone();
        result
            .functions
            .retain(|function| projection.contains(function.function));
        for function in &mut result.functions {
            projection.function(&mut function.function)?;
            projection.callable(&mut function.callable)?;
            for output in &mut function.outputs {
                projection.fact(output)?;
            }
        }
        result
            .program_points
            .retain(|facts| projection.contains(facts.point.function));
        for facts in &mut result.program_points {
            projection.point(&mut facts.point)?;
            for local in &mut facts.locals {
                projection.value(&mut local.value)?;
                if let Some(fact) = &mut local.fact {
                    projection.fact(fact)?;
                }
            }
            facts.locals.sort_by_key(|local| local.value);
        }
        result
            .operation_contracts
            .retain(|contract| projection.contains(contract.site.point.function));
        for contract in &mut result.operation_contracts {
            projection.point(&mut contract.site.point)?;
        }
        result
            .fusion_stage_contracts
            .retain(|stage| projection.contains(stage.point.function));
        for stage in &mut result.fusion_stage_contracts {
            projection.point(&mut stage.point)?;
            for value in stage
                .uses
                .iter_mut()
                .chain(&mut stage.defs)
                .chain(&mut stage.live_after)
            {
                projection.value(value)?;
            }
            stage.uses.sort_unstable();
            stage.defs.sort_unstable();
            stage.live_after.sort_unstable();
        }
        result
            .regions
            .retain(|region| projection.contains(region.contract.id.function));
        for region in &mut result.regions {
            projection.region(region)?;
        }
        for class in &mut result.classes {
            for method in &mut class.declaration.methods {
                projection.local(&mut method.function);
            }
            class
                .methods
                .retain(|function| projection.contains(*function));
            for method in &mut class.methods {
                projection.function(method)?;
            }
            class.methods.sort_unstable();
            for property in &mut class.properties {
                projection.fact(&mut property.fact)?;
            }
        }
        projection.parallel(&mut result.parallel)?;
        result.functions.sort_by_key(|function| function.function);
        result.program_points.sort_by_key(|facts| facts.point);
        result
            .operation_contracts
            .sort_by(|left, right| left.site.cmp(&right.site));
        result
            .fusion_stage_contracts
            .sort_by_key(|stage| stage.point);
        result.regions.sort_by_key(|region| region.contract.id);
        let mut bindings = BTreeSet::new();
        for function in &result.functions {
            let Some(runmat_types::CallableIdentity::BoundFunction(local)) =
                &function.callable.identity
            else {
                return Err("analysis function summary has no compiler-local self binding".into());
            };
            if !bindings.insert(*local) {
                return Err("analysis local function bindings collide".into());
            }
        }
        Ok(result)
    }
}
