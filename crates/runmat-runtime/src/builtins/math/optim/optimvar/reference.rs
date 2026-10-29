//! A public view borrows the actual original definition, never a name lookup.
use super::*;
use runmat_value::optimization_variable::{
    OptimizationVariableData, OptimizationVariablePayload, OptimizationVariableSelection,
};
pub(in crate::builtins::math::optim) struct View {
    reference: ObjectInstance,
    generation: ObjectInstance,
}
impl View {
    pub async fn new(
        object: &ObjectInstance,
        cooperation: &mut super::super::common::ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if !is_reference(object) {
            return Err(invalid("expected an optimization-variable reference"));
        }
        let Some(OptimizationVariablePayload::Reference(selection)) =
            object.optimization_variable_payload()
        else {
            unreachable!()
        };
        let generation = runmat_gc::gc_with_value(&selection.definition(), |value| match value {
            Value::Object(generation)
                if generation.class_name().is(CLASS)
                    && generation.selected_class().is_none()
                    && matches!(
                        generation.optimization_variable_payload(),
                        Some(OptimizationVariablePayload::Definition(_))
                    ) =>
            {
                Some(generation.clone())
            }
            _ => None,
        })
        .map_err(|source| gc_error("failed to access variable definition", source))?
        .ok_or_else(|| invalid("optimization-variable definition has invalid authority"))?;
        let view = Self {
            reference: object.clone(),
            generation,
        };
        let count = view.definition().element_count();
        for source in view.selection().axis_sources() {
            if matches!(source,runmat_value::optimization_variable::OptimizationVariableAxisSource::DefinitionAxis(axis) if *axis>=view.definition().shape().len())
            {
                return Err(invalid("reference axis exceeds original definition rank"));
            }
            cooperation.completed(1).await?;
        }
        if view.selection().is_whole() {
            if view.selection().element_count() != count {
                return Err(invalid(
                    "whole reference extent differs from its definition",
                ));
            }
        } else {
            for position in 0..view.selection().element_count() {
                if view
                    .selection()
                    .definition_index(position)
                    .is_none_or(|index| index >= count)
                {
                    return Err(invalid(
                        "variable reference exceeds its original definition",
                    ));
                }
                cooperation.completed(1).await?;
            }
        }
        if view.selection().role()
            == runmat_value::optimization_variable::OptimizationVariableReferenceRole::Definition
            && (!view.selection().is_whole()
                || view.selection().shape() != view.definition().shape())
        {
            return Err(invalid(
                "definition access requires its original complete shape",
            ));
        }
        Ok(view)
    }
    pub fn selection(&self) -> &OptimizationVariableSelection {
        let Some(OptimizationVariablePayload::Reference(value)) =
            self.reference.optimization_variable_payload()
        else {
            unreachable!()
        };
        value
    }
    pub fn definition(&self) -> &OptimizationVariableData {
        let Some(OptimizationVariablePayload::Definition(value)) =
            self.generation.optimization_variable_payload()
        else {
            unreachable!()
        };
        value
    }
    pub fn reference(&self) -> &ObjectInstance {
        &self.reference
    }
    pub fn publish(&self, next: ObjectInstance) -> BuiltinResult<()> {
        let Some(OptimizationVariablePayload::Definition(data)) =
            next.optimization_variable_payload()
        else {
            return Err(invalid(
                "replacement must retain the original native definition role",
            ));
        };
        if !next.class_name().is(CLASS)
            || next.selected_class().is_some()
            || data.name() != self.definition().name()
            || data.shape() != self.definition().shape()
        {
            return Err(invalid(
                "a property edit cannot replace variable identity, name or geometry",
            ));
        }
        runmat_gc::gc_with_value_mut(&self.selection().definition(), |value| {
            let Value::Object(current) = value else {
                return Err(invalid("variable definition changed representation"));
            };
            if !current.shares_generation_with(&self.generation) {
                return Err(invalid(
                    "variable definition changed during property admission",
                ));
            }
            *value = Value::Object(next);
            Ok(())
        })
        .map_err(|source| gc_error("failed to publish variable definition", source))?
    }
}
