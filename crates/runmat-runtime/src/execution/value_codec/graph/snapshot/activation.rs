use super::*;
use runmat_execution::value::GraphBinding;

impl<'a> Encoder<'a> {
    pub(super) fn process_activation(
        &mut self,
        environment: LexicalEnvironment,
        index: GraphNodeRef,
        work: &mut Vec<Task<'a>>,
    ) -> Result<(), ValueCodecError> {
        let parent = environment
            .parent()
            .map(|parent| self.push_activation(parent, work))
            .transpose()?;
        let owned = crate::call::lexical::owned_binding_ids(&environment)
            .map_err(|error| invalid(error.to_string()))?;
        let mut bindings = Vec::with_capacity(owned.len());
        for binding in owned {
            let value = crate::call::lexical::read_binding(&environment, binding)
                .map_err(|error| invalid(error.to_string()))?;
            let value = value
                .map(|value| {
                    self.push_values(vec![Input::Owned(value)], work)
                        .map(|refs| refs[0])
                })
                .transpose()?;
            bindings.push(GraphBinding { binding, value });
        }
        self.provisional_activations
            .insert(index, ProvisionalActivation { parent, bindings });
        Ok(())
    }
}
