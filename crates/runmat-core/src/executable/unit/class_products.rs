use std::{rc::Rc, sync::Arc};

use runmat_types::ClassMethodRole;
use runmat_vm::{FunctionRegistry, SelectedClassProduct};

use super::ExecutableUnit;

impl ExecutableUnit {
    /// A native entrypoint uses the same exact, source-owned class product as
    /// an interactive publication. Attaching is metadata-only: unresolved
    /// ancestry is admitted only if this constructor is actually called.
    pub(super) fn attach_runtime_class_products(
        &self,
        registry: &mut FunctionRegistry,
    ) -> Result<(), String> {
        let current_program = self
            .revision()
            .program_revision
            .identity_digest()
            .map_err(|error| error.to_string())?;
        let constructors = registry
            .functions
            .iter()
            .filter_map(|(function, body)| {
                if body.selected_class_product.is_some()
                    || body
                        .binding_origin
                        .is_none_or(|origin| origin.program != current_program)
                {
                    return None;
                }
                body.class_method_owner
                    .as_ref()
                    .filter(|owner| owner.role == ClassMethodRole::Constructor && !owner.is_static)
                    .map(|owner| (*function, owner.declaring_class.clone()))
            })
            .collect::<Vec<_>>();
        for (function, class) in constructors {
            let body = registry
                .get_current(function)
                .ok_or_else(|| "native constructor lost its selected body".to_string())?;
            let definition = Arc::clone(&body.definition_owner);
            let product = SelectedClassProduct::from_admitted_instructions(
                self.bytecode().instructions.iter().chain(
                    self.bytecode()
                        .bound_functions
                        .iter()
                        .filter(|(id, bound)| {
                            bound
                                .binding_origin
                                .is_some_and(|origin| origin.program == current_program)
                                && registry.get_current(**id).is_some_and(|selected| {
                                    Arc::ptr_eq(&selected.definition_owner, &bound.definition_owner)
                                })
                        })
                        .flat_map(|(_, bound)| bound.instructions.iter()),
                ),
                registry,
                function,
                &class,
            )
            .map_err(|error| error.to_string())?;
            registry
                .attach_current_selected_class_product(function, &definition, Rc::new(product))
                .map_err(str::to_string)?;
        }
        Ok(())
    }
}
