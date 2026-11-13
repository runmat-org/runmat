//! Publish one authenticated product into the direct invocation's code views.
use std::rc::Rc;
use std::sync::Arc;

use runmat_vm::{Bytecode, SelectedExecutableProduct};

use super::SelectedGraphPrimary;

impl SelectedGraphPrimary {
    pub(in crate::session) fn with_current_product(
        mut self,
        bytecode: &mut Bytecode,
        product: Rc<SelectedExecutableProduct>,
        constructors: &crate::executable::RuntimeConstructorAdmissions,
    ) -> Result<Self, String> {
        let canonical = self.canonical_v3()?;
        if (!Arc::ptr_eq(&canonical, product.canonical_v3())
            && canonical.as_ref() != product.canonical_v3().as_ref())
            || !self.source_owner.same_issuance(product.source_owner())
        {
            return Err("direct selected product differs from its code/source owner".into());
        }
        // Fresh product binding already checked exact receipt presence,
        // canonical bytes and source issuance; do not decode the code again.
        self.interop_receipt = product.interop_receipt().cloned();
        let ids = self
            .current_functions
            .iter()
            .copied()
            .filter(|id| {
                self.selected_registry
                    .get_current(*id)
                    .and_then(|body| body.binding_origin)
                    .is_some_and(|origin| origin.program == self.expected_program)
            })
            .collect::<Vec<_>>();
        let capture = crate::session::source_evaluation::static_capture(
            Rc::clone(&product),
            &self.static_targets,
        )?;
        let selected = Rc::get_mut(&mut self.selected_registry)
            .ok_or("direct selected registry escaped before product publication")?;
        selected
            .attach_current_selected_executable_product(ids.iter().copied(), product.clone())
            .map_err(|error| error.to_string())?;
        selected.attach_current_selected_static_capture(ids.iter().copied(), &capture)?;
        if let Some(unit) = self.unit.as_ref() {
            unit.attach_runtime_class_products(selected, constructors)?;
        }
        bytecode
            .function_registry
            .attach_current_selected_executable_product(ids.iter().copied(), product)
            .map_err(|error| error.to_string())?;
        bytecode
            .function_registry
            .attach_current_selected_static_capture(ids.iter().copied(), &capture)?;
        for id in &ids {
            let body = selected
                .get_current(*id)
                .ok_or("selected class body disappeared")?;
            if let Some(product) = body.selected_class_product.as_ref() {
                bytecode
                    .function_registry
                    .attach_current_selected_class_product(
                        *id,
                        &body.definition_owner,
                        Rc::clone(product),
                    )
                    .map_err(str::to_owned)?;
            }
        }
        for id in ids {
            if let (Some(selected), Some(bound)) = (
                bytecode.function_registry.get_current(id),
                bytecode.bound_functions.get_mut(&id),
            ) {
                if Arc::ptr_eq(&selected.definition_owner, &bound.definition_owner)
                    && selected.binding_origin == bound.binding_origin
                    && selected.source_id == bound.source_id
                {
                    bound.selected_executable_product =
                        selected.selected_executable_product.clone();
                    bound.selected_static_capture = selected.selected_static_capture.clone();
                    bound.selected_class_product = selected.selected_class_product.clone();
                }
            }
        }
        Ok(self)
    }
}
