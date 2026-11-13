use std::rc::Rc;
use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_types::ClassMethodRole;
use runmat_value::BoundFunctionAuthority;

use super::FunctionRegistry;
use crate::SelectedClassProduct;

impl FunctionRegistry {
    /// Read a body selected by the typed dependency walk without minting a
    /// counted callable lease just to inspect immutable class metadata.
    pub(crate) fn get_selected_definition(
        &self,
        function: FunctionId,
        owner: &Arc<runmat_value::FunctionDefinitionOwner>,
    ) -> Option<&super::FunctionBytecode> {
        self.functions
            .get(&function)
            .filter(|body| Arc::ptr_eq(&body.definition_owner, owner))
            .or_else(|| {
                self.retired.get(&function).and_then(|bodies| {
                    bodies
                        .iter()
                        .find(|body| Arc::ptr_eq(&body.definition_owner, owner))
                })
            })
    }

    /// Current publication may attach the same immutable product to its
    /// executing and durable registry views only when both still own the
    /// selected definition. A reused numeric ID or name is insufficient.
    pub fn attach_current_selected_class_product(
        &mut self,
        function: FunctionId,
        owner: &Arc<runmat_value::FunctionDefinitionOwner>,
        product: Rc<SelectedClassProduct>,
    ) -> Result<(), &'static str> {
        let body = self
            .functions
            .get_mut(&function)
            .filter(|body| Arc::ptr_eq(&body.definition_owner, owner))
            .ok_or("selected constructor definition changed during publication")?;
        if body
            .selected_source_owner
            .as_ref()
            .is_none_or(|source| !product.admits_constructor_source(source))
            || !body.class_method_owner.as_ref().is_some_and(|method| {
                method.role == ClassMethodRole::Constructor
                    && !method.is_static
                    && &method.declaring_class == product.class()
            })
        {
            return Err("selected class product differs from its admitted constructor");
        }
        if body
            .selected_class_product
            .as_ref()
            .is_some_and(|existing| !Rc::ptr_eq(existing, &product))
            || !body
                .definition_owner
                .admit_selected_receiver_seal(Arc::clone(product.receiver_seal()))
        {
            return Err("selected constructor already owns another class product");
        }
        body.selected_class_product = Some(product);
        Ok(())
    }

    /// Attach a selected class product only to the exact newly admitted
    /// constructor definition. Reused numeric IDs and equal source origins
    /// do not identify a retained body; its local definition lease does.
    pub fn attach_selected_class_product(
        &mut self,
        function: FunctionId,
        authority: &BoundFunctionAuthority,
        product: Rc<SelectedClassProduct>,
    ) -> Result<(), &'static str> {
        let lease = authority
            .local_lease(function.0)
            .ok_or("selected constructor lacks its exact definition lease")?;
        let body = self
            .get_owned(function, lease)
            .ok_or("selected constructor definition has retired")?;
        if body
            .selected_source_owner
            .as_ref()
            .is_none_or(|source| !product.admits_constructor_source(source))
            || !body.class_method_owner.as_ref().is_some_and(|owner| {
                owner.role == ClassMethodRole::Constructor
                    && !owner.is_static
                    && &owner.declaring_class == product.class()
            })
        {
            return Err("selected class product differs from constructor owner");
        }
        let owner = Arc::clone(&body.definition_owner);
        if body
            .selected_class_product
            .as_ref()
            .is_some_and(|existing| !Rc::ptr_eq(existing, &product))
            || !owner.admit_selected_receiver_seal(Arc::clone(product.receiver_seal()))
        {
            return Err("selected constructor already owns another class product");
        }
        let data = &mut **self;
        if let Some(selected) = data
            .functions
            .get_mut(&function)
            .filter(|body| Arc::ptr_eq(&body.definition_owner, &owner))
        {
            selected.selected_class_product = Some(product);
            return Ok(());
        }
        let selected = data
            .retired
            .get_mut(&function)
            .and_then(|bodies| {
                bodies
                    .iter_mut()
                    .find(|body| Arc::ptr_eq(&body.definition_owner, &owner))
            })
            .ok_or("selected constructor definition changed during admission")?;
        selected.selected_class_product = Some(product);
        Ok(())
    }
}
