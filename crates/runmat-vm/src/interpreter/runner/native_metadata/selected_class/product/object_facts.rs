use std::collections::HashMap;
use std::rc::Rc;

use runmat_types::{ClassMethodOwner, ClassMethodRole};
use runmat_value::{BoundFunctionBinding, SelectedObjectClassFacts};

use super::{mex, SelectedClassProduct};
use crate::FunctionRegistry;

impl SelectedClassProduct {
    pub fn validates_receiver_method(
        &self,
        facts: &SelectedObjectClassFacts,
        owner: &ClassMethodOwner,
    ) -> bool {
        let Ok(projection) = self.invocation_projection() else {
            return false;
        };
        let Some(method) = facts.method(&owner.method) else {
            return false;
        };
        if projection
            .receiver_declarations(self.class())
            .ok()
            .is_none_or(|admitted| !std::sync::Arc::ptr_eq(&admitted, facts.declarations()))
        {
            return false;
        }
        facts.class() == self.class()
            && &method.owner == owner
            && projection.receiver_method_matches(
                self.class(),
                facts.ancestry(),
                facts.is_handle_class(),
                owner,
                method.access,
                method.is_abstract,
                method.standard_substruct,
            )
    }

    /// Select receiver methods from this constructor's exact class product and
    /// its already-owned definition graph. No class name resolves a body.
    pub fn invocation_runtime_with_binding(
        &self,
        runtime: runmat_runtime::context::RuntimeContext,
        selected: &FunctionRegistry,
        binding: &BoundFunctionBinding,
    ) -> Result<runmat_runtime::context::RuntimeContext, runmat_runtime::RuntimeError> {
        if self._source_owners.is_empty()
            || self
                ._source_owners
                .iter()
                .any(|owner| !owner.is_issued_in(&runtime))
        {
            return Err(mex(
                "ClassMethodBindingInvalid",
                "selected class binding belongs to another session",
            ));
        }
        let projection = self.invocation_projection()?;
        // Graph selection already issued canonical facts. Preserve their Rc
        // identity for subsequent exact selected-method dispatch, but only
        // when the original binding, declarations and definition seal agree.
        let facts = match runtime.selected_object_class().filter(|facts| {
            facts.class() == self.class()
                && facts.constructor() == binding
                && binding
                    .authority
                    .admits_local(&self._source_owners[0].session_lease())
                && binding
                    .authority
                    .local_lease(binding.function)
                    .is_some_and(|lease| {
                        lease
                            .owner_arc()
                            .owns_selected_receiver_seal(&self.receiver_seal)
                    })
                && projection
                    .receiver_declarations(self.class())
                    .ok()
                    .is_some_and(|admitted| std::sync::Arc::ptr_eq(&admitted, facts.declarations()))
        }) {
            Some(facts) => Rc::clone(facts),
            None => self.receiver_facts_with_binding(selected, binding)?,
        };
        Ok(runtime
            .with_selected_class_projection(projection)
            .with_selected_object_class(facts))
    }

    /// Port-free receiver facts for an already leased constructor view. A
    /// response decoder may keep these exact bindings without issuing a new
    /// source or consulting the current class registry.
    pub fn receiver_facts_with_binding(
        &self,
        selected: &FunctionRegistry,
        binding: &BoundFunctionBinding,
    ) -> Result<Rc<SelectedObjectClassFacts>, runmat_runtime::RuntimeError> {
        if self._source_owners.is_empty()
            || self._source_owners.iter().any(|owner| !owner.is_live())
            || !binding
                .authority
                .admits_local(&self._source_owners[0].session_lease())
        {
            return Err(mex(
                "ClassMethodBindingInvalid",
                "selected receiver lacks its exact live issuing session",
            ));
        }
        let projection = self.invocation_projection()?;
        let declarations = projection.receiver_declarations(&self.class)?;
        let constructor_owner = binding
            .authority
            .local_lease(binding.function)
            .ok_or_else(|| {
                mex(
                    "ClassMethodBindingInvalid",
                    "constructor definition lease is absent",
                )
            })?
            .owner_arc();
        if !constructor_owner.owns_selected_receiver_seal(&self.receiver_seal)
            || !self
                .receiver_seal
                .fill(&self.receiver_key, std::sync::Arc::clone(&declarations))
        {
            return Err(mex(
                "ClassMethodBindingInvalid",
                "selected receiver declaration seal differs from its admitted product",
            ));
        }
        let ancestry = declarations.ancestry();
        let mut bodies = HashMap::<ClassMethodOwner, &crate::FunctionBytecode>::new();
        for body in selected.selected_execution_bodies() {
            let Some(owner) = body.class_method_owner.as_ref() else {
                continue;
            };
            if owner.role != ClassMethodRole::Ordinary || !ancestry.contains(&owner.declaring_class)
            {
                continue;
            }
            let lease = binding
                .authority
                .local_lease(body.function.0)
                .ok_or_else(|| {
                    mex(
                        "ClassMethodBindingInvalid",
                        "selected method is outside its constructor dependency graph",
                    )
                })?;
            let source_valid = body.selected_source_owner.as_ref().is_some_and(|owner| {
                self._source_owners
                    .iter()
                    .any(|selected| selected.same_issuance(owner))
            });
            if !lease.matches(&body.definition_owner)
                || !source_valid
                || bodies.insert(owner.clone(), body).is_some()
            {
                return Err(mex(
                    "ClassMethodBindingInvalid",
                    "selected class method differs from its admitted definition",
                ));
            }
        }
        let mut methods = HashMap::with_capacity(declarations.methods().len());
        for (name, method) in declarations.methods() {
            let binding = bodies.get(&method.owner).map(|body| BoundFunctionBinding {
                function: body.function.0,
                authority: binding.authority.clone(),
            });
            if !method.is_abstract && !method.owner.is_static && binding.is_none() {
                return Err(mex(
                    "ClassMethodBindingInvalid",
                    "selected ordinary method lacks its exact definition lease",
                ));
            }
            methods.insert(name.clone(), binding);
        }
        let facts = SelectedObjectClassFacts::new(declarations, binding.clone(), methods)
            .ok_or_else(|| {
                mex(
                    "ClassMethodBindingInvalid",
                    "selected receiver facts differ from its class declaration",
                )
            })?;
        Ok(Rc::new(facts))
    }
}

impl FunctionRegistry {
    /// Recheck receiver storage against the leased body and the exact class
    /// product in that same definition graph. This is the authority boundary
    /// shared by Core's bound service and the VM-local bound executor.
    pub fn validate_selected_receiver_method(
        &self,
        binding: &BoundFunctionBinding,
        facts: &SelectedObjectClassFacts,
        expected: &ClassMethodOwner,
    ) -> Result<Rc<SelectedClassProduct>, runmat_runtime::RuntimeError> {
        let function = runmat_hir::FunctionId(binding.function);
        let body = if let Some(lease) = binding.authority.local_lease(binding.function) {
            self.get_owned(function, lease)
        } else {
            binding.authority.portable_origin().and_then(|origin| {
                self.get(function)
                    .filter(|body| body.binding_origin == Some(origin))
            })
        }
        .ok_or_else(|| {
            mex(
                "ClassMethodBindingInvalid",
                "selected method's leased body is absent",
            )
        })?;
        if expected.role != ClassMethodRole::Ordinary
            || expected.is_static
            || body.class_method_owner.as_ref() != Some(expected)
            || body.selected_source_owner.as_ref().is_none_or(|owner| {
                body.selected_executable_product
                    .as_ref()
                    .is_none_or(|product| !product.source_owner().same_issuance(owner))
            })
            || facts.method(&expected.method).is_none_or(|method| {
                &method.owner != expected
                    || facts.method_binding(&expected.method).as_ref() != Some(binding)
            })
        {
            return Err(mex(
                "ClassMethodBindingInvalid",
                "selected method differs from its receiver's leased definition",
            ));
        }
        let mut admitted_product = None;
        for candidate in self.selected_execution_bodies() {
            let Some(product) = candidate.selected_class_product.as_ref() else {
                continue;
            };
            if product.class() != facts.class()
                || candidate.function.0 != facts.constructor().function
                || facts
                    .constructor()
                    .authority
                    .local_lease(candidate.function.0)
                    .is_none_or(|lease| !lease.matches(&candidate.definition_owner))
            {
                continue;
            }
            if admitted_product.replace(Rc::clone(product)).is_some() {
                return Err(mex(
                    "ClassMethodBindingInvalid",
                    "selected receiver has ambiguous class products",
                ));
            }
        }
        if admitted_product.as_ref().is_none_or(|product| {
            !product.validates_receiver_method(facts, expected)
                || body.selected_source_owner.as_ref().is_none_or(|source| {
                    !product
                        ._source_owners
                        .iter()
                        .any(|selected| selected.same_issuance(source))
                })
        }) {
            return Err(mex(
                "ClassMethodBindingInvalid",
                "selected receiver differs from its admitted class declaration",
            ));
        }
        admitted_product.ok_or_else(|| {
            mex(
                "ClassMethodBindingInvalid",
                "selected receiver lacks its admitted class product",
            )
        })
    }
}
