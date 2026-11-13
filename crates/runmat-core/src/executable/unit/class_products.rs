use std::{rc::Rc, sync::Arc};

use runmat_types::ClassMethodRole;
use runmat_vm::{FunctionRegistry, SelectedClassProduct};

use super::ExecutableUnit;

/// Invocation-local association produced before any selected static capture.
/// It retains only opaque definition owners, never code copies or a Runtime.
pub(crate) struct RuntimeConstructorAdmissions(Vec<ConstructorAdmission>);

struct ConstructorAdmission {
    function: runmat_hir::FunctionId,
    original: Arc<runmat_value::FunctionDefinitionOwner>,
    issued: Arc<runmat_value::FunctionDefinitionOwner>,
}

impl RuntimeConstructorAdmissions {
    fn admits(
        &self,
        original: &runmat_vm::FunctionBytecode,
        issued: &runmat_vm::FunctionBytecode,
    ) -> bool {
        self.0
            .binary_search_by_key(&issued.function, |entry| entry.function)
            .ok()
            .is_some_and(|index| {
                let entry = &self.0[index];
                Arc::ptr_eq(&entry.original, &original.definition_owner)
                    && Arc::ptr_eq(&entry.issued, &issued.definition_owner)
                    && same_constructor_source(original, issued)
            })
    }
}

/// The stream is cloned by ordinary Vec semantics. Its admission relation is
/// definition/source identity, never its allocation address or textual content.
fn same_constructor_source(
    original: &runmat_vm::FunctionBytecode,
    issued: &runmat_vm::FunctionBytecode,
) -> bool {
    original.function == issued.function
        && original.portable_function == issued.portable_function
        && original.binding_origin == issued.binding_origin
        && original.class_method_owner == issued.class_method_owner
        && original
            .source_id
            .and_then(|source| source.program())
            .is_some_and(|ordinal| {
                issued
                    .source_id
                    .and_then(|source| source.runtime())
                    .is_some_and(|source| {
                        issued.selected_source_owner.as_ref().is_some_and(|owner| {
                            owner.is_live() && owner.source_ordinal_for(source) == Some(ordinal)
                        })
                    })
            })
}

impl ExecutableUnit {
    /// Reissue only current portable compiler constructors at actual source
    /// admission. Ordinary and already-issued inherited definitions stay exact.
    /// Validate every original/mirror before creating any new definition owner.
    pub(crate) fn materialize_runtime_constructor_definitions(
        &self,
        registry: &mut FunctionRegistry,
        mut mirror: Option<
            &mut std::collections::HashMap<runmat_hir::FunctionId, runmat_vm::FunctionBytecode>,
        >,
    ) -> Result<RuntimeConstructorAdmissions, String> {
        let program = self
            .revision()
            .program_revision
            .identity_digest()
            .map_err(|error| error.to_string())?;
        let mut constructors = self
            .functions()
            .functions
            .iter()
            .filter_map(|(function, body)| {
                (body
                    .binding_origin
                    .is_some_and(|origin| origin.program == program)
                    && body.class_method_owner.as_ref().is_some_and(|owner| {
                        owner.role == ClassMethodRole::Constructor && !owner.is_static
                    })
                    && body
                        .source_id
                        .is_some_and(|source| source.program().is_some()))
                .then_some(*function)
            })
            .collect::<Vec<_>>();
        constructors.sort_unstable();
        for function in &constructors {
            let original = self
                .functions()
                .get_current(*function)
                .ok_or("original compiler constructor is absent")?;
            let issued = registry
                .get_current(*function)
                .ok_or("source-admitted constructor is absent")?;
            if !Arc::ptr_eq(&original.definition_owner, &issued.definition_owner)
                || !same_constructor_source(original, issued)
                || issued.selected_executable_product.is_some()
                || issued.selected_static_capture.is_some()
                || issued.selected_class_product.is_some()
            {
                return Err("constructor differs from its original source admission".into());
            }
            if mirror
                .as_ref()
                .and_then(|mirror| mirror.get(function))
                .is_some_and(|bound| {
                    bound.binding_origin != issued.binding_origin
                        || bound.source_id != issued.source_id
                        || !Arc::ptr_eq(&bound.definition_owner, &issued.definition_owner)
                })
            {
                return Err("constructor and bound mirror select different definitions".into());
            }
        }
        let mut admissions = Vec::with_capacity(constructors.len());
        for function in constructors {
            let body = registry
                .functions
                .get_mut(&function)
                .expect("checked constructor");
            let original = Arc::clone(&body.definition_owner);
            let issued = Arc::new(runmat_value::FunctionDefinitionOwner::default());
            body.definition_owner = Arc::clone(&issued);
            if let Some(bound) = mirror
                .as_deref_mut()
                .and_then(|mirror| mirror.get_mut(&function))
            {
                bound.definition_owner = Arc::clone(&issued);
            }
            admissions.push(ConstructorAdmission {
                function,
                original,
                issued,
            });
        }
        Ok(RuntimeConstructorAdmissions(admissions))
    }

    /// A native entrypoint uses the same exact, source-owned class product as
    /// an interactive publication. Attaching is metadata-only: unresolved
    /// ancestry is admitted only if this constructor is actually called.
    pub(crate) fn attach_runtime_class_products(
        &self,
        registry: &mut FunctionRegistry,
        admissions: &RuntimeConstructorAdmissions,
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
            let original = self
                .functions()
                .get_current(function)
                .ok_or("original class constructor is absent")?;
            if !admissions.admits(original, body) {
                return Err("constructor declaration lacks its exact source admission".into());
            }
            let definition = Arc::clone(&body.definition_owner);
            let product = SelectedClassProduct::from_admitted_instructions(
                self.bytecode().instructions.iter().chain(
                    self.bytecode()
                        .bound_functions
                        .iter()
                        .filter(|(_, bound)| {
                            bound
                                .binding_origin
                                .is_some_and(|origin| origin.program == current_program)
                        })
                        .filter_map(|(id, original)| {
                            let issued = registry.get_current(*id)?;
                            if Arc::ptr_eq(&original.definition_owner, &issued.definition_owner) {
                                Some(original.instructions.iter())
                            } else if admissions.admits(original, issued) {
                                // Consume this actual issued stream only after the
                                // source-admission receipt authenticates its clone.
                                Some(issued.instructions.iter())
                            } else {
                                None
                            }
                        })
                        .flatten(),
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
