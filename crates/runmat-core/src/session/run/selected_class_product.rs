use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_runtime::build_runtime_error;
use runmat_types::ClassMethodRole;
use runmat_vm::{Bytecode, FunctionRegistry, SelectedClassProduct, SelectedDeclarationProduct};

use crate::RunError;

fn publication_error(error: impl ToString) -> RunError {
    RunError::Runtime(
        build_runtime_error(error.to_string())
            .with_identifier("RunMat:ExecutableClassProduct")
            .build(),
    )
}

/// Keep the class defaults/access of this compilation with its newly
/// published constructor definitions. Both active and durable registries
/// must still own the very same body; neither a name nor a numeric slot is
/// enough to attach an older product after replacement.
pub(super) fn attach_published_constructors(
    bytecode: &mut Bytecode,
    current: &HashSet<FunctionId>,
    active: &mut FunctionRegistry,
    durable: &mut FunctionRegistry,
    declaration_products: &[Rc<SelectedDeclarationProduct>],
) -> Result<(), RunError> {
    for function in current {
        let Some(body) = active.get_current(*function) else {
            continue;
        };
        let Some(class) = body
            .class_method_owner
            .as_ref()
            .filter(|owner| owner.role == ClassMethodRole::Constructor && !owner.is_static)
            .map(|owner| owner.declaring_class.clone())
        else {
            continue;
        };
        let owner = Arc::clone(&body.definition_owner);
        durable
            .get_current(*function)
            .filter(|published| Arc::ptr_eq(&published.definition_owner, &owner))
            .and_then(|published| published.selected_source_owner.as_ref())
            .ok_or_else(|| publication_error("constructor lacks its issued source owner"))?;
        let product = SelectedClassProduct::from_admitted_instructions(
            bytecode.instructions.iter().chain(
                bytecode
                    .bound_functions
                    .iter()
                    .filter(|(id, bound)| {
                        current.contains(id)
                            && active.get_current(**id).is_some_and(|selected| {
                                Arc::ptr_eq(&selected.definition_owner, &bound.definition_owner)
                            })
                    })
                    .flat_map(|(_, body)| body.instructions.iter()),
            ),
            active,
            *function,
            &class,
        )
        .map_err(publication_error)?;
        // Publication may include a constructor that is never called. Keep
        // its exact declarations/source owner, but defer closed ancestry and
        // default admission to that selected constructor's invocation.
        // An unresolved parent must not be supplied by mutable same-name
        // registry state when it is eventually invoked.
        let product = Rc::new(
            product
                .with_declaration_products(declaration_products)
                .map_err(publication_error)?,
        );
        active
            .attach_current_selected_class_product(*function, &owner, Rc::clone(&product))
            .map_err(publication_error)?;
        if let Some(bound) = bytecode
            .bound_functions
            .get_mut(function)
            .filter(|bound| Arc::ptr_eq(&bound.definition_owner, &owner))
        {
            bound.selected_class_product = Some(Rc::clone(&product));
        }
        durable
            .attach_current_selected_class_product(*function, &owner, product)
            .map_err(publication_error)?;
    }
    bytecode.function_registry = active.clone();
    Ok(())
}

#[cfg(test)]
#[path = "selected_class_product/tests.rs"]
mod tests;
