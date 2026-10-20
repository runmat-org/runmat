//! Select the linked product's exact superclass body for one native frame.

use runmat_execution::FunctionBindingOrigin;
use runmat_runtime::context::{RetainedClassDeclaration, RetainedConstructorDeclaration};
use runmat_types::{ClassMethodRole, FunctionId};

use super::native_error;

pub(super) fn bind_selected(
    executor: &runmat_native_executor::NativeExecutor,
    function: FunctionId,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::context::RuntimeContext, Box<runmat_runtime::RuntimeError>> {
    let metadata = executor
        .function_metadata(function)
        .ok_or_else(|| native_error("selected superclass caller is absent from Native IR"))?;
    if !metadata.super_constructor_calls().is_empty()
        && metadata
            .class_method_owner
            .as_ref()
            .is_none_or(|owner| owner.role != ClassMethodRole::Constructor || owner.is_static)
    {
        return Err(Box::new(native_error(
            "standalone superclass call lacks a typed constructor owner",
        )));
    }
    let runtime =
        runtime.with_constructor_receiver_for_method(metadata.class_method_owner.as_ref())?;
    let caller_origin =
        FunctionBindingOrigin::from_revision(executor.program_revision(), metadata.portable_id)
            .map_err(native_error)?;
    let Some(owner) = metadata
        .class_method_owner
        .as_ref()
        .filter(|owner| owner.role == ClassMethodRole::Constructor && !owner.is_static)
    else {
        return Ok(runtime.with_retained_super_caller_origin(caller_origin));
    };
    let declaration = executor
        .class_declarations()
        .iter()
        .find(|declaration| declaration.class == owner.declaring_class)
        .ok_or_else(|| native_error("native constructor has no immutable class declaration"))?;
    let Some(parent) = &declaration.parent else {
        return Ok(runtime.with_retained_super_caller_origin(caller_origin));
    };
    let caller_declaration = admitted_declaration(declaration)?;
    let linked_parent = executor
        .class_declarations()
        .iter()
        .find(|candidate| &candidate.class == parent);
    let Some(target) = linked_parent.and_then(|candidate| candidate.constructor.as_ref()) else {
        if linked_parent.is_none()
            && !runmat_runtime::class_registry::has_intrinsic_implicit_constructor(parent)
        {
            return Err(Box::new(native_error(
                "custom superclass is absent from the linked product",
            )));
        }
        let selection = if let Some(linked_parent) = linked_parent {
            runmat_runtime::context::RetainedSuperConstructorBinding::from_admitted_linked_implicit(
                &caller_declaration,
                &admitted_declaration(linked_parent)?,
                caller_origin,
            )?
        } else {
            runmat_runtime::context::RetainedSuperConstructorBinding::from_admitted_intrinsic(
                &caller_declaration,
                caller_origin,
            )?
        };
        return runtime
            .with_retained_super_constructors(vec![selection], caller_origin)
            .map_err(Box::new);
    };
    let target_declaration = admitted_declaration(
        linked_parent.expect("target constructor came from the linked declaration"),
    )?;
    let target_metadata = executor
        .function_metadata(target.function)
        .ok_or_else(|| native_error("linked superclass constructor is absent from Native IR"))?;
    let target_owner = target_metadata
        .class_method_owner
        .as_ref()
        .ok_or_else(|| native_error("linked superclass constructor lacks a typed owner"))?;
    let target_origin = FunctionBindingOrigin::from_revision(
        executor.program_revision(),
        target_metadata.portable_id,
    )
    .map_err(native_error)?;
    let target_binding =
        runmat_runtime::user_functions::resolve_binding_origin_binding_from_runtime(
            &runtime,
            target_origin,
        )
        .filter(|binding| binding.function == target.function.0)
        .ok_or_else(|| native_error("linked superclass constructor has no exact issued binding"))?;
    runtime
        .with_retained_super_constructors(
            vec![
                runmat_runtime::context::RetainedSuperConstructorBinding::from_admitted_definition(
                    &caller_declaration,
                    &target_declaration,
                    caller_origin,
                    target_origin,
                    target_owner.clone(),
                    target_binding,
                )?,
            ],
            caller_origin,
        )
        .map_err(Box::new)
}

fn admitted_declaration(
    class: &runmat_native_codegen::NativeClassDeclaration,
) -> Result<RetainedClassDeclaration, Box<runmat_runtime::RuntimeError>> {
    let constructor = class
        .constructor
        .as_ref()
        .map(
            |constructor| -> Result<_, Box<runmat_runtime::RuntimeError>> {
                let method = class
                    .methods
                    .iter()
                    .find(|method| method.function == constructor.function)
                    .ok_or_else(|| native_error("linked constructor has no declared method"))?;
                Ok(RetainedConstructorDeclaration {
                    name: method.name.clone(),
                    is_static: method.is_static,
                    is_abstract: method.attributes.is_abstract,
                    access: method.attributes.access,
                })
            },
        )
        .transpose()?;
    Ok(RetainedClassDeclaration {
        class: class.class.clone(),
        parent: class.parent.clone(),
        constructor,
    })
}
