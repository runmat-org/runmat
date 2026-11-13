//! Invocation-local dispatch for a bound body's sealed static sites.

use std::collections::HashMap;
use std::rc::Rc;

use super::super::{BytecodeSourceRef, FunctionBytecode, FunctionRegistry};
use runmat_execution::executable::ExecutableStaticCaller;
use runmat_execution::value::GraphStaticReferenceSite;
use runmat_execution::FunctionBindingOrigin;
use runmat_hir::FunctionId;
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_value::{BoundFunctionBinding, SelectedStaticSiteBinding};

fn invalid(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message.into())
        .with_identifier("RunMat:StaticImportUnavailable")
        .build()
}

/// Bind only sites belonging to this exact selected V3 body. The escaped
/// binding already owns counted target leases; the current registry merely
/// reopens those exact bodies by owner, never by name or current numeric ID.
pub(super) fn bind_selected_body_sites(
    runtime: RuntimeContext,
    binding: &BoundFunctionBinding,
    body: &FunctionBytecode,
    registry: &FunctionRegistry,
) -> Result<RuntimeContext, RuntimeError> {
    let Some(product) = body.selected_executable_product.as_ref() else {
        if body.selected_static_capture.is_some() {
            return Err(invalid("selected static capture has no exact body product"));
        }
        return Ok(runtime);
    };
    let Some(origin) = body.binding_origin else {
        return Err(invalid("selected callable has no portable body origin"));
    };
    let source_owner = body
        .selected_source_owner
        .as_ref()
        .ok_or_else(|| invalid("selected callable has no issued source"))?;
    if !source_owner.is_live()
        || !product.source_owner().same_issuance(source_owner)
        || body
            .source_id
            .and_then(BytecodeSourceRef::runtime)
            .is_none_or(|source| !source_owner.owns_source(source))
        || !binding
            .authority
            .admits_local(&source_owner.session_lease())
        || binding
            .authority
            .local_lease(binding.function)
            .is_none_or(|lease| !lease.matches(&body.definition_owner))
    {
        return Err(invalid(
            "selected callable differs from its live source/definition",
        ));
    }
    let Some(capture) = body.selected_static_capture.as_ref() else {
        return if product.static_imports().is_empty() && product.linked_native_imports().is_empty()
        {
            Ok(runtime)
        } else {
            Err(invalid("selected callable lacks sealed static capture"))
        };
    };
    if !Rc::ptr_eq(capture.product(), product) {
        return Err(invalid(
            "selected callable capture differs from its exact product",
        ));
    }
    let mut issued = HashMap::new();
    for SelectedStaticSiteBinding {
        caller_unit,
        caller,
        site,
        target_binding,
        target_unit,
        target: target_origin,
    } in binding
        .authority
        .selected_site_bindings()
        .map_err(invalid)?
    {
        if issued
            .insert(
                (caller_unit, caller, site),
                (target_binding, target_unit, target_origin),
            )
            .is_some()
        {
            return Err(invalid("selected callable repeats one issued site lease"));
        }
    }
    let mut selected = HashMap::<
        (Option<FunctionBindingOrigin>, GraphStaticReferenceSite),
        BoundFunctionBinding,
    >::new();
    for edge in capture.dependencies() {
        if !Rc::ptr_eq(edge.caller(), product) {
            continue;
        }
        let ExecutableStaticCaller::Function(caller) = edge.import().caller else {
            continue;
        };
        if caller != origin {
            continue;
        }
        let import = edge.import();
        let (target_binding, unit_digest, target_origin) = issued
            .remove(&(product.full_digest(), Some(caller), import.site))
            .ok_or_else(|| invalid("sealed site has no retained target lease"))?;
        if unit_digest != edge.target().full_digest() || target_origin != import.target {
            return Err(invalid(
                "sealed site lease differs from its captured target",
            ));
        }
        let lease = target_binding
            .authority
            .local_definition()
            .ok_or_else(|| invalid("selected site target has no exact definition lease"))?;
        let target = registry
            .get_owned(FunctionId(target_binding.function), lease)
            .ok_or_else(|| invalid("selected site target body has retired"))?;
        if target.binding_origin != Some(import.target)
            || target
                .selected_executable_product
                .as_ref()
                .is_none_or(|selected| !Rc::ptr_eq(selected, edge.target()))
            || target
                .selected_source_owner
                .as_ref()
                .is_none_or(|owner| !owner.same_issuance(edge.target().source_owner()))
        {
            return Err(invalid(
                "selected site target changed its body/source product",
            ));
        }
        if selected
            .insert((Some(caller), import.site), target_binding)
            .is_some()
        {
            return Err(invalid("selected callable repeats one sealed site"));
        }
    }
    for site in capture.linked_native() {
        if !Rc::ptr_eq(site.caller(), product) {
            continue;
        }
        let import = site.import();
        if import.caller != ExecutableStaticCaller::Function(origin) {
            continue;
        }
        let target = site
            .binding()
            .ok_or_else(|| invalid("linked-native static site source has retired"))?;
        if selected
            .insert((Some(origin), import.site), target)
            .is_some()
        {
            return Err(invalid(
                "selected callable repeats a VM or linked-native site",
            ));
        }
    }
    let expected = product
        .static_imports()
        .iter()
        .filter(|import| import.caller == ExecutableStaticCaller::Function(origin))
        .count()
        + product
            .linked_native_imports()
            .iter()
            .filter(|import| import.caller == ExecutableStaticCaller::Function(origin))
            .count();
    if selected.len() != expected {
        return Err(invalid("selected callable omits a sealed static site"));
    }
    if selected.is_empty() {
        return Ok(runtime);
    }
    let selected = Rc::new(selected);
    Ok(
        runtime.with_selected_static_site_resolver(Rc::new(move |caller, site| {
            Ok(selected.get(&(caller, site)).cloned())
        })),
    )
}
