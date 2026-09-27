//! Request-local checks against the runtime's existing dispatch authority.
use super::{
    builtin_fusion_spec_registrations, canonical_fusion_declaration, FusionEpilogueOperation,
    FusionStructuralOperation,
};
use runmat_builtins::{
    builtin_catalog_entry_by_name, builtin_functions, builtin_implementation_provenance,
    BuiltinBindingIdentity, BuiltinImplementationAuthority,
};

/// Evidence for one selected implementation, not permission to skip operand,
/// method-overload, output-count, or numerical-domain admission.
pub struct SelectedFusionBinding {
    pub identity: SelectedFusionIdentity,
    pub epilogue: Option<FusionEpilogueOperation>,
    pub structural: Option<FusionStructuralOperation>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum SelectedFusionIdentity {
    Canonical(BuiltinBindingIdentity),
    Legacy { physical_owner: &'static str },
}

pub fn selected_fusion_binding(
    name: &str,
    expected: Option<BuiltinBindingIdentity>,
    context: Option<&crate::context::RuntimeContext>,
) -> Option<SelectedFusionBinding> {
    let active = crate::context::legacy::active();
    let service = context
        .or(active.as_ref())
        .and_then(|context| context.service_ports().builtin());
    if let Some(entry) = builtin_catalog_entry_by_name(name) {
        let [binding] = entry.bindings else {
            return None;
        };
        let identity = entry.binding_identity(binding);
        if expected.is_some_and(|expected| expected != identity) {
            return None;
        }
        let compiled = crate::builtin::runtime_builtin_bindings_by_name(name);
        let [compiled] = compiled.as_slice() else {
            return None;
        };
        if compiled.identity != identity
            || (service.is_none()
                && builtin_functions()
                    .into_iter()
                    .any(|legacy| legacy.name == name))
        {
            return None;
        }
        if let Some(service) = service {
            let selected = service.bindings_by_name(name);
            let [selected] = selected.as_slice() else {
                return None;
            };
            if selected.identity != identity
                || !std::ptr::fn_addr_eq(selected.implementation, compiled.implementation)
            {
                return None;
            }
        }
        let declaration = canonical_fusion_declaration(identity)?;
        return Some(SelectedFusionBinding {
            identity: SelectedFusionIdentity::Canonical(identity),
            epilogue: declaration.epilogue,
            structural: declaration.structural,
        });
    }
    // The dispatcher never falls back to global legacy inventory when a scoped
    // builtin service owns resolution. Retain that same boundary here.
    if service.is_some() || expected.is_some() {
        return None;
    }
    let mut runtime = builtin_functions()
        .into_iter()
        .filter(|binding| binding.name == name);
    runtime.next()?;
    if runtime.next().is_some() {
        return None;
    }
    let provenance = builtin_implementation_provenance();
    let mut implementations = provenance.iter().filter(|implementation| {
        implementation.name == name
            && implementation.authority == BuiltinImplementationAuthority::LegacyFunction
    });
    let implementation = implementations.next()?;
    if implementations.next().is_some() {
        return None;
    }
    let mut declarations = builtin_fusion_spec_registrations()
        .filter(|spec| spec.builtin_path == implementation.builtin_path && spec.spec.name == name);
    let declaration = declarations.next()?;
    if declarations.next().is_some() {
        return None;
    }
    Some(SelectedFusionBinding {
        identity: SelectedFusionIdentity::Legacy {
            physical_owner: implementation.builtin_path,
        },
        epilogue: declaration.epilogue,
        structural: declaration.structural,
    })
}
