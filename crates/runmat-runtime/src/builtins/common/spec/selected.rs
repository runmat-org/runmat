//! Request-local checks against the runtime's existing dispatch authority.
use super::{
    canonical_fusion_declaration, registration_index::with_registrations, FusionEpilogueOperation,
    FusionStructuralOperation,
};
use runmat_builtins::{
    builtin_catalog_entry_by_name, with_builtin_functions_by_exact_name,
    with_builtin_implementation_provenance_by_name, BuiltinBindingIdentity,
    BuiltinImplementationAuthority,
};

/// Evidence for one selected implementation, not permission to skip operand,
/// method-overload, output-count, or numerical-domain admission.
pub struct SelectedFusionBinding {
    pub identity: SelectedFusionIdentity,
    pub canonical: Option<super::CanonicalFusionDeclaration>,
    pub epilogue: Option<FusionEpilogueOperation>,
    pub structural: Option<FusionStructuralOperation>,
    pub reduction: Option<&'static super::FusionReductionDeclaration>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum SelectedFusionIdentity {
    Canonical(BuiltinBindingIdentity),
    Legacy { physical_owner: &'static str },
}

/// Resolve a canonical implementation against the current invocation's dispatch
/// authority. This is not operand, overload, or numerical-domain admission.
pub fn selected_builtin_binding(
    name: &str,
    expected: Option<BuiltinBindingIdentity>,
    context: Option<&crate::context::RuntimeContext>,
) -> Option<crate::builtin::RuntimeBuiltinBinding> {
    let service = context.and_then(|context| context.service_ports().builtin());
    let entry = builtin_catalog_entry_by_name(name)?;
    let [binding] = entry.bindings else {
        return None;
    };
    let identity = entry.binding_identity(binding);
    if expected.is_some_and(|expected| expected != identity) {
        return None;
    }
    let compiled =
        crate::builtin::with_runtime_builtin_bindings_by_name(name, |bindings| match bindings {
            [binding] => Some(*binding),
            _ => None,
        })?;
    if compiled.identity != identity
        || (service.is_none()
            && with_builtin_functions_by_exact_name(name, |entries| !entries.is_empty()))
    {
        return None;
    }
    if let Some(service) = service {
        let selected = service.bindings_by_name(name);
        let [selected] = selected.as_slice() else {
            return None;
        };
        if selected.identity != identity
            || selected.provenance != compiled.provenance
            || !std::ptr::fn_addr_eq(selected.implementation, compiled.implementation)
        {
            return None;
        }
    }
    Some(compiled)
}

pub fn selected_fusion_binding(
    name: &str,
    expected: Option<BuiltinBindingIdentity>,
    context: Option<&crate::context::RuntimeContext>,
) -> Option<SelectedFusionBinding> {
    let service = context.and_then(|context| context.service_ports().builtin());
    if builtin_catalog_entry_by_name(name).is_some() {
        let identity = selected_builtin_binding(name, expected, context)?.identity;
        let declaration = canonical_fusion_declaration(identity)?;
        return Some(SelectedFusionBinding {
            identity: SelectedFusionIdentity::Canonical(identity),
            canonical: Some(declaration),
            epilogue: declaration.epilogue,
            structural: declaration.structural,
            reduction: declaration.reduction,
        });
    }
    // The dispatcher never falls back to global legacy inventory when a scoped
    // builtin service owns resolution. Retain that same boundary here.
    if service.is_some() || expected.is_some() {
        return None;
    }
    if !with_builtin_functions_by_exact_name(name, |entries| entries.len() == 1) {
        return None;
    }
    let implementation = with_builtin_implementation_provenance_by_name(name, |provenance| {
        let mut implementations = provenance.iter().copied().filter(|implementation| {
            implementation.authority == BuiltinImplementationAuthority::LegacyFunction
        });
        let implementation = implementations.next()?;
        implementations.next().is_none().then_some(implementation)
    })?;
    let declaration = with_registrations(implementation.builtin_path, |registrations| {
        let mut declarations = registrations
            .iter()
            .copied()
            .filter(|spec| spec.spec.name == name);
        let declaration = declarations.next()?;
        declarations.next().is_none().then_some(declaration)
    })?;
    Some(SelectedFusionBinding {
        canonical: None,
        identity: SelectedFusionIdentity::Legacy {
            physical_owner: implementation.builtin_path,
        },
        epilogue: declaration.epilogue,
        structural: declaration.structural,
        reduction: declaration.reduction,
    })
}
