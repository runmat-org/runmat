use runmat_hir::FunctionId;
use runmat_value::{
    BoundFunctionAuthority, BoundFunctionBinding, FunctionDefinitionOwner, LocalExport,
};

use super::FunctionRegistry;

impl FunctionRegistry {
    /// Issue one local callable only from the exact selected definition and
    /// its complete retained dependency graph. A reused numeric function ID
    /// or matching display name is never sufficient to select this binding.
    pub fn exact_bound_binding(&self, function: FunctionId) -> Option<BoundFunctionBinding> {
        let body = self.get(function)?;
        let origin = body.binding_origin?;
        let owner = body.selected_source_owner.as_ref()?;
        if !owner.is_live() {
            return None;
        }
        let source = body.source_id?.runtime()?;
        owner.source_ordinal_for(source)?;
        let static_sites = self.static_site_leases_for_body(body)?;
        let mut dependencies = self.definition_dependency_owners(function)?;
        let position = dependencies.iter().position(|(id, selected)| {
            *id == function && std::sync::Arc::ptr_eq(selected, &body.definition_owner)
        })?;
        let (_, selected) = dependencies.remove(position);
        let definition = FunctionDefinitionOwner::issue(&selected, function.0);
        let dependencies = dependencies
            .into_iter()
            .map(|(id, selected)| FunctionDefinitionOwner::issue(&selected, id.0))
            .collect();
        Some(BoundFunctionBinding {
            function: function.0,
            authority: BoundFunctionAuthority::local_with_static_sites(
                owner.session_lease(),
                definition,
                dependencies,
                static_sites,
                LocalExport::Exact(origin),
            ),
        })
    }
}
