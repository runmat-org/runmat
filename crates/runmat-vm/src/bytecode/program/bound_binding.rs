use runmat_hir::FunctionId;
use runmat_value::{
    BoundFunctionAuthority, BoundFunctionBinding, FunctionDefinitionOwner, LocalExport,
};

use super::FunctionRegistry;

#[path = "bound_binding/selected_sites.rs"]
mod selected_sites;

impl FunctionRegistry {
    /// A source-admitted constructor enters the canonical bound constructor
    /// policy even when its compiler call site carries a local function ID.
    /// Bare VM embeddings retain their separate unselected metadata path.
    pub fn requires_selected_constructor_admission(&self, function: FunctionId) -> bool {
        self.get(function).is_some_and(|body| {
            body.class_method_owner
                .as_ref()
                .is_some_and(|owner| owner.role == runmat_types::ClassMethodRole::Constructor)
                && (body.selected_source_owner.is_some()
                    || body.selected_executable_product.is_some()
                    || body.selected_class_product.is_some())
        })
    }

    /// Bind a selected body's proved imports once at frame admission. The
    /// resulting invocation map retains only code/source/definition leases.
    pub fn bind_selected_body_sites(
        &self,
        runtime: runmat_runtime::context::RuntimeContext,
        binding: &BoundFunctionBinding,
        body: &super::FunctionBytecode,
    ) -> Result<runmat_runtime::context::RuntimeContext, runmat_runtime::RuntimeError> {
        selected_sites::bind_selected_body_sites(runtime, binding, body, self)
    }

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
