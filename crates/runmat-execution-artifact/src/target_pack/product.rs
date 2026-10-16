//! Derive selected product metadata only from the exact verified retained
//! bytes. This does not produce an execution/provider qualification token.
use super::*;
use runmat_execution::host::{
    NativeSelectedProductBinding, NativeSelectedServiceBinding,
    NATIVE_SELECTED_PRODUCT_SCHEMA_VERSION,
};

impl VerifiedTargetPackContents<'_> {
    pub fn selected_product_binding(&self) -> ArtifactResult<NativeSelectedProductBinding> {
        let manifest = self.manifest();
        let selection = self.selection();
        let index = components::index(&manifest.components)?;
        let mut services = Vec::new();
        for kind in selection
            .reached_effects()
            .0
            .iter()
            .copied()
            .filter_map(TargetPackServiceKind::for_effect)
            .chain(selection.required_services().iter().copied())
            .collect::<std::collections::BTreeSet<_>>()
        {
            let service = manifest
                .services
                .iter()
                .find(|service| service.kind == kind)
                .ok_or_else(|| invalid("selected product service declaration is missing"))?;
            if !selection.runtime_components().contains(&service.component) {
                return Err(invalid("selected product omits its service implementation"));
            }
            services.push(NativeSelectedServiceBinding {
                kind,
                abi: service.abi,
                implementation: index[&service.component].object.digest,
                behavior: service.behavior,
            });
        }
        services.sort_by_key(|service| service.kind);
        let binding = NativeSelectedProductBinding {
            schema_version: NATIVE_SELECTED_PRODUCT_SCHEMA_VERSION,
            pack: selection.pack(),
            selection: selection.fingerprint(),
            target: manifest.target.clone(),
            abi: manifest.abi.clone(),
            runtime: manifest.runtime,
            catalog: manifest.catalog,
            runtime_component: index[&manifest.runtime_component].object.digest,
            platform_service_abi: manifest.platform_service_abi,
            numerical: manifest.numerical.clone(),
            optimization: manifest.optimization,
            profile: selection.profile(),
            reached_capabilities: selection.reached_capabilities().clone(),
            available_runtime_capabilities: manifest.capabilities.clone(),
            authorized_runtime_capabilities: selection.authorized_runtime_capabilities().clone(),
            reached_effects: selection.reached_effects().clone(),
            required_services: selection.required_services().clone(),
            services,
        };
        binding.validate().map_err(contract)?;
        Ok(binding)
    }
}
