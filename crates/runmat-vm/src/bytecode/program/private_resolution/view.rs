use std::collections::HashMap;

use runmat_hir::FunctionId;

use super::{resolve_private_name, FunctionBytecode, FunctionRegistry, PrivateNameFact};
use crate::bytecode::PrivateSourceDirectory;

/// Sendable, immutable name-selection metadata. The invocation-local registry
/// retains executable bodies and source leases; this projection carries only
/// the fields used by the canonical private-name rule in the parent module.
pub(crate) struct PrivateNameResolutionView {
    names: PublishedNameResolutionView,
    bodies: HashMap<FunctionId, PrivateNameMetadata>,
    static_overlay: bool,
}

/// Sendable published names without local source or definition owners.
/// Private resolution reuses this exact name projection.
pub struct PublishedNameResolutionView {
    names: HashMap<String, FunctionId>,
}

impl PublishedNameResolutionView {
    pub fn resolve_name(&self, name: &str) -> Option<FunctionId> {
        self.names.get(name).copied()
    }
}

struct PrivateNameMetadata {
    function: FunctionId,
    display_name: String,
    source_file: Option<String>,
    private_owner_scope: String,
    private_source_owner: Option<PrivateSourceDirectory>,
}

impl PrivateNameMetadata {
    fn from_body(body: &FunctionBytecode) -> Self {
        Self {
            function: body.function,
            display_name: body.display_name.clone(),
            source_file: body.source_file.clone(),
            private_owner_scope: body.private_owner_scope.clone(),
            private_source_owner: body.private_source_owner.clone(),
        }
    }

    fn fact(&self) -> PrivateNameFact<'_> {
        PrivateNameFact {
            function: self.function,
            display_name: &self.display_name,
            source_file: self.source_file.as_deref(),
            private_owner_scope: &self.private_owner_scope,
            private_source_owner: self.private_source_owner.as_ref(),
        }
    }
}

impl PrivateNameResolutionView {
    pub(crate) fn resolve_private_name_for_caller(
        &self,
        caller: FunctionId,
        name: &str,
    ) -> Option<FunctionId> {
        let caller = self.bodies.get(&caller)?.fact();
        resolve_private_name(
            caller,
            name,
            self.static_overlay,
            |scoped| self.names.resolve_name(scoped),
            self.bodies.values().map(PrivateNameMetadata::fact),
        )
    }

    pub(crate) fn resolve_name(&self, name: &str) -> Option<FunctionId> {
        self.names.resolve_name(name)
    }
}

impl FunctionRegistry {
    pub fn published_name_resolution_view(&self) -> PublishedNameResolutionView {
        let names = self
            .names
            .iter()
            .filter_map(|(name, _)| self.resolve_name(name).map(|id| (name.clone(), id)))
            .collect();
        PublishedNameResolutionView { names }
    }

    pub(crate) fn private_name_resolution_view(&self) -> PrivateNameResolutionView {
        let names = self.published_name_resolution_view();
        let bodies = self
            .selected_execution_bodies()
            .into_iter()
            .map(|body| (body.function, PrivateNameMetadata::from_body(body)))
            .collect();
        PrivateNameResolutionView {
            names,
            bodies,
            static_overlay: self.static_overlay.is_some(),
        }
    }
}
