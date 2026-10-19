use std::sync::Arc;

use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_value::FunctionDefinitionOwner;

use super::RuntimeBoundExecutableUnit;

/// The code/source half of a successfully staged imported unit. A generation
/// may retain this while one of its exact definition handles lives, without
/// retaining the invocation RuntimeContext or any execution service port.
pub struct RetainedExecutableSourceOwner {
    canonical_v3: Arc<[u8]>,
    source_owner: SelectedProgramSourceOwner,
    definitions: Vec<Arc<FunctionDefinitionOwner>>,
}

impl std::fmt::Debug for RetainedExecutableSourceOwner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedExecutableSourceOwner")
            .field("v3_bytes", &self.canonical_v3.len())
            .field("definition_count", &self.definitions.len())
            .finish_non_exhaustive()
    }
}

impl RetainedExecutableSourceOwner {
    /// Retain a locally compiled, already source-bound V3 product without
    /// reissuing its source lease or storing its invocation RuntimeContext.
    /// This is used when dynamic source evaluation publishes a provisional
    /// definition generation: only exact escaped definition owners survive its
    /// later rollback of named visibility.
    pub fn from_selected_product(
        canonical_v3: Arc<[u8]>,
        source_owner: SelectedProgramSourceOwner,
        selected: &crate::FunctionRegistry,
    ) -> Result<Self, String> {
        if !source_owner.is_live() {
            return Err("selected product source issuance has retired".into());
        }
        let admitted =
            super::admit_executable_unit(&canonical_v3).map_err(|error| error.to_string())?;
        if admitted.admitted_source_map() != source_owner.portable_map() {
            return Err("selected product source map differs from its issuing owner".into());
        }
        let program = admitted
            .envelope()
            .manifest
            .identity
            .program
            .identity_digest()
            .map_err(|error| error.to_string())?;
        let expected = admitted
            .retained_callable_origin_fingerprints()
            .map_err(|error| error.to_string())?;
        let bodies = selected
            .selected_execution_bodies()
            .into_iter()
            .filter(|body| {
                body.binding_origin
                    .is_some_and(|origin| origin.program == program)
                    && body
                        .selected_source_owner
                        .as_ref()
                        .is_some_and(|owner| owner.same_issuance(&source_owner))
                    && body
                        .selected_executable_product
                        .as_ref()
                        .is_some_and(|product| Arc::ptr_eq(product.canonical_v3(), &canonical_v3))
            })
            .collect::<Vec<_>>();
        let ids = bodies.iter().map(|body| body.function).collect::<Vec<_>>();
        let mut actual = std::collections::HashMap::new();
        for (origin, fingerprint) in selected
            .selected_runtime_body_fingerprints_with_imports(&ids, &admitted.fingerprint_imports())
            .map_err(|error| error.to_string())?
            .into_values()
        {
            if actual.insert(origin, fingerprint).is_some() {
                return Err("selected dynamic unit repeats one bound origin".into());
            }
        }
        if actual != expected {
            return Err("selected dynamic definitions differ from the admitted V3 unit".into());
        }
        for body in &bodies {
            if !body.source_id.is_some_and(|source| {
                source
                    .runtime()
                    .is_some_and(|runtime| source_owner.owns_source(runtime))
            }) {
                return Err("selected dynamic definition has no issued source member".into());
            }
        }
        Ok(Self {
            canonical_v3,
            source_owner,
            definitions: bodies
                .into_iter()
                .map(|body| Arc::clone(&body.definition_owner))
                .collect(),
        })
    }

    pub(crate) fn capture(unit: &RuntimeBoundExecutableUnit) -> Result<Self, &'static str> {
        unit.validate_runtime_owner(unit.runtime())?;
        let canonical_v3 = Arc::clone(
            unit.canonical_v3()
                .ok_or("staged imported unit has no canonical V3 product")?,
        );
        let source_owner = unit
            .selected_source_owner()
            .ok_or("staged imported unit has no selected source owner")?
            .clone();
        if !source_owner.is_issued_in(unit.runtime())
            || unit.selected_source_map() != Some(source_owner.portable_map())
        {
            return Err("staged imported unit differs from its issuing source map");
        }
        let definitions = unit
            .registry()
            .selected_execution_bodies()
            .into_iter()
            .map(|body| Arc::clone(&body.definition_owner))
            .collect();
        Ok(Self {
            canonical_v3,
            source_owner,
            definitions,
        })
    }

    pub fn canonical_v3(&self) -> &Arc<[u8]> {
        &self.canonical_v3
    }

    pub fn selected_source_owner(&self) -> &SelectedProgramSourceOwner {
        &self.source_owner
    }

    pub fn has_live_definition_owners(&self) -> bool {
        self.definitions.iter().any(|owner| owner.has_handles())
    }
}
