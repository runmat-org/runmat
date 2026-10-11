use std::{collections::HashMap, sync::Arc};

use super::{FunctionBytecode, FunctionRegistry};
use runmat_hir::FunctionId;

impl FunctionRegistry {
    /// Capture exact selected-definition name facts while source authority is
    /// installed. The callback later reads only these facts and never
    /// reconstructs a reused numeric ID's path from a newer publication.
    pub fn semantic_name_candidate_facts_in(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        functions: impl IntoIterator<Item = FunctionId>,
    ) -> HashMap<FunctionId, runmat_runtime::user_functions::SemanticNameCandidate> {
        functions
            .into_iter()
            .filter_map(|id| {
                let body = self.get(id)?;
                Some((
                    id,
                    runmat_runtime::user_functions::SemanticNameCandidate {
                        function: id.0,
                        source: self.execution_source_fact_in(runtime, id),
                        definition_owner: Some(Arc::clone(&body.definition_owner)),
                    },
                ))
            })
            .collect()
    }

    /// Facts from the selected executable body, never from a same-numbered
    /// later publication. The dependency graph is issued on demand only when
    /// a callable value is created.
    pub fn execution_source_fact_in(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        function: FunctionId,
    ) -> Option<runmat_runtime::user_functions::SourceFunctionInfo> {
        let body = self.get(function)?;
        source_fact(runtime, body, None)
    }

    /// Resolve an immutable portable origin only when it identifies exactly
    /// one body in this selected registry. Display names and destination-local
    /// function numbers are not origin authority.
    pub fn execution_source_fact_for_origin_in(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        origin: runmat_execution::FunctionBindingOrigin,
    ) -> Option<runmat_runtime::user_functions::SourceFunctionInfo> {
        let mut bodies = self
            .functions
            .values()
            .filter(|body| body.binding_origin == Some(origin));
        let body = bodies.next()?;
        if bodies.next().is_some() {
            return None;
        }
        source_fact(runtime, body, None)
    }

    pub fn execution_definition_owners(
        &self,
        function: FunctionId,
    ) -> Option<runmat_runtime::user_functions::DefinitionOwnerGraph> {
        self.definition_dependency_owners(function).map(owners_arc)
    }

    /// Source members share one immutable owner graph. A retained view only
    /// enumerates definitions captured by that view, never published replacements.
    pub fn execution_source_group_facts_in(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        source: runmat_types::SourceId,
    ) -> Vec<runmat_runtime::user_functions::SourceFunctionInfo> {
        let Some(source) =
            runmat_runtime::source_context::issued_source_ref_in_context(runtime, source)
        else {
            return Vec::new();
        };
        let bodies = self.execution_source_group(runtime, source);
        let owners = bodies
            .first()
            .and_then(|body| self.definition_dependency_owners(body.function))
            .map(owners_arc);
        bodies
            .into_iter()
            .filter_map(|body| source_fact(runtime, body, owners.clone()))
            .collect()
    }
}

fn owners_arc(
    owners: Vec<(FunctionId, Arc<runmat_value::FunctionDefinitionOwner>)>,
) -> runmat_runtime::user_functions::DefinitionOwnerGraph {
    Arc::new(
        owners
            .into_iter()
            .map(|(id, owner)| (id.0, owner))
            .collect(),
    )
}

fn source_fact(
    runtime: &runmat_runtime::context::RuntimeContext,
    body: &FunctionBytecode,
    definition_owners: Option<runmat_runtime::user_functions::DefinitionOwnerGraph>,
) -> Option<runmat_runtime::user_functions::SourceFunctionInfo> {
    let source_id = body.source_id?.resolve_in(runtime)?;
    let source_file = runmat_runtime::source_context::source_info_in_context(runtime, source_id)
        .and_then(|source| source.fullpath_name.map(|path| path.to_string()))
        .or_else(|| body.source_file.clone());
    Some(runmat_runtime::user_functions::SourceFunctionInfo {
        source_id,
        name: body.display_name.clone(),
        function: body.function.0,
        role: body.source_role,
        binding_origin: body.binding_origin,
        source_file,
        definition_owners,
    })
}
