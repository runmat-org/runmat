use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_types::SourceId;

use super::{FunctionBytecode, FunctionRegistry};

impl FunctionRegistry {
    /// Facts from the selected executable body, never from a same-numbered
    /// later publication. The dependency graph is issued on demand only when
    /// a callable value is created.
    pub fn execution_source_fact(
        &self,
        function: FunctionId,
    ) -> Option<runmat_runtime::user_functions::SourceFunctionInfo> {
        let body = self.get(function)?;
        source_fact(body, None)
    }

    pub fn execution_definition_owners(
        &self,
        function: FunctionId,
    ) -> Option<runmat_runtime::user_functions::DefinitionOwnerGraph> {
        self.definition_dependency_owners(function).map(owners_arc)
    }

    /// Source members share one immutable owner graph. A retained view only
    /// enumerates definitions captured by that view, never published replacements.
    pub fn execution_source_group_facts(
        &self,
        source: SourceId,
    ) -> Vec<runmat_runtime::user_functions::SourceFunctionInfo> {
        let bodies = self.execution_source_group(source);
        let owners = bodies
            .first()
            .and_then(|body| self.definition_dependency_owners(body.function))
            .map(owners_arc);
        bodies
            .into_iter()
            .filter_map(|body| source_fact(body, owners.clone()))
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
    body: &FunctionBytecode,
    definition_owners: Option<runmat_runtime::user_functions::DefinitionOwnerGraph>,
) -> Option<runmat_runtime::user_functions::SourceFunctionInfo> {
    Some(runmat_runtime::user_functions::SourceFunctionInfo {
        source_id: body.source_id?,
        name: body.display_name.clone(),
        function: body.function.0,
        role: body.source_role,
        binding_origin: body.binding_origin,
        source_file: body.source_file.clone(),
        definition_owners,
    })
}
