use std::collections::HashMap;

use runmat_execution::value::{
    GraphBoundOwner, GraphLexicalLayout, GraphProgramUnitId, PayloadGraph, ValueLimits,
};
use runmat_execution::ExecutableSourceMapV2;
use runmat_value::Value;

use super::RuntimeBoundExecutableUnit;

#[cfg(test)]
#[path = "graph_codec/tests.rs"]
mod tests;

/// A source/layout projection issued only by one bound, admitted executable.
/// Payload metadata and destination display names cannot construct this owner.
pub struct GraphCodecAdmission<'a> {
    runtime: &'a runmat_runtime::context::RuntimeContext,
    source_map: &'a ExecutableSourceMapV2,
    layouts: HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: HashMap<GraphBoundOwner, runmat_runtime::execution::value_codec::GraphCallableProof>,
    bindings: HashMap<GraphBoundOwner, runmat_value::BoundFunctionBinding>,
}

/// Immutable callable facts from one already-selected registry. A deferred
/// snapshot may retain these without retaining its runtime or resolver.
pub struct SelectedGraphCallableFacts {
    layouts: HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: HashMap<GraphBoundOwner, runmat_runtime::execution::value_codec::GraphCallableProof>,
    bindings: HashMap<GraphBoundOwner, runmat_value::BoundFunctionBinding>,
}

impl SelectedGraphCallableFacts {
    pub fn into_parts(
        self,
    ) -> (
        HashMap<GraphBoundOwner, GraphLexicalLayout>,
        HashMap<GraphBoundOwner, runmat_runtime::execution::value_codec::GraphCallableProof>,
        HashMap<GraphBoundOwner, runmat_value::BoundFunctionBinding>,
    ) {
        (self.layouts, self.proofs, self.bindings)
    }
}

impl RuntimeBoundExecutableUnit {
    pub fn graph_codec_admission(&self) -> Result<GraphCodecAdmission<'_>, String> {
        self.validate_runtime_owner(&self.runtime)
            .map_err(str::to_owned)?;
        let source_map = self
            .selected_source_map()
            .ok_or("bound executable has no admitted portable source map")?;
        let SelectedGraphCallableFacts {
            layouts,
            proofs,
            bindings,
        } = selected_graph_callable_facts(&self.registry, GraphProgramUnitId(0))?;
        Ok(GraphCodecAdmission {
            runtime: &self.runtime,
            source_map,
            layouts,
            proofs,
            bindings,
        })
    }
}

#[path = "graph_codec/facts.rs"]
mod facts;
#[cfg(test)]
use facts::{capture_proof, nearest_activation_parent};
pub use facts::{selected_graph_callable_facts, selected_graph_callable_facts_for_owner};

impl GraphCodecAdmission<'_> {
    pub fn snapshot(
        &self,
        values: &[Value],
        limits: ValueLimits,
    ) -> Result<PayloadGraph, runmat_runtime::execution::value_codec::ValueCodecError> {
        runmat_runtime::execution::value_codec::encode_payload_graph_in_context(
            values,
            self.runtime,
            self.source_map,
            &self.layouts,
            &self.proofs,
            limits,
        )
    }

    pub fn reconstruct(
        &self,
        graph: &PayloadGraph,
        limits: ValueLimits,
    ) -> Result<
        runmat_runtime::execution::RootedValues,
        runmat_runtime::execution::value_codec::ValueCodecError,
    > {
        runmat_runtime::execution::value_codec::decode_payload_graph_in_context(
            graph,
            self.runtime,
            self.source_map,
            &self.layouts,
            &self.proofs,
            &self.bindings,
            limits,
        )
    }
}
