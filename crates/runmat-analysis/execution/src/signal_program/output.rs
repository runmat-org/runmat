//! Transfer one already-committed worker result into generator canonicalization.
//! Scheduler/response admission belongs to the existing execution owner; this
//! adapter neither creates that authority nor replays the callable.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, GeneratorChunkOutput};
use runmat_execution::{
    value::{GraphNode, ValueLimits},
    Digest,
};
use runmat_execution_runner::AttemptSuccess;

/// Consume one already-committed scalar-output result without a numerical
/// payload copy, ready for the original generator's canonical shape checks.
pub fn generator_chunk_output_from_success(
    identity: Digest,
    success: AttemptSuccess,
) -> AnalysisBridgeResult<GeneratorChunkOutput> {
    let invalid = || {
        AnalysisBridgeError::Invalid(
            "generator worker must return one closed portable numeric value".into(),
        )
    };
    let value = match success {
        AttemptSuccess::Graph { graph, .. } => {
            let [root] = graph.roots.as_slice() else {
                return Err(invalid());
            };
            let root = root.0 as usize;
            let Some(GraphNode::Leaf { value }) = graph.nodes.into_iter().nth(root) else {
                return Err(invalid());
            };
            value
        }
        success => {
            let Some((mut values, objects)) = success.into_values() else {
                return Err(invalid());
            };
            if values.len() != 1 || !objects.is_empty() {
                return Err(invalid());
            }
            values.remove(0)
        }
    };
    value.validate(ValueLimits::default())?;
    Ok(GeneratorChunkOutput {
        chunk_identity: identity,
        value,
    })
}
