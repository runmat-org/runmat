//! Exact compiler-folded source leaves, not permission to skip arbitrary producers.

use crate::graph::{AccelGraph, AccelGraphTag, ValueId, ValueOrigin};
use runmat_execution::Digest;

/// Admit only a compiler-marked immutable producer that precedes this use.
/// The represented payload is reconstructed from exact source literals before
/// it can become a digest-bearing external leaf.
pub(super) fn digest(graph: &AccelGraph, id: ValueId, before_pc: usize) -> Option<Digest> {
    let info = graph.value(id)?;
    let ValueOrigin::NodeOutput { node, output: 0 } = &info.origin else {
        return None;
    };
    let producer = graph.node(*node)?;
    if producer.id != *node
        || producer.outputs.as_slice() != [id]
        || producer.span.start > producer.span.end
        || producer.span.end >= before_pc
    {
        return None;
    }
    let represented = info.constant.as_ref()?;
    if producer.tags.as_slice() == [AccelGraphTag::FoldedLiteralArray] {
        graph.exact_folded_literal_array(id)?;
    } else if producer.tags.contains(&AccelGraphTag::UnaryScalarFold) {
        graph.exact_folded_unary_scalar(id)?;
    } else {
        return None;
    }
    crate::fusion::semantics::exact_constant_digest(represented).ok()
}
