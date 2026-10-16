//! Exact compiler-folded source leaves, not permission to skip arbitrary producers.

use runmat_builtins::Type;
use runmat_builtins::SINGLE_CATALOG_ENTRY;
use runmat_execution::Digest;
use runmat_value::Value;

use crate::graph::{
    AccelGraph, AccelGraphTag, AccelNodeLabel, AccelOpCategory, FusionCallOrigin, ShapeInfo,
    ValueId, ValueOrigin,
};

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
        if producer.category != AccelOpCategory::Elementwise
            || producer.tags.as_slice()
                != [AccelGraphTag::Elementwise, AccelGraphTag::UnaryScalarFold]
            || info.shape != ShapeInfo::Tensor(vec![Some(1), Some(1)])
            || info.ty
                != (Type::Tensor {
                    shape: Some(vec![Some(1), Some(1)]),
                })
        {
            return None;
        }
        let AccelNodeLabel::Builtin {
            name,
            origin: FusionCallOrigin::Named { binding },
        } = &producer.label
        else {
            return None;
        };
        let entry = runmat_builtins::builtin_catalog_entry_by_name(name)?;
        if entry.identity != SINGLE_CATALOG_ENTRY.identity
            || entry.identity != binding.builtin
            || !entry
                .bindings
                .iter()
                .any(|variant| entry.binding_identity(variant) == *binding)
        {
            return None;
        }
        let [source] = producer.inputs.as_slice() else {
            return None;
        };
        let source_info = graph.value(*source)?;
        if !matches!(&source_info.origin, ValueOrigin::Constant) {
            return None;
        }
        let source = match &source_info.constant {
            Some(Value::Num(value)) => *value as f32,
            Some(Value::Bool(value)) => {
                if *value {
                    1.0
                } else {
                    0.0
                }
            }
            _ => return None,
        };
        let Value::Tensor(result) = represented else {
            return None;
        };
        if result.shape() != [1, 1] {
            return None;
        }
        let Some([result]) = result.as_f32_slice() else {
            return None;
        };
        if result.to_bits() != source.to_bits() {
            return None;
        }
    } else {
        return None;
    }
    crate::fusion::semantics::exact_constant_digest(represented).ok()
}
