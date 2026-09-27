use std::collections::HashMap;

use runmat_builtins::BuiltinCatalogEntry;
use runmat_runtime::builtins::common::spec::{
    canonical_fusion_spec, FusionExprContext, ScalarType,
};

use crate::graph::ValueId;

/// Canonical expression declarations are authoritative. A missing or ambiguous
/// declaration must not fall through to the legacy name-based implementation.
pub(super) fn canonical_expression(
    entry: &BuiltinCatalogEntry,
    inputs: &[ValueId],
    expressions: &HashMap<ValueId, String>,
    scalar_ty: ScalarType,
) -> Option<String> {
    let [binding] = entry.bindings else {
        return None;
    };
    let spec = canonical_fusion_spec(entry.binding_identity(binding))?;
    let template = spec.elementwise.as_ref()?;
    if !template.scalar_precisions.contains(&scalar_ty) {
        return None;
    }
    let inputs = inputs
        .iter()
        .map(|id| expressions.get(id).map(String::as_str))
        .collect::<Option<Vec<_>>>()?;
    (template.wgsl_body)(&FusionExprContext {
        scalar_ty,
        inputs: &inputs,
        constants: &[],
    })
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        AccelGraph, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, ShapeInfo, ValueInfo,
        ValueOrigin,
    };

    #[test]
    fn canonical_single_rounds_even_inside_a_double_kernel() {
        let entry = runmat_builtins::builtin_catalog_entry_by_name("single").unwrap();
        let expressions = HashMap::from([(0, "source".to_owned())]);
        assert_eq!(
            canonical_expression(entry, &[0], &expressions, ScalarType::F64).as_deref(),
            Some("f64(f32(source))")
        );
        assert_eq!(
            canonical_expression(entry, &[0], &expressions, ScalarType::F32).as_deref(),
            Some("source")
        );
    }

    #[test]
    fn constant_and_value_tracing_do_not_erase_conversion_rounding() {
        let mut graph = AccelGraph {
            nodes: Vec::new(),
            values: vec![ValueInfo {
                id: 0,
                origin: ValueOrigin::Constant,
                ty: runmat_builtins::Type::Num,
                shape: ShapeInfo::Scalar,
                constant: Some(runmat_value::Value::Num(16_777_217.0)),
            }],
            var_bindings: HashMap::new(),
            node_bindings: HashMap::new(),
        };
        for (node_id, name) in ["single", "double"].into_iter().enumerate() {
            let node_id = node_id as u32;
            graph.nodes.push(AccelNode {
                id: node_id,
                label: AccelNodeLabel::Builtin { name: name.into() },
                category: AccelOpCategory::Elementwise,
                inputs: vec![node_id],
                outputs: vec![node_id + 1],
                span: InstrSpan { start: 0, end: 0 },
                tags: vec![],
            });
            graph.values.push(ValueInfo {
                id: node_id + 1,
                origin: ValueOrigin::NodeOutput {
                    node: node_id,
                    output: 0,
                },
                ty: runmat_builtins::Type::Num,
                shape: ShapeInfo::Scalar,
                constant: None,
            });
        }
        assert_eq!(
            super::super::resolve_scalar_constant(&graph, 0),
            Some(16_777_217.0)
        );
        assert!(super::super::resolve_scalar_constant(&graph, 1).is_none());
        assert!(super::super::resolve_scalar_constant(&graph, 2).is_none());
    }
}
