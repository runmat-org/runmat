#[cfg(test)]
use std::collections::HashMap;

use runmat_builtins::BuiltinCatalogEntry;
use runmat_runtime::builtins::common::spec::{
    canonical_fusion_declaration, CanonicalFusionDeclaration, FusionError, FusionExprContext,
    FusionExprInput, ScalarType,
};

use super::ExpressionLookup;
use crate::graph::ValueId;

/// Canonical expression declarations are authoritative. A missing or ambiguous
/// declaration must not fall through to the legacy name-based implementation.
pub(super) fn canonical_expression(
    entry: &BuiltinCatalogEntry,
    inputs: &[ValueId],
    expressions: &impl ExpressionLookup,
    scalar_ty: ScalarType,
) -> Option<String> {
    let [binding] = entry.bindings else {
        return None;
    };
    let declaration = canonical_fusion_declaration(entry.binding_identity(binding))?;
    // Untyped legacy/reduction emission has no operation-owned admission or
    // helper publication boundary. It cannot bypass a declared requirement.
    if declaration
        .semantics
        .is_some_and(|semantics| semantics.admit.is_some() || !semantics.helpers.is_empty())
    {
        return None;
    }
    admitted_expression(declaration, inputs, expressions, scalar_ty)
}

pub(super) fn admitted_expression(
    declaration: CanonicalFusionDeclaration,
    inputs: &[ValueId],
    expressions: &impl ExpressionLookup,
    scalar_ty: ScalarType,
) -> Option<String> {
    admitted_expression_checked(declaration, inputs, expressions, scalar_ty)
        .ok()
        .flatten()
}

/// Selected typed producers retain allocation/overflow errors so finite
/// admission cannot mistake resource failure for an unsupported declaration.
pub(super) fn admitted_expression_checked(
    declaration: CanonicalFusionDeclaration,
    inputs: &[ValueId],
    expressions: &impl ExpressionLookup,
    scalar_ty: ScalarType,
) -> Result<Option<String>, FusionError> {
    let Some(template) = declaration.spec.elementwise.as_ref() else {
        return Ok(None);
    };
    if !template.scalar_precisions.contains(&scalar_ty) {
        return Ok(None);
    }
    // The finite producer admits at most two scalar operands. Its canonical
    // declaration writer receives borrowed stack views, not an extra Vec.
    if inputs.len() <= 2 {
        let mut projected = [FusionExprInput::text(""); 2];
        for (slot, id) in projected.iter_mut().zip(inputs) {
            let Some(expression) = expressions.get(id) else {
                return Ok(None);
            };
            *slot = FusionExprInput::text(expression);
        }
        return template
            .wgsl_body
            .render(&FusionExprContext {
                scalar_ty,
                inputs: &projected[..inputs.len()],
                constants: &[],
            })
            .map(Some);
    }
    // Wider legacy declarations retain their established admission policy;
    // the finite producer has no positive byte certificate for this branch.
    let Some(inputs) = inputs
        .iter()
        .map(|id| expressions.get(id).map(|text| FusionExprInput::text(text)))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    template
        .wgsl_body
        .render(&FusionExprContext {
            scalar_ty,
            inputs: &inputs,
            constants: &[],
        })
        .map(Some)
}

pub(super) fn structural_support(plan: &super::FusionGroupPlan) -> bool {
    plan.operations
        .iter()
        .all(|operation| structural_operation_support(plan, operation))
}

pub(super) fn structural_operation_support(
    plan: &super::FusionGroupPlan,
    operation: &super::FusionOp,
) -> bool {
    match operation {
        super::FusionOp::Primitive { .. } => true,
        super::FusionOp::Builtin { name, .. } => {
            let Some(entry) = runmat_builtins::builtin_catalog_entry_by_name(name) else {
                return plan.generate_wgsl("f32").is_some() || plan.generate_wgsl("f64").is_some();
            };
            let [binding] = entry.bindings else {
                return false;
            };
            canonical_fusion_declaration(entry.binding_identity(binding)).is_some_and(
                |declaration| {
                    declaration
                        .spec
                        .elementwise
                        .as_ref()
                        .is_some_and(|template| !template.scalar_precisions.is_empty())
                        || declaration
                            .semantics
                            .and_then(|semantics| semantics.scalar_operation)
                            == Some(runmat_types::ScalarOperation::Operator(
                                runmat_types::OperatorKind::MatrixMultiply,
                            ))
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        AccelGraph, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, ShapeInfo, ValueInfo,
        ValueOrigin,
    };

    #[test]
    fn remainder_requires_helpers_but_has_an_admitted_typed_expression() {
        let expressions = HashMap::from([(0, "left".to_owned()), (1, "right".to_owned())]);
        for (name, modulo) in [("mod", true), ("rem", false)] {
            let entry = runmat_builtins::builtin_catalog_entry_by_name(name).unwrap();
            let [binding] = entry.bindings else {
                panic!("expected a single canonical remainder binding");
            };
            let declaration =
                canonical_fusion_declaration(entry.binding_identity(binding)).unwrap();
            let helper =
                runmat_runtime::builtins::common::spec::FusionMathHelper::FloatingRemainder;
            assert!(declaration.semantics.unwrap().helpers.contains(&helper));
            for precision in [ScalarType::F32, ScalarType::F64] {
                assert!(canonical_expression(entry, &[0, 1], &expressions, precision).is_none());
                assert_eq!(
                    admitted_expression(declaration, &[0, 1], &expressions, precision),
                    Some(format!(
                        "{}(left, right, {modulo})",
                        helper.entrypoint(precision).unwrap()
                    ))
                );
            }
        }
    }

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
                label: AccelNodeLabel::Builtin {
                    name: name.into(),
                    origin: Default::default(),
                },
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
