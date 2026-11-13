use std::collections::{BTreeMap, BTreeSet};

use runmat_hir::FunctionId;

use crate::{MirAssembly, MirCallee, MirOperand, MirRvalue, MirStmt, MirStmtKind};

/// Statically named lexical calls and explicit function handles both demand
/// the target's shared bindings in the caller's frame. Textual names do not.
pub(crate) fn lexical_reference_graph(
    assembly: &MirAssembly,
) -> BTreeMap<FunctionId, BTreeSet<FunctionId>> {
    let mut graph = BTreeMap::new();
    for (function, body) in &assembly.bodies {
        let targets = graph.entry(*function).or_default();
        for block in &body.blocks {
            for statement in &block.statements {
                collect_statement_calls(statement, targets);
                crate::rvalue::visit_stmt_operands(
                    statement,
                    &mut |operand| local_handle(operand, targets),
                    true,
                );
            }
            if let crate::MirTerminatorKind::For { iterable, .. }
            | crate::MirTerminatorKind::ParFor { iterable, .. } = &block.terminator.kind
            {
                collect_local_calls(iterable, targets);
            }
            if let crate::MirTerminatorKind::ParFor {
                maximum_workers: Some(maximum_workers),
                ..
            } = &block.terminator.kind
            {
                collect_local_calls(maximum_workers, targets);
            }
            if let crate::MirTerminatorKind::Spmd { header, .. } = &block.terminator.kind {
                match header.as_ref() {
                    crate::parallel::MirSpmdHeader::Default => {}
                    crate::parallel::MirSpmdHeader::One(first) => {
                        collect_local_calls(first, targets)
                    }
                    crate::parallel::MirSpmdHeader::Two(first, second) => {
                        collect_local_calls(first, targets);
                        collect_local_calls(second, targets);
                    }
                    crate::parallel::MirSpmdHeader::Three(first, second, third) => {
                        collect_local_calls(first, targets);
                        collect_local_calls(second, targets);
                        collect_local_calls(third, targets);
                    }
                }
            }
            let mut handle = |operand: &MirOperand| local_handle(operand, targets);
            match &block.terminator.kind {
                crate::MirTerminatorKind::Return(values) => {
                    for value in values {
                        handle(value);
                    }
                }
                crate::MirTerminatorKind::For { iterable, .. } => {
                    iterable.visit_operands(&mut handle);
                }
                crate::MirTerminatorKind::ParFor {
                    iterable,
                    maximum_workers,
                    ..
                } => {
                    iterable.visit_operands(&mut handle);
                    if let Some(maximum_workers) = maximum_workers {
                        maximum_workers.visit_operands(&mut handle);
                    }
                }
                crate::MirTerminatorKind::Branch { cond, .. } => handle(cond),
                crate::MirTerminatorKind::Switch { discr, cases, .. } => {
                    handle(discr);
                    for (value, _) in cases {
                        handle(value);
                    }
                }
                crate::MirTerminatorKind::Await { future, .. } => handle(future),
                crate::MirTerminatorKind::Spmd { header, .. } => match header.as_ref() {
                    crate::parallel::MirSpmdHeader::Default => {}
                    crate::parallel::MirSpmdHeader::One(first) => first.visit_operands(&mut handle),
                    crate::parallel::MirSpmdHeader::Two(first, second) => {
                        first.visit_operands(&mut handle);
                        second.visit_operands(&mut handle);
                    }
                    crate::parallel::MirSpmdHeader::Three(first, second, third) => {
                        first.visit_operands(&mut handle);
                        second.visit_operands(&mut handle);
                        third.visit_operands(&mut handle);
                    }
                },
                _ => {}
            }
        }
    }
    graph
}

fn local_handle(operand: &MirOperand, targets: &mut BTreeSet<FunctionId>) {
    if let MirOperand::FunctionHandle(runmat_hir::CallableIdentity::BoundFunction(function)) =
        operand
    {
        targets.insert(*function);
    }
}

fn collect_statement_calls(statement: &MirStmt, targets: &mut BTreeSet<FunctionId>) {
    match &statement.kind {
        MirStmtKind::PrepareFixedAssignment(owner) => {
            for step in owner.steps() {
                match step {
                    crate::MirExpressionStep::Let { value, .. } => {
                        collect_local_calls(value, targets)
                    }
                    crate::MirExpressionStep::CaptureSequence { source, .. } => {
                        collect_expansion_calls(source, targets)
                    }
                }
            }
            owner.visit_expression_regions(|region| {
                for step in region.steps() {
                    match step {
                        crate::MirExpressionStep::Let { value, .. } => {
                            collect_local_calls(value, targets)
                        }
                        crate::MirExpressionStep::CaptureSequence { source, .. } => {
                            collect_expansion_calls(source, targets)
                        }
                    }
                }
            });
        }
        MirStmtKind::PreparedAssign { value, .. }
        | MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::SequenceAssign { value, .. }
        | MirStmtKind::Expr(value) => collect_local_calls(value, targets),
        MirStmtKind::CaptureSequence { source, .. } => collect_expansion_calls(source, targets),
        _ => {}
    }
}

fn collect_expansion_calls(source: &crate::MirExpansionSource, targets: &mut BTreeSet<FunctionId>) {
    source.visit_direct_expression_steps_dyn(&mut |steps| {
        for step in steps {
            match step {
                crate::MirExpressionStep::Let { value, .. } => collect_local_calls(value, targets),
                crate::MirExpressionStep::CaptureSequence { source, .. } => {
                    collect_expansion_calls(source, targets)
                }
            }
        }
    });
}

fn collect_local_calls(value: &MirRvalue, targets: &mut BTreeSet<FunctionId>) {
    if !matches!(value, MirRvalue::ShortCircuit { .. }) {
        value.visit_direct_expression_steps_dyn(&mut |steps| {
            for step in steps {
                match step {
                    crate::MirExpressionStep::Let { value, .. } => {
                        collect_local_calls(value, targets)
                    }
                    crate::MirExpressionStep::CaptureSequence { source, .. } => {
                        collect_expansion_calls(source, targets);
                    }
                }
            }
        });
    }
    match value {
        MirRvalue::Call(call) => {
            if let MirCallee::Static(runmat_hir::CallableIdentity::BoundFunction(function)) =
                &call.callee
            {
                targets.insert(*function);
            }
        }
        MirRvalue::Future { function, .. } => {
            targets.insert(*function);
        }
        MirRvalue::ShortCircuit { right_temps, .. } => {
            for statement in right_temps {
                collect_statement_calls(statement, targets);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contextual_source(
        statements: Vec<MirStmt>,
        result: MirOperand,
    ) -> crate::MirExpansionSource {
        let region = crate::MirExpressionRegion::from_lowered(statements, result)
            .expect("valid contextual selector region");
        crate::MirExpansionSource::CellContents {
            base: MirOperand::Constant(crate::MirConstant::EmptyArray),
            indexing: crate::MirIndexing {
                argument_spans: None,
                kind: runmat_hir::IndexKind::Brace,
                plan: crate::MirIndexPlan::Cell,
                components: vec![crate::MirIndexComponent::ContextualExpr(region)],
                result_context: runmat_hir::IndexResultContext::ReadCommaList,
                cell_expand_all: false,
            },
        }
    }

    fn nested_selector(
        source: crate::MirExpansionSource,
        sequence: usize,
        contextual_local: usize,
        local: usize,
    ) -> crate::MirExpansionSource {
        contextual_source(
            vec![
                MirStmt {
                    kind: MirStmtKind::Assign {
                        place: crate::MirPlace::Local(crate::MirLocalId(contextual_local)),
                        value: MirRvalue::End,
                    },
                    span: runmat_hir::Span::default(),
                },
                MirStmt {
                    kind: MirStmtKind::CaptureSequence {
                        destination: crate::MirSequenceLocalId(sequence),
                        source,
                    },
                    span: runmat_hir::Span::default(),
                },
                MirStmt {
                    kind: MirStmtKind::Assign {
                        place: crate::MirPlace::Local(crate::MirLocalId(local)),
                        value: MirRvalue::Aggregate {
                            kind: crate::MirAggregateKind::Tensor,
                            row_lengths: vec![2],
                            elements: vec![
                                crate::MirAggregateElement::Single(MirOperand::Local(
                                    crate::MirLocalId(contextual_local),
                                )),
                                crate::MirAggregateElement::CapturedSequence(
                                    crate::MirSequenceLocalId(sequence),
                                ),
                            ],
                        },
                    },
                    span: runmat_hir::Span::default(),
                },
            ],
            MirOperand::Local(crate::MirLocalId(local)),
        )
    }

    fn nested_capture_depth(source: &crate::MirExpansionSource) -> usize {
        let mut depth = 0;
        source.visit_direct_expression_steps_dyn(&mut |steps| {
            for step in steps {
                if let crate::MirExpressionStep::CaptureSequence { source, .. } = step {
                    depth = depth.max(1 + nested_capture_depth(source));
                }
            }
        });
        depth
    }

    #[test]
    fn deeply_nested_sequence_preparation_retains_lexical_call_edge() {
        let parsed = runmat_parser::parse(
            "function out = outer(values)\n shared = 1;\n out = shared;\n function result = inner(index)\n result = shared + index;\n end\n end",
        )
        .expect("parse nested preparation");
        let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
            .expect("lower nested preparation");
        let mut mir =
            crate::lowering::lower_assembly(&hir.assembly).expect("lower nested preparation MIR");
        let outer = mir
            .functions
            .iter()
            .find_map(|(id, metadata)| (metadata.name.0 == "outer").then_some(*id))
            .expect("outer function");
        let inner = mir
            .functions
            .iter()
            .find_map(|(id, metadata)| (metadata.name.0 == "inner").then_some(*id))
            .expect("nested function");
        let body = mir.bodies.get_mut(&outer).expect("outer MIR body");
        assert!(body
            .locals
            .iter()
            .enumerate()
            .all(|(index, local)| local.id.0 == index));
        let first_private = body.locals.len();
        for offset in 0..6 {
            body.locals.push(crate::MirLocal {
                id: crate::MirLocalId(first_private + offset),
                binding: None,
                kind: crate::MirLocalKind::Temporary,
                span: runmat_hir::Span::default(),
            });
        }
        // Source lowering may flatten selectors into adjacent statements. Build
        // an actual validated MIR selector region whose preparation is nested
        // twice, so this tests the graph visitor's recursive branch itself.
        let leaf = contextual_source(
            vec![
                MirStmt {
                    kind: MirStmtKind::Assign {
                        place: crate::MirPlace::Local(crate::MirLocalId(first_private)),
                        value: MirRvalue::End,
                    },
                    span: runmat_hir::Span::default(),
                },
                MirStmt {
                    kind: MirStmtKind::Assign {
                        place: crate::MirPlace::Local(crate::MirLocalId(first_private + 1)),
                        value: MirRvalue::Call(crate::MirCall {
                            callee: MirCallee::Static(runmat_hir::CallableIdentity::BoundFunction(
                                inner,
                            )),
                            args: vec![crate::MirCallArg::Single(MirOperand::Local(
                                crate::MirLocalId(first_private),
                            ))],
                            arg_spans: vec![runmat_hir::Span::default()],
                            syntax: runmat_hir::CallSyntax::Command,
                            requested_outputs: runmat_hir::RequestedOutputCount::Exactly(1),
                            fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                            workspace_first_name: None,
                            bare_identifier: false,
                            async_behavior: crate::AsyncBehaviorFact::MaySuspend,
                            effects: runmat_builtins::BuiltinEffects::none(),
                            workspace_effect: None,
                            environment_effect: None,
                            purity: runmat_builtins::BuiltinPurity::Impure,
                            semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
                        }),
                    },
                    span: runmat_hir::Span::default(),
                },
            ],
            MirOperand::Local(crate::MirLocalId(first_private + 1)),
        );
        let source = nested_selector(
            nested_selector(leaf, 0, first_private + 2, first_private + 3),
            1,
            first_private + 4,
            first_private + 5,
        );
        body.blocks[0].statements.push(MirStmt {
            kind: MirStmtKind::CaptureSequence {
                destination: crate::MirSequenceLocalId(2),
                source,
            },
            span: runmat_hir::Span::default(),
        });
        body.blocks[0].statements.push(MirStmt {
            kind: MirStmtKind::Expr(MirRvalue::Aggregate {
                kind: crate::MirAggregateKind::Tensor,
                row_lengths: vec![1],
                elements: vec![crate::MirAggregateElement::CapturedSequence(
                    crate::MirSequenceLocalId(2),
                )],
            }),
            span: runmat_hir::Span::default(),
        });
        body.validate_expression_regions()
            .expect("nested preparation is a valid MIR body");
        let deepest = mir.bodies[&outer]
            .blocks
            .iter()
            .flat_map(|block| &block.statements)
            .filter_map(|statement| match &statement.kind {
                MirStmtKind::CaptureSequence { source, .. } => Some(nested_capture_depth(source)),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        assert!(deepest >= 2, "fixture must enter two nested captures");
        assert!(lexical_reference_graph(&mir)[&outer].contains(&inner));
    }

    #[test]
    fn external_function_id_collision_is_not_a_lexical_capture_edge() {
        let parsed = runmat_parser::parse(
            "function y = outer()\n shared = 1;\n y = caller();\n function z = reader()\n z = shared;\n end\n function z = caller()\n z = reader();\n end\n end",
        )
        .expect("parse lexical fixture");
        let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
            .expect("lower lexical fixture");
        let mut mir = crate::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
        let by_name = mir
            .functions
            .iter()
            .map(|(id, metadata)| (metadata.name.0.clone(), *id))
            .collect::<BTreeMap<_, _>>();
        let caller = by_name["caller"];
        let reader = by_name["reader"];
        assert!(lexical_reference_graph(&mir)[&caller].contains(&reader));

        let mut replaced = false;
        for statement in mir
            .bodies
            .get_mut(&caller)
            .expect("caller body")
            .blocks
            .iter_mut()
            .flat_map(|block| block.statements.iter_mut())
        {
            if let MirStmtKind::Assign {
                value: MirRvalue::Call(call),
                ..
            } = &mut statement.kind
            {
                call.callee = MirCallee::Static(runmat_hir::CallableIdentity::ExternalFunction {
                    function: reader,
                    display_name: "foreign.reader".into(),
                });
                replaced = true;
            }
        }
        assert!(replaced, "fixture has a static caller site");
        assert!(
            !lexical_reference_graph(&mir)[&caller].contains(&reader),
            "an external identity with a colliding ordinal is not a lexical edge"
        );
    }
}
