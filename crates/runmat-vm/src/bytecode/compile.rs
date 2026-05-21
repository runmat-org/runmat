#[cfg(feature = "native-accel")]
use crate::accel::graph::build_accel_graph;
#[cfg(feature = "native-accel")]
use crate::accel::stack_layout::annotate_fusion_groups_with_stack_layout;
#[cfg(feature = "native-accel")]
use crate::bytecode::instr::Instr;
use crate::bytecode::program::SemanticFunctionBytecode;
use crate::bytecode::{Bytecode, SemanticFunctionRegistry};
use crate::compiler::{CompileError, Compiler};
use crate::layout::derive_layout;
#[cfg(feature = "native-accel")]
use runmat_builtins::{builtin_functions, AccelTag, BuiltinSemanticKind};
use runmat_hir::{EntrypointId, FunctionId, HirAssembly};
use runmat_mir::MirAssembly;
use runmat_mir::{MirRvalue, MirStmtKind, MirTerminatorKind};
use std::collections::HashMap;
#[cfg(feature = "native-accel")]
use std::collections::HashSet;

pub fn compile(
    hir: &HirAssembly,
    mir: &MirAssembly,
    entrypoint: EntrypointId,
) -> Result<Bytecode, CompileError> {
    let layout = derive_layout(hir, mir)
        .map_err(|err| CompileError::new(format!("failed to derive VM layout: {err:?}")))?;
    let mut c = Compiler::new(hir, mir, layout, entrypoint)?;
    c.compile()?;
    let semantic_functions =
        compile_semantic_functions(hir, mir, c.layout.as_ref().unwrap(), Some(entrypoint))?;
    let semantic_function_registry = SemanticFunctionRegistry::new(semantic_functions.clone());
    let var_names = c
        .layout
        .as_ref()
        .and_then(|layout| layout.entrypoints.get(&entrypoint))
        .map(|entrypoint_layout| {
            entrypoint_layout
                .exports
                .iter()
                .map(|export| (export.slot.0, export.name.clone()))
                .collect()
        })
        .unwrap_or_default();
    let entrypoint_target = hir
        .entrypoints
        .iter()
        .find(|candidate| candidate.id == entrypoint)
        .map(|candidate| candidate.target);
    #[cfg(feature = "native-accel")]
    let mut semantic_fusion_metadata = derive_semantic_fusion_metadata(mir, entrypoint_target);
    let semantic_instruction_windows = derive_semantic_fusion_instruction_windows(
        &c.instructions,
        &c.instr_spans,
        &semantic_fusion_metadata.mir_fusion_candidate_groups,
    );
    semantic_fusion_metadata.semantic_instruction_window_count = semantic_instruction_windows.len();
    semantic_fusion_metadata.semantic_instruction_windows = semantic_instruction_windows.clone();
    #[cfg(feature = "native-accel")]
    let (accel_graph, fusion_groups) = if semantic_fusion_metadata.mir_fusion_candidate_group_count
        == 0
        || semantic_instruction_windows.is_empty()
    {
        (None, Vec::new())
    } else {
        let accel_graph = build_accel_graph(&c.instructions, &c.var_types);
        // Compile-time ownership is semantic-window scaffolding only; runtime fusion plan
        // preparation performs node reconciliation against the accel graph.
        let mut fusion_groups =
            derive_semantic_fusion_groups_from_instruction_windows(&semantic_instruction_windows);
        if !fusion_groups.is_empty() {
            annotate_fusion_groups_with_stack_layout(
                &c.instructions,
                &accel_graph,
                &mut fusion_groups,
            );
            fusion_groups.retain(|group| {
                fusion_group_within_semantic_candidate_spans(
                    group,
                    &c.instr_spans,
                    &semantic_fusion_metadata.mir_fusion_candidate_groups,
                )
            });
        }
        // Preserve accel graph whenever semantic candidate/window scaffolds exist.
        // Runtime planning owns final executable-group reconciliation and may still
        // recover groups from semantic windows when compile groups are empty.
        (Some(accel_graph), fusion_groups)
    };
    let semantic_async_metadata = derive_semantic_async_metadata(mir, entrypoint_target);

    Ok(Bytecode {
        instructions: c.instructions,
        instr_spans: c.instr_spans,
        call_arg_spans: c.call_arg_spans,
        source_id: None,
        var_count: c.var_count,
        semantic_functions,
        semantic_function_registry,
        var_types: c.var_types,
        var_names,
        layout: c.layout,
        semantic_async_metadata,
        #[cfg(feature = "native-accel")]
        accel_graph,
        #[cfg(feature = "native-accel")]
        fusion_groups,
        #[cfg(feature = "native-accel")]
        semantic_fusion_metadata,
    })
}

fn derive_semantic_async_metadata(
    mir: &MirAssembly,
    entrypoint_target: Option<FunctionId>,
) -> crate::bytecode::SemanticAsyncMetadata {
    let mut spawn_sites = Vec::new();
    let mut await_sites = Vec::new();
    let mut function_ids: Vec<_> = if let Some(function) = entrypoint_target {
        vec![function]
    } else {
        mir.bodies.keys().copied().collect()
    };
    function_ids.sort_by_key(|id| id.0);
    for function_id in function_ids {
        let Some(body) = mir.bodies.get(&function_id) else {
            continue;
        };
        for block in &body.blocks {
            for (stmt_index, stmt) in block.statements.iter().enumerate() {
                let value = match &stmt.kind {
                    MirStmtKind::Assign { value, .. }
                    | MirStmtKind::MultiAssign { value, .. }
                    | MirStmtKind::Expr(value) => value,
                    MirStmtKind::PlaceMutation(_)
                    | MirStmtKind::WorkspaceEffect { .. }
                    | MirStmtKind::EnvironmentEffect(_) => continue,
                };
                if matches!(value, MirRvalue::Spawn(_)) {
                    spawn_sites.push(crate::bytecode::SemanticSpawnSite {
                        function: body.function,
                        block: block.id,
                        stmt_index,
                    });
                }
            }
            if let MirTerminatorKind::Await { resume, .. } = &block.terminator.kind {
                await_sites.push(crate::bytecode::SemanticAwaitSite {
                    function: body.function,
                    block: block.id,
                    resume: *resume,
                });
            }
        }
    }
    crate::bytecode::SemanticAsyncMetadata {
        mir_spawn_site_count: spawn_sites.len(),
        mir_spawn_sites: spawn_sites,
        mir_await_site_count: await_sites.len(),
        mir_await_sites: await_sites,
        runtime_model:
            crate::bytecode::program::SemanticAsyncRuntimeModel::LazyFutureDescriptorLane,
    }
}

#[cfg(feature = "native-accel")]
fn derive_semantic_fusion_metadata(
    mir: &MirAssembly,
    entrypoint_target: Option<FunctionId>,
) -> crate::bytecode::SemanticFusionMetadata {
    let (mir_fusion_signal_count, mir_fusion_candidate_groups) =
        derive_semantic_fusion_candidate_groups(mir, entrypoint_target);
    crate::bytecode::SemanticFusionMetadata {
        mir_fusion_signal_count,
        mir_fusion_candidate_group_count: mir_fusion_candidate_groups.len(),
        mir_fusion_candidate_groups,
        semantic_instruction_window_count: 0,
        semantic_instruction_windows: Vec::new(),
    }
}

#[cfg(feature = "native-accel")]
fn derive_semantic_fusion_candidate_groups(
    mir: &MirAssembly,
    entrypoint_target: Option<FunctionId>,
) -> (usize, Vec<crate::bytecode::SemanticFusionCandidateGroup>) {
    let mut signal_count = 0usize;
    let mut groups = Vec::new();
    let mut function_ids: Vec<_> = if let Some(function) = entrypoint_target {
        vec![function]
    } else {
        mir.bodies.keys().copied().collect()
    };
    function_ids.sort_by_key(|id| id.0);
    for function_id in function_ids {
        let Some(body) = mir.bodies.get(&function_id) else {
            continue;
        };
        for block in &body.blocks {
            let mut run_len = 0usize;
            for (stmt_index, stmt) in block.statements.iter().enumerate() {
                let value = match &stmt.kind {
                    MirStmtKind::Assign { value, .. }
                    | MirStmtKind::MultiAssign { value, .. }
                    | MirStmtKind::Expr(value) => value,
                    MirStmtKind::PlaceMutation(_)
                    | MirStmtKind::WorkspaceEffect { .. }
                    | MirStmtKind::EnvironmentEffect(_) => {
                        if run_len >= 2 {
                            let stmt_start = stmt_index - run_len;
                            let stmt_end = stmt_index;
                            groups.push(crate::bytecode::SemanticFusionCandidateGroup {
                                id: groups.len(),
                                signal_count: run_len,
                                function: body.function,
                                block: block.id,
                                stmt_start,
                                stmt_end,
                                source_span: merge_stmt_run_span(block, stmt_start, stmt_end),
                            });
                        }
                        run_len = 0;
                        continue;
                    }
                };
                if rvalue_has_fusion_signal(value) {
                    signal_count += 1;
                    run_len += 1;
                } else {
                    if run_len >= 2 {
                        let stmt_start = stmt_index - run_len;
                        let stmt_end = stmt_index;
                        groups.push(crate::bytecode::SemanticFusionCandidateGroup {
                            id: groups.len(),
                            signal_count: run_len,
                            function: body.function,
                            block: block.id,
                            stmt_start,
                            stmt_end,
                            source_span: merge_stmt_run_span(block, stmt_start, stmt_end),
                        });
                    }
                    run_len = 0;
                }
            }
            if run_len >= 2 {
                let stmt_start = block.statements.len() - run_len;
                let stmt_end = block.statements.len();
                groups.push(crate::bytecode::SemanticFusionCandidateGroup {
                    id: groups.len(),
                    signal_count: run_len,
                    function: body.function,
                    block: block.id,
                    stmt_start,
                    stmt_end,
                    source_span: merge_stmt_run_span(block, stmt_start, stmt_end),
                });
            }
        }
    }
    (signal_count, groups)
}

#[cfg(feature = "native-accel")]
fn merge_stmt_run_span(
    block: &runmat_mir::BasicBlock,
    stmt_start: usize,
    stmt_end: usize,
) -> runmat_hir::Span {
    let mut iter = block.statements[stmt_start..stmt_end]
        .iter()
        .map(|stmt| stmt.span);
    let Some(first) = iter.next() else {
        return runmat_hir::Span::default();
    };
    iter.fold(first, runmat_hir::merge_span)
}

#[cfg(feature = "native-accel")]
fn source_span_contains(outer: runmat_hir::Span, inner: runmat_hir::Span) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}

#[cfg(all(feature = "native-accel", test))]
fn semantic_candidates_touch_accel_capable_instruction(
    instructions: &[Instr],
    instr_spans: &[runmat_hir::Span],
    semantic_candidate_groups: &[crate::bytecode::SemanticFusionCandidateGroup],
) -> bool {
    if instructions.is_empty() || instr_spans.is_empty() || semantic_candidate_groups.is_empty() {
        return false;
    }
    instructions
        .iter()
        .enumerate()
        .filter(|(index, _)| *index < instr_spans.len())
        .any(|(index, instr)| {
            let span = instr_spans[index];
            semantic_candidate_groups
                .iter()
                .any(|candidate| source_span_contains(candidate.source_span, span))
                && instr_is_accel_capable(instr)
        })
}

#[cfg(all(feature = "native-accel", test))]
fn instr_is_accel_capable(instr: &Instr) -> bool {
    match instr {
        Instr::Add
        | Instr::Sub
        | Instr::Mul
        | Instr::RightDiv
        | Instr::LeftDiv
        | Instr::Pow
        | Instr::Neg
        | Instr::UPlus
        | Instr::Transpose
        | Instr::ConjugateTranspose
        | Instr::ElemMul
        | Instr::ElemDiv
        | Instr::ElemPow
        | Instr::ElemLeftDiv
        | Instr::LessEqual
        | Instr::Less
        | Instr::Greater
        | Instr::GreaterEqual
        | Instr::Equal
        | Instr::NotEqual => true,
        Instr::CallBuiltinMulti(name, _, _) => builtin_functions()
            .iter()
            .find(|func| func.name == name.as_str())
            .map(|func| {
                func.accel_tags.iter().any(|tag| {
                    matches!(
                        tag,
                        AccelTag::Unary
                            | AccelTag::Elementwise
                            | AccelTag::Reduction
                            | AccelTag::MatMul
                            | AccelTag::Transpose
                    )
                })
            })
            .unwrap_or(false),
        _ => false,
    }
}

#[cfg(feature = "native-accel")]
fn fusion_group_within_semantic_candidate_spans(
    group: &runmat_accelerate::fusion::FusionGroup,
    instr_spans: &[runmat_hir::Span],
    semantic_candidate_groups: &[crate::bytecode::SemanticFusionCandidateGroup],
) -> bool {
    if instr_spans.is_empty()
        || group.span.start > group.span.end
        || group.span.start >= instr_spans.len()
    {
        return false;
    }
    let end = group.span.end.min(instr_spans.len().saturating_sub(1));
    let candidate_spans: Vec<_> = semantic_candidate_groups
        .iter()
        .map(|group| group.source_span)
        .collect();
    candidate_spans.iter().any(|candidate| {
        instr_spans[group.span.start..=end]
            .iter()
            .all(|span| source_span_contains(*candidate, *span))
    })
}

#[cfg(all(feature = "native-accel", test))]
fn derive_semantic_fusion_groups_from_candidates(
    semantic_instruction_windows: &[crate::bytecode::SemanticFusionInstructionWindow],
    accel_graph: &runmat_accelerate::graph::AccelGraph,
) -> Vec<runmat_accelerate::fusion::FusionGroup> {
    let mut groups = Vec::new();
    let mut assigned_nodes = HashSet::new();

    for window in semantic_instruction_windows {
        let nodes = accel_nodes_for_instruction_window(accel_graph, &window, &assigned_nodes);
        if nodes.is_empty() {
            continue;
        }
        for node_id in &nodes {
            assigned_nodes.insert(*node_id);
        }
        let kind = infer_semantic_fusion_kind(window.kind);
        groups.push(runmat_accelerate::fusion::FusionGroup {
            id: groups.len(),
            kind,
            nodes,
            shape: runmat_accelerate::graph::ShapeInfo::Unknown,
            span: window.span.clone(),
            pattern: None,
            stack_layout: None,
        });
    }

    groups
}

#[cfg(all(feature = "native-accel", test))]
fn derive_semantic_fusion_groups_preserving_unmapped_windows(
    semantic_instruction_windows: &[crate::bytecode::SemanticFusionInstructionWindow],
    accel_graph: &runmat_accelerate::graph::AccelGraph,
) -> Vec<runmat_accelerate::fusion::FusionGroup> {
    let mut groups = Vec::new();
    let mut assigned_nodes = HashSet::new();

    for window in semantic_instruction_windows {
        let nodes = accel_nodes_for_instruction_window(accel_graph, window, &assigned_nodes);
        for node_id in &nodes {
            assigned_nodes.insert(*node_id);
        }
        groups.push(runmat_accelerate::fusion::FusionGroup {
            id: groups.len(),
            kind: infer_semantic_fusion_kind(window.kind),
            nodes,
            shape: runmat_accelerate::graph::ShapeInfo::Unknown,
            span: window.span.clone(),
            pattern: None,
            stack_layout: None,
        });
    }

    groups
}

#[cfg(feature = "native-accel")]
fn derive_semantic_fusion_groups_from_instruction_windows(
    semantic_instruction_windows: &[crate::bytecode::SemanticFusionInstructionWindow],
) -> Vec<runmat_accelerate::fusion::FusionGroup> {
    semantic_instruction_windows
        .iter()
        .enumerate()
        .map(|(id, window)| runmat_accelerate::fusion::FusionGroup {
            id,
            kind: infer_semantic_fusion_kind(window.kind),
            nodes: Vec::new(),
            shape: runmat_accelerate::graph::ShapeInfo::Unknown,
            span: window.span.clone(),
            pattern: None,
            stack_layout: None,
        })
        .collect()
}

#[cfg(all(feature = "native-accel", test))]
fn accel_nodes_for_instruction_window(
    accel_graph: &runmat_accelerate::graph::AccelGraph,
    window: &crate::bytecode::SemanticFusionInstructionWindow,
    assigned_nodes: &HashSet<runmat_accelerate::graph::NodeId>,
) -> Vec<runmat_accelerate::graph::NodeId> {
    let mut nodes: Vec<_> = accel_graph
        .nodes
        .iter()
        .filter(|node| {
            !assigned_nodes.contains(&node.id)
                && accel_node_matches_semantic_window_kind(node, window.kind)
                && accel_node_span_matches_instruction_window(node, window)
        })
        .map(|node| node.id)
        .collect();
    nodes.sort_unstable_by_key(|node_id| {
        accel_graph
            .node(*node_id)
            .map(|node| (node.span.start, node.span.end, node.id))
            .unwrap_or((usize::MAX, usize::MAX, *node_id))
    });
    nodes.dedup();
    nodes
}

#[cfg(all(feature = "native-accel", test))]
fn accel_node_span_matches_instruction_window(
    node: &runmat_accelerate::graph::AccelNode,
    window: &crate::bytecode::SemanticFusionInstructionWindow,
) -> bool {
    // Compile-time mapping is strict: only contained spans are assigned.
    // Broader reconciliation remains runtime-owned in fusion plan sanitization.
    node.span.start >= window.span.start && node.span.end <= window.span.end
}

#[cfg(all(feature = "native-accel", test))]
fn accel_node_has_semantic_signal(node: &runmat_accelerate::graph::AccelNode) -> bool {
    node.tags.iter().any(|tag| {
        matches!(
            tag,
            runmat_accelerate::graph::AccelGraphTag::Unary
                | runmat_accelerate::graph::AccelGraphTag::Elementwise
                | runmat_accelerate::graph::AccelGraphTag::Reduction
                | runmat_accelerate::graph::AccelGraphTag::MatMul
                | runmat_accelerate::graph::AccelGraphTag::Transpose
        )
    })
}

#[cfg(all(feature = "native-accel", test))]
fn accel_node_matches_semantic_window_kind(
    node: &runmat_accelerate::graph::AccelNode,
    kind: crate::bytecode::SemanticFusionInstructionKind,
) -> bool {
    let has_semantic_signal = accel_node_has_semantic_signal(node);
    if !has_semantic_signal {
        // If graph tags are absent, fall back to accel category compatibility
        // instead of admitting any span-matched node.
        return match kind {
            crate::bytecode::SemanticFusionInstructionKind::Elementwise => matches!(
                node.category,
                runmat_accelerate::graph::AccelOpCategory::Elementwise
                    | runmat_accelerate::graph::AccelOpCategory::Transpose
            ),
            crate::bytecode::SemanticFusionInstructionKind::Reduction => {
                matches!(
                    node.category,
                    runmat_accelerate::graph::AccelOpCategory::Reduction
                )
            }
            crate::bytecode::SemanticFusionInstructionKind::Matmul => {
                matches!(
                    node.category,
                    runmat_accelerate::graph::AccelOpCategory::MatMul
                )
            }
        };
    }
    let has_reduction = node
        .tags
        .iter()
        .any(|tag| matches!(tag, runmat_accelerate::graph::AccelGraphTag::Reduction));
    let has_matmul = node
        .tags
        .iter()
        .any(|tag| matches!(tag, runmat_accelerate::graph::AccelGraphTag::MatMul));
    match kind {
        crate::bytecode::SemanticFusionInstructionKind::Elementwise => {
            !has_reduction && !has_matmul
        }
        crate::bytecode::SemanticFusionInstructionKind::Reduction => !has_matmul,
        crate::bytecode::SemanticFusionInstructionKind::Matmul => true,
    }
}

#[cfg(feature = "native-accel")]
fn instruction_within_semantic_candidate_span(
    instruction_index: usize,
    instr_spans: &[runmat_hir::Span],
    semantic_candidate_span: runmat_hir::Span,
) -> bool {
    if instr_spans.is_empty() || instruction_index >= instr_spans.len() {
        return false;
    }
    source_span_contains(semantic_candidate_span, instr_spans[instruction_index])
}

#[cfg(feature = "native-accel")]
fn derive_semantic_fusion_instruction_windows(
    instructions: &[Instr],
    instr_spans: &[runmat_hir::Span],
    semantic_candidate_groups: &[crate::bytecode::SemanticFusionCandidateGroup],
) -> Vec<crate::bytecode::SemanticFusionInstructionWindow> {
    if instructions.is_empty() || instr_spans.is_empty() || semantic_candidate_groups.is_empty() {
        return Vec::new();
    }

    let mut windows = Vec::new();
    let mut assigned_instructions = HashSet::new();

    for candidate in semantic_candidate_groups {
        let mut run_start: Option<usize> = None;
        let mut run_kind: Option<crate::bytecode::SemanticFusionInstructionKind> = None;
        for (index, instr) in instructions.iter().enumerate() {
            if index >= instr_spans.len() || assigned_instructions.contains(&index) {
                if let Some(start) = run_start.take() {
                    windows.push(crate::bytecode::SemanticFusionInstructionWindow {
                        span: runmat_accelerate::graph::InstrSpan {
                            start,
                            end: index.saturating_sub(1),
                        },
                        kind: run_kind
                            .unwrap_or(crate::bytecode::SemanticFusionInstructionKind::Elementwise),
                    });
                    run_kind = None;
                }
                continue;
            }
            if !instruction_within_semantic_candidate_span(
                index,
                instr_spans,
                candidate.source_span,
            ) {
                if let Some(start) = run_start.take() {
                    windows.push(crate::bytecode::SemanticFusionInstructionWindow {
                        span: runmat_accelerate::graph::InstrSpan {
                            start,
                            end: index.saturating_sub(1),
                        },
                        kind: run_kind
                            .unwrap_or(crate::bytecode::SemanticFusionInstructionKind::Elementwise),
                    });
                    run_kind = None;
                }
                continue;
            }
            let Some(signal_kind) = instr_fusion_signal_kind(instr) else {
                if let Some(start) = run_start.take() {
                    windows.push(crate::bytecode::SemanticFusionInstructionWindow {
                        span: runmat_accelerate::graph::InstrSpan {
                            start,
                            end: index.saturating_sub(1),
                        },
                        kind: run_kind
                            .unwrap_or(crate::bytecode::SemanticFusionInstructionKind::Elementwise),
                    });
                    run_kind = None;
                }
                continue;
            };

            if run_start.is_none() {
                run_start = Some(index);
                run_kind = Some(signal_kind);
            } else if matches!(
                signal_kind,
                crate::bytecode::SemanticFusionInstructionKind::Matmul
            ) {
                run_kind = Some(crate::bytecode::SemanticFusionInstructionKind::Matmul);
            } else if matches!(
                signal_kind,
                crate::bytecode::SemanticFusionInstructionKind::Reduction
            ) && !matches!(
                run_kind,
                Some(crate::bytecode::SemanticFusionInstructionKind::Matmul)
            ) {
                run_kind = Some(crate::bytecode::SemanticFusionInstructionKind::Reduction);
            }
            assigned_instructions.insert(index);
        }
        if let Some(start) = run_start.take() {
            windows.push(crate::bytecode::SemanticFusionInstructionWindow {
                span: runmat_accelerate::graph::InstrSpan {
                    start,
                    end: instructions.len().saturating_sub(1),
                },
                kind: run_kind
                    .unwrap_or(crate::bytecode::SemanticFusionInstructionKind::Elementwise),
            });
        }
    }

    windows
}

#[cfg(feature = "native-accel")]
fn instr_fusion_signal_kind(
    instr: &Instr,
) -> Option<crate::bytecode::SemanticFusionInstructionKind> {
    match instr {
        Instr::Add
        | Instr::Sub
        | Instr::Mul
        | Instr::RightDiv
        | Instr::LeftDiv
        | Instr::Pow
        | Instr::Neg
        | Instr::UPlus
        | Instr::Transpose
        | Instr::ConjugateTranspose
        | Instr::ElemMul
        | Instr::ElemDiv
        | Instr::ElemPow
        | Instr::ElemLeftDiv
        | Instr::LessEqual
        | Instr::Less
        | Instr::Greater
        | Instr::GreaterEqual
        | Instr::Equal
        | Instr::NotEqual => Some(crate::bytecode::SemanticFusionInstructionKind::Elementwise),
        Instr::CallBuiltinMulti(name, _, _) => builtin_functions()
            .iter()
            .find(|func| func.name == name.as_str())
            .and_then(|func| {
                let has_matmul = func
                    .accel_tags
                    .iter()
                    .any(|tag| matches!(tag, AccelTag::MatMul));
                if has_matmul {
                    return Some(crate::bytecode::SemanticFusionInstructionKind::Matmul);
                }
                let has_reduction = func
                    .accel_tags
                    .iter()
                    .any(|tag| matches!(tag, AccelTag::Reduction));
                if has_reduction {
                    return Some(crate::bytecode::SemanticFusionInstructionKind::Reduction);
                }
                let has_elementwise = func.accel_tags.iter().any(|tag| {
                    matches!(
                        tag,
                        AccelTag::Unary | AccelTag::Elementwise | AccelTag::Transpose
                    )
                });
                has_elementwise
                    .then_some(crate::bytecode::SemanticFusionInstructionKind::Elementwise)
            }),
        _ => None,
    }
}

#[cfg(feature = "native-accel")]
fn infer_semantic_fusion_kind(
    kind_hint: crate::bytecode::SemanticFusionInstructionKind,
) -> runmat_accelerate::fusion::FusionKind {
    match kind_hint {
        crate::bytecode::SemanticFusionInstructionKind::Matmul => {
            runmat_accelerate::fusion::FusionKind::MatmulEpilogue
        }
        crate::bytecode::SemanticFusionInstructionKind::Reduction => {
            runmat_accelerate::fusion::FusionKind::Reduction
        }
        crate::bytecode::SemanticFusionInstructionKind::Elementwise => {
            runmat_accelerate::fusion::FusionKind::ElementwiseChain
        }
    }
}

#[cfg(feature = "native-accel")]
fn rvalue_has_fusion_signal(value: &MirRvalue) -> bool {
    match value {
        MirRvalue::Unary(_, _) | MirRvalue::Binary(_, _, _) => true,
        MirRvalue::Call(call) => matches!(
            call.semantic_kind,
            BuiltinSemanticKind::Elementwise
                | BuiltinSemanticKind::Reduction
                | BuiltinSemanticKind::LinearAlgebra
                | BuiltinSemanticKind::ShapeTransform(_)
        ),
        MirRvalue::Use(_)
        | MirRvalue::Range { .. }
        | MirRvalue::Aggregate { .. }
        | MirRvalue::Index { .. }
        | MirRvalue::Member { .. }
        | MirRvalue::DynamicMember { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End
        | MirRvalue::Future { .. }
        | MirRvalue::Spawn(_) => false,
    }
}

pub fn compile_semantic_function_registry(
    hir: &HirAssembly,
    mir: &MirAssembly,
) -> Result<HashMap<FunctionId, SemanticFunctionBytecode>, CompileError> {
    let layout = derive_layout(hir, mir)
        .map_err(|err| CompileError::new(format!("failed to derive VM layout: {err:?}")))?;
    compile_semantic_functions(hir, mir, &layout, None)
}

fn compile_semantic_functions(
    hir: &HirAssembly,
    mir: &MirAssembly,
    layout: &crate::layout::VmAssemblyLayout,
    entrypoint: Option<EntrypointId>,
) -> Result<HashMap<FunctionId, SemanticFunctionBytecode>, CompileError> {
    let entry_target = entrypoint
        .and_then(|entrypoint| layout.entrypoints.get(&entrypoint))
        .map(|entry| entry.target);
    let mut functions = HashMap::new();
    for function in &hir.functions {
        if Some(function.id) == entry_target {
            continue;
        }
        let mut compiler = Compiler::new_for_function(hir, mir, layout.clone(), function.id)?;
        compiler.compile()?;
        let function_layout = layout.functions.get(&function.id).ok_or_else(|| {
            CompileError::new(format!("missing VM layout for function {:?}", function.id))
        })?;
        functions.insert(
            function.id,
            SemanticFunctionBytecode {
                function: function.id,
                display_name: function_layout.display_name.clone(),
                source_id: None,
                instructions: compiler.instructions,
                instr_spans: compiler.instr_spans,
                call_arg_spans: compiler.call_arg_spans,
                var_count: compiler.var_count,
                input_slots: function_layout
                    .frame_abi
                    .fixed_inputs
                    .iter()
                    .filter(|slot| Some(**slot) != function_layout.frame_abi.varargin)
                    .map(|slot| slot.0)
                    .collect(),
                varargin_slot: function_layout.frame_abi.varargin.map(|slot| slot.0),
                output_slots: function_layout
                    .frame_abi
                    .fixed_outputs
                    .iter()
                    .filter(|slot| Some(**slot) != function_layout.frame_abi.varargout)
                    .map(|slot| slot.0)
                    .collect(),
                varargout_slot: function_layout.frame_abi.varargout.map(|slot| slot.0),
                capture_slots: function_layout
                    .captures
                    .iter()
                    .map(|capture| capture.slot.0)
                    .collect(),
            },
        );
    }
    Ok(functions)
}

#[cfg(test)]
mod tests {
    use super::compile;
    use crate::Instr;
    use futures::executor::block_on;
    #[cfg(feature = "native-accel")]
    use runmat_accelerate::fusion::prepare_fusion_plan;
    use runmat_builtins::Value;
    use runmat_hir::{
        lower, AssignmentCreationPolicy, BuiltinId, CallableFallbackPolicy, CallableIdentity,
        DefPath, DefPathSegment, FunctionId, IndexResultContext, LoweringContext, MethodId,
        OperatorKind, PackageName, QualifiedName, RequestedOutputCount, SymbolName,
    };
    use runmat_mir::lowering::lower_assembly;
    use runmat_mir::{
        MirAggregateKind, MirCallee, MirConstant, MirIndexComponent, MirIndexPlan, MirOperand,
        MirOutputTarget, MirPlace, MirRvalue, MirStmtKind, MirTerminatorKind,
    };
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn primary_compile_attaches_derived_layout() {
        let ast = runmat_parser::parse("x = 1 + 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        let layout = bytecode.layout.as_ref().expect("layout");
        let entrypoint_layout = &layout.entrypoints[&entrypoint];
        let function_layout = &layout.functions[&entrypoint_layout.target];
        assert_eq!(bytecode.var_count, function_layout.local_count);
        assert_eq!(bytecode.var_types.len(), function_layout.local_count);
    }

    #[test]
    fn primary_compile_lowers_simple_assignment_arithmetic() {
        let ast = runmat_parser::parse("x = 1 + 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert_eq!(bytecode.instructions.len(), 4);
        assert!(matches!(bytecode.instructions[0], Instr::LoadConst(1.0)));
        assert!(matches!(bytecode.instructions[1], Instr::LoadConst(2.0)));
        assert!(matches!(bytecode.instructions[2], Instr::Add));
        assert!(matches!(bytecode.instructions[3], Instr::StoreVar(_)));
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn primary_compile_records_semantic_fusion_metadata() {
        let ast = runmat_parser::parse("x = 1 + 2; y = x * 3;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert!(
            bytecode.semantic_fusion_metadata.mir_fusion_signal_count > 0,
            "expected non-zero MIR fusion signal count"
        );
        assert!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count
                > 0,
            "expected non-zero MIR fusion candidate group count"
        );
        assert!(
            !bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_groups
                .is_empty(),
            "expected non-empty MIR fusion candidate groups"
        );
        assert!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_groups
                .iter()
                .all(|group| group.stmt_end > group.stmt_start),
            "expected candidate groups to carry non-empty statement spans"
        );
        assert!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_groups
                .iter()
                .all(|group| group.source_span.end > group.source_span.start),
            "expected candidate groups to carry non-empty source spans"
        );
        assert!(
            bytecode
                .semantic_fusion_metadata
                .semantic_instruction_window_count
                > 0,
            "expected non-zero semantic instruction window count"
        );
        assert_eq!(
            bytecode
                .semantic_fusion_metadata
                .semantic_instruction_window_count,
            bytecode
                .semantic_fusion_metadata
                .semantic_instruction_windows
                .len(),
            "window count should match serialized semantic instruction window entries"
        );
        assert!(
            bytecode
                .semantic_fusion_metadata
                .semantic_instruction_windows
                .iter()
                .all(|window| window.span.end >= window.span.start),
            "expected semantic instruction windows to carry valid instruction spans"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn primary_compile_emits_semantic_window_scaffolds_and_runtime_plan_reconciles_nodes() {
        let ast = runmat_parser::parse("x = 1 + 2; y = x * 3;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let runtime_groups = bytecode.runtime_fusion_groups();
        let runtime_graph = bytecode.runtime_accel_graph_for_fusion(&runtime_groups);
        assert!(
            runtime_graph.is_some(),
            "expected runtime accel graph when semantic fusion candidates/windows exist"
        );
        assert!(
            !bytecode.fusion_groups.is_empty(),
            "expected semantic-window fusion scaffolds"
        );
        assert!(
            bytecode
                .fusion_groups
                .iter()
                .all(|group| group.nodes.is_empty()),
            "compile should not assign accel node IDs to semantic-window groups"
        );

        let runtime_groups = if let Some(graph) = runtime_graph.as_ref() {
            bytecode.runtime_fusion_groups_for_graph(graph)
        } else {
            bytecode.fusion_groups.clone()
        };
        let runtime_plan = prepare_fusion_plan(
            runtime_graph.as_ref(),
            &runtime_groups,
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count,
        )
        .expect("runtime fusion planning should reconcile executable groups");
        assert!(
            runtime_plan
                .groups
                .iter()
                .any(|group| !group.group.nodes.is_empty()),
            "runtime fusion planning should reconcile node IDs from accel graph"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn primary_compile_keeps_multi_window_groups_node_empty_before_runtime_reconciliation() {
        let ast =
            runmat_parser::parse("a = 1 + 2; b = a * 3; marker = 'x'; c = b - 4; d = c ./ 2;")
                .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(
            bytecode.fusion_groups.len() >= 2,
            "expected multiple semantic-window fusion groups for split accel-capable runs"
        );
        assert!(
            bytecode
                .fusion_groups
                .iter()
                .all(|group| group.nodes.is_empty()),
            "compile-time semantic-window groups must remain node-empty"
        );

        let runtime_groups = bytecode.runtime_fusion_groups();
        let runtime_graph = bytecode.runtime_accel_graph_for_fusion(&runtime_groups);
        let runtime_groups = if let Some(graph) = runtime_graph.as_ref() {
            bytecode.runtime_fusion_groups_for_graph(graph)
        } else {
            bytecode.fusion_groups.clone()
        };
        let runtime_plan = prepare_fusion_plan(
            runtime_graph.as_ref(),
            &runtime_groups,
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count,
        )
        .expect("runtime fusion planning should reconcile executable groups");
        assert!(
            runtime_plan
                .groups
                .iter()
                .any(|group| !group.group.nodes.is_empty()),
            "runtime reconciliation should assign accel nodes"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn primary_compile_semantically_gates_bytecode_fusion_groups() {
        let ast = runmat_parser::parse("x = 1;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let runtime_graph =
            bytecode.runtime_accel_graph_for_fusion(&bytecode.runtime_fusion_groups());

        assert_eq!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count,
            0,
            "expected no semantic fusion candidate groups"
        );
        assert!(
            runtime_graph.is_none(),
            "expected runtime accel graph to be omitted when semantic candidate groups are absent"
        );
        assert!(
            bytecode.fusion_groups.is_empty(),
            "expected bytecode fusion groups to be gated off when semantic candidates are absent"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn primary_compile_omits_accel_graph_when_signals_exist_but_no_candidate_group() {
        let ast = runmat_parser::parse("x = 1 + 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let runtime_graph =
            bytecode.runtime_accel_graph_for_fusion(&bytecode.runtime_fusion_groups());

        assert!(
            bytecode.semantic_fusion_metadata.mir_fusion_signal_count > 0,
            "expected non-zero fusion signal count for arithmetic operation"
        );
        assert_eq!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count,
            0,
            "expected no semantic candidate groups for a single-operation run"
        );
        assert!(
            runtime_graph.is_none(),
            "expected runtime accel graph omission to follow semantic candidate-group gating"
        );
        assert!(
            bytecode.fusion_groups.is_empty(),
            "expected no executable bytecode fusion groups without semantic candidates"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn primary_compile_omits_accel_graph_when_candidates_overlap_only_logical_ops() {
        let ast =
            runmat_parser::parse("a = true; b = false; c = a & b; d = c | a;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let runtime_graph =
            bytecode.runtime_accel_graph_for_fusion(&bytecode.runtime_fusion_groups());

        assert!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count
                > 0,
            "logical chain should still produce semantic candidate groups"
        );
        assert!(
            runtime_graph.is_none(),
            "expected runtime accel graph omission when candidate overlap is non-accelerable logical ops"
        );
        assert!(
            bytecode.fusion_groups.is_empty(),
            "expected no executable fusion groups for logical-only candidate overlap"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_candidate_accel_capability_gate_rejects_logical_ops() {
        let instructions = vec![Instr::LogicalAnd, Instr::LogicalOr];
        let instr_spans = vec![
            runmat_hir::Span { start: 10, end: 20 },
            runmat_hir::Span { start: 21, end: 30 },
        ];
        let candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 2,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 2,
            source_span: runmat_hir::Span { start: 10, end: 30 },
        }];
        assert!(
            !super::semantic_candidates_touch_accel_capable_instruction(
                &instructions,
                &instr_spans,
                &candidates,
            ),
            "logical ops should not trigger accel-graph construction gate"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_candidate_accel_capability_gate_accepts_binary_ops() {
        let instructions = vec![Instr::Add, Instr::ElemMul];
        let instr_spans = vec![
            runmat_hir::Span { start: 10, end: 20 },
            runmat_hir::Span { start: 21, end: 30 },
        ];
        let candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 2,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 2,
            source_span: runmat_hir::Span { start: 10, end: 30 },
        }];
        assert!(
            super::semantic_candidates_touch_accel_capable_instruction(
                &instructions,
                &instr_spans,
                &candidates,
            ),
            "elementwise arithmetic ops should trigger accel-graph construction gate"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_candidate_accel_capability_gate_rejects_partial_span_overlap() {
        let instructions = vec![Instr::Add];
        let instr_spans = vec![runmat_hir::Span { start: 10, end: 20 }];
        let candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 1,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 1,
            source_span: runmat_hir::Span { start: 19, end: 25 },
        }];
        assert!(
            !super::semantic_candidates_touch_accel_capable_instruction(
                &instructions,
                &instr_spans,
                &candidates,
            ),
            "partial boundary overlap should not satisfy accel-capability semantic gate"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_candidate_accel_capability_gate_accepts_reduction_builtin() {
        let instructions = vec![Instr::CallBuiltinMulti("sum".to_string(), 1, 1)];
        let instr_spans = vec![runmat_hir::Span { start: 10, end: 20 }];
        let candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 2,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 2,
            source_span: runmat_hir::Span { start: 10, end: 20 },
        }];
        assert!(
            super::semantic_candidates_touch_accel_capable_instruction(
                &instructions,
                &instr_spans,
                &candidates,
            ),
            "reduction builtin call should trigger accel-graph construction gate"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_candidate_accel_capability_gate_rejects_control_assert_builtin() {
        let instructions = vec![Instr::CallBuiltinMulti("assert".to_string(), 1, 0)];
        let instr_spans = vec![runmat_hir::Span { start: 10, end: 20 }];
        let candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 2,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 2,
            source_span: runmat_hir::Span { start: 10, end: 20 },
        }];
        assert!(
            !super::semantic_candidates_touch_accel_capable_instruction(
                &instructions,
                &instr_spans,
                &candidates,
            ),
            "control/assertion builtin should not trigger accel-graph construction gate"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_candidate_accel_capability_gate_rejects_sink_builtins() {
        for builtin in ["disp", "fprintf"] {
            let instructions = vec![Instr::CallBuiltinMulti(builtin.to_string(), 1, 0)];
            let instr_spans = vec![runmat_hir::Span { start: 10, end: 20 }];
            let candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
                id: 0,
                signal_count: 2,
                function: runmat_hir::FunctionId(0),
                block: runmat_mir::BasicBlockId(0),
                stmt_start: 0,
                stmt_end: 2,
                source_span: runmat_hir::Span { start: 10, end: 20 },
            }];
            assert!(
                !super::semantic_candidates_touch_accel_capable_instruction(
                    &instructions,
                    &instr_spans,
                    &candidates,
                ),
                "sink builtin `{builtin}` should not trigger accel-graph construction gate"
            );
        }
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn primary_compile_scopes_semantic_fusion_metadata_to_entrypoint_target() {
        let source = "x = 1; function z = helper(a); t = a + 1; z = t * 2; end;";
        let ast = runmat_parser::parse(source).expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let runtime_graph =
            bytecode.runtime_accel_graph_for_fusion(&bytecode.runtime_fusion_groups());

        assert_eq!(
            bytecode.semantic_fusion_metadata.mir_fusion_signal_count, 0,
            "non-entrypoint helper MIR bodies should not drive entrypoint fusion signal metadata"
        );
        assert_eq!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count,
            0,
            "non-entrypoint helper MIR bodies should not drive entrypoint fusion candidate metadata"
        );
        assert!(
            runtime_graph.is_none(),
            "entrypoint with no semantic candidates should omit runtime accel graph even if helper bodies are fusible"
        );
        assert!(
            bytecode.fusion_groups.is_empty(),
            "entrypoint with no semantic candidates should not emit executable fusion groups"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn fusion_group_semantic_span_filter_requires_full_group_coverage() {
        let instr_spans = vec![
            runmat_hir::Span { start: 0, end: 2 },
            runmat_hir::Span { start: 2, end: 4 },
            runmat_hir::Span { start: 4, end: 6 },
        ];
        let group = runmat_accelerate::fusion::FusionGroup {
            id: 0,
            kind: runmat_accelerate::fusion::FusionKind::ElementwiseChain,
            nodes: vec![],
            shape: runmat_accelerate::graph::ShapeInfo::Scalar,
            span: runmat_accelerate::graph::InstrSpan { start: 1, end: 2 },
            pattern: None,
            stack_layout: None,
        };
        let fully_covering_candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 3,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 2,
            source_span: runmat_hir::Span { start: 2, end: 6 },
        }];
        let partially_covering_candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 2,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 1,
            source_span: runmat_hir::Span { start: 0, end: 3 },
        }];
        let non_overlapping_candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 2,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 1,
            source_span: runmat_hir::Span { start: 8, end: 10 },
        }];

        assert!(
            super::fusion_group_within_semantic_candidate_spans(
                &group,
                &instr_spans,
                &fully_covering_candidates
            ),
            "expected full coverage when all instruction source spans intersect semantic candidate spans"
        );
        assert!(
            !super::fusion_group_within_semantic_candidate_spans(
                &group,
                &instr_spans,
                &partially_covering_candidates
            ),
            "expected group rejection when only part of the instruction span range intersects semantic candidate spans"
        );
        assert!(
            !super::fusion_group_within_semantic_candidate_spans(
                &group,
                &instr_spans,
                &non_overlapping_candidates
            ),
            "expected no overlap when instruction source spans are disjoint from semantic candidate spans"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn fusion_group_semantic_span_filter_rejects_multi_candidate_union_coverage() {
        let instr_spans = vec![
            runmat_hir::Span { start: 0, end: 2 },
            runmat_hir::Span { start: 2, end: 4 },
        ];
        let group = runmat_accelerate::fusion::FusionGroup {
            id: 0,
            kind: runmat_accelerate::fusion::FusionKind::ElementwiseChain,
            nodes: vec![],
            shape: runmat_accelerate::graph::ShapeInfo::Scalar,
            span: runmat_accelerate::graph::InstrSpan { start: 0, end: 1 },
            pattern: None,
            stack_layout: None,
        };
        let split_candidates = vec![
            crate::bytecode::SemanticFusionCandidateGroup {
                id: 0,
                signal_count: 1,
                function: runmat_hir::FunctionId(0),
                block: runmat_mir::BasicBlockId(0),
                stmt_start: 0,
                stmt_end: 1,
                source_span: runmat_hir::Span { start: 0, end: 2 },
            },
            crate::bytecode::SemanticFusionCandidateGroup {
                id: 1,
                signal_count: 1,
                function: runmat_hir::FunctionId(0),
                block: runmat_mir::BasicBlockId(0),
                stmt_start: 1,
                stmt_end: 2,
                source_span: runmat_hir::Span { start: 2, end: 4 },
            },
        ];

        assert!(
            !super::fusion_group_within_semantic_candidate_spans(&group, &instr_spans, &split_candidates),
            "expected rejection when bytecode group coverage requires unioning multiple semantic candidate spans"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_candidates_build_fusion_groups_from_accel_graph_nodes() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![
                runmat_accelerate::graph::AccelNode {
                    id: 0,
                    label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                        runmat_accelerate::graph::PrimitiveOp::Add,
                    ),
                    category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                    inputs: vec![0, 0],
                    outputs: vec![1],
                    span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                    tags: vec![runmat_accelerate::graph::AccelGraphTag::Elementwise],
                },
                runmat_accelerate::graph::AccelNode {
                    id: 1,
                    label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                        runmat_accelerate::graph::PrimitiveOp::ElemMul,
                    ),
                    category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                    inputs: vec![1, 0],
                    outputs: vec![2],
                    span: runmat_accelerate::graph::InstrSpan { start: 1, end: 1 },
                    tags: vec![runmat_accelerate::graph::AccelGraphTag::Elementwise],
                },
            ],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 2,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 1,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let instr_spans = vec![
            runmat_hir::Span { start: 10, end: 11 },
            runmat_hir::Span { start: 11, end: 12 },
        ];
        let semantic_candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 2,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 2,
            source_span: runmat_hir::Span { start: 10, end: 12 },
        }];

        let windows = super::derive_semantic_fusion_instruction_windows(
            &[Instr::Add, Instr::ElemMul],
            &instr_spans,
            &semantic_candidates,
        );
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert_eq!(groups.len(), 1, "expected one semantic-driven fusion group");
        assert_eq!(groups[0].nodes, vec![0, 1]);
        assert_eq!(
            groups[0].kind,
            runmat_accelerate::fusion::FusionKind::ElementwiseChain
        );
        assert_eq!(
            groups[0].shape,
            runmat_accelerate::graph::ShapeInfo::Unknown
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_windows_fallback_to_empty_node_groups_when_mapping_drops_all_nodes() {
        let windows = vec![crate::bytecode::SemanticFusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: 7, end: 9 },
            kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
        }];
        let groups = super::derive_semantic_fusion_groups_from_instruction_windows(&windows);
        assert_eq!(
            groups.len(),
            1,
            "semantic windows fallback should preserve executable-group scaffolding even when graph mapping is unavailable"
        );
        assert_eq!(
            groups[0].nodes,
            Vec::<runmat_accelerate::graph::NodeId>::new()
        );
        assert_eq!(groups[0].span.start, 7);
        assert_eq!(groups[0].span.end, 9);
        assert_eq!(
            groups[0].kind,
            runmat_accelerate::fusion::FusionKind::ElementwiseChain
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_windows_preserve_unmapped_windows_alongside_mapped_groups() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Add,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                tags: vec![],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let windows = vec![
            crate::bytecode::SemanticFusionInstructionWindow {
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
            },
            crate::bytecode::SemanticFusionInstructionWindow {
                span: runmat_accelerate::graph::InstrSpan { start: 5, end: 5 },
                kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
            },
        ];
        let groups = super::derive_semantic_fusion_groups_preserving_unmapped_windows(
            &windows,
            &accel_graph,
        );
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].nodes, vec![0]);
        assert_eq!(
            groups[1].nodes,
            Vec::<runmat_accelerate::graph::NodeId>::new()
        );
        assert_eq!(groups[0].span.start, 0);
        assert_eq!(groups[1].span.start, 5);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_windows_map_accel_nodes_without_semantic_tags() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Add,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                tags: vec![],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let windows = vec![crate::bytecode::SemanticFusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
            kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
        }];
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert_eq!(
            groups.len(),
            1,
            "semantic windows should still map accel nodes when graph semantic tags are absent"
        );
        assert_eq!(groups[0].nodes, vec![0]);
        assert_eq!(
            groups[0].kind,
            runmat_accelerate::fusion::FusionKind::ElementwiseChain
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_windows_without_tags_reject_category_mismatch() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Builtin {
                    name: "sum".to_string(),
                },
                category: runmat_accelerate::graph::AccelOpCategory::Reduction,
                inputs: vec![0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                tags: vec![],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let windows = vec![crate::bytecode::SemanticFusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
            kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
        }];
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert!(
            groups.is_empty(),
            "missing tags should not bypass category mismatch for semantic window mapping"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_windows_reject_covering_node_span_at_compile_mapping_stage() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Add,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 2 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Elementwise],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let windows = vec![crate::bytecode::SemanticFusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: 1, end: 1 },
            kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
        }];
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert!(
            groups.is_empty(),
            "compile-time mapping should reject covering node spans and defer reconciliation to runtime sanitization"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_windows_reject_overly_wide_covering_node_spans() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Add,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 4 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Elementwise],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let windows = vec![crate::bytecode::SemanticFusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: 2, end: 2 },
            kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
        }];
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert!(
            groups.is_empty(),
            "semantic windows should reject overly broad covering node spans"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_windows_reject_partial_overlap_at_compile_mapping_stage() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Add,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 1 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Elementwise],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let windows = vec![crate::bytecode::SemanticFusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: 1, end: 2 },
            kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
        }];
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert!(
            groups.is_empty(),
            "compile-time mapping should reject partial overlap and defer reconciliation to runtime sanitization"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_windows_reject_partial_overlap_with_large_boundary_shift() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Add,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 2 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Elementwise],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let windows = vec![crate::bytecode::SemanticFusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: 2, end: 3 },
            kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
        }];
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert!(
            groups.is_empty(),
            "semantic windows should reject partial overlap when boundary shift exceeds tolerance"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_windows_reject_disjoint_gap_at_compile_mapping_stage() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Add,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Elementwise],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let windows = vec![crate::bytecode::SemanticFusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: 1, end: 1 },
            kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
        }];
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert!(
            groups.is_empty(),
            "compile-time mapping should reject disjoint graph/window spans and leave reconciliation to runtime sanitization"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_windows_reject_accel_nodes_with_large_disjoint_gap() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Add,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Elementwise],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let windows = vec![crate::bytecode::SemanticFusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: 3, end: 3 },
            kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
        }];
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert!(
            groups.is_empty(),
            "semantic windows should reject accel-node mapping when disjoint span gap exceeds tolerance"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_window_kind_is_not_overridden_by_graph_category() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Add,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Reduction,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Elementwise],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let instr_spans = vec![runmat_hir::Span { start: 10, end: 11 }];
        let semantic_candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 1,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 1,
            source_span: runmat_hir::Span { start: 10, end: 11 },
        }];

        let windows = super::derive_semantic_fusion_instruction_windows(
            &[Instr::Add],
            &instr_spans,
            &semantic_candidates,
        );
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert_eq!(groups.len(), 1, "expected one semantic fusion group");
        assert_eq!(
            groups[0].kind,
            runmat_accelerate::fusion::FusionKind::ElementwiseChain,
            "semantic instruction-window kind should drive fusion kind classification"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_candidates_build_fusion_groups_from_transpose_nodes() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Transpose,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Transpose,
                inputs: vec![0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Transpose],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let instr_spans = vec![runmat_hir::Span { start: 10, end: 11 }];
        let semantic_candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 1,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 1,
            source_span: runmat_hir::Span { start: 10, end: 11 },
        }];

        let windows = super::derive_semantic_fusion_instruction_windows(
            &[Instr::Transpose],
            &instr_spans,
            &semantic_candidates,
        );
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert_eq!(
            groups.len(),
            1,
            "expected transpose-tagged accel node to participate in semantic fusion-group mapping"
        );
        assert_eq!(groups[0].nodes, vec![0]);
        assert_eq!(
            groups[0].kind,
            runmat_accelerate::fusion::FusionKind::ElementwiseChain
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_elementwise_window_excludes_reduction_nodes() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Builtin {
                    name: "sum".to_string(),
                },
                category: runmat_accelerate::graph::AccelOpCategory::Reduction,
                inputs: vec![0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Reduction],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let windows = vec![crate::bytecode::SemanticFusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
            kind: crate::bytecode::SemanticFusionInstructionKind::Elementwise,
        }];
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert!(
            groups.is_empty(),
            "elementwise semantic windows should not absorb reduction-tagged accel nodes"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_reduction_window_accepts_reduction_nodes() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Builtin {
                    name: "sum".to_string(),
                },
                category: runmat_accelerate::graph::AccelOpCategory::Reduction,
                inputs: vec![0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Reduction],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let windows = vec![crate::bytecode::SemanticFusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
            kind: crate::bytecode::SemanticFusionInstructionKind::Reduction,
        }];
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert_eq!(
            groups.len(),
            1,
            "reduction semantic windows should include reduction-tagged accel nodes"
        );
        assert_eq!(groups[0].nodes, vec![0]);
        assert_eq!(
            groups[0].kind,
            runmat_accelerate::fusion::FusionKind::Reduction
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_candidate_instruction_windows_split_on_non_accel_ops() {
        let instructions = vec![Instr::Add, Instr::LoadConst(1.0), Instr::ElemMul];
        let instr_spans = vec![
            runmat_hir::Span { start: 10, end: 11 },
            runmat_hir::Span { start: 11, end: 12 },
            runmat_hir::Span { start: 12, end: 13 },
        ];
        let semantic_candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 3,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 3,
            source_span: runmat_hir::Span { start: 10, end: 13 },
        }];

        let windows = super::derive_semantic_fusion_instruction_windows(
            &instructions,
            &instr_spans,
            &semantic_candidates,
        );
        assert_eq!(
            windows.len(),
            2,
            "expected non-accel instruction boundary to split semantic instruction windows"
        );
        assert_eq!(windows[0].span.start, 0);
        assert_eq!(windows[0].span.end, 0);
        assert_eq!(windows[1].span.start, 2);
        assert_eq!(windows[1].span.end, 2);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_candidates_without_overlap_do_not_build_fusion_groups() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Add,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Elementwise],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let instr_spans = vec![runmat_hir::Span { start: 10, end: 11 }];
        let semantic_candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 1,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 1,
            source_span: runmat_hir::Span {
                start: 100,
                end: 101,
            },
        }];

        let windows = super::derive_semantic_fusion_instruction_windows(
            &[Instr::Add],
            &instr_spans,
            &semantic_candidates,
        );
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert!(
            groups.is_empty(),
            "expected no semantic-driven fusion groups when candidate spans do not overlap instruction source spans"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn semantic_candidates_with_partial_overlap_do_not_build_fusion_groups() {
        let accel_graph = runmat_accelerate::graph::AccelGraph {
            nodes: vec![runmat_accelerate::graph::AccelNode {
                id: 0,
                label: runmat_accelerate::graph::AccelNodeLabel::Primitive(
                    runmat_accelerate::graph::PrimitiveOp::Add,
                ),
                category: runmat_accelerate::graph::AccelOpCategory::Elementwise,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: runmat_accelerate::graph::InstrSpan { start: 0, end: 0 },
                tags: vec![runmat_accelerate::graph::AccelGraphTag::Elementwise],
            }],
            values: vec![
                runmat_accelerate::graph::ValueInfo {
                    id: 0,
                    origin: runmat_accelerate::graph::ValueOrigin::Variable {
                        kind: runmat_accelerate::graph::VarKind::Global,
                        index: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
                runmat_accelerate::graph::ValueInfo {
                    id: 1,
                    origin: runmat_accelerate::graph::ValueOrigin::NodeOutput {
                        node: 0,
                        output: 0,
                    },
                    ty: runmat_builtins::Type::Num,
                    shape: runmat_accelerate::graph::ShapeInfo::Scalar,
                    constant: None,
                },
            ],
            var_bindings: std::collections::HashMap::new(),
            node_bindings: std::collections::HashMap::new(),
        };
        let instr_spans = vec![runmat_hir::Span { start: 10, end: 20 }];
        let semantic_candidates = vec![crate::bytecode::SemanticFusionCandidateGroup {
            id: 0,
            signal_count: 1,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 1,
            source_span: runmat_hir::Span { start: 19, end: 25 },
        }];

        let windows = super::derive_semantic_fusion_instruction_windows(
            &[Instr::Add],
            &instr_spans,
            &semantic_candidates,
        );
        let groups = super::derive_semantic_fusion_groups_from_candidates(&windows, &accel_graph);
        assert!(
            groups.is_empty(),
            "expected no semantic-driven fusion groups when candidate spans only partially overlap instruction spans"
        );
    }

    #[test]
    fn primary_compile_records_semantic_spawn_site_metadata() {
        let ast = runmat_parser::parse("fut = make(); task = spawn(fut);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert!(
            bytecode.semantic_async_metadata.mir_spawn_site_count > 0,
            "expected non-zero spawn site count"
        );
        assert!(
            !bytecode.semantic_async_metadata.mir_spawn_sites.is_empty(),
            "expected spawn site metadata entries"
        );
        assert_eq!(
            bytecode.semantic_async_metadata.mir_spawn_site_count,
            bytecode.semantic_async_metadata.mir_spawn_sites.len(),
            "spawn site count should match listed sites"
        );
        let unique_sites = bytecode
            .semantic_async_metadata
            .mir_spawn_sites
            .iter()
            .map(|site| (site.function, site.block, site.stmt_index))
            .collect::<std::collections::HashSet<_>>();
        assert!(
            unique_sites.len() == bytecode.semantic_async_metadata.mir_spawn_sites.len(),
            "spawn site metadata entries should be distinct"
        );
        assert_eq!(
            bytecode.semantic_async_metadata.runtime_model,
            crate::bytecode::program::SemanticAsyncRuntimeModel::LazyFutureDescriptorLane,
            "semantic async metadata should surface the current lazy-future runtime model"
        );
        assert_eq!(
            bytecode.semantic_async_metadata.mir_await_site_count, 0,
            "spawn-only program should not report await sites"
        );
        assert!(
            bytecode.semantic_async_metadata.mir_await_sites.is_empty(),
            "spawn-only program should have an empty await-site list"
        );
    }

    #[test]
    fn primary_compile_scopes_spawn_site_metadata_to_entrypoint_target() {
        let source = "x = 1; function z = helper(a); fut = make(); z = spawn(fut); end;";
        let ast = runmat_parser::parse(source).expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert_eq!(
            bytecode.semantic_async_metadata.mir_spawn_site_count,
            0,
            "spawn sites in non-entrypoint helper bodies should not be attributed to the entrypoint bytecode artifact"
        );
        assert!(
            bytecode.semantic_async_metadata.mir_spawn_sites.is_empty(),
            "spawn site list should be empty when only helper bodies contain spawn expressions"
        );
        assert_eq!(
            bytecode.semantic_async_metadata.mir_await_site_count, 0,
            "program without entrypoint await should not report await sites"
        );
    }

    #[test]
    fn primary_compile_records_semantic_await_site_metadata() {
        let source = "async function y = inc(x); y = x + 1; end; t = inc(2); z = await(t);";
        let ast = runmat_parser::parse(source).expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert!(
            bytecode.semantic_async_metadata.mir_await_site_count > 0,
            "expected non-zero await site count"
        );
        assert!(
            !bytecode.semantic_async_metadata.mir_await_sites.is_empty(),
            "expected await site metadata entries"
        );
        assert_eq!(
            bytecode.semantic_async_metadata.mir_await_site_count,
            bytecode.semantic_async_metadata.mir_await_sites.len(),
            "await site count should match listed sites"
        );
        let unique_sites = bytecode
            .semantic_async_metadata
            .mir_await_sites
            .iter()
            .map(|site| (site.function, site.block, site.resume))
            .collect::<std::collections::HashSet<_>>();
        assert!(
            unique_sites.len() == bytecode.semantic_async_metadata.mir_await_sites.len(),
            "await site metadata entries should be distinct"
        );
        assert_eq!(
            bytecode.semantic_async_metadata.runtime_model,
            crate::bytecode::program::SemanticAsyncRuntimeModel::LazyFutureDescriptorLane,
            "semantic async metadata should surface the current lazy-future runtime model"
        );
    }

    #[test]
    fn primary_compile_scopes_await_site_metadata_to_entrypoint_target() {
        let source = "x = 1; async function z = helper(a); z = await(a); end;";
        let ast = runmat_parser::parse(source).expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert_eq!(
            bytecode.semantic_async_metadata.mir_await_site_count,
            0,
            "await sites in non-entrypoint helper bodies should not be attributed to the entrypoint bytecode artifact"
        );
        assert!(
            bytecode.semantic_async_metadata.mir_await_sites.is_empty(),
            "await site list should be empty when only helper bodies contain await expressions"
        );
    }

    #[test]
    fn primary_compile_interprets_visible_assignment() {
        let ast = runmat_parser::parse("x = 1 + 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let export = &layout.entrypoints[&entrypoint].exports[0];

        assert_eq!(bytecode.var_names[&export.slot.0], "x");

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_interprets_builtin_assignment() {
        let ast = runmat_parser::parse("x = sqrt(9);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let export = &layout.entrypoints[&entrypoint].exports[0];

        assert!(matches!(
            bytecode.instructions.as_slice(),
            [
                Instr::LoadConst(9.0),
                Instr::CallBuiltinMulti(name, 1, 1),
                Instr::StoreVar(_),
            ] if name == "sqrt"
        ));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_interprets_matrix_literal_assignment() {
        let ast = runmat_parser::parse("x = [1 2; 3 4];").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let export = &layout.entrypoints[&entrypoint].exports[0];

        assert!(matches!(
            bytecode.instructions.as_slice(),
            [
                Instr::LoadConst(1.0),
                Instr::LoadConst(2.0),
                Instr::LoadConst(3.0),
                Instr::LoadConst(4.0),
                Instr::CreateMatrix(2, 2),
                Instr::StoreVar(_),
            ]
        ));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn primary_compile_interprets_simple_matrix_indexing() {
        let ast = runmat_parser::parse("x = [1 2; 3 4]; y = x(2, 1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let y_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "y")
            .expect("y export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::Index(2))));
        assert!(!bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::IndexSlice(2, 2, 0, 0))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[y_export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_interprets_simple_colon_slice() {
        let ast = runmat_parser::parse("x = [1 2; 3 4]; y = x(:, 2);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let y_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "y")
            .expect("y export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::IndexSlice(2, 1, 1, 0))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[y_export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(tensor.data, vec![2.0, 4.0]);
    }

    #[test]
    fn primary_compile_lowers_ambiguous_local_index_to_slice() {
        let ast =
            runmat_parser::parse("x = [10 20 30 40]; a = find([0 1 1]); idx = a + 1; y = x(idx);")
                .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let y_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "y")
            .expect("y export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::IndexSlice(1, _, _, _))));
        assert!(!bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::Index(1))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[y_export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(tensor.data, vec![30.0, 40.0]);
    }

    #[test]
    fn primary_compile_lowers_ambiguous_local_store_index_to_slice() {
        let ast =
            runmat_parser::parse("x = [10 20 30 40]; idx = [2 4]; x(idx) = [9 8];").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::StoreSlice(1, _, _, _))));
        assert!(!bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::StoreIndex(1))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[x_export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 4]);
        assert_eq!(tensor.data, vec![10.0, 9.0, 30.0, 8.0]);
    }

    #[test]
    fn primary_compile_rejects_invalid_scalar_index_plan_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3]; y = x(2);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Index { indexing, .. },
                    ..
                } = &mut stmt.kind
                {
                    indexing.plan = MirIndexPlan::Scalar;
                    indexing.components = vec![MirIndexComponent::Colon];
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirScalarIndexPlanInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_invalid_slice_index_plan_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3]; y = x(end);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Index { indexing, .. },
                    ..
                } = &mut stmt.kind
                {
                    indexing.plan = MirIndexPlan::Slice;
                    indexing.components = vec![MirIndexComponent::End {
                        dim: Some(0),
                        offset: 1,
                    }];
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirSliceIndexPlanInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_slice_plan_selector_dimension_beyond_mask_width() {
        let ast = runmat_parser::parse("x = [1 2 3]; y = x(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Index { indexing, .. },
                    ..
                } = &mut stmt.kind
                {
                    indexing.plan = MirIndexPlan::Slice;
                    let seed = indexing.components.first().cloned().expect("seed selector");
                    let mut components = vec![seed; 33];
                    components[32] = MirIndexComponent::Colon;
                    indexing.components = components;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirSliceIndexPlanInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_slice_plan_end_dimension_beyond_mask_width() {
        let ast = runmat_parser::parse("x = [1 2 3]; y = x(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Index { indexing, .. },
                    ..
                } = &mut stmt.kind
                {
                    indexing.plan = MirIndexPlan::Slice;
                    let seed = indexing.components.first().cloned().expect("seed selector");
                    let mut components = vec![seed; 33];
                    components[32] = MirIndexComponent::End {
                        dim: Some(32),
                        offset: 0,
                    };
                    indexing.components = components;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirSliceIndexPlanInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_scalar_plan_with_range_expr_component_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3 4]; y = x(1:end);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Index { indexing, .. },
                    ..
                } = &mut stmt.kind
                {
                    indexing.plan = MirIndexPlan::Scalar;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirScalarIndexPlanInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_slice_plan_with_range_expr_component_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3 4]; y = x(1:end);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Index { indexing, .. },
                    ..
                } = &mut stmt.kind
                {
                    indexing.plan = MirIndexPlan::Slice;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirSliceIndexPlanInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_invalid_paren_cell_plan_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3]; y = x(2);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Index { indexing, .. },
                    ..
                } = &mut stmt.kind
                {
                    indexing.plan = MirIndexPlan::Cell;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirParenCellPlanInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_index_assignment_with_read_context_identifier() {
        let ast = runmat_parser::parse("x=[1,2,3]; x(1)=4;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    place: MirPlace::Index(_, indexing),
                    ..
                } = &mut stmt.kind
                {
                    indexing.result_context = IndexResultContext::ReadSingle;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed assignment place in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirDeletionContextWithoutDeleteInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_index_assignment_with_deletion_context_identifier() {
        let ast = runmat_parser::parse("x=[1,2,3]; x(1)=4;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    place: MirPlace::Index(_, indexing),
                    ..
                } = &mut stmt.kind
                {
                    indexing.result_context = IndexResultContext::DeletionTarget;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed assignment place in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirDeletionContextWithoutDeleteInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_invalid_cell_expand_all_shape_with_identifier() {
        let ast = runmat_parser::parse("c = {1,2;3,4}; [a,b] = c{:,2};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::MultiAssign {
                    value: MirRvalue::Index { indexing, .. },
                    ..
                } = &mut stmt.kind
                {
                    indexing.cell_expand_all = true;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected multi-assign cell expansion in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirCellExpandPlanInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_invalid_mir_aggregate_shape_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3];").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Aggregate { rows, cols: _, .. },
                    ..
                } = &mut stmt.kind
                {
                    *rows = 2;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected aggregate assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirAggregateShapeInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_invalid_cell_index_component_with_identifier() {
        let ast = runmat_parser::parse("c = {1}; c{1} = 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::Index(_, indexing) = place {
                        if matches!(indexing.kind, runmat_hir::IndexKind::Brace) {
                            indexing.components = vec![MirIndexComponent::Colon];
                            patched = true;
                            break;
                        }
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected brace index assignment place in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirCellIndexPlanInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_cell_assignment_colon_selector_from_source_with_identifier() {
        let ast = runmat_parser::parse("c = {1,2;3,4}; c{:,2} = 9;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirCellIndexPlanInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_mismatched_cell_index_context_with_identifier() {
        let ast = runmat_parser::parse("c = {1}; c{1} = 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::Index(_, indexing) = place {
                        if matches!(indexing.kind, runmat_hir::IndexKind::Brace) {
                            indexing.result_context = runmat_hir::IndexResultContext::ReadSingle;
                            patched = true;
                            break;
                        }
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected brace index assignment place in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_member_store_back_brace_index_with_read_context_identifier() {
        let ast = runmat_parser::parse("c = {struct('x', 1)}; c{1}.x = 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::Member(base, _) = place {
                        if let MirPlace::Index(_, indexing) = base.as_mut() {
                            if matches!(indexing.kind, runmat_hir::IndexKind::Brace) {
                                indexing.result_context =
                                    runmat_hir::IndexResultContext::ReadSingle;
                                patched = true;
                                break;
                            }
                        }
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected member-over-brace assignment in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_member_store_back_brace_index_with_deletion_context_identifier() {
        let ast = runmat_parser::parse("c = {struct('x', 1)}; c{1}.x = 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::Member(base, _) = place {
                        if let MirPlace::Index(_, indexing) = base.as_mut() {
                            if matches!(indexing.kind, runmat_hir::IndexKind::Brace) {
                                indexing.result_context =
                                    runmat_hir::IndexResultContext::DeletionTarget;
                                patched = true;
                                break;
                            }
                        }
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected member-over-brace assignment in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_member_store_back_paren_index_with_read_context_identifier() {
        let ast = runmat_parser::parse("s = struct('x', {1, 2}); s(1).x = 3;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::Member(base, _) = place {
                        if let MirPlace::Index(_, indexing) = base.as_mut() {
                            if matches!(indexing.kind, runmat_hir::IndexKind::Paren) {
                                indexing.result_context =
                                    runmat_hir::IndexResultContext::ReadSingle;
                                patched = true;
                                break;
                            }
                        }
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected member-over-paren assignment in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_member_store_back_paren_index_with_deletion_context_identifier() {
        let ast = runmat_parser::parse("s = struct('x', {1, 2}); s(1).x = 3;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::Member(base, _) = place {
                        if let MirPlace::Index(_, indexing) = base.as_mut() {
                            if matches!(indexing.kind, runmat_hir::IndexKind::Paren) {
                                indexing.result_context =
                                    runmat_hir::IndexResultContext::DeletionTarget;
                                patched = true;
                                break;
                            }
                        }
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected member-over-paren assignment in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_interprets_member_store_back_paren_assignment() {
        let ast = runmat_parser::parse(
            "s = struct('x', {1, 2}); s(2).x = 9; t = s(2); y = getfield(t, 'x');",
        )
        .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let y_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "y")
            .expect("y export");

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[y_export.slot.0], Value::Num(9.0));
    }

    #[test]
    fn primary_compile_rejects_dynamic_member_store_back_paren_index_with_read_context_identifier()
    {
        let ast =
            runmat_parser::parse("s = struct('x', {1, 2}); f = 'x'; s(1).(f) = 3;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::DynamicMember(base, _) = place {
                        if let MirPlace::Index(_, indexing) = base.as_mut() {
                            if matches!(indexing.kind, runmat_hir::IndexKind::Paren) {
                                indexing.result_context =
                                    runmat_hir::IndexResultContext::ReadSingle;
                                patched = true;
                                break;
                            }
                        }
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected dynamic-member-over-paren assignment in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_dynamic_member_store_back_paren_index_with_deletion_context_identifier(
    ) {
        let ast =
            runmat_parser::parse("s = struct('x', {1, 2}); f = 'x'; s(1).(f) = 3;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::DynamicMember(base, _) = place {
                        if let MirPlace::Index(_, indexing) = base.as_mut() {
                            if matches!(indexing.kind, runmat_hir::IndexKind::Paren) {
                                indexing.result_context =
                                    runmat_hir::IndexResultContext::DeletionTarget;
                                patched = true;
                                break;
                            }
                        }
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected dynamic-member-over-paren assignment in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_interprets_dynamic_member_store_back_paren_assignment() {
        let ast = runmat_parser::parse(
            "s = struct('x', {1, 2}); f = 'x'; s(2).(f) = 9; t = s(2); y = getfield(t, f);",
        )
        .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let y_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "y")
            .expect("y export");

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[y_export.slot.0], Value::Num(9.0));
    }

    #[test]
    fn primary_compile_rejects_dynamic_member_store_back_brace_index_with_read_context_identifier()
    {
        let ast =
            runmat_parser::parse("c = {struct('x', 1)}; f = 'x'; c{1}.(f) = 3;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::DynamicMember(base, _) = place {
                        if let MirPlace::Index(_, indexing) = base.as_mut() {
                            if matches!(indexing.kind, runmat_hir::IndexKind::Brace) {
                                indexing.result_context =
                                    runmat_hir::IndexResultContext::ReadSingle;
                                patched = true;
                                break;
                            }
                        }
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected dynamic-member-over-brace assignment in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_dynamic_member_store_back_brace_index_with_deletion_context_identifier(
    ) {
        let ast =
            runmat_parser::parse("c = {struct('x', 1)}; f = 'x'; c{1}.(f) = 3;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::DynamicMember(base, _) = place {
                        if let MirPlace::Index(_, indexing) = base.as_mut() {
                            if matches!(indexing.kind, runmat_hir::IndexKind::Brace) {
                                indexing.result_context =
                                    runmat_hir::IndexResultContext::DeletionTarget;
                                patched = true;
                                break;
                            }
                        }
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected dynamic-member-over-brace assignment in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_interprets_dynamic_member_store_back_brace_assignment() {
        let ast = runmat_parser::parse(
            "c = {struct('x', 1)}; f = 'x'; c{1}.(f) = 9; t = c{1}; y = getfield(t, f);",
        )
        .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let y_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "y")
            .expect("y export");

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[y_export.slot.0], Value::Num(9.0));
    }

    #[test]
    fn primary_compile_interprets_dynamic_member_nested_index_delete_store_back() {
        let ast = runmat_parser::parse(
            "s = struct(); s.x = {1, 2, 3}; f = 'x'; s(1).(f)(2) = []; z = getfield(s(1), f); y = z{2};",
        )
        .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let y_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "y")
            .expect("y export");

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[y_export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_rejects_multi_assign_call_output_count_mismatch_with_identifier() {
        let ast = runmat_parser::parse("[a, b] = deal(1, 2);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::MultiAssign {
                    value: MirRvalue::Call(call),
                    ..
                } = &mut stmt.kind
                {
                    call.requested_outputs = RequestedOutputCount::One;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected multi-assign call in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirMultiAssignOutputCountMismatch")
        );
    }

    #[test]
    fn primary_compile_rejects_multi_assign_index_target_context_mismatch_with_identifier() {
        let ast = runmat_parser::parse("a(1)=0; [x, b] = deal(1, 2);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut indexed_place_for_target: Option<MirPlace> = None;
        for block in &body.blocks {
            for stmt in &block.statements {
                if let MirStmtKind::Assign { place, .. } = &stmt.kind {
                    let mut cloned = place.clone();
                    if let MirPlace::Index(_, ref mut idx) = cloned {
                        idx.result_context = IndexResultContext::ReadSingle;
                        indexed_place_for_target = Some(cloned);
                    }
                }
                if indexed_place_for_target.is_some() {
                    break;
                }
            }
            if indexed_place_for_target.is_some() {
                break;
            }
        }

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::MultiAssign { targets, .. } = &mut stmt.kind {
                    let place = indexed_place_for_target
                        .clone()
                        .expect("expected indexed place in lowered MIR");
                    if let Some(target) = targets.targets.first_mut() {
                        *target = MirOutputTarget::Place(place);
                        patched = true;
                    }
                }
                if patched {
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected indexed multi-assign output target in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_multi_assign_index_target_deletion_context_identifier() {
        let ast = runmat_parser::parse("a(1)=0; [x, b] = deal(1, 2);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut indexed_place_for_target: Option<MirPlace> = None;
        for block in &body.blocks {
            for stmt in &block.statements {
                if let MirStmtKind::Assign { place, .. } = &stmt.kind {
                    let mut cloned = place.clone();
                    if let MirPlace::Index(_, ref mut idx) = cloned {
                        idx.result_context = IndexResultContext::DeletionTarget;
                        indexed_place_for_target = Some(cloned);
                    }
                }
                if indexed_place_for_target.is_some() {
                    break;
                }
            }
            if indexed_place_for_target.is_some() {
                break;
            }
        }

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::MultiAssign { targets, .. } = &mut stmt.kind {
                    let place = indexed_place_for_target
                        .clone()
                        .expect("expected indexed place in lowered MIR");
                    if let Some(target) = targets.targets.first_mut() {
                        *target = MirOutputTarget::Place(place);
                        patched = true;
                    }
                }
                if patched {
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(
            patched,
            "expected indexed multi-assign output target in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_nonempty_delete_rhs_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3]; x(2) = [];").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, value } = &mut stmt.kind {
                    if matches!(place, MirPlace::Index(_, _)) {
                        *value = MirRvalue::Aggregate {
                            kind: MirAggregateKind::Tensor,
                            rows: 1,
                            cols: 1,
                            elements: vec![MirOperand::Constant(MirConstant::Number(
                                "1".to_string(),
                            ))],
                        };
                        patched = true;
                        break;
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed delete assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirDeleteAssignmentRhsInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_delete_place_mismatch_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3]; x(2) = [];").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::Index(base, _) = place {
                        *place = (**base).clone();
                        patched = true;
                        break;
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed delete assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirDeleteAssignmentPlaceMismatch")
        );
    }

    #[test]
    fn primary_compile_rejects_delete_on_nonindexed_target_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3]; x(2) = [];").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let replacement = body
            .blocks
            .iter()
            .flat_map(|block| block.statements.iter())
            .find_map(|stmt| match &stmt.kind {
                MirStmtKind::Assign { place, .. } => match place {
                    MirPlace::Index(base, _) => Some((**base).clone()),
                    _ => None,
                },
                _ => None,
            })
            .expect("expected indexed delete assignment target");

        let mut patched_assign = false;
        let mut patched_mutation = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                match &mut stmt.kind {
                    MirStmtKind::Assign { place, .. } => {
                        if matches!(place, MirPlace::Index(_, _)) {
                            *place = replacement.clone();
                            patched_assign = true;
                        }
                    }
                    MirStmtKind::PlaceMutation(mutation) => {
                        if matches!(mutation.kind, runmat_hir::PlaceMutationKind::Delete) {
                            mutation.place = replacement.clone();
                            patched_mutation = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        assert!(
            patched_assign,
            "expected indexed delete assign stmt in lowered MIR"
        );
        assert!(
            patched_mutation,
            "expected delete place mutation stmt in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirDeleteAssignmentTargetInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_delete_on_brace_index_target_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3]; x(2) = [];").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched_assign = false;
        let mut patched_mutation = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                match &mut stmt.kind {
                    MirStmtKind::Assign { place, .. } => {
                        if let MirPlace::Index(_, indexing) = place {
                            indexing.kind = runmat_hir::IndexKind::Brace;
                            patched_assign = true;
                        }
                    }
                    MirStmtKind::PlaceMutation(mutation) => {
                        if let MirPlace::Index(_, indexing) = &mut mutation.place {
                            indexing.kind = runmat_hir::IndexKind::Brace;
                            patched_mutation = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        assert!(
            patched_assign,
            "expected indexed delete assign stmt in lowered MIR"
        );
        assert!(
            patched_mutation,
            "expected delete place mutation stmt in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirDeleteAssignmentIndexKindInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_delete_with_nondeletion_index_context_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3]; x(2) = [];").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched_assign = false;
        let mut patched_mutation = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                match &mut stmt.kind {
                    MirStmtKind::Assign { place, .. } => {
                        if let MirPlace::Index(_, indexing) = place {
                            indexing.result_context = IndexResultContext::AssignmentTarget;
                            patched_assign = true;
                        }
                    }
                    MirStmtKind::PlaceMutation(mutation) => {
                        if let MirPlace::Index(_, indexing) = &mut mutation.place {
                            indexing.result_context = IndexResultContext::AssignmentTarget;
                            patched_mutation = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        assert!(
            patched_assign,
            "expected indexed delete assign stmt in lowered MIR"
        );
        assert!(
            patched_mutation,
            "expected delete place mutation stmt in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirDeleteAssignmentContextInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_deletion_context_without_delete_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3]; x(2) = 9;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched_assign = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { place, .. } = &mut stmt.kind {
                    if let MirPlace::Index(_, indexing) = place {
                        indexing.result_context = IndexResultContext::DeletionTarget;
                        patched_assign = true;
                        break;
                    }
                }
            }
            if patched_assign {
                break;
            }
        }
        assert!(
            patched_assign,
            "expected indexed non-delete assign stmt in lowered MIR"
        );

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirDeletionContextWithoutDeleteInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_delete_with_nonexisting_creation_policy_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3]; x(2) = [];").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::PlaceMutation(mutation) = &mut stmt.kind {
                    if matches!(mutation.kind, runmat_hir::PlaceMutationKind::Delete) {
                        mutation.creation_policy = AssignmentCreationPolicy::CreateArrayByIndex;
                        patched = true;
                        break;
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected delete place mutation in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirDeleteAssignmentCreationPolicyInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_invalid_read_index_context_with_identifier() {
        let ast = runmat_parser::parse("x = [1 2 3]; y = x(2);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Index { indexing, .. },
                    ..
                } = &mut stmt.kind
                {
                    indexing.result_context = IndexResultContext::AssignmentTarget;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected indexed read assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirIndexContextInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_unsupported_mir_unary_operator_with_identifier() {
        let ast = runmat_parser::parse("x = -1;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Unary(op, _),
                    ..
                } = &mut stmt.kind
                {
                    *op = OperatorKind::Add;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected unary assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirOperatorUnsupported")
        );
    }

    #[test]
    fn primary_compile_rejects_unsupported_mir_binary_operator_with_identifier() {
        let ast = runmat_parser::parse("x = 1 + 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Binary(_, op, _),
                    ..
                } = &mut stmt.kind
                {
                    *op = OperatorKind::Transpose;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected binary assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirOperatorUnsupported")
        );
    }

    #[test]
    fn primary_compile_rejects_unknown_mir_builtin_id_with_identifier() {
        let ast = runmat_parser::parse("x = sin(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Call(call),
                    ..
                } = &mut stmt.kind
                {
                    call.callee = MirCallee::Static(CallableIdentity::Builtin(BuiltinId(
                        "__not_a_builtin".into(),
                    )));
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected call assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(err.identifier.as_deref(), Some("RunMat:MirBuiltinUnknown"));
    }

    #[test]
    fn primary_compile_rejects_invalid_mir_number_literal_with_identifier() {
        let ast = runmat_parser::parse("x = 1;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { value, .. } = &mut stmt.kind {
                    *value = MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
                        "not_a_number".into(),
                    )));
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirNumberLiteralInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_unknown_mir_constant_with_identifier() {
        let ast = runmat_parser::parse("x = pi;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { value, .. } = &mut stmt.kind {
                    *value = MirRvalue::Use(MirOperand::Constant(MirConstant::Symbol(
                        runmat_hir::SymbolName("definitely_missing_constant".to_string()),
                    )));
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(err.identifier.as_deref(), Some("RunMat:MirConstantUnknown"));
    }

    #[test]
    fn primary_compile_rejects_missing_mir_function_handle_runtime_name_with_identifier() {
        let ast = runmat_parser::parse("f = @sin;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { value, .. } = &mut stmt.kind {
                    *value = MirRvalue::Use(MirOperand::FunctionHandle(
                        CallableIdentity::ExternalName(QualifiedName(vec![
                            SymbolName("pkg".to_string()),
                            SymbolName(String::new()),
                            SymbolName("broken".to_string()),
                        ])),
                    ));
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirFunctionHandleNameMissing")
        );
    }

    #[test]
    fn primary_compile_rejects_single_segment_external_function_handle_name_with_identifier() {
        let ast = runmat_parser::parse("f = @sin;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { value, .. } = &mut stmt.kind {
                    *value =
                        MirRvalue::Use(MirOperand::FunctionHandle(CallableIdentity::ExternalName(
                            QualifiedName(vec![SymbolName("pkg".to_string())]),
                        )));
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirFunctionHandleNameMissing")
        );
    }

    #[test]
    fn primary_compile_rejects_empty_dynamic_function_handle_name_with_identifier() {
        let ast = runmat_parser::parse("f = @sin;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { value, .. } = &mut stmt.kind {
                    *value = MirRvalue::Use(MirOperand::FunctionHandle(
                        CallableIdentity::DynamicName(SymbolName(String::new())),
                    ));
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirFunctionHandleNameMissing")
        );
    }

    #[test]
    fn primary_compile_rejects_empty_builtin_function_handle_name_with_identifier() {
        let ast = runmat_parser::parse("f = @sin;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { value, .. } = &mut stmt.kind {
                    *value = MirRvalue::Use(MirOperand::FunctionHandle(CallableIdentity::Builtin(
                        BuiltinId(String::new()),
                    )));
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirFunctionHandleNameMissing")
        );
    }

    #[test]
    fn primary_compile_rejects_method_function_handle_target_with_identifier() {
        let ast = runmat_parser::parse("f = @sin;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { value, .. } = &mut stmt.kind {
                    *value = MirRvalue::Use(MirOperand::FunctionHandle(CallableIdentity::Method(
                        MethodId("m".to_string()),
                    )));
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirFunctionHandleTargetUnsupported")
        );
    }

    #[test]
    fn primary_compile_rejects_empty_imported_module_function_handle_name_with_identifier() {
        let ast = runmat_parser::parse("f = @sin;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { value, .. } = &mut stmt.kind {
                    *value = MirRvalue::Use(MirOperand::FunctionHandle(
                        CallableIdentity::Imported(DefPath {
                            package: PackageName("pkg".to_string()),
                            module: QualifiedName(vec![]),
                            item: vec![DefPathSegment::Function(SymbolName("target".to_string()))],
                        }),
                    ));
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirFunctionHandleNameMissing")
        );
    }

    #[test]
    fn primary_compile_rejects_imported_function_handle_missing_item_with_identifier() {
        let ast = runmat_parser::parse("f = @sin;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { value, .. } = &mut stmt.kind {
                    *value = MirRvalue::Use(MirOperand::FunctionHandle(
                        CallableIdentity::Imported(DefPath {
                            package: PackageName("pkg".to_string()),
                            module: QualifiedName(vec![
                                SymbolName("pkg".to_string()),
                                SymbolName("mod".to_string()),
                            ]),
                            item: vec![],
                        }),
                    ));
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirFunctionHandleNameMissing")
        );
    }

    #[test]
    fn primary_compile_rejects_imported_function_handle_mismatched_item_with_identifier() {
        let ast = runmat_parser::parse("f = @sin;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign { value, .. } = &mut stmt.kind {
                    *value = MirRvalue::Use(MirOperand::FunctionHandle(
                        CallableIdentity::Imported(DefPath {
                            package: PackageName("pkg".to_string()),
                            module: QualifiedName(vec![
                                SymbolName("pkg".to_string()),
                                SymbolName("target".to_string()),
                            ]),
                            item: vec![DefPathSegment::Function(SymbolName(
                                "different".to_string(),
                            ))],
                        }),
                    ));
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirFunctionHandleNameMissing")
        );
    }

    #[test]
    fn primary_compile_rejects_unsupported_mir_static_call_fallback_policy_with_identifier() {
        let ast = runmat_parser::parse("x = sin(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Call(call),
                    ..
                } = &mut stmt.kind
                {
                    call.callee = MirCallee::Static(CallableIdentity::Method(MethodId("m".into())));
                    call.fallback_policy = CallableFallbackPolicy::ObjectDispatch;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected call assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirCallFallbackPolicyUnsupported")
        );
    }

    #[test]
    fn primary_compile_rejects_static_call_with_mismatched_imported_identity_name_shape() {
        let ast = runmat_parser::parse("x = sin(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Call(call),
                    ..
                } = &mut stmt.kind
                {
                    call.callee = MirCallee::Static(CallableIdentity::Imported(DefPath {
                        package: PackageName("pkg".to_string()),
                        module: QualifiedName(vec![
                            SymbolName("pkg".to_string()),
                            SymbolName("target".to_string()),
                        ]),
                        item: vec![DefPathSegment::Function(SymbolName(
                            "different".to_string(),
                        ))],
                    }));
                    call.fallback_policy = CallableFallbackPolicy::RuntimeNameResolution;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected call assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirCallTargetNameInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_static_call_with_single_segment_external_identity() {
        let ast = runmat_parser::parse("x = sin(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Call(call),
                    ..
                } = &mut stmt.kind
                {
                    call.callee =
                        MirCallee::Static(CallableIdentity::ExternalName(QualifiedName(vec![
                            SymbolName("sqrt".to_string()),
                        ])));
                    call.fallback_policy = CallableFallbackPolicy::ExternalBoundary;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected call assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirCallTargetNameInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_static_call_with_method_identity_name_shape() {
        let ast = runmat_parser::parse("x = sin(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::Assign {
                    value: MirRvalue::Call(call),
                    ..
                } = &mut stmt.kind
                {
                    call.callee = MirCallee::Static(CallableIdentity::Method(MethodId(
                        "remote_inc".to_string(),
                    )));
                    call.fallback_policy = CallableFallbackPolicy::RuntimeNameResolution;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected call assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirCallTargetNameInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_multi_assign_static_call_with_invalid_name_shape() {
        let ast = runmat_parser::parse("[a, b] = max([1,2]);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::MultiAssign {
                    value: MirRvalue::Call(call),
                    ..
                } = &mut stmt.kind
                {
                    call.callee = MirCallee::Static(CallableIdentity::Imported(DefPath {
                        package: PackageName("pkg".to_string()),
                        module: QualifiedName(vec![
                            SymbolName("pkg".to_string()),
                            SymbolName("target".to_string()),
                        ]),
                        item: vec![DefPathSegment::Function(SymbolName(
                            "different".to_string(),
                        ))],
                    }));
                    call.fallback_policy = CallableFallbackPolicy::RuntimeNameResolution;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected multi-assign call in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirCallTargetNameInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_multi_assign_static_call_with_method_identity_name_shape() {
        let ast = runmat_parser::parse("[a, b] = max([1,2]);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::MultiAssign {
                    value: MirRvalue::Call(call),
                    ..
                } = &mut stmt.kind
                {
                    call.callee = MirCallee::Static(CallableIdentity::Method(MethodId(
                        "remote_pair".to_string(),
                    )));
                    call.fallback_policy = CallableFallbackPolicy::RuntimeNameResolution;
                    patched = true;
                    break;
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected multi-assign call in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirCallTargetNameInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_unsupported_mir_method_call_fallback_policy_with_identifier() {
        let ast = runmat_parser::parse("obj = 1; obj.method(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                let maybe_call = match &mut stmt.kind {
                    MirStmtKind::Assign {
                        value: MirRvalue::Call(call),
                        ..
                    }
                    | MirStmtKind::Expr(MirRvalue::Call(call)) => Some(call),
                    _ => None,
                };
                if let Some(call) = maybe_call {
                    if matches!(
                        call.syntax,
                        runmat_hir::CallSyntax::Method | runmat_hir::CallSyntax::DottedInvoke
                    ) {
                        call.fallback_policy = CallableFallbackPolicy::ExternalBoundary;
                        patched = true;
                        break;
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected method call assignment in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirMethodFallbackPolicyUnsupported")
        );
    }

    #[test]
    fn primary_compile_rejects_missing_mir_method_call_receiver_with_identifier() {
        let ast = runmat_parser::parse("obj = 1; obj.method(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                let maybe_call = match &mut stmt.kind {
                    MirStmtKind::Assign {
                        value: MirRvalue::Call(call),
                        ..
                    }
                    | MirStmtKind::Expr(MirRvalue::Call(call)) => Some(call),
                    _ => None,
                };
                if let Some(call) = maybe_call {
                    if matches!(
                        call.syntax,
                        runmat_hir::CallSyntax::Method | runmat_hir::CallSyntax::DottedInvoke
                    ) {
                        call.args.clear();
                        patched = true;
                        break;
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected method call in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirMethodCallReceiverMissing")
        );
    }

    #[test]
    fn primary_compile_rejects_invalid_mir_method_call_callee_with_identifier() {
        let ast = runmat_parser::parse("obj = 1; obj.method(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                let maybe_call = match &mut stmt.kind {
                    MirStmtKind::Assign {
                        value: MirRvalue::Call(call),
                        ..
                    }
                    | MirStmtKind::Expr(MirRvalue::Call(call)) => Some(call),
                    _ => None,
                };
                if let Some(call) = maybe_call {
                    if matches!(
                        call.syntax,
                        runmat_hir::CallSyntax::Method | runmat_hir::CallSyntax::DottedInvoke
                    ) {
                        call.callee = MirCallee::Dynamic(MirOperand::Constant(
                            MirConstant::Number("1".into()),
                        ));
                        patched = true;
                        break;
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected method call in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirMethodCallCalleeInvalid")
        );
    }

    #[test]
    fn primary_compile_rejects_invalid_mir_multi_assign_method_call_callee_with_identifier() {
        let ast = runmat_parser::parse("obj = 1; [a, b] = obj.method(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");

        let mut patched = false;
        for block in &mut body.blocks {
            for stmt in &mut block.statements {
                if let MirStmtKind::MultiAssign {
                    value: MirRvalue::Call(call),
                    ..
                } = &mut stmt.kind
                {
                    if matches!(
                        call.syntax,
                        runmat_hir::CallSyntax::Method | runmat_hir::CallSyntax::DottedInvoke
                    ) {
                        call.callee = MirCallee::Dynamic(MirOperand::Constant(
                            MirConstant::Number("1".into()),
                        ));
                        patched = true;
                        break;
                    }
                }
            }
            if patched {
                break;
            }
        }
        assert!(patched, "expected multi-assign method call in lowered MIR");

        let err = compile(&hir.assembly, &mir, entrypoint).expect_err("compile should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:MirMethodCallCalleeInvalid")
        );
    }

    #[test]
    fn primary_compile_lowers_statement_semantic_call_to_zero_outputs() {
        let ast =
            runmat_parser::parse("function y = f(x); y = nargout(); end; f(10);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::CallSemanticFunctionMulti(_, _, 0))));
    }

    #[test]
    fn primary_compile_lowers_method_calls_with_explicit_object_dispatch_policy() {
        let ast = runmat_parser::parse("obj = 1; obj.method(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(bytecode.instructions.iter().any(|instr| matches!(
            instr,
            Instr::CallMethodOrMemberIndexMulti {
                fallback_policy: CallableFallbackPolicy::ObjectDispatch,
                ..
            }
        )));
    }

    #[test]
    fn primary_compile_interprets_simple_indexed_assignment() {
        let ast = runmat_parser::parse("x = [1 2; 3 4]; x(1, 2) = 9;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::StoreIndex(2))));
        assert!(!bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::StoreSlice(2, 2, 0, 0))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[x_export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 3.0, 9.0, 4.0]);
    }

    #[test]
    fn primary_compile_interprets_simple_slice_assignment() {
        let ast = runmat_parser::parse("x = [1 2; 3 4]; x(:, 2) = [9; 8];").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::StoreSlice(2, 1, 1, 0))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[x_export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 3.0, 9.0, 8.0]);
    }

    #[test]
    fn primary_compile_interprets_simple_cell_indexing() {
        let ast = runmat_parser::parse("c = {1, 2}; x = c{2};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(2.0));
    }

    #[test]
    fn primary_compile_interprets_simple_cell_indexed_assignment() {
        let ast = runmat_parser::parse("c = {1, 2}; c{2} = 9; x = c{2};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::StoreIndexCell { num_indices: 1, .. })));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(9.0));
    }

    #[test]
    fn primary_compile_carries_cell_end_selector_metadata_for_reads() {
        let ast = runmat_parser::parse("c = {1, 2, 3}; x = c{end};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        assert!(bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                Instr::IndexCell {
                    num_indices: 1,
                    end_offsets,
                    ..
                } if end_offsets == &vec![(0, 0)]
            ) || matches!(
                instr,
                Instr::IndexCellList {
                    num_indices: 1,
                    end_offsets,
                    ..
                } if end_offsets == &vec![(0, 0)]
            )
        }));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_carries_cell_end_selector_metadata_for_stores() {
        let ast = runmat_parser::parse("c = {1, 2, 3}; c{end} = 9; x = c{3};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        assert!(bytecode.instructions.iter().any(|instr| matches!(
            instr,
            Instr::StoreIndexCell {
                num_indices: 1,
                end_offsets,
                ..
            } if end_offsets == &vec![(0, 0)]
        )));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(9.0));
    }

    #[test]
    fn primary_compile_carries_cell_end_offset_selector_metadata_in_semantic_function_reads() {
        let ast = runmat_parser::parse(
            "function y = tail_cell(c); y = c{end-1}; end; c = {1, 2, 3}; x = tail_cell(c);",
        )
        .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let function_id = bytecode
            .semantic_function_registry
            .resolve_name("tail_cell")
            .expect("tail_cell semantic function id");
        let function = bytecode
            .semantic_function_registry
            .get(function_id)
            .expect("tail_cell semantic bytecode");
        let read_offsets: Vec<Vec<(usize, isize)>> = function
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::IndexCell {
                    num_indices: 1,
                    end_offsets,
                    ..
                }
                | Instr::IndexCellList {
                    num_indices: 1,
                    end_offsets,
                    ..
                } => Some(end_offsets.clone()),
                _ => None,
            })
            .collect();
        assert!(
            read_offsets.iter().any(|offsets| offsets == &vec![(0, -1)]),
            "expected semantic function end-1 metadata offset; actual offsets: {read_offsets:?}; instructions: {:?}",
            function.instructions
        );

        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(2.0));
    }

    #[test]
    fn primary_compile_carries_cell_end_offset_selector_metadata_in_semantic_function_stores() {
        let ast = runmat_parser::parse(
            "function y = patch_cell(c, v); c{end-1} = v; y = c{2}; end; c = {1, 2, 3}; x = patch_cell(c, 9);",
        )
        .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let function_id = bytecode
            .semantic_function_registry
            .resolve_name("patch_cell")
            .expect("patch_cell semantic function id");
        let function = bytecode
            .semantic_function_registry
            .get(function_id)
            .expect("patch_cell semantic bytecode");
        let store_offsets: Vec<Vec<(usize, isize)>> = function
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreIndexCell {
                    num_indices: 1,
                    end_offsets,
                    ..
                } => Some(end_offsets.clone()),
                _ => None,
            })
            .collect();
        assert!(
            store_offsets
                .iter()
                .any(|offsets| offsets == &vec![(0, -1)]),
            "expected semantic function end-1 metadata offset; actual offsets: {store_offsets:?}"
        );

        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(9.0));
    }

    #[test]
    fn primary_compile_supports_general_cell_end_expression_reads() {
        let ast = runmat_parser::parse("c = {10, 20, 30, 40}; x = c{end/2};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                Instr::IndexCell {
                    num_indices: 1,
                    end_exprs,
                    ..
                } if end_exprs.iter().any(|(pos, expr)| {
                    *pos == 0
                        && matches!(
                            expr,
                            crate::bytecode::EndExpr::Div(left, right)
                                if matches!(left.as_ref(), crate::bytecode::EndExpr::End)
                                    && matches!(right.as_ref(), crate::bytecode::EndExpr::Const(v) if (*v - 2.0).abs() < f64::EPSILON)
                        )
                })
            ) || matches!(
                instr,
                Instr::IndexCellList {
                    num_indices: 1,
                    end_exprs,
                    ..
                } if end_exprs.iter().any(|(pos, expr)| {
                    *pos == 0
                        && matches!(
                            expr,
                            crate::bytecode::EndExpr::Div(left, right)
                                if matches!(left.as_ref(), crate::bytecode::EndExpr::End)
                                    && matches!(right.as_ref(), crate::bytecode::EndExpr::Const(v) if (*v - 2.0).abs() < f64::EPSILON)
                        )
                })
            )
        }));
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(20.0));
    }

    #[test]
    fn primary_compile_rejects_fractional_cell_end_expression_read_index() {
        let ast = runmat_parser::parse("c = {10, 20, 30, 40, 50}; x = c{end/2};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let err = block_on(crate::interpret(&bytecode))
            .expect_err("fractional cell end expression selector should fail");
        assert_eq!(err.identifier(), Some("RunMat:UnsupportedIndexType"));
    }

    #[test]
    fn primary_compile_supports_general_cell_end_expression_stores() {
        let ast = runmat_parser::parse("c = {1, 2, 3, 4}; c{floor(end/2)} = 9; x = c{2};")
            .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                Instr::StoreIndexCell {
                    num_indices: 1,
                    end_exprs,
                    ..
                } if end_exprs.iter().any(|(pos, expr)| {
                    *pos == 0
                        && matches!(
                            expr,
                            crate::bytecode::EndExpr::ResolvedCall { args, .. } if args.len() == 1
                        )
                })
            )
        }));
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(9.0));
    }

    #[test]
    fn primary_compile_supports_cell_brace_end_plus_one_growth_for_vectors() {
        let ast = runmat_parser::parse("c = {1, 2}; c{end+1} = 9; x = c{3};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                Instr::StoreIndexCell {
                    num_indices: 1,
                    end_offsets,
                    ..
                } if end_offsets == &vec![(0, 1)]
            )
        }));
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(9.0));
    }

    #[test]
    fn primary_compile_supports_cell_brace_linear_gap_growth_for_vectors() {
        let ast = runmat_parser::parse(
            "c = {1, 2}; c{5} = 9; a = isempty(c{3}); b = isempty(c{4}); x = c{5};",
        )
        .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let c_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "c")
            .expect("c export");
        let a_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "a")
            .expect("a export");
        let b_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "b")
            .expect("b export");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[a_export.slot.0], Value::Bool(true));
        assert_eq!(vars[b_export.slot.0], Value::Bool(true));
        assert_eq!(vars[x_export.slot.0], Value::Num(9.0));
        match &vars[c_export.slot.0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 5);
            }
            other => panic!("expected cell export, got {other:?}"),
        }
    }

    #[test]
    fn primary_compile_supports_cell_brace_linear_end_plus_k_growth_for_vectors() {
        let ast = runmat_parser::parse("c = {1, 2}; c{end+3} = 9; a = isempty(c{3}); x = c{5};")
            .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                Instr::StoreIndexCell {
                    num_indices: 1,
                    end_offsets,
                    ..
                } if end_offsets == &vec![(0, 3)]
            )
        }));
        let layout = bytecode.layout.as_ref().expect("layout");
        let a_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "a")
            .expect("a export");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[a_export.slot.0], Value::Bool(true));
        assert_eq!(vars[x_export.slot.0], Value::Num(9.0));
    }

    #[test]
    fn primary_compile_linear_cell_growth_from_5_by_0_normalizes_to_row_vector() {
        let ast = runmat_parser::parse("c = cell(5,0); c{3} = 2; v = c{3};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let c_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "c")
            .expect("c export");
        let v_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "v")
            .expect("v export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[v_export.slot.0], Value::Num(2.0));
        match &vars[c_export.slot.0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 3);
            }
            other => panic!("expected cell export, got {other:?}"),
        }
    }

    #[test]
    fn primary_compile_linear_cell_growth_from_0_by_5_normalizes_to_row_vector() {
        let ast = runmat_parser::parse("c = cell(0,5); c{3} = 2; v = c{3};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let c_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "c")
            .expect("c export");
        let v_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "v")
            .expect("v export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[v_export.slot.0], Value::Num(2.0));
        match &vars[c_export.slot.0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 3);
            }
            other => panic!("expected cell export, got {other:?}"),
        }
    }

    #[test]
    fn primary_compile_rejects_cell_brace_end_plus_one_growth_for_matrix_linear_assignment() {
        let ast = runmat_parser::parse("c = {1, 2; 3, 4}; c{end+1} = 9;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let err = block_on(crate::interpret(&bytecode))
            .expect_err("matrix linear brace end+1 growth should be rejected");
        assert_eq!(err.identifier(), Some("RunMat:UnsupportedCellGrowth"));
    }

    #[test]
    fn primary_compile_supports_cell_brace_subscript_growth_with_empty_fillers() {
        let ast =
            runmat_parser::parse("c = {1, 2; 3, 4}; c{3,3} = 9; a = c{3,3}; b = isempty(c{2,3});")
                .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let c_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "c")
            .expect("c export");
        let a_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "a")
            .expect("a export");
        let b_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "b")
            .expect("b export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");

        assert_eq!(vars[a_export.slot.0], Value::Num(9.0));
        assert_eq!(vars[b_export.slot.0], Value::Bool(true));
        match &vars[c_export.slot.0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 3);
                assert_eq!(ca.cols, 3);
            }
            other => panic!("expected cell export, got {other:?}"),
        }
    }

    #[test]
    fn primary_compile_supports_cell_brace_end_plus_one_subscript_growth() {
        let ast =
            runmat_parser::parse("c = {1, 2; 3, 4}; c{end,end+1} = 8; x = c{2,3};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                Instr::StoreIndexCell {
                    num_indices: 2,
                    end_offsets,
                    ..
                } if end_offsets.iter().any(|entry| *entry == (1, 1))
            )
        }));
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(8.0));
    }

    #[test]
    fn primary_compile_supports_mixed_cell_colon_expansion() {
        let ast = runmat_parser::parse("c = {1,2;3,4}; [a,b] = c{:,2}; z = a + b;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let z_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "z")
            .expect("z export");

        assert!(bytecode.instructions.iter().any(|instr| matches!(
            instr,
            Instr::IndexCellExpand {
                num_indices,
                out_count,
                ..
            } if *num_indices == 2 && *out_count == 2
        )));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[z_export.slot.0], Value::Num(6.0));
    }

    #[test]
    fn primary_compile_3d_slice_roundtrip_uses_slice_expr_paths() {
        let ast = runmat_parser::parse(
            r#"
            A = reshape([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24], 3, 4, 2);
            S = A(1:2, 2:3, end);
            A(1:2, 2:3, end) = S;
        "#,
        )
        .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let mut saw_index_expr = false;
        let mut saw_store_expr = false;
        for instr in &bytecode.instructions {
            if let Instr::IndexSliceExpr {
                dims,
                end_mask,
                end_numeric_exprs,
                range_dims,
                ..
            } = instr
            {
                if *dims == 3 {
                    saw_index_expr = true;
                    assert_eq!(*end_mask, 0);
                    assert_eq!(end_numeric_exprs.len(), 1);
                    assert_eq!(range_dims, &vec![0, 1]);
                }
            }
            if let Instr::StoreSliceExpr {
                dims,
                numeric_count,
                colon_mask,
                end_mask,
                range_dims,
                range_has_step,
                end_numeric_exprs,
                ..
            } = instr
            {
                if *dims == 3 {
                    saw_store_expr = true;
                    assert_eq!(*numeric_count, 1);
                    assert_eq!(*colon_mask, 0);
                    assert_eq!(*end_mask, 0);
                    assert_eq!(range_dims, &vec![0, 1]);
                    assert_eq!(range_has_step, &vec![false, false]);
                    assert_eq!(end_numeric_exprs.len(), 1);
                }
            }
        }
        assert!(saw_index_expr);
        assert!(saw_store_expr);

        let run = block_on(crate::interpret(&bytecode));
        assert!(
            run.is_ok(),
            "roundtrip script should interpret successfully: {run:?}"
        );
    }

    #[test]
    fn primary_compile_interprets_basic_if_statement() {
        let ast = runmat_parser::parse("if 1; x = 2; else; x = 3; end").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::JumpIfFalse(_))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(2.0));
    }

    #[test]
    fn primary_compile_interprets_basic_switch_statement() {
        let ast =
            runmat_parser::parse("switch 2; case 1; x = 1; case 2; x = 2; otherwise; x = 3; end")
                .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::Equal)));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(2.0));
    }

    #[test]
    fn primary_compile_lowers_unreachable_terminator() {
        let ast = runmat_parser::parse("x = 1;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");
        body.blocks.last_mut().expect("entry block").terminator.kind =
            MirTerminatorKind::Unreachable;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::Return)));
    }

    #[test]
    fn primary_compile_external_semantic_function_handle_keeps_identity() {
        let ast = runmat_parser::parse("h = @remote_inc; y = feval(h, 2);").expect("parse");
        let mut semantic_functions = HashMap::new();
        semantic_functions.insert("remote_inc".to_string(), FunctionId(9001));
        let context = LoweringContext::empty().with_semantic_functions(&semantic_functions);
        let hir = lower(&ast, &context).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(bytecode.instructions.iter().any(|instr| matches!(
            instr,
            Instr::CreateSemanticFunctionHandle(FunctionId(9001), _)
        )));

        let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(
            Some(Arc::new(|name| {
                if name == "remote_inc" {
                    Some(9001)
                } else {
                    None
                }
            })),
        );
        let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(
            Some(Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 9001);
                assert_eq!(args, &[Value::Num(2.0)]);
                assert_eq!(requested_outputs, 1);
                Box::pin(async move { Ok(Value::Num(3.0)) })
            })),
        );

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert!(vars
            .iter()
            .any(|value| matches!(value, Value::Num(n) if (*n - 3.0).abs() < 1e-12)));
    }

    #[test]
    fn primary_compile_external_semantic_direct_call_uses_host_invoker() {
        let ast = runmat_parser::parse("y = remote_inc(2);").expect("parse");
        let mut semantic_functions = HashMap::new();
        semantic_functions.insert("remote_inc".to_string(), FunctionId(9001));
        let context = LoweringContext::empty().with_semantic_functions(&semantic_functions);
        let hir = lower(&ast, &context).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(bytecode.instructions.iter().any(|instr| matches!(
            instr,
            Instr::CallSemanticFunctionMulti(FunctionId(9001), 1, 1)
        )));

        let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(
            Some(Arc::new(|name| {
                if name == "remote_inc" {
                    Some(9001)
                } else {
                    None
                }
            })),
        );
        let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(
            Some(Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 9001);
                assert_eq!(args, &[Value::Num(2.0)]);
                assert_eq!(requested_outputs, 1);
                Box::pin(async move { Ok(Value::Num(3.0)) })
            })),
        );

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert!(vars
            .iter()
            .any(|value| matches!(value, Value::Num(n) if (*n - 3.0).abs() < 1e-12)));
    }

    #[test]
    fn primary_compile_interprets_async_call_and_await_via_semantic_future_lane() {
        let source = "async function y = inc(x); y = x + 1; end; t = inc(2); z = await(t);";
        let ast = runmat_parser::parse(source).expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(
            bytecode
                .instructions
                .iter()
                .any(|instr| matches!(instr, Instr::CreateSemanticFuture(FunctionId(_), 1, 1))),
            "expected async direct call lowering to create a semantic future descriptor"
        );
        let layout = bytecode.layout.as_ref().expect("layout");
        let z_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "z")
            .expect("z export");

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[z_export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_emits_explicit_spawn_instruction() {
        let source = "async function y = inc(x); y = x + 1; end; t = spawn(inc(2)); z = await(t);";
        let ast = runmat_parser::parse(source).expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(
            bytecode
                .instructions
                .iter()
                .any(|instr| matches!(instr, Instr::Spawn)),
            "expected MIR spawn lowering to emit an explicit spawn instruction"
        );
        assert!(
            bytecode
                .instructions
                .iter()
                .any(|instr| matches!(instr, Instr::Await)),
            "expected MIR await lowering to emit an explicit await instruction"
        );

        let layout = bytecode.layout.as_ref().expect("layout");
        let z_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "z")
            .expect("z export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[z_export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_lowers_async_expansion_call_to_future_expand_instruction() {
        let source =
            "async function y = inc(x); y = x + 1; end; args = {2}; t = inc(args{:}); z = await(t);";
        let ast = runmat_parser::parse(source).expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(
            bytecode
                .instructions
                .iter()
                .any(|instr| matches!(
                    instr,
                    Instr::CreateSemanticFutureExpandMultiOutput(FunctionId(_), _, 1)
                )),
            "expected async expansion call lowering to create a semantic future expansion descriptor"
        );
        let layout = bytecode.layout.as_ref().expect("layout");
        let z_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "z")
            .expect("z export");
        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[z_export.slot.0], Value::Num(3.0));
    }
}
