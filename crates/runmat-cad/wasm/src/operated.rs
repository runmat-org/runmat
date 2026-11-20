//! Component displays with the assembly's features applied ([`runmat_cad::assembly::operations`]).
//!
//! Each component an assembly feature affects is regenerated on the installed kernel in its
//! own configuration, then each feature's tool (built in assembly coordinates and moved into
//! the component's by the inverse of its placement) is applied to its bodies in the order the
//! features are written: a cut and holes remove material, a weld bead adds a body to the first
//! component it names. The faces keep their names through each step, so picks and mates on a
//! cut component still read. The part file doesn't change.

use crate::runtime::part_cache;
use runmat_cad::assembly::model::{AssemblyDocument, DefinitionKind};
use runmat_cad::assembly::operations::{tool, AssemblyOperation, OperationKind, OperationTool};
use runmat_cad::assembly::resolve::ResolvedAssembly;
use runmat_cad::display::{part_display, PartDisplay};
use runmat_cad::ids::FeatureId;
use runmat_cad::model::PartDocument;
use runmat_cad::refs::naming::{name_outputs, NamingContext, OperandView};
use runmat_cad::regen::engine::LiveBody;
use runmat_cad::regen::kernel::{Combine, KernelOp, ToolOp, ToolShape};
use runmat_cad::regen::RegenResult;
use std::collections::BTreeMap;

/// What the assembly's features made: per affected node, its display and its digest; and per
/// feature that failed, why.
#[derive(Default)]
pub struct Operated {
    pub displays: BTreeMap<u32, PartDisplay>,
    pub digests: BTreeMap<u32, String>,
    pub problems: Vec<(String, String)>,
}

/// The top-level component node `id` sits under (itself at the top level), by name.
fn top_name(resolved: &ResolvedAssembly, mut id: u32) -> Option<&str> {
    loop {
        let node = resolved.node(id);
        match node.parent {
            Some(0) => return Some(node.segment.name.as_str()),
            Some(p) => id = p,
            None => return None,
        }
    }
}

/// The part nodes `op` affects.
fn affected(resolved: &ResolvedAssembly, op: &AssemblyOperation) -> Vec<u32> {
    resolved
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.definition.kind == DefinitionKind::Part && !n.represented_by_ancestor)
        .filter(|(i, _)| {
            op.components.is_empty()
                || top_name(resolved, *i as u32)
                    .is_some_and(|t| op.components.iter().any(|c| c == t))
        })
        .map(|(i, _)| i as u32)
        .collect()
}

/// Each affected component's tools in its own coordinates, in feature order, by feature name.
type NodeTools = BTreeMap<u32, Vec<(String, OperationTool)>>;

/// The tools of `document`'s features, per node they apply to; problems by feature name.
pub fn node_tools(
    resolved: &ResolvedAssembly,
    document: &AssemblyDocument,
) -> (NodeTools, Vec<(String, String)>) {
    let mut out: NodeTools = BTreeMap::new();
    let mut problems = Vec::new();
    for op in document.operations.iter().filter(|o| !o.suppressed) {
        let Some(sketch) = resolved.sketch(op.op.sketch()) else {
            problems.push((
                op.name.clone(),
                format!("the sketch `{}` is missing or didn't solve", op.op.sketch()),
            ));
            continue;
        };
        let made = match tool(&op.op, sketch) {
            Ok(t) => t,
            Err(e) => {
                problems.push((op.name.clone(), e));
                continue;
            }
        };
        for c in &op.components {
            if !document.occurrences.iter().any(|o| &o.name == c) {
                problems.push((
                    op.name.clone(),
                    format!("the assembly has no component `{c}`"),
                ));
            }
        }
        let nodes = affected(resolved, op);
        if nodes.is_empty() {
            problems.push((op.name.clone(), "it affects no part component".into()));
            continue;
        }
        // A weld bead is one body, drawn with the first component it names.
        let nodes = match op.op {
            OperationKind::Weld { .. } => nodes.into_iter().take(1).collect(),
            _ => nodes,
        };
        for id in nodes {
            let to_local = resolved.node(id).world.inverse();
            out.entry(id)
                .or_default()
                .push((op.name.clone(), made.transformed(&to_local)));
        }
    }
    (out, problems)
}

fn kernel_op(t: &OperationTool) -> KernelOp {
    let (shape, combine) = match t.clone() {
        OperationTool::Cut {
            profile,
            start,
            end,
            start_bound,
            end_bound,
        } => (
            ToolShape::Prism {
                profile,
                start,
                end,
                start_bound,
                end_bound,
                draft: None,
                thin: None,
            },
            Combine::Cut,
        ),
        OperationTool::Holes {
            frame,
            positions,
            diameter,
            end,
            counterbore,
            countersink,
        } => (
            ToolShape::Holes {
                frame,
                positions,
                diameter,
                end,
                counterbore,
                countersink,
                drill_point: None,
                taper: None,
                thread: None,
                slot: None,
                stages: Vec::new(),
            },
            Combine::Cut,
        ),
        OperationTool::Bead { profile, path } => (
            ToolShape::Sweep {
                profile,
                path,
                path_curves: Vec::new(),
                follow_path: true,
            },
            Combine::New,
        ),
    };
    KernelOp::Tool(ToolOp {
        shape,
        combine,
        references: 0,
        profile_face: None,
    })
}

/// Applies `tools` to `result`'s bodies on `kernel`, naming the outputs as feature `name` would.
fn apply(
    result: &mut RegenResult,
    kernel: &dyn crate::kernel::DesignKernel,
    tools: &[(String, OperationTool)],
) -> Result<(), (String, String)> {
    for (name, t) in tools {
        let op = kernel_op(t);
        let fail = |e: String| (name.clone(), e);
        let naming = NamingContext {
            feature: FeatureId::new(name),
            // The tool's profile curves are named as if feature `name` sketched them.
            profiles: vec![vec![FeatureId::new(name)]],
            seeds: Vec::new(),
            step: None,
            restore: Vec::new(),
        };
        if matches!(t, OperationTool::Bead { .. }) {
            let out = kernel.execute(&op, &[]).map_err(|e| fail(e.to_string()))?;
            let outs: Vec<_> = out.bodies.iter().map(|b| &b.topology).collect();
            let named =
                name_outputs(&naming, &[], &outs, &out.history).map_err(|e| fail(e.to_string()))?;
            let _ = named.intern_into(&mut result.state.names);
            for (b, names) in out.bodies.into_iter().zip(named.bodies) {
                result.state.bodies.push(LiveBody {
                    handle: b.handle,
                    topology: b.topology,
                    names,
                });
            }
            continue;
        }
        let mut next = Vec::new();
        for body in std::mem::take(&mut result.state.bodies) {
            let out = kernel
                .execute(&op, &[&body.handle])
                .map_err(|e| fail(e.to_string()))?;
            let view = OperandView {
                names: &body.names,
                topology: &body.topology,
            };
            let outs: Vec<_> = out.bodies.iter().map(|b| &b.topology).collect();
            let named = name_outputs(&naming, &[view], &outs, &out.history)
                .map_err(|e| fail(e.to_string()))?;
            let _ = named.intern_into(&mut result.state.names);
            for (b, names) in out.bodies.into_iter().zip(named.bodies) {
                next.push(LiveBody {
                    handle: b.handle,
                    topology: b.topology,
                    names,
                });
            }
        }
        result.state.bodies = next;
    }
    Ok(())
}

/// The displays of the components `document`'s features change, regenerated with the features
/// applied. `documents` holds each part file's document.
pub fn operated_displays(
    resolved: &ResolvedAssembly,
    document: &AssemblyDocument,
    documents: &BTreeMap<String, PartDocument>,
) -> Operated {
    let (tools, problems) = node_tools(resolved, document);
    let mut out = Operated {
        problems,
        ..Default::default()
    };
    for (id, tools) in tools {
        let node = resolved.node(id);
        let file = node.source_key();
        let Some(part) = documents.get(&file) else {
            continue;
        };
        let key = format!(
            "{}\0operated\0{:?}",
            part_cache::configured_key(&file, part, &node.configuration),
            tools
        );
        let mut failed = None;
        let made = part_cache::configured_part(key.clone(), || {
            let (mut result, kernel) =
                crate::configured::regenerate_configured(part, &node.configuration);
            if let Err(e) = apply(&mut result, kernel.as_ref(), &tools) {
                failed = Some(e);
            }
            let mut display = part_display(
                &result,
                kernel.as_ref(),
                &crate::runtime::ASSEMBLY_TESSELLATION,
            );
            runmat_cad::appearance::apply_document_appearance(part, &result, &mut display);
            // A failed feature isn't cached: the next compile reports it again.
            let dynamic = failed.is_some();
            (part_cache::ConfiguredPart { display }, dynamic)
        });
        if let Some(problem) = failed {
            out.problems.push(problem);
        }
        out.digests.insert(id, part_cache::digest(&key));
        out.displays.insert(id, made.display.clone());
    }
    out
}
