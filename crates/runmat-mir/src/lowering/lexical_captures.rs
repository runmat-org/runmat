//! Carry shared lexical bindings through frames that call or create handles to
//! nested functions, even when an intervening frame never names the binding.

use std::collections::BTreeMap;

use runmat_hir::{BindingOwner, CapturedBinding, FunctionId, HirAssembly, HirError};
use runmat_types::FunctionSourceRole;

use crate::{analysis::lexical_reference_graph, MirAssembly, MirLocal, MirLocalId, MirLocalKind};

pub(super) fn propagate(hir: &HirAssembly, assembly: &mut MirAssembly) -> Result<(), HirError> {
    let references = lexical_reference_graph(assembly);
    let binding_owners = hir
        .bindings
        .iter()
        .filter_map(|binding| match &binding.owner {
            BindingOwner::Function(owner) => Some((binding.id, *owner)),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();

    // A callee may itself acquire a pass-through capture from another callee.
    // The finite binding/function product bounds the monotone fixed point.
    let mut changed = true;
    while changed {
        changed = false;
        for (caller, targets) in &references {
            for target in targets {
                let Some(callee) = assembly.functions.get(target) else {
                    continue; // an imported or externally bound definition
                };
                if callee.source_role != FunctionSourceRole::Nested {
                    continue;
                }
                let parent = callee
                    .parent
                    .ok_or_else(|| HirError::new("nested callable has no lexical parent"))?;
                if frames_to_owner(assembly, *caller, parent).is_none() {
                    return Err(HirError::new(
                        "nested callable is outside the caller's lexical scope",
                    ));
                }
                let captures = callee.captures.clone();
                for capture in captures {
                    let owner = binding_owners
                        .get(&capture.binding)
                        .copied()
                        .ok_or_else(|| {
                            HirError::new("nested capture has no function binding owner")
                        })?;
                    let frames = frames_to_owner(assembly, *caller, owner).ok_or_else(|| {
                        HirError::new("nested capture owner is outside the caller's ancestry")
                    })?;
                    for frame in frames {
                        let metadata = assembly
                            .functions
                            .get_mut(&frame)
                            .ok_or_else(|| HirError::new("lexical frame metadata is absent"))?;
                        let demanded = CapturedBinding {
                            binding: capture.binding,
                            from_function: owner,
                        };
                        if !metadata.captures.contains(&demanded) {
                            if metadata.modifiers.isolated {
                                return Err(HirError::new(
                                    "isolated function cannot carry a nested lexical capture",
                                )
                                .with_span(metadata.span));
                            }
                            metadata.captures.push(demanded);
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    for (function, metadata) in &assembly.functions {
        let body = assembly
            .bodies
            .get_mut(function)
            .ok_or_else(|| HirError::new("lexical function body is absent"))?;
        for capture in &metadata.captures {
            if body
                .locals
                .iter()
                .any(|local| local.binding == Some(capture.binding))
            {
                continue;
            }
            body.locals.push(MirLocal {
                id: MirLocalId(body.locals.len()),
                binding: Some(capture.binding),
                kind: MirLocalKind::Capture,
                span: metadata.span,
            });
        }
    }
    Ok(())
}

/// Caller and intervening frames up to, but not including, the binding owner.
fn frames_to_owner(
    assembly: &MirAssembly,
    caller: FunctionId,
    owner: FunctionId,
) -> Option<Vec<FunctionId>> {
    let mut frames = Vec::new();
    let mut current = Some(caller);
    for _ in 0..=assembly.functions.len() {
        let frame = current?;
        if frame == owner {
            return Some(frames);
        }
        frames.push(frame);
        current = assembly.functions.get(&frame)?.parent;
    }
    None
}
