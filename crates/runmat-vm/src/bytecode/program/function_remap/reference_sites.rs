//! Stable reference-site enumeration shared by selected-graph production and
//! admitted-unit validation. It follows the same instruction grammar as
//! session-ID remapping; the ordinal is within one instruction's references.

use std::collections::{HashMap, HashSet};

use runmat_execution::value::GraphStaticReferenceSite;
use runmat_hir::FunctionId;
use runmat_types::CallableIdentity;

use super::grammar::with_instruction_function_references;
use crate::bytecode::{BytecodeSubscriptStep, Instr};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FunctionReferenceKind {
    Local,
    External,
}

/// Return all compiler-bound function references, including dotted identity
/// fallbacks, in grammar order. A caller must still prove the selected body,
/// source owner and definition lease before publishing an external edge.
pub fn instruction_reference_sites(
    instruction: &Instr,
    pc: usize,
) -> Result<Vec<(GraphStaticReferenceSite, FunctionId, FunctionReferenceKind)>, &'static str> {
    let instruction_id = u32::try_from(pc).map_err(|_| "instruction site exceeds graph domain")?;
    let mut references = Vec::new();
    with_instruction_function_references!(
        instruction,
        &mut references,
        visit_local,
        visit_external,
        visit_identity,
        visit_steps
    );
    references
        .into_iter()
        .enumerate()
        .map(|(ordinal, (function, kind))| {
            Ok((
                GraphStaticReferenceSite {
                    instruction: instruction_id,
                    occurrence: u16::try_from(ordinal)
                        .map_err(|_| "instruction has too many function references")?,
                },
                function,
                kind,
            ))
        })
        .collect()
}

type References = Vec<(FunctionId, FunctionReferenceKind)>;

fn visit_local(references: &mut References, function: &FunctionId) {
    references.push((*function, FunctionReferenceKind::Local));
}

fn visit_external(references: &mut References, function: &FunctionId) {
    references.push((*function, FunctionReferenceKind::External));
}

fn visit_identity(references: &mut References, identity: &CallableIdentity) {
    match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function) => visit_local(references, function),
        CallableIdentity::ExternalFunction { function, .. } => visit_external(references, function),
        CallableIdentity::Builtin(_)
        | CallableIdentity::Imported(_)
        | CallableIdentity::Method(_)
        | CallableIdentity::DynamicName(_)
        | CallableIdentity::ExternalName(_) => {}
    }
}

fn visit_steps(references: &mut References, steps: &[BytecodeSubscriptStep]) {
    for step in steps {
        if let BytecodeSubscriptStep::DottedInvoke {
            semantic_fallback: Some(identity),
            ..
        } = step
        {
            visit_identity(references, identity);
        }
    }
}

/// Replace proved reference sites, not numeric IDs. An external reference may
/// have the same session-local number as a different local definition; the
/// site's typed instruction and sealed import decide its destination.
pub fn remap_reference_sites(
    instructions: &mut [Instr],
    replacements: &HashMap<GraphStaticReferenceSite, FunctionId>,
) -> Result<(), &'static str> {
    let mut sites = HashSet::new();
    for (pc, instruction) in instructions.iter().enumerate() {
        sites.extend(
            instruction_reference_sites(instruction, pc)?
                .into_iter()
                .map(|(site, _, _)| site),
        );
    }
    if replacements.keys().any(|site| !sites.contains(site)) {
        return Err("reference-site remap names a nonexistent instruction reference");
    }
    for (pc, instruction) in instructions.iter_mut().enumerate() {
        let instruction_id =
            u32::try_from(pc).map_err(|_| "instruction site exceeds graph domain")?;
        let mut context = SiteRemapContext {
            instruction: instruction_id,
            occurrence: 0,
            replacements,
        };
        with_instruction_function_references!(
            instruction,
            &mut context,
            remap_direct,
            remap_direct,
            remap_identity,
            remap_steps
        );
    }
    Ok(())
}

struct SiteRemapContext<'a> {
    instruction: u32,
    occurrence: usize,
    replacements: &'a HashMap<GraphStaticReferenceSite, FunctionId>,
}

impl SiteRemapContext<'_> {
    fn replace(&mut self, function: &mut FunctionId) {
        let site = GraphStaticReferenceSite {
            instruction: self.instruction,
            occurrence: u16::try_from(self.occurrence)
                .expect("reference-site count was validated before mutation"),
        };
        if let Some(destination) = self.replacements.get(&site) {
            *function = *destination;
        }
        self.occurrence += 1;
    }
}

fn remap_direct(context: &mut SiteRemapContext<'_>, function: &mut FunctionId) {
    context.replace(function);
}

fn remap_identity(context: &mut SiteRemapContext<'_>, identity: &mut CallableIdentity) {
    match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function)
        | CallableIdentity::ExternalFunction { function, .. } => context.replace(function),
        CallableIdentity::Builtin(_)
        | CallableIdentity::Imported(_)
        | CallableIdentity::Method(_)
        | CallableIdentity::DynamicName(_)
        | CallableIdentity::ExternalName(_) => {}
    }
}

fn remap_steps(context: &mut SiteRemapContext<'_>, steps: &mut [BytecodeSubscriptStep]) {
    for step in steps {
        if let BytecodeSubscriptStep::DottedInvoke {
            semantic_fallback: Some(identity),
            ..
        } = step
        {
            match identity.as_ref() {
                CallableIdentity::BoundFunction(function)
                | CallableIdentity::AnonymousFunction(function)
                | CallableIdentity::ExternalFunction { function, .. } => {
                    let mut next = *function;
                    context.replace(&mut next);
                    if next != *function {
                        match std::sync::Arc::make_mut(identity) {
                            CallableIdentity::BoundFunction(function)
                            | CallableIdentity::AnonymousFunction(function)
                            | CallableIdentity::ExternalFunction { function, .. } => {
                                *function = next
                            }
                            _ => unreachable!("retained identity keeps its typed variant"),
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::FunctionSourceRole;

    #[test]
    fn external_and_local_handles_keep_distinct_site_kinds() {
        let external = Instr::CreateExternalBoundFunctionHandle(
            FunctionId(7),
            "parent".into(),
            FunctionSourceRole::Local,
        );
        let local = Instr::CreateBoundFunctionHandle(
            FunctionId(7),
            "parent".into(),
            FunctionSourceRole::Local,
        );
        assert_eq!(
            instruction_reference_sites(&external, 12).expect("external site"),
            vec![(
                GraphStaticReferenceSite {
                    instruction: 12,
                    occurrence: 0,
                },
                FunctionId(7),
                FunctionReferenceKind::External,
            )]
        );
        assert_eq!(
            instruction_reference_sites(&local, 12).expect("local site")[0].2,
            FunctionReferenceKind::Local
        );
    }

    #[test]
    fn site_remap_distinguishes_equal_numeric_local_and_external_references() {
        let mut instructions = vec![
            Instr::CreateBoundFunctionHandle(
                FunctionId(7),
                "local".into(),
                FunctionSourceRole::Local,
            ),
            Instr::CreateExternalBoundFunctionHandle(
                FunctionId(7),
                "parent".into(),
                FunctionSourceRole::Local,
            ),
        ];
        let replacements = HashMap::from([
            (
                GraphStaticReferenceSite {
                    instruction: 0,
                    occurrence: 0,
                },
                FunctionId(3),
            ),
            (
                GraphStaticReferenceSite {
                    instruction: 1,
                    occurrence: 0,
                },
                FunctionId(9),
            ),
        ]);
        remap_reference_sites(&mut instructions, &replacements).expect("typed remap");
        assert!(matches!(
            instructions[0],
            Instr::CreateBoundFunctionHandle(FunctionId(3), ..)
        ));
        assert!(matches!(
            instructions[1],
            Instr::CreateExternalBoundFunctionHandle(FunctionId(9), ..)
        ));
    }
}
