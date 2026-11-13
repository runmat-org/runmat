//! Lexically scoped unresolved calls in an already-lowered closure body.
//!
//! This is a classification pass, never a source/name authority. In
//! particular, a parameter or local binding is not reclassified by looking
//! up its display spelling in a cache.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use runmat_hir::{
    FunctionId, FunctionKind, HirAssembly, HirCallableRef, HirExpr, HirExprKind, HirFunction,
    HirPlace, HirStmtKind,
};

use crate::bytecode::{Bytecode, Instr};
use crate::call::builtins::DynamicEvalCompileProduct;
use crate::interpreter::dynamic_admission::PendingClosureSibling;

#[path = "eval_hir_sites/abi.rs"]
mod abi;
pub(crate) use abi::{ClosureIssuanceAbi, SelectedClosureCapture};
#[path = "eval_hir_sites/visitor.rs"]
mod visitor;
use visitor::visit_block;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HirUnresolvedSite {
    pub caller: FunctionId,
    pub expression: runmat_hir::ExprId,
    pub name: String,
}

/// A closure is pending only when the compiler actually emits its creation
/// opcode. Other HIR functions (including an uncalled nested definition) do
/// not cause an eager source observation merely because their syntax contains
/// an unresolved call. The keys are compiler-local IDs; the eval projection
/// must apply its checked function remap before VM dispatch uses them.
#[derive(Clone, Debug, Default)]
pub(super) struct PendingClosureSites {
    by_function: HashMap<FunctionId, Vec<HirUnresolvedSite>>,
}

/// Invocation-local, nonserialized compiler/source authority for an eval
/// closure. The plan has code metadata and a retained source lease, never a
/// workspace Value, RuntimeContext, selected host, or mutable name table.
pub(crate) struct PendingClosurePlan {
    pub(crate) ast: Rc<runmat_parser::Program>,
    pub(crate) source: runmat_runtime::source_context::RuntimeSourceLease,
    pub(crate) caller_source: Option<runmat_types::SourceId>,
    pub(crate) requested_outputs: usize,
    pub(crate) source_ref: Option<runmat_runtime::source_context::RuntimeSourceRef>,
    pub(super) options: super::DynamicEvalOptions,
    pub(crate) alias_floor: usize,
    /// Exact precompile registry used to assign the running frame's local
    /// function IDs. Recompilation must not treat those very IDs as new
    /// inherited collisions and silently move the closure identity.
    pub(crate) inherited: crate::bytecode::FunctionRegistry,
    functions: HashMap<FunctionId, PendingClosureFunction>,
}

pub(crate) struct PendingClosureFunction {
    pub(crate) local_id: FunctionId,
    pub(crate) sites: Vec<HirUnresolvedSite>,
    pub(crate) abi: ClosureIssuanceAbi,
    pub(crate) execution_owner: std::sync::Arc<runmat_value::FunctionDefinitionOwner>,
    pub(crate) execution_source: Option<crate::BytecodeSourceRef>,
    pub(crate) siblings: Vec<PendingClosureSibling>,
}

pub(super) fn sibling_hir_digest(
    hir: &HirAssembly,
    function: FunctionId,
) -> Option<runmat_execution::Digest> {
    let function = hir
        .functions
        .iter()
        .find(|candidate| candidate.id == function)?;
    let canonical = serde_json::to_vec(function).ok()?;
    Some(runmat_execution::Digest::sha256(&canonical))
}

fn issued_eval_body_source(
    compiled: Option<crate::BytecodeSourceRef>,
    source_ref: Option<runmat_runtime::source_context::RuntimeSourceRef>,
    lease: &runmat_runtime::source_context::RuntimeSourceLease,
) -> Result<crate::BytecodeSourceRef, &'static str> {
    if compiled
        != Some(crate::BytecodeSourceRef::Program(
            runmat_types::ProgramSourceId(0),
        ))
    {
        return Err("pending eval body has another compiler source ordinal");
    }
    let source = source_ref.ok_or("pending eval body lacks its issued source")?;
    if source != lease.source_ref() {
        return Err("pending eval body source differs from its exact eval lease");
    }
    Ok(crate::BytecodeSourceRef::Runtime(source))
}

impl PendingClosurePlan {
    pub(crate) fn from_compiled(
        ast: Rc<runmat_parser::Program>,
        source: runmat_runtime::source_context::RuntimeSourceLease,
        caller_source: Option<runmat_types::SourceId>,
        product: &DynamicEvalCompileProduct,
        inherited: &crate::bytecode::FunctionRegistry,
    ) -> Result<Self, &'static str> {
        let pending = PendingClosureSites::from_compiled(&product.hir, &product.canonical_bytecode);
        let alias_floor = product
            .canonical_bytecode
            .function_registry
            .functions
            .keys()
            .chain(
                product
                    .canonical_bytecode
                    .layout
                    .iter()
                    .flat_map(|layout| layout.functions.keys()),
            )
            .map(|id| id.0)
            .max()
            .unwrap_or(0);
        let mut functions = HashMap::new();
        for (local_id, sites) in pending.by_function {
            let body = product
                .canonical_bytecode
                .function_registry
                .get_current(local_id)
                .ok_or("pending eval closure has no compiled body")?;
            let abi = ClosureIssuanceAbi::capture(&product.hir, body, local_id)
                .ok_or("pending eval closure has no exact compiler ABI")?;
            let execution_id = product
                .function_remap
                .get(&local_id)
                .copied()
                .unwrap_or(local_id);
            let execution_body = product
                .execution_bytecode
                .function_registry
                .get_current(execution_id)
                .ok_or("pending eval closure has no projected execution body")?;
            let mut local_targets = HashSet::new();
            for (pc, instruction) in body.instructions.iter().enumerate() {
                for (_, target, kind) in crate::instruction_reference_sites(instruction, pc)? {
                    if kind == crate::FunctionReferenceKind::Local && target != local_id {
                        local_targets.insert(target);
                    }
                }
            }
            let mut siblings = Vec::with_capacity(local_targets.len());
            for target in local_targets {
                product
                    .canonical_bytecode
                    .function_registry
                    .get_current(target)
                    .ok_or("pending closure Local target lacks its original compiled body")?;
                let execution_function = product
                    .function_remap
                    .get(&target)
                    .copied()
                    .unwrap_or(target);
                let selected = product
                    .execution_bytecode
                    .function_registry
                    .get_current(execution_function)
                    .ok_or("pending closure Local target lacks its original execution body")?;
                siblings.push(PendingClosureSibling {
                    local_function: target,
                    execution_function,
                    expected_owner: std::sync::Arc::clone(&selected.definition_owner),
                    expected_source: Some(issued_eval_body_source(
                        selected.source_id,
                        product.source_ref,
                        &source,
                    )?),
                    hir_digest: sibling_hir_digest(&product.hir, target)
                        .ok_or("pending closure Local target has no exact HIR declaration")?,
                });
            }
            siblings.sort_by_key(|sibling| sibling.local_function.0);
            if functions
                .insert(
                    execution_id,
                    PendingClosureFunction {
                        local_id,
                        sites,
                        abi,
                        execution_owner: std::sync::Arc::clone(&execution_body.definition_owner),
                        execution_source: Some(issued_eval_body_source(
                            execution_body.source_id,
                            product.source_ref,
                            &source,
                        )?),
                        siblings,
                    },
                )
                .is_some()
            {
                return Err("pending eval closures share one execution ID");
            }
        }
        Ok(Self {
            ast,
            source,
            caller_source,
            requested_outputs: product.requested_outputs,
            source_ref: product.source_ref,
            options: super::current_dynamic_eval_options(),
            alias_floor,
            inherited: inherited.clone(),
            functions,
        })
    }

    pub(crate) fn function(&self, execution_id: FunctionId) -> Option<&PendingClosureFunction> {
        self.functions.get(&execution_id)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }
}

impl PendingClosureSites {
    pub(super) fn from_compiled(hir: &HirAssembly, bytecode: &Bytecode) -> Self {
        let mut created = HashSet::new();
        let mut scan = |instructions: &[Instr]| {
            for instruction in instructions {
                if let Instr::CreateSemanticClosure(function, ..) = instruction {
                    created.insert(*function);
                }
            }
        };
        scan(&bytecode.instructions);
        for body in bytecode.function_registry.functions.values() {
            scan(&body.instructions);
        }
        let by_function = hir
            .functions
            .iter()
            .filter(|function| created.contains(&function.id))
            .filter(|function| function.kind == FunctionKind::Anonymous)
            .filter_map(|function| {
                // Even when an unresolved call is not safe to preselect at
                // closure creation, the already-running opcode still needs
                // a fresh lexical lowering to capture workspace values that
                // earlier effects assigned before this issuance point.
                let all_sites = all_unresolved_sites(function);
                (!all_sites.is_empty())
                    .then(|| (function.id, pre_effect_sites(function, all_sites)))
            })
            .collect();
        Self { by_function }
    }
}

pub(super) fn pre_effect_sites(
    function: &HirFunction,
    mut sites: Vec<HirUnresolvedSite>,
) -> Vec<HirUnresolvedSite> {
    // The source observer runs when this closure is created, not when its
    // body later runs. Only a call that is the body's first possible effect
    // can use that observation. A preceding call, conditional branch, or
    // effectful argument may change path/source state first; those sites
    // retain ordinary dynamic dispatch instead of being selected early.
    let Some(frontier) = direct_pre_effect_call(function) else {
        return Vec::new();
    };
    sites.retain(|site| site.expression == frontier.id);
    sites
}

pub(super) fn all_unresolved_sites(function: &HirFunction) -> Vec<HirUnresolvedSite> {
    let mut sites = Vec::new();
    visit_block(&function.body, function.id, &mut sites);
    sites
}

/// Numeric text is one bracketed expression. Its direct single-element call
/// is an entry-frontier site only when every argument is inert; selection is
/// still performed by the live source observer, never by this HIR inspection.
/// A wider matrix expression can have earlier effects, so it keeps ordinary
/// dynamic dispatch instead of borrowing entry-time source authority.
pub(super) fn numeric_text_entry_sites(function: &HirFunction) -> Vec<HirUnresolvedSite> {
    if function.kind != FunctionKind::SyntheticEntrypoint {
        return Vec::new();
    }
    let [statement] = function.body.statements.as_slice() else {
        return Vec::new();
    };
    let HirStmtKind::ExprStmt(expression, _) = &statement.kind else {
        return Vec::new();
    };
    let HirExprKind::Tensor(rows) = &expression.kind else {
        return Vec::new();
    };
    let [row] = rows.as_slice() else {
        return Vec::new();
    };
    let [element] = row.as_slice() else {
        return Vec::new();
    };
    let HirExprKind::Call(call) = &element.kind else {
        return Vec::new();
    };
    if !matches!(&call.callee, HirCallableRef::Unresolved(_))
        || !call.args.iter().all(side_effect_free_input)
    {
        return Vec::new();
    }
    all_unresolved_sites(function)
        .into_iter()
        .filter(|site| site.expression == element.id)
        .collect()
}

fn direct_pre_effect_call(function: &HirFunction) -> Option<&HirExpr> {
    // Named/nested callable issuance is not its invocation boundary. A later
    // caller may mutate source state before entering it, and argument
    // validators can run before the first body statement.
    if function.kind != FunctionKind::Anonymous || !function.argument_validations.is_empty() {
        return None;
    }
    let [statement] = function.body.statements.as_slice() else {
        return None;
    };
    let expression = match &statement.kind {
        HirStmtKind::Assign(HirPlace::Binding(_), expression, _)
        | HirStmtKind::ExprStmt(expression, _) => expression,
        HirStmtKind::SequenceAssign {
            target,
            value,
            suppressed: _,
        } if anonymous_output_forwarding_target(function, target) => value,
        _ => return None,
    };
    let HirExprKind::Call(call) = &expression.kind else {
        return None;
    };
    if !matches!(&call.callee, HirCallableRef::Unresolved(_))
        || !call.args.iter().all(side_effect_free_input)
    {
        return None;
    }
    Some(expression)
}

/// Unknown-output anonymous calls lower to `varargout{1:nargout} = call()`.
/// Only this body's ABI-owned destination is inert before the call. A general
/// sequence/index assignment may evaluate user code to determine cardinality.
fn anonymous_output_forwarding_target(
    function: &HirFunction,
    target: &runmat_hir::HirSequenceTarget,
) -> bool {
    let runmat_hir::HirSequenceTarget::CellContents { base, indexing } = target else {
        return false;
    };
    let (Some(output), Some(count)) = (function.abi.varargout, function.abi.implicit_nargout)
    else {
        return false;
    };
    if !matches!(base.kind, HirExprKind::Binding(binding) if binding == output)
        || indexing.kind != runmat_hir::IndexKind::Brace
        || indexing.result_context != runmat_hir::IndexResultContext::AssignmentTarget
    {
        return false;
    }
    let [runmat_hir::IndexComponent::Expr(range)] = indexing.components.as_slice() else {
        return false;
    };
    matches!(&range.kind,
        HirExprKind::Range(first, None, last)
            if matches!(&first.kind, HirExprKind::Number(number) if number == "1")
                && matches!(last.kind, HirExprKind::Binding(binding) if binding == count))
}

fn side_effect_free_input(expression: &HirExpr) -> bool {
    matches!(
        &expression.kind,
        HirExprKind::Binding(_)
            | HirExprKind::Number(_)
            | HirExprKind::ImaginaryNumber(_)
            | HirExprKind::IntegerLiteral(_)
            | HirExprKind::String(_)
            | HirExprKind::Constant(_)
    )
}

#[cfg(test)]
mod frontier_tests {
    use super::*;

    #[test]
    fn anonymous_forwarded_outputs_preserve_only_the_inert_entry_call_frontier() {
        for (source, selected) in [
            ("h = @() helper();", true),
            ("h = @(x) helper(x);", true),
            ("h = @() helper(other());", false),
            ("h = @() other() + helper();", false),
        ] {
            let ast = runmat_parser::parse(source).unwrap();
            let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
            let mut function = hir
                .assembly
                .functions
                .into_iter()
                .find(|f| f.kind == FunctionKind::Anonymous)
                .unwrap();
            let sites = all_unresolved_sites(&function);
            assert!(!sites.is_empty());
            assert_eq!(
                !pre_effect_sites(&function, sites).is_empty(),
                selected,
                "{source}"
            );
            if selected {
                // A different destination cannot borrow this ABI's inertness.
                function.abi.varargout = None;
                assert!(pre_effect_sites(&function, all_unresolved_sites(&function)).is_empty());
            }
        }
    }
}
