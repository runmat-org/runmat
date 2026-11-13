//! Invocation-local admission of a dynamically compiled VM product.
//!
//! The VM owns the compiler result and the execution projection. A host may
//! authenticate and publish its exact code/source product without making the
//! compiler, registry, or runtime execution ports ambient capabilities.

use crate::bytecode::{Bytecode, FunctionRegistry, Instr};
use crate::call::builtins::{DynamicEvalCompileProduct, DynamicEvalResult};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::source_context::RuntimeSourceLease;
use runmat_runtime::RuntimeError;
use std::rc::Rc;
use std::sync::Arc;

/// A deliberately small, typed proof that this invocation cannot create or
/// submit an isolated callback. Unknown instructions, user dispatch, overloads,
/// and local definitions all require a selected product. This does not depend
/// on builtin display names or assume an indirect call is harmless.
pub fn proves_no_graph_demand(canonical: &Bytecode) -> bool {
    canonical.function_registry.functions.is_empty()
        && canonical.instructions.iter().all(|instruction| {
            matches!(
                instruction,
                Instr::LoadConst(_)
                    | Instr::LoadInt(_)
                    | Instr::LoadComplex(_, _)
                    | Instr::LoadBool(_)
                    | Instr::LoadString(_)
                    | Instr::LoadCharRow(_)
                    | Instr::LoadVar(_)
                    | Instr::StoreVar(_)
                    | Instr::Pop
                    | Instr::Return
                    | Instr::ReturnValue
            )
        })
}

/// Selected code admission for one VM invocation, never a global registry.
pub trait DynamicEvalProductAdmission {
    /// Retain an already-observed semantic definition from this session's
    /// current publication. The numeric ID only narrows the lookup; the exact
    /// observed owner must agree before its dependency graph can be issued.
    fn selected_semantic_source_binding(
        &self,
        runtime: &RuntimeContext,
        function: runmat_hir::FunctionId,
        expected_owner: &std::sync::Arc<runmat_value::FunctionDefinitionOwner>,
    ) -> Result<runmat_value::BoundFunctionBinding, RuntimeError>;

    /// Read the exact admitted body signature selected by the no-invoke
    /// source loader. Numeric ID and display name alone are never authority.
    fn selected_source_output_arity(
        &self,
        binding: &runmat_value::BoundFunctionBinding,
    ) -> Result<(usize, bool), RuntimeError>;

    /// Called after the single compiler pass. A host must select whenever it
    /// cannot prove that execution is incapable of an isolated callback.
    fn requires_selected_product(
        &self,
        canonical: &Bytecode,
        inherited: &FunctionRegistry,
    ) -> Result<bool, RuntimeError>;

    /// Authenticate the exact issued source and both compiler projections,
    /// then publish definitions transactionally before interpretation.
    fn admit(
        &self,
        runtime: &RuntimeContext,
        source: &RuntimeSourceLease,
        source_text: &str,
        caller_source: Option<runmat_types::SourceId>,
        product: DynamicEvalCompileProduct,
        inherited: &FunctionRegistry,
    ) -> Result<AdmittedDynamicEval, RuntimeError>;

    /// Admit exactly the body whose closure-creation opcode is executing.
    /// The surrounding eval script has already run to this point; it and
    /// sibling functions are compiler provenance, not new publications.
    fn admit_pending_closure(
        &self,
        request: PendingClosureAdmissionRequest<'_>,
    ) -> Result<AdmittedPendingClosure, RuntimeError>;
}

/// A single exact compiler/invocation join. The precompile registry assigns
/// stable eval-local IDs; the executing view proves the original closure
/// opcode still addresses its issuing definition after earlier effects.
pub struct PendingClosureAdmissionRequest<'a> {
    pub runtime: &'a RuntimeContext,
    pub source: &'a RuntimeSourceLease,
    pub caller_source: Option<runmat_types::SourceId>,
    /// The selected compiler already boxed the canonical and execution
    /// bytecode views. Preserve that owner across this synchronous admission
    /// call, which runs inside a suspended interpreter instruction.
    pub product: Box<DynamicEvalCompileProduct>,
    pub compile_inherited: &'a FunctionRegistry,
    pub executing: &'a FunctionRegistry,
    pub local_function: runmat_hir::FunctionId,
    /// The issuing CreateSemanticClosure opcode's original execution ID.
    /// Recompilation after source observation cannot choose this identity.
    pub execution_function: runmat_hir::FunctionId,
    pub expected_issuing_owner: &'a std::sync::Arc<runmat_value::FunctionDefinitionOwner>,
    pub expected_source: Option<crate::BytecodeSourceRef>,
    /// Compiler-local Local targets proved by the original eval product.
    /// Each target is reopened only through its exact running owner/source.
    pub siblings: &'a [PendingClosureSibling],
}

/// One originally compiled Local reference into the executing eval unit.
/// A fresh function-only closure product may import this exact selected body,
/// but cannot republish or recompile it under a matching numeric ID.
#[derive(Clone)]
pub struct PendingClosureSibling {
    pub local_function: runmat_hir::FunctionId,
    pub execution_function: runmat_hir::FunctionId,
    pub expected_owner: Arc<runmat_value::FunctionDefinitionOwner>,
    pub expected_source: Option<crate::BytecodeSourceRef>,
    pub hir_digest: runmat_execution::Digest,
}

/// A provisional definition generation. Dropping it restores the caller's
/// named registry and lets the canonical generation owner retain only handles
/// that actually escaped through committed workspace/output effects.
pub trait DynamicEvalPublication {}

pub struct AdmittedDynamicEval {
    pub runtime: RuntimeContext,
    pub execution_bytecode: Bytecode,
    pub result: Option<DynamicEvalResult>,
    pub publication: Box<dyn DynamicEvalPublication>,
}

pub struct AdmittedPendingClosure {
    pub binding: runmat_value::BoundFunctionBinding,
    pub source_file: Option<String>,
    /// Keep provisional code/source visible until the new handle owns its
    /// exact definition lease. Drop then performs ordinary escape retention.
    pub publication: Box<dyn DynamicEvalPublication>,
}

#[derive(Clone)]
pub struct DynamicEvalAdmissionPort {
    host: Rc<dyn DynamicEvalProductAdmission>,
    pub(crate) pending: Option<Rc<crate::call::builtins::PendingClosurePlan>>,
}

impl DynamicEvalAdmissionPort {
    pub fn new(host: Rc<dyn DynamicEvalProductAdmission>) -> Self {
        Self {
            host,
            pending: None,
        }
    }

    pub(crate) fn host(&self) -> &dyn DynamicEvalProductAdmission {
        self.host.as_ref()
    }

    pub(crate) fn with_pending(
        mut self,
        pending: Rc<crate::call::builtins::PendingClosurePlan>,
    ) -> Self {
        self.pending = Some(pending);
        self
    }
}

impl std::fmt::Debug for DynamicEvalAdmissionPort {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamicEvalAdmissionPort")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_assignment_has_typed_no_graph_proof_but_unknown_effect_does_not() {
        let scalar = Bytecode::with_instructions(
            vec![Instr::LoadConst(1.0), Instr::StoreVar(0), Instr::Return],
            1,
        );
        assert!(proves_no_graph_demand(&scalar));
        let overloaded = Bytecode::with_instructions(
            vec![
                Instr::LoadVar(0),
                Instr::LoadVar(1),
                Instr::Add,
                Instr::ReturnValue,
            ],
            2,
        );
        assert!(!proves_no_graph_demand(&overloaded));
    }
}
