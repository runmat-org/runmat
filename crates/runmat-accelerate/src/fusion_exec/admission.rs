//! Selected-binding evidence exists only for this invocation. It is acquired
//! before domain probes, placement, compilation, uploads or provider execution.
use super::{unbound_constant_values, FusionExecutionRequest};
use crate::fusion::semantic_bindings::{selected_host_binding_for_primitive, SemanticBinding};
use crate::{
    fusion::FusionOp,
    graph::{FusionCallOrigin, PrimitiveOp},
};
use runmat_builtins::builtin_catalog_entry_by_name;
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::{
    canonical_fusion_declaration, selected_fusion_binding, FusionEpilogueOperation,
    FusionStructuralOperation, SelectedFusionIdentity,
};
use runmat_value::Value;
use serde::Serialize;

#[derive(Serialize)]
enum Identity {
    Primitive(PrimitiveOp),
    Call {
        origin: FusionCallOrigin,
        binding: SelectedFusionIdentity,
    },
}
struct Operation {
    identity: Identity,
    structural: Option<FusionStructuralOperation>,
    epilogue: Option<FusionEpilogueOperation>,
    reduction: Option<&'static runmat_runtime::builtins::common::spec::FusionReductionDeclaration>,
    semantic: Option<SemanticBinding>,
}
pub(super) struct AdmittedBindings {
    operations: Vec<Operation>,
}

impl AdmittedBindings {
    /// Intrinsic VM numeric owners retain their operator identity. Unary minus
    /// also authenticates its real canonical `times` dependency; scalar matrix
    /// multiplication deliberately does not borrow a named mtimes override.
    pub fn host_operations(
        &self,
        runtime: Option<&runmat_runtime::context::RuntimeContext>,
    ) -> Option<Vec<runmat_runtime::context::VmFusionHostOperation>> {
        use runmat_runtime::context::VmFusionHostOperation as H;
        let mut operations = Vec::new();
        for operation in &self.operations {
            match &operation.identity {
                Identity::Primitive(
                    op @ (PrimitiveOp::Neg | PrimitiveOp::UPlus | PrimitiveOp::Mul),
                ) => {
                    operations.push(H::Intrinsic(op.operator_kind()));
                    if *op == PrimitiveOp::Neg {
                        let times =
                            runmat_types::OperatorKind::ElementwiseMultiply.overload_name()?;
                        operations.push(H::Builtin(
                            crate::fusion::semantic_bindings::selected_canonical_host_binding(
                                times.display_name(),
                                runtime,
                            )?,
                        ));
                    }
                }
                Identity::Primitive(op) => operations.push(H::Builtin(
                    selected_host_binding_for_primitive(*op, runtime)?,
                )),
                Identity::Call {
                    binding: SelectedFusionIdentity::Canonical(binding),
                    ..
                } => operations.push(H::Builtin(*binding)),
                Identity::Call { .. } => return None,
            }
        }
        Some(operations)
    }
    /// Exact canonical implementations already captured for the selected
    /// fusion program. Legacy or opaque entries cannot certify a VM host
    /// region's inner execution.
    pub fn canonical_bindings(
        &self,
        runtime: Option<&runmat_runtime::context::RuntimeContext>,
    ) -> Option<Vec<runmat_builtins::BuiltinBindingIdentity>> {
        let mut bindings = Vec::new();
        for operation in &self.operations {
            match &operation.identity {
                Identity::Primitive(op) => {
                    bindings.push(selected_host_binding_for_primitive(*op, runtime)?);
                }
                Identity::Call {
                    binding: SelectedFusionIdentity::Canonical(binding),
                    ..
                } => {
                    bindings.push(*binding);
                }
                Identity::Call {
                    binding: SelectedFusionIdentity::Legacy { .. },
                    ..
                } => {
                    return None;
                }
            }
        }
        Some(bindings)
    }
    pub fn new(request: &FusionExecutionRequest<'_>) -> Option<Self> {
        if request.inputs.len() != request.plan.inputs.len()
            || !request
                .inputs
                .iter()
                .chain(unbound_constant_values(request.plan))
                .all(closed_value)
        {
            return None;
        }
        let operations = request
            .plan
            .operations
            .iter()
            .map(|operation| {
                let (name, origin) = match operation {
                    FusionOp::Primitive { op, .. } => {
                        let semantic = SemanticBinding::capture(operation).ok()?;
                        let epilogue = semantic.and_then(|binding| binding.declaration.epilogue);
                        return Some(Operation {
                            identity: Identity::Primitive(*op),
                            structural: None,
                            epilogue,
                            reduction: None,
                            semantic,
                        });
                    }
                    FusionOp::Builtin { name, origin, .. } => (name, *origin),
                };
                if let FusionCallOrigin::Operator(operator) = origin {
                    let expected = operator.overload_name()?;
                    if expected.display_name() != name {
                        return None;
                    }
                    let entry = builtin_catalog_entry_by_name(expected.display_name())?;
                    let [binding] = entry.bindings else {
                        return None;
                    };
                    let binding = entry.binding_identity(binding);
                    let declaration = canonical_fusion_declaration(binding)?;
                    return Some(Operation {
                        identity: Identity::Call {
                            origin,
                            binding: SelectedFusionIdentity::Canonical(binding),
                        },
                        structural: declaration.structural,
                        epilogue: declaration.epilogue,
                        reduction: declaration.reduction,
                        semantic: Some(SemanticBinding::selected(declaration)?),
                    });
                }
                let expected = match origin {
                    FusionCallOrigin::Named { binding } => Some(binding),
                    _ => None,
                };
                let selected = selected_fusion_binding(name, expected, request.runtime.as_ref())?;
                let semantic = match selected.canonical {
                    Some(declaration) => Some(SemanticBinding::selected(declaration)?),
                    None => None,
                };
                Some(Operation {
                    identity: Identity::Call {
                        origin,
                        binding: selected.identity,
                    },
                    structural: selected.structural,
                    epilogue: selected.epilogue,
                    reduction: selected.reduction,
                    semantic,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self { operations })
    }

    pub fn structural(&self, index: usize) -> Option<FusionStructuralOperation> {
        self.operations.get(index)?.structural
    }
    pub fn semantic(&self, index: usize) -> Option<SemanticBinding> {
        self.operations.get(index)?.semantic
    }
    pub fn epilogue(&self, index: usize) -> Option<FusionEpilogueOperation> {
        self.operations.get(index)?.epilogue
    }
    pub fn reduction(
        &self,
        index: usize,
    ) -> Option<&'static runmat_runtime::builtins::common::spec::FusionReductionDeclaration> {
        self.operations.get(index)?.reduction
    }
    pub fn identity(&self, semantic: Option<Digest>) -> anyhow::Result<Digest> {
        let identities: Vec<_> = self
            .operations
            .iter()
            .map(|operation| {
                (
                    &operation.identity,
                    operation
                        .reduction
                        .map(|declaration| (declaration.revision, declaration.operation)),
                )
            })
            .collect();
        Ok(Digest::sha256(serde_json::to_vec(&(
            "selected-fusion-bindings-v2",
            identities,
            semantic,
        ))?))
    }
}

// Closed numerical values and literal controls cannot invoke user methods.
// Object, symbolic, foreign, distributed and callable inputs retain runtime
// dispatch; their methods/effects must not be bypassed by a structural plan.
fn closed_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Complex(_, _)
            | Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::SparseTensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
            | Value::CharArray(_)
            | Value::String(_)
            | Value::StringArray(_)
    )
}

#[cfg(test)]
mod tests;
