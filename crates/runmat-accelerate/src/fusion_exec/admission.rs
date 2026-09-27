//! Selected-binding evidence exists only for this invocation. It is acquired
//! before domain probes, placement, compilation, uploads or provider execution.
use super::FusionExecutionRequest;
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
}
pub(super) struct AdmittedBindings {
    operations: Vec<Operation>,
}

impl AdmittedBindings {
    pub fn new(request: &FusionExecutionRequest<'_>) -> Option<Self> {
        if request.inputs.len() != request.plan.inputs.len()
            || !request
                .inputs
                .iter()
                .chain(request.plan.const_values.values())
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
                        let epilogue = if *op == PrimitiveOp::ElemPow {
                            let entry = &runmat_builtins::POWER_CATALOG_ENTRY;
                            let [binding] = entry.bindings else {
                                return None;
                            };
                            canonical_fusion_declaration(entry.binding_identity(binding))?.epilogue
                        } else {
                            None
                        };
                        return Some(Operation {
                            identity: Identity::Primitive(*op),
                            structural: None,
                            epilogue,
                            reduction: None,
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
                    });
                }
                let expected = match origin {
                    FusionCallOrigin::Named { binding } => Some(binding),
                    _ => None,
                };
                let selected = selected_fusion_binding(name, expected, request.runtime.as_ref())?;
                Some(Operation {
                    identity: Identity::Call {
                        origin,
                        binding: selected.identity,
                    },
                    structural: selected.structural,
                    epilogue: selected.epilogue,
                    reduction: selected.reduction,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self { operations })
    }

    pub fn structural(&self, index: usize) -> Option<FusionStructuralOperation> {
        self.operations.get(index)?.structural
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
                    operation.reduction.map(|declaration| declaration.revision),
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
