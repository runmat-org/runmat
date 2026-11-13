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

/// Exact requested payload for the captured operation vector. The caller
/// checks the multiplication before any binding capture can allocate.
pub(crate) fn binding_storage_bytes(count: usize) -> Option<u64> {
    count
        .checked_mul(std::mem::size_of::<Operation>())
        .and_then(|bytes| u64::try_from(bytes).ok())
}

impl AdmittedBindings {
    /// Numerical output ownership follows the actual VM consumer. Unary plus
    /// returns its input unchanged, so it has no output carrier to consume.
    pub(crate) fn host_numerical_operations(
        &self,
        runtime: Option<&runmat_runtime::context::RuntimeContext>,
    ) -> Option<Vec<runmat_runtime::context::VmFusionHostOperation>> {
        use runmat_runtime::context::VmFusionHostOperation as HostOperation;
        let mut operations = Vec::with_capacity(self.operations.len());
        for operation in &self.operations {
            operations.push(match &operation.identity {
                Identity::Primitive(
                    op @ (PrimitiveOp::UPlus | PrimitiveOp::Neg | PrimitiveOp::Mul),
                ) => HostOperation::Intrinsic(op.operator_kind()),
                Identity::Primitive(op) => {
                    HostOperation::Builtin(selected_host_binding_for_primitive(*op, runtime)?)
                }
                Identity::Call {
                    binding: SelectedFusionIdentity::Canonical(binding),
                    ..
                } => HostOperation::Builtin(*binding),
                Identity::Call { .. } => return None,
            });
        }
        Some(operations)
    }

    pub(super) fn host_facts(
        &self,
        request: &FusionExecutionRequest<'_>,
        precision: runmat_accelerate_api::ProviderPrecision,
    ) -> Option<crate::fusion::semantics::ResolvedProgram> {
        let mut snapshots = Vec::with_capacity(self.operations.len());
        snapshots.extend(self.operations.iter().map(|operation| operation.semantic));
        let mut inputs = Vec::with_capacity(request.inputs.len());
        inputs.extend(request.inputs.iter());
        crate::fusion::semantics::ResolvedProgram::resolve_host_static_captured(
            request.plan,
            &inputs,
            precision,
            &snapshots,
        )
        .ok()?
    }

    pub(super) fn host_resource_bound(
        &self,
        index: usize,
        runtime: Option<&runmat_runtime::context::RuntimeContext>,
        query: &runmat_runtime::builtins::common::spec::FusionHostResourceQuery<'_>,
    ) -> Option<runmat_runtime::builtins::common::spec::FusionHostResourceBound> {
        let operation = self.operations.get(index)?;
        let binding = match &operation.identity {
            Identity::Primitive(
                op @ (PrimitiveOp::Neg | PrimitiveOp::UPlus | PrimitiveOp::Mul),
            ) => {
                return runmat_runtime::builtins::math::host_resources::intrinsic(
                    op.operator_kind(),
                    query,
                );
            }
            Identity::Primitive(op) => selected_host_binding_for_primitive(*op, runtime)?,
            Identity::Call {
                binding: SelectedFusionIdentity::Canonical(binding),
                ..
            } => *binding,
            Identity::Call { .. } => return None,
        };
        canonical_fusion_declaration(binding)?.host_resources?(query)
    }
    /// Intrinsic VM numeric owners retain their operator identity. Unary minus
    /// also authenticates its real canonical `times` dependency; scalar matrix
    /// multiplication deliberately does not borrow a named mtimes override.
    pub fn host_operations(
        &self,
        runtime: Option<&runmat_runtime::context::RuntimeContext>,
    ) -> Option<Vec<runmat_runtime::context::VmFusionHostOperation>> {
        let mut operations = Vec::with_capacity(self.operations.len());
        self.visit_host_operations(runtime, |operation| operations.push(operation))?;
        Some(operations)
    }

    pub(super) fn has_host_operations(
        &self,
        runtime: Option<&runmat_runtime::context::RuntimeContext>,
    ) -> bool {
        self.visit_host_operations(runtime, |_| {}).is_some()
    }

    fn visit_host_operations(
        &self,
        runtime: Option<&runmat_runtime::context::RuntimeContext>,
        mut visit: impl FnMut(runmat_runtime::context::VmFusionHostOperation),
    ) -> Option<()> {
        use runmat_runtime::context::VmFusionHostOperation as H;
        for operation in &self.operations {
            match &operation.identity {
                Identity::Primitive(
                    op @ (PrimitiveOp::Neg | PrimitiveOp::UPlus | PrimitiveOp::Mul),
                ) => {
                    visit(H::Intrinsic(op.operator_kind()));
                    if *op == PrimitiveOp::Neg {
                        let times =
                            runmat_types::OperatorKind::ElementwiseMultiply.overload_name()?;
                        visit(H::Builtin(
                            crate::fusion::semantic_bindings::selected_canonical_host_binding(
                                times.display_name(),
                                runtime,
                            )?,
                        ));
                    }
                }
                Identity::Primitive(op) => visit(H::Builtin(selected_host_binding_for_primitive(
                    *op, runtime,
                )?)),
                Identity::Call {
                    binding: SelectedFusionIdentity::Canonical(binding),
                    ..
                } => visit(H::Builtin(*binding)),
                Identity::Call { .. } => return None,
            }
        }
        Some(())
    }
    /// Exact canonical implementations already captured for the selected
    /// fusion program. Legacy or opaque entries cannot certify a VM host
    /// region's inner execution.
    pub fn canonical_bindings(
        &self,
        runtime: Option<&runmat_runtime::context::RuntimeContext>,
    ) -> Option<Vec<runmat_builtins::BuiltinBindingIdentity>> {
        let mut bindings = Vec::with_capacity(self.operations.len());
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
        let mut operations = Vec::with_capacity(request.plan.operations.len());
        for operation in &request.plan.operations {
            let selected = (|| {
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
            })()?;
            operations.push(selected);
        }
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
        struct Identities<'a>(&'a [Operation]);
        impl Serialize for Identities<'_> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeSeq;
                let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
                for operation in self.0 {
                    sequence.serialize_element(&(
                        &operation.identity,
                        operation
                            .reduction
                            .map(|declaration| (declaration.revision, declaration.operation)),
                    ))?;
                }
                sequence.end()
            }
        }
        Ok(Digest::sha256_serialized(&(
            "selected-fusion-bindings-v2",
            Identities(&self.operations),
            semantic,
        ))?)
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
