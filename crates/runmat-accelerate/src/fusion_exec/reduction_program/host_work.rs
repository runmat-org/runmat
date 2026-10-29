//! Optional original-source host pricing; no provider feasibility or payload probes.
use super::inputs::Inputs;
use crate::fusion::{semantic_bindings::SemanticBinding, semantics::ResolvedProgram};
use runmat_accelerate_api::ProviderPrecision;
use runmat_runtime::builtins::common::spec::PreparedFusionReduction;
use runmat_types::FusionReductionOperation;
use runmat_value::Value;

pub(super) fn derive(
    mapped: &Inputs<'_>,
    values: &[&Value],
    precision: ProviderPrecision,
    snapshots: &[Option<SemanticBinding>],
    prepared: &dyn PreparedFusionReduction,
    operation: FusionReductionOperation,
) -> Option<u64> {
    // Exact here describes original CPU source classes, not the GPU owner's
    // current downcast policy. Static admission cannot read resident payloads.
    let original = ResolvedProgram::resolve_host_static_captured(
        &mapped.producer,
        values,
        precision,
        snapshots,
    )
    .ok()??;
    let contract = prepared.resolve(original.fact(mapped.root).ok()?).ok()??;
    if contract.operation != operation {
        return None;
    }
    original
        .host_scalar_work_units(&mapped.producer)
        .ok()?
        .checked_add(contract.algorithmic_work_prior()?.checked_units()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::{FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp};
    use crate::graph::{FusionCallOrigin, InstrSpan, PrimitiveOp, ShapeInfo};
    use runmat_runtime::builtins::common::spec::{
        canonical_fusion_declaration, FusionReductionControls,
    };
    use runmat_value::Tensor;
    use std::collections::HashMap;

    fn mapped(operation: FusionOp) -> Inputs<'static> {
        Inputs {
            producer: FusionGroupPlan {
                index: 0,
                group: FusionGroup {
                    id: 0,
                    kind: FusionKind::ElementwiseChain,
                    nodes: vec![],
                    shape: ShapeInfo::Scalar,
                    span: InstrSpan { start: 0, end: 1 },
                    pattern: None,
                    stack_layout: None,
                },
                operations: vec![operation],
                inputs: vec![0, 1],
                stack_pattern: vec![0, 1],
                constants: HashMap::new(),
                const_values: HashMap::new(),
                materialized_stores: vec![],
                output: Some(2),
                kernel: FusionKernelSpec {
                    kind: FusionKind::ElementwiseChain,
                    supported: true,
                },
                reduction_data: None,
                reduction_dim: None,
                reduction_flavor: None,
                reduction_axes: None,
                pattern: None,
            },
            data: vec![],
            controls: vec![],
            _mapping_claim: None,
            root: 2,
            terminal: 1,
            producer_indices: vec![0],
        }
    }

    #[test]
    fn original_class_host_prior_includes_its_own_terminal_contract() {
        runmat_gc::gc_test_context(|| {
            #[cfg(target_arch = "wasm32")]
            runmat_runtime::builtins::wasm_registry::register_all();
            let entry = &runmat_builtins::SUM_CATALOG_ENTRY;
            let declaration =
                canonical_fusion_declaration(entry.binding_identity(&entry.bindings[0]))
                    .unwrap()
                    .reduction
                    .unwrap();
            let prepared =
                futures::executor::block_on((declaration.prepare)(FusionReductionControls {
                    values: &[],
                    requested_outputs: 1,
                }))
                .unwrap()
                .unwrap();
            let source = mapped(FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![0, 1],
                output: Some(2),
            });
            for (left_single, right_single, expected) in
                [(false, false, 9), (true, true, 12), (true, false, 16)]
            {
                let value = |single| {
                    if single {
                        Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap())
                    } else {
                        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap())
                    }
                };
                let left = value(left_single);
                let right = value(right_single);
                for owner in [ProviderPrecision::F32, ProviderPrecision::F64] {
                    // Mixed arithmetic has two conversions plus the add for
                    // each producer element; its Single sum independently
                    // widens input samples and narrows final publication.
                    assert_eq!(
                        derive(
                            &source,
                            &[&left, &right],
                            owner,
                            &[None],
                            prepared.as_ref(),
                            declaration.operation
                        ),
                        Some(expected)
                    );
                }
            }

            let mut unresolved = mapped(FusionOp::Builtin {
                name: "acosh".into(),
                origin: FusionCallOrigin::named("acosh"),
                inputs: vec![0],
                output: Some(2),
            });
            unresolved.producer.inputs = vec![0];
            let snapshots =
                crate::fusion::semantic_bindings::capture(&unresolved.producer).unwrap();
            let input = Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap());
            assert_eq!(
                derive(
                    &unresolved,
                    &[&input],
                    ProviderPrecision::F64,
                    &snapshots,
                    prepared.as_ref(),
                    declaration.operation
                ),
                None,
                "missing static domain proof is no host price, not zero or a payload scan"
            );
        });
    }
}
