use super::*;
use crate::fusion::{FusionGroup, FusionKernelSpec, FusionKind};
use crate::graph::{InstrSpan, ShapeInfo};
use runmat_runtime::builtins::common::spec::ScalarType;
use std::collections::HashMap;

#[path = "semantics_tests/exact_product.rs"]
mod exact_product_tests;

fn plan(operations: Vec<FusionOp>, output: ValueId) -> FusionGroupPlan {
    // Direct fusion fixtures bypass the runtime entrypoint that initializes
    // browser declarations; native inventory initializes them automatically.
    #[cfg(target_arch = "wasm32")]
    runmat_runtime::builtins::wasm_registry::register_all();

    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: vec![],
            shape: ShapeInfo::Scalar,
            span: InstrSpan { start: 0, end: 4 },
            pattern: None,
            stack_layout: None,
        },
        operations,
        inputs: vec![0, 1],
        stack_pattern: vec![0, 1],
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: vec![],
        output: Some(output),
        kernel: FusionKernelSpec {
            kind: FusionKind::ElementwiseChain,
            supported: true,
        },
        reduction_data: None,
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: None,
        pattern: None,
    }
}

fn cast(name: &str, input: ValueId, output: ValueId) -> FusionOp {
    FusionOp::Builtin {
        origin: Default::default(),
        name: name.into(),
        inputs: vec![input],
        output: Some(output),
    }
}

fn resolve(
    plan: &FusionGroupPlan,
    inputs: &[Value],
) -> std::result::Result<ResolvedProgram, FusionResolveFailure> {
    resolve_policy(plan, inputs, ProviderPrecision::F64, PrecisionPolicy::Exact)
}

fn resolve_policy(
    plan: &FusionGroupPlan,
    inputs: &[Value],
    precision: ProviderPrecision,
    policy: PrecisionPolicy,
) -> std::result::Result<ResolvedProgram, FusionResolveFailure> {
    futures::executor::block_on(ResolvedProgram::resolve_with_policy(
        plan, inputs, precision, policy,
    ))
}

#[path = "semantics_tests/domain_arithmetic.rs"]
mod domain_arithmetic;
#[path = "semantics_tests/input_representation.rs"]
mod input_representation;
#[path = "semantics_tests/power.rs"]
mod power;
#[path = "semantics_tests/precision_policy.rs"]
mod precision_policy;
#[path = "semantics_tests/sign_chain.rs"]
mod sign_chain;
#[path = "semantics_tests/source_identity.rs"]
mod source_identity;
#[path = "semantics_tests/resident_inputs.rs"]
mod resident_inputs;
#[path = "semantics_tests/source_plan_identity.rs"]
mod source_plan_identity;
