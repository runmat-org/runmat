use super::*;
use crate::fusion::{FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind};
use crate::fusion_exec::*;
use crate::graph::{InstrSpan, ShapeInfo};
use runmat_runtime::{
    builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture},
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
};
use std::{collections::HashMap, rc::Rc};

struct Selected(Vec<RuntimeBuiltinBinding>);
impl RuntimeBuiltinService for Selected {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        self.0
            .iter()
            .filter(|binding| binding.identity.builtin.name == name)
            .copied()
            .collect()
    }
}
fn replacement(_: &[Value]) -> RuntimeBuiltinFuture {
    Box::pin(async { Ok(Value::Num(99.0)) })
}
fn context(bindings: Vec<RuntimeBuiltinBinding>) -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(Selected(bindings))))
}
fn named(name: &str, origin: FusionCallOrigin) -> FusionOp {
    FusionOp::Builtin {
        name: name.into(),
        origin,
        inputs: vec![0, 1],
        output: Some(2),
    }
}
fn make_plan(operations: Vec<FusionOp>) -> FusionGroupPlan {
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
    }
}
fn request<'a>(plan: &'a FusionGroupPlan, runtime: &RuntimeContext) -> FusionExecutionRequest<'a> {
    FusionExecutionRequest {
        plan,
        inputs: vec![Value::Num(2.), Value::Num(3.)],
        placement: None,
        runtime: Some(runtime.clone()),
    }
}

#[test]
fn selected_named_bindings_reject_replacements_absence_and_ambiguity() {
    runmat_gc::gc_test_context(|| {
        for name in ["sin", "mtimes", "power"] {
            let original = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
            assert_eq!(original.len(), 1, "{name}");
            let plan = make_plan(vec![named(name, FusionCallOrigin::named(name))]);
            assert!(AdmittedBindings::new(&request(&plan, &context(original.clone()))).is_some());
            let mut changed = original.clone();
            changed[0].implementation = replacement;
            let selected = context(changed);
            assert!(AdmittedBindings::new(&request(&plan, &selected)).is_none());
            // The retained route really reaches the selected implementation.
            let result = futures::executor::block_on(selected.scope(
                runmat_runtime::call_builtin_async_with_outputs(
                    name,
                    &[Value::Num(2.), Value::Num(3.)],
                    1,
                ),
            ))
            .unwrap();
            assert_eq!(result, Value::Num(99.));
            assert!(AdmittedBindings::new(&request(&plan, &context(vec![]))).is_none());
            assert!(AdmittedBindings::new(&request(
                &plan,
                &context(vec![original[0], original[0]])
            ))
            .is_none());
        }
    });
}

#[test]
fn primitive_matrix_syntax_is_not_a_named_service_call_and_identity_retains_origin() {
    runmat_gc::gc_test_context(|| {
        let name = "mtimes";
        let original = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        let original_context = context(original.clone());
        let mut overridden = original;
        overridden[0].implementation = replacement;
        let overridden = context(overridden);
        let primitive = make_plan(vec![named(
            name,
            FusionCallOrigin::Operator(runmat_types::OperatorKind::MatrixMultiply),
        )]);
        let named = make_plan(vec![named(name, FusionCallOrigin::named(name))]);
        let primitive_proof = AdmittedBindings::new(&request(&primitive, &overridden)).unwrap();
        assert!(AdmittedBindings::new(&request(&named, &overridden)).is_none());
        let named_proof = AdmittedBindings::new(&request(&named, &original_context)).unwrap();
        assert_ne!(
            primitive_proof.identity(None).unwrap(),
            named_proof.identity(None).unwrap()
        );
        let mut object = request(&primitive, &overridden);
        object.inputs[0] = Value::Object(runmat_value::ObjectInstance::new("OverloadedMatrix"));
        assert!(AdmittedBindings::new(&object).is_none());
        let forged = make_plan(vec![FusionOp::Builtin {
            name: "sin".into(),
            origin: FusionCallOrigin::Operator(runmat_types::OperatorKind::MatrixMultiply),
            inputs: vec![0, 1],
            output: Some(2),
        }]);
        assert!(AdmittedBindings::new(&request(&forged, &original_context)).is_none());
    });
}

#[test]
fn every_placed_family_returns_to_runtime_before_provider_or_domain_work() {
    runmat_gc::gc_test_context(|| {
        let mut selected = runmat_runtime::builtin::runtime_builtin_bindings_by_name("sin");
        selected[0].implementation = replacement;
        let context = context(selected);
        let plan = make_plan(vec![named("sin", FusionCallOrigin::named("sin"))]);
        // An arbitrary specialized shape/pattern cannot authorize bypassing a
        // selected override. No provider is installed or needed for this gate.
        futures::executor::block_on(async {
            assert!(matches!(
                execute_elementwise_placed(request(&plan, &context))
                    .await
                    .unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
            assert!(matches!(
                execute_matmul_epilogue_placed(request(&plan, &context))
                    .await
                    .unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
            assert!(matches!(
                execute_centered_gram_placed(request(&plan, &context))
                    .await
                    .unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
            assert!(matches!(
                execute_power_step_normalize_placed(request(&plan, &context))
                    .await
                    .unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
            assert!(matches!(
                execute_explained_variance_placed(request(&plan, &context))
                    .await
                    .unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
            assert!(matches!(
                execute_image_normalize_placed(request(&plan, &context))
                    .await
                    .unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
        });
        assert!(matches!(
            execute_reduction_with_shape_placed(request(&plan, &context), 1, 1, &[1, 1], 64)
                .unwrap(),
            FusionExecutionOutcome::SharedRuntime(_)
        ));
        let legacy = make_plan(vec![named("sum", FusionCallOrigin::NamedUnresolved)]);
        assert!(AdmittedBindings::new(&request(&legacy, &context)).is_none());
        let composite = make_plan(vec![
            named(
                "mtimes",
                FusionCallOrigin::Operator(runmat_types::OperatorKind::MatrixMultiply),
            ),
            FusionOp::Builtin {
                name: "sin".into(),
                origin: FusionCallOrigin::named("sin"),
                inputs: vec![2],
                output: Some(3),
            },
        ]);
        assert!(AdmittedBindings::new(&request(&composite, &context)).is_none());
    });
}
