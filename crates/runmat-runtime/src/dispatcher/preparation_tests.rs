use super::*;
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use std::{cell::RefCell, rc::Rc};

#[derive(Default)]
struct Probe(RefCell<Vec<String>>);
impl RuntimeAccelerationService for Probe {
    fn supports_operation(&self, _: &str) -> bool {
        true
    }
    fn prepare_legacy(
        &self,
        _: RuntimeContext,
        name: String,
        args: Vec<Value>,
    ) -> RuntimeServiceFuture<Result<Vec<Value>, RuntimeError>> {
        self.0.borrow_mut().push(name);
        Box::pin(async move { Ok(args) })
    }
}
fn runtime(probe: &Rc<Probe>) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_acceleration(probe.clone()))
}

struct Override;
impl crate::context::RuntimeBuiltinService for Override {
    fn bindings_by_name(&self, name: &str) -> Vec<crate::builtin::RuntimeBuiltinBinding> {
        let Some(entry) = runmat_builtins::builtin_catalog_primary_entry_by_name(name) else {
            return vec![];
        };
        vec![crate::builtin::RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            |args| {
                let value = args[0].clone();
                assert!(matches!(value, Value::Tensor(_)));
                Box::pin(async { Ok(value) })
            },
        )]
    }
}

#[test]
fn selected_override_receives_original_operands_without_legacy_preparation() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let ports = context
        .service_ports()
        .clone()
        .with_builtin(Rc::new(Override));
    let context = context.with_service_ports(ports);
    let _scope = context.enter();
    let input = Value::Tensor(runmat_value::Tensor::new(vec![1.; 4], vec![1, 4]).unwrap());
    assert_eq!(
        call_builtin("logical", std::slice::from_ref(&input)).unwrap(),
        input
    );
    assert!(probe.0.borrow().is_empty());
}

#[test]
fn canonical_execution_never_enters_legacy_preparation() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let _scope = context.enter();
    call_builtin("full", &[Value::Num(3.)]).unwrap();
    assert!(probe.0.borrow().is_empty());
}

#[test]
fn inventory_execution_enters_legacy_preparation_after_selection() {
    assert!(runmat_builtins::builtin_function_by_name("mean").is_some());
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let _scope = context.enter();
    call_builtin("mean", &[Value::Num(3.)]).unwrap();
    assert_eq!(&*probe.0.borrow(), &["mean"]);
}

#[test]
fn registered_instance_method_sees_original_operand_and_skips_legacy_service() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let _scope = context.enter();
    let class = "PreparationOrderFixture";
    let mut methods = std::collections::HashMap::new();
    methods.insert(
        "mean".into(),
        crate::class_registry::RuntimeMethod {
            name: "mean".into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: runmat_types::MemberAccess::Public,
            function_name: "preparation_fixture_mean".into(),
            implicit_class_argument: None,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: class.into(),
        parent: None,
        properties: Default::default(),
        methods,
    });
    let _resolver = crate::user_functions::install_semantic_function_resolver(Some(
        std::sync::Arc::new(|name| (name == "preparation_fixture_mean").then_some(29021)),
    ));
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
        std::sync::Arc::new(|_, args, _| {
            assert!(matches!(&args[0], Value::Object(_)));
            assert!(matches!(&args[1], Value::Tensor(_)));
            Box::pin(async { crate::sequence::single_value_sequence(Value::Num(9.)) })
        }),
    ));
    let args = [
        Value::Object(runmat_value::ObjectInstance::new(class)),
        Value::Tensor(runmat_value::Tensor::new(vec![1.; 4], vec![1, 4]).unwrap()),
    ];
    assert_eq!(call_builtin("mean", &args).unwrap(), Value::Num(9.));
    assert!(probe.0.borrow().is_empty());
}
