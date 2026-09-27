use super::*;
use crate::{
    builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture},
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
};
use runmat_value::Value;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

struct ChangingService {
    bindings: RefCell<Vec<RuntimeBuiltinBinding>>,
    calls: Cell<usize>,
}

impl RuntimeBuiltinService for ChangingService {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        self.calls.set(self.calls.get() + 1);
        self.bindings
            .borrow()
            .iter()
            .copied()
            .filter(|binding| binding.identity.builtin.name.eq_ignore_ascii_case(name))
            .collect()
    }
}

fn replacement(_: &[Value]) -> RuntimeBuiltinFuture {
    Box::pin(async { Ok(Value::Num(99.0)) })
}

#[test]
fn indexed_declarations_never_cache_scoped_selection_permission() {
    runmat_gc::gc_test_context(|| {
        let original = crate::builtin::runtime_builtin_bindings_by_name("sin");
        assert_eq!(original.len(), 1);
        let service = Rc::new(ChangingService {
            bindings: RefCell::new(original.clone()),
            calls: Cell::new(0),
        });
        let context =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(RuntimeServicePorts::default().with_builtin(service.clone()));
        let expected = Some(original[0].identity);
        assert!(selected_fusion_binding("sin", expected, Some(&context)).is_some());
        service.bindings.borrow_mut()[0].implementation = replacement;
        assert!(selected_fusion_binding("sin", expected, Some(&context)).is_none());
        *service.bindings.borrow_mut() = vec![original[0], original[0]];
        assert!(selected_fusion_binding("sin", expected, Some(&context)).is_none());
        service.bindings.borrow_mut().clear();
        assert!(selected_fusion_binding("sin", expected, Some(&context)).is_none());
        *service.bindings.borrow_mut() = original;
        assert!(selected_fusion_binding("SIN", expected, Some(&context)).is_some());
        assert_eq!(service.calls.get(), 5);
        // A separate invocation authority cannot inherit this service's result.
        let absent = Rc::new(ChangingService {
            bindings: RefCell::new(vec![]),
            calls: Cell::new(0),
        });
        let other = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_builtin(absent.clone()));
        assert!(selected_fusion_binding("sin", expected, Some(&other)).is_none());
        assert_eq!(absent.calls.get(), 1);
        assert_eq!(service.calls.get(), 5);
    });
}
