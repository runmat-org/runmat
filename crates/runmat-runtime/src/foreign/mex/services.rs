use runmat_mex::{MexCancellationScope, MexDiagnostic, MexEngineCompletion, MexHostServices};
use runmat_value::Value;
use std::rc::Rc;
use std::sync::{atomic::AtomicBool, Arc};

use crate::context::RuntimeContext;

/// Runtime-owned callback authority for one active MEX invocation.
///
/// The exact invocation context is retained so native execution sees its
/// scoped workspace service instead of falling back to a process-global
/// interpreter workspace.
pub struct RuntimeMexHostServices {
    runtime: RuntimeContext,
}

impl RuntimeMexHostServices {
    pub fn new(runtime: RuntimeContext) -> Self {
        Self { runtime }
    }
}

impl MexHostServices for RuntimeMexHostServices {
    fn cancellation_scope(&self, cancellation: Arc<AtomicBool>) -> Box<dyn MexCancellationScope> {
        let previous = self.runtime.state().cancellation.replace(cancellation);
        Box::new(RuntimeMexCancellationScope {
            state: Rc::clone(self.runtime.state()),
            previous: Some(previous),
        })
    }

    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        pollster::block_on(self.runtime.scope(crate::call_builtin_async_with_outputs(
            "eval",
            &[Value::String(command.to_string())],
            0,
        )))
        .map(|_| ())
        .map_err(runtime_diagnostic)
    }

    fn eval_captured(&self, command: &str) -> MexEngineCompletion<()> {
        captured_completion(|| self.eval(command))
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic> {
        let value = pollster::block_on(self.runtime.scope(crate::call_feval_async_with_outputs(
            Value::from(function),
            &arguments,
            requested_outputs,
        )))
        .map_err(runtime_diagnostic)?;
        let sequence = crate::call::arguments::adapt_legacy_builtin_result(value)
            .map_err(runtime_diagnostic)?;
        crate::sequence::ResolveValueSequence::resolve(
            sequence,
            runmat_types::SequenceUse::SelectPrefix {
                count: requested_outputs,
            },
            crate::sequence::SequenceResolutionContext::default(),
        )
        .map_err(runtime_diagnostic)
    }

    fn call_captured(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> MexEngineCompletion<Vec<Value>> {
        captured_completion(|| self.call(function, arguments, requested_outputs))
    }

    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<Value>, MexDiagnostic> {
        validate_workspace(workspace)?;
        if workspace == "global" {
            let _scope = self.runtime.enter();
            return Ok(crate::workspace::session::global_value(name));
        }
        if let Some(service) = self.runtime.service_ports().workspace() {
            Ok(service.lookup(name))
        } else {
            let _scope = self.runtime.enter();
            Ok(crate::workspace::lookup(name))
        }
    }

    fn put_variable(&self, workspace: &str, name: &str, value: Value) -> Result<(), MexDiagnostic> {
        validate_workspace(workspace)?;
        if workspace == "global" {
            let _scope = self.runtime.enter();
            crate::workspace::session::store_global_named(name, value);
            return Ok(());
        }
        if let Some(service) = self.runtime.service_ports().workspace() {
            service.assign(name, value).map_err(runtime_diagnostic)
        } else {
            let _scope = self.runtime.enter();
            crate::workspace::assign(name, value).map_err(|message| MexDiagnostic {
                identifier: Some("RunMat:MEX:Workspace".into()),
                message,
            })
        }
    }

    fn get_object_property(&self, object: Value, name: &str) -> Result<Value, MexDiagnostic> {
        pollster::block_on(
            self.runtime
                .scope(crate::object::resolve::load_member_with_context(
                    Some(&self.runtime),
                    object,
                    name.to_string(),
                    false,
                    None,
                )),
        )
        .map_err(runtime_diagnostic)
    }

    fn get_object_property_at(
        &self,
        object: Value,
        index: usize,
        name: &str,
    ) -> Result<Value, MexDiagnostic> {
        let object = select_object_element(object, index)?;
        self.get_object_property(object, name)
    }

    fn set_object_property(
        &self,
        object: Value,
        name: &str,
        value: Value,
    ) -> Result<Value, MexDiagnostic> {
        pollster::block_on(self.runtime.scope(crate::object::resolve::store_member(
            object,
            name.to_string(),
            value,
            false,
            None,
            |_, _| {},
        )))
        .map_err(runtime_diagnostic)
    }

    fn set_object_property_at(
        &self,
        object: Value,
        index: usize,
        name: &str,
        value: Value,
    ) -> Result<Value, MexDiagnostic> {
        match object {
            Value::ObjectArray(array) => {
                let class_name = array.class_name().to_string();
                let shape = array.shape().to_vec();
                let mut elements = array.into_data();
                let selected = elements
                    .get(index)
                    .cloned()
                    .ok_or_else(|| object_index_diagnostic(index, elements.len()))?;
                elements[index] = self.set_object_property(selected, name, value)?;
                runmat_value::ObjectArray::new(class_name, elements, shape)
                    .map(Value::ObjectArray)
                    .map_err(|message| MexDiagnostic {
                        identifier: Some("RunMat:MEX:ObjectArray".into()),
                        message,
                    })
            }
            object if index == 0 => self.set_object_property(object, name, value),
            _ => Err(object_index_diagnostic(index, 1)),
        }
    }
}

fn captured_completion<T>(
    operation: impl FnOnce() -> Result<T, MexDiagnostic>,
) -> MexEngineCompletion<T> {
    let capture = crate::console::begin_capture();
    let result = operation();
    let entries = capture.finish_entries();
    let mut stdout = String::new();
    let mut stderr = String::new();
    for entry in entries {
        match entry.stream {
            crate::console::ConsoleStream::Stdout => stdout.push_str(&entry.text),
            crate::console::ConsoleStream::Stderr => stderr.push_str(&entry.text),
            crate::console::ConsoleStream::ClearScreen => {}
        }
    }
    MexEngineCompletion {
        result,
        stdout,
        stderr,
    }
}

struct RuntimeMexCancellationScope {
    state: Rc<crate::context::RuntimeContextState>,
    previous: Option<Arc<AtomicBool>>,
}

impl MexCancellationScope for RuntimeMexCancellationScope {}

impl Drop for RuntimeMexCancellationScope {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            self.state.cancellation.replace(previous);
        }
    }
}

fn select_object_element(object: Value, index: usize) -> Result<Value, MexDiagnostic> {
    match object {
        Value::ObjectArray(array) => array
            .get_linear(index)
            .cloned()
            .ok_or_else(|| object_index_diagnostic(index, array.len())),
        object if index == 0 => Ok(object),
        _ => Err(object_index_diagnostic(index, 1)),
    }
}

fn object_index_diagnostic(index: usize, length: usize) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:ObjectIndex".into()),
        message: format!(
            "zero-based object index {index} is outside an object array with {length} elements"
        ),
    }
}

fn validate_workspace(workspace: &str) -> Result<(), MexDiagnostic> {
    if matches!(workspace, "base" | "caller" | "global") {
        Ok(())
    } else {
        Err(MexDiagnostic {
            identifier: Some("RunMat:MEX:Workspace".into()),
            message: format!("unknown MEX workspace '{workspace}'"),
        })
    }
}

fn runtime_diagnostic(error: crate::RuntimeError) -> MexDiagnostic {
    MexDiagnostic {
        identifier: error.identifier().map(str::to_string),
        message: error.message().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RuntimeServicePorts, RuntimeWorkspaceService};
    use crate::execution::RuntimeExecutionService;
    use runmat_value::{HandleRef, ObjectInstance};
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::fs;
    use std::rc::Rc;

    #[derive(Default)]
    struct ScopedWorkspace {
        values: RefCell<BTreeMap<String, Value>>,
    }

    impl RuntimeWorkspaceService for ScopedWorkspace {
        fn lookup(&self, name: &str) -> Option<Value> {
            self.values.borrow().get(name).cloned()
        }

        fn snapshot(&self) -> Vec<(String, Value)> {
            self.values
                .borrow()
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect()
        }

        fn global_names(&self) -> Vec<String> {
            Vec::new()
        }

        fn assign(&self, name: &str, value: Value) -> Result<(), crate::RuntimeError> {
            self.values.borrow_mut().insert(name.to_string(), value);
            Ok(())
        }

        fn clear(&self) -> Result<(), crate::RuntimeError> {
            self.values.borrow_mut().clear();
            Ok(())
        }

        fn remove(&self, name: &str) -> Result<(), crate::RuntimeError> {
            self.values.borrow_mut().remove(name);
            Ok(())
        }
    }

    #[test]
    fn workspace_callbacks_use_the_exact_invocation_service() {
        let workspace = Rc::new(ScopedWorkspace::default());
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_workspace(workspace.clone()));
        let services = RuntimeMexHostServices::new(runtime);

        services
            .put_variable("caller", "answer", Value::Num(42.0))
            .expect("write scoped caller workspace");
        assert_eq!(
            services
                .get_variable("base", "answer")
                .expect("read scoped base workspace"),
            Some(Value::Num(42.0))
        );
        assert_eq!(workspace.lookup("answer"), Some(Value::Num(42.0)));
    }

    #[test]
    fn workspace_callbacks_reject_unknown_workspace_names() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let services = RuntimeMexHostServices::new(runtime);
        let error = services
            .get_variable("workspace-typo", "answer")
            .expect_err("unknown workspace must fail");

        assert_eq!(error.identifier.as_deref(), Some("RunMat:MEX:Workspace"));
    }

    #[test]
    fn asynchronous_callback_cancellation_is_scoped_and_restored() {
        let original = Arc::new(AtomicBool::new(false));
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            Arc::clone(&original),
        );
        let services = RuntimeMexHostServices::new(runtime.clone());
        let request = Arc::new(AtomicBool::new(false));

        {
            let _scope = services.cancellation_scope(Arc::clone(&request));
            assert!(Arc::ptr_eq(&runtime.cancellation(), &request));
            request.store(true, std::sync::atomic::Ordering::Release);
            assert!(runtime.state().is_cancelled());
        }

        assert!(Arc::ptr_eq(&runtime.cancellation(), &original));
        assert!(!runtime.state().is_cancelled());
    }

    #[test]
    fn object_callbacks_preserve_handle_identity_and_mutate_its_target() {
        let mut target = ObjectInstance::new("FixtureHandle");
        target.properties.insert("Value".into(), Value::Num(4.0));
        let target = runmat_gc::gc_allocate(Value::Object(target)).expect("allocate handle target");
        let handle = HandleRef {
            class_name: "FixtureHandle".into(),
            target,
            valid: true,
        };
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let services = RuntimeMexHostServices::new(runtime);

        assert_eq!(
            services
                .get_object_property(Value::HandleObject(handle.clone()), "Value")
                .expect("read handle property"),
            Value::Num(4.0)
        );
        let updated = services
            .set_object_property(
                Value::HandleObject(handle.clone()),
                "Value",
                Value::Num(9.0),
            )
            .expect("write handle property");
        let Value::HandleObject(updated) = updated else {
            panic!("handle property assignment must return the same handle kind");
        };
        assert_eq!(updated, handle);
        assert_eq!(
            services
                .get_object_property(Value::HandleObject(handle), "Value")
                .expect("read updated handle property"),
            Value::Num(9.0)
        );
    }

    #[test]
    fn indexed_object_callbacks_select_exact_zero_based_element() {
        let make_handle = |value| {
            let mut target = ObjectInstance::new("FixtureHandle");
            target.properties.insert("Value".into(), Value::Num(value));
            let target = runmat_gc::gc_allocate(Value::Object(target))
                .expect("allocate indexed handle target");
            HandleRef {
                class_name: "FixtureHandle".into(),
                target,
                valid: true,
            }
        };
        let first = make_handle(2.0);
        let second = make_handle(4.0);
        let array = runmat_value::ObjectArray::from_handles(
            "FixtureHandle",
            vec![first.clone(), second.clone()],
            vec![1, 2],
        )
        .expect("valid handle array");
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let services = RuntimeMexHostServices::new(runtime);

        assert_eq!(
            services
                .get_object_property_at(Value::ObjectArray(array.clone()), 1, "Value")
                .expect("read indexed property"),
            Value::Num(4.0)
        );
        let updated = services
            .set_object_property_at(Value::ObjectArray(array), 1, "Value", Value::Num(9.0))
            .expect("write indexed property");
        let Value::ObjectArray(updated) = updated else {
            panic!("indexed property write must preserve the object array");
        };
        assert_eq!(updated.shape(), &[1, 2]);
        runmat_gc::gc_with_value(&first.target, |value| {
            let Value::Object(value) = value else {
                panic!("first indexed target must remain an object");
            };
            assert_eq!(value.properties.get("Value"), Some(&Value::Num(2.0)));
        })
        .expect("inspect first indexed target");
        runmat_gc::gc_with_value(&second.target, |value| {
            let Value::Object(value) = value else {
                panic!("second indexed target must remain an object");
            };
            assert_eq!(value.properties.get("Value"), Some(&Value::Num(9.0)));
        })
        .expect("inspect second indexed target");
    }

    #[test]
    fn compiled_cpp_engine_properties_preserve_runtime_handle_identity() {
        let directory = tempfile::tempdir().expect("temporary C++ MEX directory");
        let source = directory.path().join("handle_properties.cpp");
        fs::write(
            &source,
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        matlab::data::Array object = inputs[0];
        if (object.getType() != matlab::data::ArrayType::HANDLE_OBJECT_REF) {
            throw matlab::Exception("input is not a handle object");
        }
        auto engine = getEngine();
        matlab::data::TypedArray<double> before =
            engine->getProperty(object, u"Value");
        if (static_cast<double>(before[0]) != 4.0) {
            throw matlab::Exception("handle property was not retained");
        }
        matlab::data::ArrayFactory factory;
        engine->setProperty(object, u"Value", factory.createScalar<double>(9.0));
        outputs[0] = object;
    }
};
"#,
        )
        .expect("write C++ MEX fixture");

        let mut target = ObjectInstance::new("FixtureHandle");
        target.properties.insert("Value".into(), Value::Num(4.0));
        let target = runmat_gc::gc_allocate(Value::Object(target)).expect("allocate handle target");
        let handle = HandleRef {
            class_name: "FixtureHandle".into(),
            target,
            valid: true,
        };
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let services = Rc::new(RuntimeMexHostServices::new(runtime));
        let artifact = runmat_mex::MexBuild::new(&source, directory.path())
            .compile()
            .expect("compile C++ MEX fixture");
        let module = runmat_mex::MexModule::load(&artifact.module).expect("load C++ MEX fixture");
        let invocation = module
            .invoke_with_services(
                &[Value::HandleObject(handle.clone())],
                1,
                module.api_mode(),
                services,
            )
            .expect("invoke C++ MEX fixture");
        let [Value::HandleObject(output)] = invocation.outputs.as_slice() else {
            panic!("C++ MEX output must retain handle-object identity");
        };
        assert_eq!(output, &handle);
        runmat_gc::gc_with_value(&handle.target, |value| {
            let Value::Object(value) = value else {
                panic!("handle target must remain an object");
            };
            assert_eq!(value.properties.get("Value"), Some(&Value::Num(9.0)));
        })
        .expect("inspect updated handle target");
    }
}
