use runmat_execution::cancellation::CancellationToken;
use runmat_mex::{MexCancellationScope, MexDiagnostic, MexEngineCompletion, MexHostServices};
use runmat_value::Value;

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
    fn cancellation_scope(&self, cancellation: CancellationToken) -> Box<dyn MexCancellationScope> {
        let combined = self.runtime.cancellation().or(&cancellation);
        let previous = self.runtime.replace_cancellation_token(combined);
        Box::new(RuntimeMexCancellationScope {
            runtime: self.runtime.clone(),
            previous: Some(previous),
        })
    }

    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        pollster::block_on(crate::call_builtin_async_with_outputs_in_context(
            &self.runtime,
            "eval",
            &[Value::String(command.to_string())],
            0,
        ))
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
        let value = pollster::block_on(crate::call_feval_async_with_outputs_in_context(
            &self.runtime,
            Value::from(function),
            &arguments,
            requested_outputs,
        ))
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
        let workspace = MexWorkspace::parse(workspace)?;
        let _scope = self.runtime.enter();
        match workspace {
            MexWorkspace::Caller => self
                .runtime
                .service_ports()
                .require_workspace("read the MEX caller workspace")
                .map_err(capability_diagnostic)?
                .try_lookup(name)
                .map_err(runtime_diagnostic),
            MexWorkspace::Base => self
                .runtime
                .service_ports()
                .require_base_workspace("read the MEX base workspace")
                .map_err(capability_diagnostic)?
                .owner()
                .lookup(name)
                .map_err(crate::workspace::base_workspace_error_to_runtime)
                .map_err(runtime_diagnostic),
            MexWorkspace::Global => Ok(crate::workspace::session::global_value(name)),
        }
    }

    fn put_variable(&self, workspace: &str, name: &str, value: Value) -> Result<(), MexDiagnostic> {
        let workspace = MexWorkspace::parse(workspace)?;
        let _scope = self.runtime.enter();
        match workspace {
            MexWorkspace::Caller => self
                .runtime
                .service_ports()
                .require_workspace("write the MEX caller workspace")
                .map_err(capability_diagnostic)?
                .assign(name, value)
                .map_err(runtime_diagnostic),
            MexWorkspace::Base => {
                let mut delta = crate::workspace::WorkspaceDelta::new();
                delta
                    .assign(name, value)
                    .map_err(crate::workspace::base_workspace_error_to_runtime)
                    .map_err(runtime_diagnostic)?;
                self.runtime
                    .service_ports()
                    .require_base_workspace("write the MEX base workspace")
                    .map_err(capability_diagnostic)?
                    .owner()
                    .apply(delta)
                    .map(|_| ())
                    .map_err(crate::workspace::base_workspace_error_to_runtime)
                    .map_err(runtime_diagnostic)
            }
            MexWorkspace::Global => {
                crate::workspace::session::store_global_named(name, value);
                Ok(())
            }
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
        pollster::block_on(
            self.runtime
                .scope(crate::object::resolve::store_member_in_context(
                    &self.runtime,
                    object,
                    name.to_string(),
                    value,
                    false,
                    None,
                    |_, _| {},
                )),
        )
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
    runtime: RuntimeContext,
    previous: Option<CancellationToken>,
}

impl MexCancellationScope for RuntimeMexCancellationScope {}

impl Drop for RuntimeMexCancellationScope {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            self.runtime.replace_cancellation_token(previous);
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MexWorkspace {
    Base,
    Caller,
    Global,
}

impl MexWorkspace {
    fn parse(workspace: &str) -> Result<Self, MexDiagnostic> {
        match workspace {
            "base" => Ok(Self::Base),
            "caller" => Ok(Self::Caller),
            "global" => Ok(Self::Global),
            _ => Err(MexDiagnostic {
                identifier: Some("RunMat:MEX:Workspace".into()),
                message: format!("unknown MEX workspace '{workspace}'"),
            }),
        }
    }
}

fn capability_diagnostic(error: crate::context::RuntimeCapabilityError) -> MexDiagnostic {
    runtime_diagnostic(error.into_runtime_error())
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
        fn visit_retained_values(
            &self,
            visitor: &mut dyn FnMut(&Value) -> Option<()>,
        ) -> Option<()> {
            for value in self.values.try_borrow().ok()?.values() {
                visitor(value)?;
            }
            Some(())
        }
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
    fn workspace_callbacks_keep_base_caller_and_global_namespaces_distinct() {
        let caller = Rc::new(ScopedWorkspace::default());
        let base_owner = crate::workspace::BaseWorkspaceOwner::new();
        let base = Rc::new(crate::workspace::BaseWorkspaceService::new(
            base_owner.clone(),
        ));
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_workspace(caller.clone())
                    .with_base_workspace(base),
            );
        let services = RuntimeMexHostServices::new(runtime);

        services
            .put_variable("caller", "answer", Value::Num(42.0))
            .expect("write scoped caller workspace");
        services
            .put_variable("base", "answer", Value::Num(7.0))
            .expect("write session base workspace");
        services
            .put_variable("global", "answer", Value::Num(3.0))
            .expect("write session global workspace");
        assert_eq!(
            services
                .get_variable("base", "answer")
                .expect("read scoped base workspace"),
            Some(Value::Num(7.0))
        );
        assert_eq!(
            services
                .get_variable("caller", "answer")
                .expect("read scoped caller workspace"),
            Some(Value::Num(42.0))
        );
        assert_eq!(
            services
                .get_variable("global", "answer")
                .expect("read scoped global workspace"),
            Some(Value::Num(3.0))
        );
        assert_eq!(caller.lookup("answer"), Some(Value::Num(42.0)));
        assert_eq!(base_owner.lookup("answer").unwrap(), Some(Value::Num(7.0)));
    }

    #[test]
    fn base_workspace_callbacks_never_fall_back_to_the_caller() {
        let caller = Rc::new(ScopedWorkspace::default());
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_workspace(caller.clone()));
        let services = RuntimeMexHostServices::new(runtime);

        services
            .put_variable("caller", "answer", Value::Num(42.0))
            .expect("caller workspace remains available");
        let error = services
            .get_variable("base", "answer")
            .expect_err("missing base authority must not read the caller");
        assert_eq!(
            error.identifier.as_deref(),
            Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
        );
        assert_eq!(caller.lookup("answer"), Some(Value::Num(42.0)));
    }

    #[test]
    fn base_revocation_does_not_revoke_the_retained_caller_service() {
        let caller = Rc::new(ScopedWorkspace::default());
        let base_owner = crate::workspace::BaseWorkspaceOwner::new();
        let base = Rc::new(crate::workspace::BaseWorkspaceService::new(
            base_owner.clone(),
        ));
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_workspace(caller.clone())
                    .with_base_workspace(base),
            );
        let services = RuntimeMexHostServices::new(runtime);
        base_owner.close().unwrap();

        services
            .put_variable("caller", "answer", Value::Num(42.0))
            .expect("caller remains independently live");
        let error = services
            .put_variable("base", "answer", Value::Num(7.0))
            .expect_err("closed base authority must reject mutation");
        assert_eq!(error.identifier.as_deref(), Some("RunMat:Workspace:Closed"));
        assert_eq!(caller.lookup("answer"), Some(Value::Num(42.0)));
    }

    #[test]
    fn missing_caller_authority_does_not_hide_the_explicit_base_port() {
        let base_owner = crate::workspace::BaseWorkspaceOwner::new();
        let base = Rc::new(crate::workspace::BaseWorkspaceService::new(
            base_owner.clone(),
        ));
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_base_workspace(base));
        let services = RuntimeMexHostServices::new(runtime);

        services
            .put_variable("base", "answer", Value::Num(7.0))
            .expect("explicit base authority remains available");
        let error = services
            .get_variable("caller", "answer")
            .expect_err("missing caller authority must not fall back to base");
        assert_eq!(
            error.identifier.as_deref(),
            Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
        );
        assert_eq!(base_owner.lookup("answer").unwrap(), Some(Value::Num(7.0)));
    }

    #[test]
    fn global_workspace_callbacks_are_session_scoped() {
        let first = RuntimeMexHostServices::new(RuntimeContext::new(Rc::new(
            RuntimeExecutionService::new(),
        )));
        let second = RuntimeMexHostServices::new(RuntimeContext::new(Rc::new(
            RuntimeExecutionService::new(),
        )));

        first
            .put_variable("global", "answer", Value::Num(3.0))
            .unwrap();
        assert_eq!(
            first.get_variable("global", "answer").unwrap(),
            Some(Value::Num(3.0))
        );
        assert_eq!(second.get_variable("global", "answer").unwrap(), None);
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
        let original = runmat_execution::cancellation::CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            original.token(),
        );
        let services = RuntimeMexHostServices::new(runtime.clone());
        let request = runmat_execution::cancellation::CancellationSource::new();

        {
            let _scope = services.cancellation_scope(request.token());
            assert!(!runtime.is_cancelled());
            request.cancel(runmat_execution::CancellationReason::User);
            assert!(runtime.is_cancelled());
        }

        assert!(!runtime.is_cancelled());
        original.cancel(runmat_execution::CancellationReason::Shutdown);
        assert!(runtime.is_cancelled());
    }

    #[test]
    fn asynchronous_callback_keeps_parent_and_nested_cancellation_authorities() {
        let parent = runmat_execution::cancellation::CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            parent.token(),
        );
        let services = RuntimeMexHostServices::new(runtime.clone());
        let request = runmat_execution::cancellation::CancellationSource::new();
        let nested = runmat_execution::cancellation::CancellationSource::new();
        {
            let _request = services.cancellation_scope(request.token());
            {
                let _nested = services.cancellation_scope(nested.token());
                nested.cancel(runmat_execution::CancellationReason::Deadline);
                assert!(runtime.is_cancelled());
            }
            assert!(!runtime.is_cancelled(), "nested cancellation stays scoped");
            parent.cancel(runmat_execution::CancellationReason::Shutdown);
            assert!(
                runtime.is_cancelled(),
                "parent remains visible during request"
            );
        }
        assert_eq!(
            runtime.cancellation().reason(),
            Some(runmat_execution::CancellationReason::Shutdown)
        );
        assert!(
            !request.token().is_cancelled(),
            "composition does not cancel siblings"
        );
    }

    fn register_fixture_handle_class(runtime: &RuntimeContext) {
        pollster::block_on(runtime.scope(async {
            crate::compatibility::set_runmat_extensions_enabled(false);
            let mut class = crate::class_registry::RuntimeClass {
                name: "FixtureHandle".into(),
                parent: None,
                properties: Default::default(),
                methods: Default::default(),
            };
            class.properties.insert(
                "Value".into(),
                crate::class_registry::RuntimeProperty {
                    name: "Value".into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: runmat_types::MemberAccess::Public,
                    set_access: runmat_types::MemberAccess::Public,
                    default_value: None,
                },
            );
            crate::class_registry::register_class(class);
        }));
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
        register_fixture_handle_class(&runtime);
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
        register_fixture_handle_class(&runtime);
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
        register_fixture_handle_class(&runtime);
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
