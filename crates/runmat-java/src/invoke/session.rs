use super::{
    conversion::binary_name,
    error::jni_error,
    reflection::{constructor_candidates, field_descriptor, method_candidates, ReflectedCallable},
    JavaInvocationError, JavaValue,
};
use crate::{
    select_overload, JavaArgumentType, JavaCallableCandidate, JavaObjectHandle, JavaObjectRegistry,
    JvmProcess,
};
use jni::objects::GlobalRef;
use std::cell::RefCell;

pub struct JavaSession {
    pub(super) process: JvmProcess,
    pub(super) objects: RefCell<JavaObjectRegistry<GlobalRef>>,
}

impl std::fmt::Debug for JavaSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JavaSession")
            .field("installation", self.process.installation())
            .field("object_count", &self.objects.borrow().len())
            .finish_non_exhaustive()
    }
}

impl JavaSession {
    pub fn new(process: JvmProcess) -> Self {
        Self {
            process,
            objects: RefCell::new(JavaObjectRegistry::default()),
        }
    }

    pub fn construct(
        &self,
        class_name: &str,
        signature: &str,
        arguments: &[JavaValue],
    ) -> Result<JavaValue, JavaInvocationError> {
        self.process.with_attached(|environment| {
            let prepared = self.prepare_arguments(environment, arguments)?;
            let values = prepared.values();
            let object = environment
                .new_object(binary_name(class_name), signature, &values)
                .map_err(|error| jni_error(environment, error))?;
            self.capture_object(environment, object)
        })
    }

    pub fn construct_resolved(
        &self,
        class_name: &str,
        arguments: &[JavaValue],
    ) -> Result<JavaValue, JavaInvocationError> {
        let callable = self.resolve_constructor(class_name, arguments)?;
        self.construct(class_name, &callable.descriptor, arguments)
    }

    pub fn call_static(
        &self,
        class_name: &str,
        method_name: &str,
        signature: &str,
        arguments: &[JavaValue],
    ) -> Result<JavaValue, JavaInvocationError> {
        self.process.with_attached(|environment| {
            let prepared = self.prepare_arguments(environment, arguments)?;
            let values = prepared.values();
            let result = environment
                .call_static_method(binary_name(class_name), method_name, signature, &values)
                .map_err(|error| jni_error(environment, error))?;
            self.capture_value(environment, result)
        })
    }

    pub fn call_static_resolved(
        &self,
        class_name: &str,
        method_name: &str,
        arguments: &[JavaValue],
    ) -> Result<JavaValue, JavaInvocationError> {
        let callable = self.resolve_method(class_name, method_name, true, arguments)?;
        self.call_static(class_name, method_name, &callable.descriptor, arguments)
    }

    pub fn call_method(
        &self,
        receiver: JavaObjectHandle,
        method_name: &str,
        signature: &str,
        arguments: &[JavaValue],
    ) -> Result<JavaValue, JavaInvocationError> {
        let receiver = self.global_reference(receiver)?;
        self.process.with_attached(|environment| {
            let prepared = self.prepare_arguments(environment, arguments)?;
            let values = prepared.values();
            let result = environment
                .call_method(receiver.as_obj(), method_name, signature, &values)
                .map_err(|error| jni_error(environment, error))?;
            self.capture_value(environment, result)
        })
    }

    pub fn call_method_resolved(
        &self,
        receiver: JavaObjectHandle,
        method_name: &str,
        arguments: &[JavaValue],
    ) -> Result<JavaValue, JavaInvocationError> {
        let class_name = {
            let objects = self.objects.borrow();
            objects.get(receiver)?.0.class_name.clone()
        };
        let callable = self.resolve_method(&class_name, method_name, false, arguments)?;
        self.call_method(receiver, method_name, &callable.descriptor, arguments)
    }

    pub fn get_static_field_resolved(
        &self,
        class_name: &str,
        field_name: &str,
    ) -> Result<JavaValue, JavaInvocationError> {
        self.process.with_attached(|environment| {
            let descriptor = field_descriptor(environment, class_name, field_name, true)?;
            let value = environment
                .get_static_field(binary_name(class_name), field_name, descriptor)
                .map_err(|error| jni_error(environment, error))?;
            self.capture_value(environment, value)
        })
    }

    pub fn get_field_resolved(
        &self,
        receiver: JavaObjectHandle,
        field_name: &str,
    ) -> Result<JavaValue, JavaInvocationError> {
        let receiver_ref = self.global_reference(receiver)?;
        let class_name = self.objects.borrow().get(receiver)?.0.class_name.clone();
        self.process.with_attached(|environment| {
            let descriptor = field_descriptor(environment, &class_name, field_name, false)?;
            let value = environment
                .get_field(receiver_ref.as_obj(), field_name, descriptor)
                .map_err(|error| jni_error(environment, error))?;
            self.capture_value(environment, value)
        })
    }

    pub fn set_field_resolved(
        &self,
        receiver: JavaObjectHandle,
        field_name: &str,
        value: &JavaValue,
    ) -> Result<(), JavaInvocationError> {
        let receiver_ref = self.global_reference(receiver)?;
        let class_name = self.objects.borrow().get(receiver)?.0.class_name.clone();
        self.process.with_attached(|environment| {
            let descriptor = field_descriptor(environment, &class_name, field_name, false)?;
            let prepared = self.prepare_arguments(environment, std::slice::from_ref(value))?;
            let values = prepared.values();
            environment
                .set_field(receiver_ref.as_obj(), field_name, descriptor, values[0])
                .map_err(|error| jni_error(environment, error))
        })
    }

    fn resolve_constructor(
        &self,
        class_name: &str,
        arguments: &[JavaValue],
    ) -> Result<ReflectedCallable, JavaInvocationError> {
        self.process.with_attached(|environment| {
            choose_callable(constructor_candidates(environment, class_name)?, arguments)
        })
    }

    fn resolve_method(
        &self,
        class_name: &str,
        method_name: &str,
        require_static: bool,
        arguments: &[JavaValue],
    ) -> Result<ReflectedCallable, JavaInvocationError> {
        self.process.with_attached(|environment| {
            choose_callable(
                method_candidates(environment, class_name, method_name, require_static)?,
                arguments,
            )
        })
    }

    pub fn release(&self, handle: JavaObjectHandle) -> Result<(), JavaInvocationError> {
        self.objects.borrow_mut().remove(handle)?;
        Ok(())
    }

    pub fn restart_session(&self) {
        self.objects.borrow_mut().restart();
    }

    pub(super) fn global_reference(
        &self,
        handle: JavaObjectHandle,
    ) -> Result<GlobalRef, JavaInvocationError> {
        Ok(self.objects.borrow().get(handle)?.1.clone())
    }
}

fn choose_callable(
    candidates: Vec<ReflectedCallable>,
    arguments: &[JavaValue],
) -> Result<ReflectedCallable, JavaInvocationError> {
    let argument_types = arguments.iter().map(java_argument_type).collect::<Vec<_>>();
    let declarations = candidates
        .iter()
        .map(|candidate| JavaCallableCandidate {
            identity: candidate.identity.clone(),
            parameters: candidate.parameters.clone(),
            varargs: candidate.varargs,
        })
        .collect::<Vec<_>>();
    let selected = select_overload(&declarations, &argument_types)?;
    Ok(candidates[selected.candidate_index].clone())
}

fn java_argument_type(value: &JavaValue) -> JavaArgumentType {
    match value {
        JavaValue::Null => JavaArgumentType::Null,
        JavaValue::Boolean(_) => JavaArgumentType::Boolean,
        JavaValue::Byte(_) => JavaArgumentType::Byte,
        JavaValue::Short(_) => JavaArgumentType::Short,
        JavaValue::Int(_) => JavaArgumentType::Int,
        JavaValue::Long(_) => JavaArgumentType::Long,
        JavaValue::Float(_) => JavaArgumentType::Float,
        JavaValue::Double(_) => JavaArgumentType::Double,
        JavaValue::Char(_) => JavaArgumentType::Char,
        JavaValue::String(_) => JavaArgumentType::String,
        JavaValue::Object { class_name, .. } => JavaArgumentType::Object(class_name.clone()),
        JavaValue::Array { component, .. } => {
            JavaArgumentType::Array(Box::new(parameter_as_argument(component)))
        }
    }
}

fn parameter_as_argument(parameter: &crate::JavaParameterType) -> JavaArgumentType {
    match parameter {
        crate::JavaParameterType::Boolean => JavaArgumentType::Boolean,
        crate::JavaParameterType::Byte => JavaArgumentType::Byte,
        crate::JavaParameterType::Short => JavaArgumentType::Short,
        crate::JavaParameterType::Int => JavaArgumentType::Int,
        crate::JavaParameterType::Long => JavaArgumentType::Long,
        crate::JavaParameterType::Float => JavaArgumentType::Float,
        crate::JavaParameterType::Double => JavaArgumentType::Double,
        crate::JavaParameterType::Char => JavaArgumentType::Char,
        crate::JavaParameterType::String => JavaArgumentType::String,
        crate::JavaParameterType::Object(name) => JavaArgumentType::Object(name.clone()),
        crate::JavaParameterType::Array(component) => {
            JavaArgumentType::Array(Box::new(parameter_as_argument(component)))
        }
    }
}
