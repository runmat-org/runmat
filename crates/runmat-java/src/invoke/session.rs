use super::{
    error::jni_error,
    reflection::{constructor_candidates, field_descriptor, method_candidates, ReflectedCallable},
    JavaInvocationError, JavaValue,
};
use crate::{
    candidate_conversion_cost, JavaArgumentType, JavaCallableCandidate, JavaObjectHandle,
    JavaObjectRegistry, JvmProcess, SessionClasspath,
};
use jni::objects::GlobalRef;
use std::cell::RefCell;

pub struct JavaSession {
    pub(super) process: JvmProcess,
    pub(super) objects: RefCell<JavaObjectRegistry<GlobalRef>>,
    pub(super) classpath: RefCell<SessionClasspath>,
    pub(super) class_loader: RefCell<Option<GlobalRef>>,
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
        Self::with_classpath(process, SessionClasspath::default())
    }

    pub fn with_classpath(process: JvmProcess, classpath: SessionClasspath) -> Self {
        Self {
            process,
            objects: RefCell::new(JavaObjectRegistry::default()),
            classpath: RefCell::new(classpath),
            class_loader: RefCell::new(None),
        }
    }

    pub fn installation(&self) -> &crate::JvmInstallation {
        self.process.installation()
    }

    pub fn construct(
        &self,
        class_name: &str,
        signature: &str,
        arguments: &[JavaValue],
    ) -> Result<JavaValue, JavaInvocationError> {
        self.process.with_attached(|environment| {
            let class = self.load_class(environment, class_name)?;
            let prepared = self.prepare_arguments(environment, arguments)?;
            let values = prepared.values();
            let object = environment
                .new_object(class, signature, &values)
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
            let class = self.load_class(environment, class_name)?;
            let prepared = self.prepare_arguments(environment, arguments)?;
            let values = prepared.values();
            let result = environment
                .call_static_method(class, method_name, signature, &values)
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
        let receiver_ref = self.global_reference(receiver)?;
        let callable = self.process.with_attached(|environment| {
            let class = environment
                .get_object_class(receiver_ref.as_obj())
                .map_err(|error| jni_error(environment, error))?;
            let candidates =
                method_candidates(environment, &class, &class_name, method_name, false)?;
            choose_callable(self, environment, candidates, arguments)
        })?;
        self.call_method(receiver, method_name, &callable.descriptor, arguments)
    }

    pub fn get_static_field_resolved(
        &self,
        class_name: &str,
        field_name: &str,
    ) -> Result<JavaValue, JavaInvocationError> {
        self.process.with_attached(|environment| {
            let class = self.load_class(environment, class_name)?;
            let descriptor = field_descriptor(environment, &class, class_name, field_name, true)?;
            let value = environment
                .get_static_field(class, field_name, descriptor)
                .map_err(|error| jni_error(environment, error))?;
            self.capture_value(environment, value)
        })
    }

    pub fn new_object_array(
        &self,
        class_name: &str,
        dimensions: &[usize],
    ) -> Result<JavaValue, JavaInvocationError> {
        if dimensions.is_empty() {
            return Err(JavaInvocationError::UnsupportedValue(
                "Java array requires at least one dimension".into(),
            ));
        }
        let dimensions = dimensions
            .iter()
            .map(|dimension| {
                i32::try_from(*dimension).map_err(|_| {
                    JavaInvocationError::UnsupportedValue(
                        "Java array dimension exceeds the signed 32-bit limit".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.process.with_attached(|environment| {
            let class = self.load_class(environment, class_name)?;
            let dimension_array = environment
                .new_int_array(i32::try_from(dimensions.len()).expect("dimension count fits i32"))
                .map_err(|error| jni_error(environment, error))?;
            environment
                .set_int_array_region(&dimension_array, 0, &dimensions)
                .map_err(|error| jni_error(environment, error))?;
            let class_object = jni::objects::JObject::from(class);
            let dimensions_object = jni::objects::JObject::from(dimension_array);
            let array = environment
                .call_static_method(
                    "java/lang/reflect/Array",
                    "newInstance",
                    "(Ljava/lang/Class;[I)Ljava/lang/Object;",
                    &[
                        jni::objects::JValue::Object(&class_object),
                        jni::objects::JValue::Object(&dimensions_object),
                    ],
                )
                .and_then(jni::objects::JValueOwned::l)
                .map_err(|error| jni_error(environment, error))?;
            let array_class = environment
                .get_object_class(&array)
                .map_err(|error| jni_error(environment, error))?;
            let class_name = environment
                .call_method(array_class, "getName", "()Ljava/lang/String;", &[])
                .and_then(jni::objects::JValueOwned::l)
                .map_err(|error| jni_error(environment, error))?;
            let class_name = jni::objects::JString::from(class_name);
            let class_name = environment
                .get_string(&class_name)
                .map_err(|error| jni_error(environment, error))?
                .into();
            self.capture_reference(environment, array, class_name)
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
            let class = environment
                .get_object_class(receiver_ref.as_obj())
                .map_err(|error| jni_error(environment, error))?;
            let descriptor = field_descriptor(environment, &class, &class_name, field_name, false)?;
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
            let class = environment
                .get_object_class(receiver_ref.as_obj())
                .map_err(|error| jni_error(environment, error))?;
            let descriptor = field_descriptor(environment, &class, &class_name, field_name, false)?;
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
            let class = self.load_class(environment, class_name)?;
            let candidates = constructor_candidates(environment, &class, class_name)?;
            choose_callable(self, environment, candidates, arguments)
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
            let class = self.load_class(environment, class_name)?;
            let candidates =
                method_candidates(environment, &class, class_name, method_name, require_static)?;
            choose_callable(self, environment, candidates, arguments)
        })
    }

    pub fn release(&self, handle: JavaObjectHandle) -> Result<(), JavaInvocationError> {
        self.process.with_attached(|_| {
            self.objects.borrow_mut().remove(handle)?;
            Ok(())
        })
    }

    pub fn restart_session(&self) -> Result<(), JavaInvocationError> {
        self.process.with_attached(|_| {
            self.objects.borrow_mut().restart();
            self.class_loader.borrow_mut().take();
            Ok(())
        })
    }

    pub(super) fn global_reference(
        &self,
        handle: JavaObjectHandle,
    ) -> Result<GlobalRef, JavaInvocationError> {
        Ok(self.objects.borrow().get(handle)?.1.clone())
    }
}

impl Drop for JavaSession {
    fn drop(&mut self) {
        let _ = self.process.with_attached::<_, crate::JvmError>(|_| {
            self.objects.get_mut().restart();
            self.class_loader.get_mut().take();
            Ok(())
        });
    }
}

fn choose_callable(
    session: &JavaSession,
    environment: &mut jni::JNIEnv<'_>,
    mut candidates: Vec<ReflectedCallable>,
    arguments: &[JavaValue],
) -> Result<ReflectedCallable, JavaInvocationError> {
    let argument_types = arguments.iter().map(java_argument_type).collect::<Vec<_>>();
    let mut feasible = Vec::new();
    for (index, candidate) in candidates.iter_mut().enumerate() {
        let penalty =
            specialize_assignable_parameters(session, environment, candidate, &argument_types)?;
        let declaration = JavaCallableCandidate {
            identity: candidate.identity.clone(),
            parameters: candidate.parameters.clone(),
            varargs: candidate.varargs,
        };
        if let Some(cost) = candidate_conversion_cost(&declaration, &argument_types) {
            feasible.push((index, cost + penalty));
        }
    }
    let Some(best_cost) = feasible.iter().map(|(_, cost)| *cost).min() else {
        return Err(crate::OverloadError::NoMatch.into());
    };
    let tied = feasible
        .into_iter()
        .filter_map(|(index, cost)| (cost == best_cost).then_some(index))
        .collect::<Vec<_>>();
    if tied.len() == 1 {
        return Ok(candidates[tied[0]].clone());
    }
    let mut dominant = Vec::new();
    for &candidate in &tied {
        let mut dominates = true;
        for &other in &tied {
            if candidate != other
                && !callable_more_specific(
                    session,
                    environment,
                    &candidates[candidate],
                    &candidates[other],
                )?
            {
                dominates = false;
                break;
            }
        }
        if dominates {
            dominant.push(candidate);
        }
    }
    if dominant.len() == 1 {
        return Ok(candidates[dominant[0]].clone());
    }
    Err(crate::OverloadError::Ambiguous(
        tied.into_iter()
            .map(|index| candidates[index].identity.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    )
    .into())
}

fn specialize_assignable_parameters(
    session: &JavaSession,
    environment: &mut jni::JNIEnv<'_>,
    candidate: &mut ReflectedCallable,
    arguments: &[JavaArgumentType],
) -> Result<u32, JavaInvocationError> {
    let mut penalty = 0;
    let fixed_count = if candidate.varargs {
        candidate.parameters.len().saturating_sub(1)
    } else {
        candidate.parameters.len()
    };
    for (argument, parameter) in arguments
        .iter()
        .zip(candidate.parameters.iter_mut())
        .take(fixed_count)
    {
        penalty += specialize_assignable_parameter(session, environment, argument, parameter)?;
    }
    if candidate.varargs {
        let Some(crate::JavaParameterType::Array(component)) = candidate.parameters.last_mut()
        else {
            return Ok(penalty);
        };
        for argument in &arguments[fixed_count..] {
            penalty += specialize_assignable_parameter(session, environment, argument, component)?;
        }
    }
    Ok(penalty)
}

fn specialize_assignable_parameter(
    session: &JavaSession,
    environment: &mut jni::JNIEnv<'_>,
    argument: &JavaArgumentType,
    parameter: &mut crate::JavaParameterType,
) -> Result<u32, JavaInvocationError> {
    let actual = match argument {
        JavaArgumentType::String => Some("java.lang.String"),
        JavaArgumentType::Object(name) => Some(name.as_str()),
        _ => None,
    };
    let Some(actual) = actual else {
        return Ok(0);
    };
    let crate::JavaParameterType::Object(expected) = parameter else {
        return Ok(0);
    };
    if actual == expected || expected == "java.lang.Object" {
        return Ok(0);
    }
    let expected_class = session.load_class(environment, expected)?;
    let actual_class = session.load_class(environment, actual)?;
    if environment
        .is_assignable_from(&actual_class, &expected_class)
        .map_err(|error| super::error::jni_error(environment, error))?
    {
        *parameter = match argument {
            JavaArgumentType::String => crate::JavaParameterType::String,
            JavaArgumentType::Object(name) => crate::JavaParameterType::Object(name.clone()),
            _ => unreachable!(),
        };
        return Ok(1);
    }
    Ok(0)
}

fn callable_more_specific(
    session: &JavaSession,
    environment: &mut jni::JNIEnv<'_>,
    candidate: &ReflectedCallable,
    other: &ReflectedCallable,
) -> Result<bool, JavaInvocationError> {
    if candidate.parameters.len() != other.parameters.len() {
        return Ok(false);
    }
    let mut strict = false;
    for (candidate, other) in candidate.parameters.iter().zip(&other.parameters) {
        if candidate == other {
            continue;
        }
        let Some(candidate_name) = reference_parameter_name(candidate) else {
            return Ok(false);
        };
        let Some(other_name) = reference_parameter_name(other) else {
            return Ok(false);
        };
        let candidate_class = session.load_class(environment, candidate_name)?;
        let other_class = session.load_class(environment, other_name)?;
        if !environment
            .is_assignable_from(&candidate_class, &other_class)
            .map_err(|error| super::error::jni_error(environment, error))?
        {
            return Ok(false);
        }
        strict = true;
    }
    Ok(strict)
}

fn reference_parameter_name(parameter: &crate::JavaParameterType) -> Option<&str> {
    match parameter {
        crate::JavaParameterType::String => Some("java.lang.String"),
        crate::JavaParameterType::Object(name) => Some(name),
        _ => None,
    }
}

fn java_argument_type(value: &JavaValue) -> JavaArgumentType {
    match value {
        JavaValue::Null => JavaArgumentType::Null,
        JavaValue::Boolean(_) => JavaArgumentType::Boolean,
        JavaValue::Byte(_) => JavaArgumentType::Byte,
        JavaValue::Short(_) => JavaArgumentType::Short,
        JavaValue::Int(_) => JavaArgumentType::Int,
        JavaValue::Long(_) => JavaArgumentType::Long,
        JavaValue::UnsignedLong(_) => JavaArgumentType::Object("java.math.BigInteger".into()),
        JavaValue::Float(_) => JavaArgumentType::Float,
        JavaValue::Double(_) => JavaArgumentType::Double,
        JavaValue::Char(_) => JavaArgumentType::Char,
        JavaValue::String(_) => JavaArgumentType::String,
        JavaValue::Object { class_name, .. } => class_name_argument_type(class_name),
        JavaValue::Array { component, .. } => {
            JavaArgumentType::Array(Box::new(parameter_as_argument(component)))
        }
    }
}

fn class_name_argument_type(class_name: &str) -> JavaArgumentType {
    let Some(component) = class_name.strip_prefix('[') else {
        return JavaArgumentType::Object(class_name.to_string());
    };
    let component = match component {
        "Z" => JavaArgumentType::Boolean,
        "B" => JavaArgumentType::Byte,
        "S" => JavaArgumentType::Short,
        "I" => JavaArgumentType::Int,
        "J" => JavaArgumentType::Long,
        "F" => JavaArgumentType::Float,
        "D" => JavaArgumentType::Double,
        "C" => JavaArgumentType::Char,
        value if value.starts_with('[') => class_name_argument_type(value),
        value if value.starts_with('L') && value.ends_with(';') => {
            JavaArgumentType::Object(value[1..value.len() - 1].replace('/', "."))
        }
        _ => JavaArgumentType::Object("java.lang.Object".into()),
    };
    JavaArgumentType::Array(Box::new(component))
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
