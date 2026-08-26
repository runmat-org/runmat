use jni::objects::{JClass, JObject, JObjectArray, JString, JValue, JValueOwned};

use crate::JavaParameterType;

use super::{conversion::binary_name, error::jni_error, JavaInvocationError};

#[derive(Debug, Clone)]
pub(super) struct ReflectedCallable {
    pub identity: String,
    pub descriptor: String,
    pub parameters: Vec<JavaParameterType>,
    pub varargs: bool,
}

pub(super) fn field_descriptor(
    environment: &mut jni::JNIEnv<'_>,
    class: &JClass<'_>,
    class_name: &str,
    field_name: &str,
    require_static: bool,
) -> Result<String, JavaInvocationError> {
    let fields = environment
        .call_method(class, "getFields", "()[Ljava/lang/reflect/Field;", &[])
        .and_then(JValueOwned::l)
        .map_err(|error| jni_error(environment, error))?;
    let fields = JObjectArray::from(fields);
    let length = environment
        .get_array_length(&fields)
        .map_err(|error| jni_error(environment, error))?;
    for index in 0..length {
        let field = environment
            .get_object_array_element(&fields, index)
            .map_err(|error| jni_error(environment, error))?;
        if string_method(environment, &field, "getName")? != field_name {
            continue;
        }
        let modifiers = environment
            .call_method(&field, "getModifiers", "()I", &[])
            .and_then(JValueOwned::i)
            .map_err(|error| jni_error(environment, error))?;
        let is_static = environment
            .call_static_method(
                "java/lang/reflect/Modifier",
                "isStatic",
                "(I)Z",
                &[JValue::Int(modifiers)],
            )
            .and_then(JValueOwned::z)
            .map_err(|error| jni_error(environment, error))?;
        if is_static != require_static {
            continue;
        }
        let field_type = environment
            .call_method(&field, "getType", "()Ljava/lang/Class;", &[])
            .and_then(JValueOwned::l)
            .map_err(|error| jni_error(environment, error))?;
        return class_descriptor(environment, &field_type);
    }
    Err(JavaInvocationError::UnsupportedValue(format!(
        "public {}field {class_name}.{field_name} was not found",
        if require_static { "static " } else { "" }
    )))
}

pub(super) fn constructor_candidates(
    environment: &mut jni::JNIEnv<'_>,
    class: &JClass<'_>,
    class_name: &str,
) -> Result<Vec<ReflectedCallable>, JavaInvocationError> {
    let members = environment
        .call_method(
            class,
            "getConstructors",
            "()[Ljava/lang/reflect/Constructor;",
            &[],
        )
        .and_then(JValueOwned::l)
        .map_err(|error| jni_error(environment, error))?;
    reflected_callables(
        environment,
        JObjectArray::from(members),
        class_name,
        "<init>",
        false,
        true,
    )
}

pub(super) fn method_candidates(
    environment: &mut jni::JNIEnv<'_>,
    class: &JClass<'_>,
    class_name: &str,
    method_name: &str,
    require_static: bool,
) -> Result<Vec<ReflectedCallable>, JavaInvocationError> {
    let members = environment
        .call_method(class, "getMethods", "()[Ljava/lang/reflect/Method;", &[])
        .and_then(JValueOwned::l)
        .map_err(|error| jni_error(environment, error))?;
    reflected_callables(
        environment,
        JObjectArray::from(members),
        class_name,
        method_name,
        require_static,
        false,
    )
}

fn reflected_callables(
    environment: &mut jni::JNIEnv<'_>,
    members: JObjectArray<'_>,
    class_name: &str,
    requested_name: &str,
    require_static: bool,
    constructors: bool,
) -> Result<Vec<ReflectedCallable>, JavaInvocationError> {
    let length = environment
        .get_array_length(&members)
        .map_err(|error| jni_error(environment, error))?;
    let mut callables: Vec<(ReflectedCallable, bool)> = Vec::new();
    for index in 0..length {
        let member = environment
            .get_object_array_element(&members, index)
            .map_err(|error| jni_error(environment, error))?;
        let name = if constructors {
            requested_name.to_string()
        } else {
            string_method(environment, &member, "getName")?
        };
        if name != requested_name {
            continue;
        }
        let generated = if constructors {
            false
        } else {
            let bridge = environment
                .call_method(&member, "isBridge", "()Z", &[])
                .and_then(JValueOwned::z)
                .map_err(|error| jni_error(environment, error))?;
            let synthetic = environment
                .call_method(&member, "isSynthetic", "()Z", &[])
                .and_then(JValueOwned::z)
                .map_err(|error| jni_error(environment, error))?;
            let modifiers = environment
                .call_method(&member, "getModifiers", "()I", &[])
                .and_then(JValueOwned::i)
                .map_err(|error| jni_error(environment, error))?;
            let is_static = environment
                .call_static_method(
                    "java/lang/reflect/Modifier",
                    "isStatic",
                    "(I)Z",
                    &[JValue::Int(modifiers)],
                )
                .and_then(JValueOwned::z)
                .map_err(|error| jni_error(environment, error))?;
            if is_static != require_static {
                continue;
            }
            bridge || synthetic
        };
        let parameter_classes = environment
            .call_method(&member, "getParameterTypes", "()[Ljava/lang/Class;", &[])
            .and_then(JValueOwned::l)
            .map_err(|error| jni_error(environment, error))?;
        let parameter_classes = JObjectArray::from(parameter_classes);
        let parameter_count = environment
            .get_array_length(&parameter_classes)
            .map_err(|error| jni_error(environment, error))?;
        let mut parameters = Vec::new();
        let mut parameter_descriptors = Vec::new();
        for parameter_index in 0..parameter_count {
            let class = environment
                .get_object_array_element(&parameter_classes, parameter_index)
                .map_err(|error| jni_error(environment, error))?;
            let descriptor = class_descriptor(environment, &class)?;
            parameters.push(parameter_type(&descriptor)?);
            parameter_descriptors.push(descriptor);
        }
        let return_descriptor = if constructors {
            "V".to_string()
        } else {
            let class = environment
                .call_method(&member, "getReturnType", "()Ljava/lang/Class;", &[])
                .and_then(JValueOwned::l)
                .map_err(|error| jni_error(environment, error))?;
            class_descriptor(environment, &class)?
        };
        let varargs = environment
            .call_method(&member, "isVarArgs", "()Z", &[])
            .and_then(JValueOwned::z)
            .map_err(|error| jni_error(environment, error))?;
        let descriptor = format!("({}){return_descriptor}", parameter_descriptors.join(""));
        let callable = ReflectedCallable {
            identity: format!("{class_name}.{requested_name}{descriptor}"),
            descriptor,
            parameters,
            varargs,
        };
        if let Some((existing, existing_generated)) = callables.iter_mut().find(|(existing, _)| {
            existing.parameters == callable.parameters && existing.varargs == callable.varargs
        }) {
            if *existing_generated && !generated {
                *existing = callable;
                *existing_generated = false;
            }
        } else {
            callables.push((callable, generated));
        }
    }
    Ok(callables
        .into_iter()
        .map(|(callable, _generated)| callable)
        .collect())
}

fn class_descriptor(
    environment: &mut jni::JNIEnv<'_>,
    class: &JObject<'_>,
) -> Result<String, JavaInvocationError> {
    let name = string_method(environment, class, "getName")?;
    Ok(match name.as_str() {
        "void" => "V".into(),
        "boolean" => "Z".into(),
        "byte" => "B".into(),
        "short" => "S".into(),
        "int" => "I".into(),
        "long" => "J".into(),
        "float" => "F".into(),
        "double" => "D".into(),
        "char" => "C".into(),
        _ if name.starts_with('[') => name.replace('.', "/"),
        _ => format!("L{};", binary_name(&name)),
    })
}

fn parameter_type(descriptor: &str) -> Result<JavaParameterType, JavaInvocationError> {
    Ok(match descriptor {
        "Z" => JavaParameterType::Boolean,
        "B" => JavaParameterType::Byte,
        "S" => JavaParameterType::Short,
        "I" => JavaParameterType::Int,
        "J" => JavaParameterType::Long,
        "F" => JavaParameterType::Float,
        "D" => JavaParameterType::Double,
        "C" => JavaParameterType::Char,
        "Ljava/lang/String;" => JavaParameterType::String,
        descriptor if descriptor.starts_with('L') && descriptor.ends_with(';') => {
            JavaParameterType::Object(descriptor[1..descriptor.len() - 1].replace('/', "."))
        }
        descriptor if descriptor.starts_with('[') => {
            JavaParameterType::Array(Box::new(parameter_type(&descriptor[1..])?))
        }
        _ => {
            return Err(JavaInvocationError::UnsupportedValue(format!(
                "unsupported reflected descriptor {descriptor}"
            )))
        }
    })
}

fn string_method(
    environment: &mut jni::JNIEnv<'_>,
    object: &JObject<'_>,
    method: &str,
) -> Result<String, JavaInvocationError> {
    let value = environment
        .call_method(object, method, "()Ljava/lang/String;", &[])
        .and_then(JValueOwned::l)
        .map_err(|error| jni_error(environment, error))?;
    let value = JString::from(value);
    let value = environment
        .get_string(&value)
        .map_err(|error| jni_error(environment, error))?
        .into();
    Ok(value)
}
