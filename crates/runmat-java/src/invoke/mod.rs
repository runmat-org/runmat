use std::cell::RefCell;

use jni::objects::{GlobalRef, JObject, JString, JValue, JValueOwned};

use crate::{
    JavaObjectHandle, JavaObjectMetadata, JavaObjectRegistry, JvmError, JvmProcess,
    ObjectRegistryError,
};

#[derive(Debug, Clone, PartialEq)]
pub enum JavaValue {
    Null,
    Boolean(bool),
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Char(u16),
    String(String),
    Object {
        handle: JavaObjectHandle,
        class_name: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum JavaInvocationError {
    #[error(transparent)]
    Jvm(#[from] JvmError),
    #[error(transparent)]
    Object(#[from] ObjectRegistryError),
    #[error("Java invocation failed: {0}")]
    Jni(String),
    #[error("Java invocation returned an unsupported value: {0}")]
    UnsupportedValue(String),
}

pub struct JavaSession {
    process: JvmProcess,
    objects: RefCell<JavaObjectRegistry<GlobalRef>>,
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
                .map_err(jni_error)?;
            self.capture_object(environment, object)
        })
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
                .map_err(jni_error)?;
            self.capture_value(environment, result)
        })
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
                .map_err(jni_error)?;
            self.capture_value(environment, result)
        })
    }

    pub fn release(&self, handle: JavaObjectHandle) -> Result<(), JavaInvocationError> {
        self.objects.borrow_mut().remove(handle)?;
        Ok(())
    }

    pub fn restart_session(&self) {
        self.objects.borrow_mut().restart();
    }

    fn global_reference(&self, handle: JavaObjectHandle) -> Result<GlobalRef, JavaInvocationError> {
        Ok(self.objects.borrow().get(handle)?.1.clone())
    }

    fn prepare_arguments<'local>(
        &self,
        environment: &mut jni::JNIEnv<'local>,
        arguments: &[JavaValue],
    ) -> Result<PreparedArguments<'local>, JavaInvocationError> {
        let mut prepared = Vec::with_capacity(arguments.len());
        let mut objects = Vec::new();
        for argument in arguments {
            let value = match argument {
                JavaValue::Null => PreparedValue::Null,
                JavaValue::Boolean(value) => PreparedValue::Boolean(*value),
                JavaValue::Byte(value) => PreparedValue::Byte(*value),
                JavaValue::Short(value) => PreparedValue::Short(*value),
                JavaValue::Int(value) => PreparedValue::Int(*value),
                JavaValue::Long(value) => PreparedValue::Long(*value),
                JavaValue::Float(value) => PreparedValue::Float(*value),
                JavaValue::Double(value) => PreparedValue::Double(*value),
                JavaValue::Char(value) => PreparedValue::Char(*value),
                JavaValue::String(value) => {
                    let string = environment.new_string(value).map_err(jni_error)?;
                    objects.push(JObject::from(string));
                    PreparedValue::Object(objects.len() - 1)
                }
                JavaValue::Object { handle, .. } => {
                    let reference = self.global_reference(*handle)?;
                    objects.push(
                        environment
                            .new_local_ref(reference.as_obj())
                            .map_err(jni_error)?,
                    );
                    PreparedValue::Object(objects.len() - 1)
                }
            };
            prepared.push(value);
        }
        Ok(PreparedArguments {
            prepared,
            objects,
            null: JObject::null(),
        })
    }

    fn capture_value(
        &self,
        environment: &mut jni::JNIEnv<'_>,
        value: JValueOwned<'_>,
    ) -> Result<JavaValue, JavaInvocationError> {
        match value {
            JValueOwned::Void => Ok(JavaValue::Null),
            JValueOwned::Bool(value) => Ok(JavaValue::Boolean(value != 0)),
            JValueOwned::Byte(value) => Ok(JavaValue::Byte(value)),
            JValueOwned::Char(value) => Ok(JavaValue::Char(value)),
            JValueOwned::Short(value) => Ok(JavaValue::Short(value)),
            JValueOwned::Int(value) => Ok(JavaValue::Int(value)),
            JValueOwned::Long(value) => Ok(JavaValue::Long(value)),
            JValueOwned::Float(value) => Ok(JavaValue::Float(value)),
            JValueOwned::Double(value) => Ok(JavaValue::Double(value)),
            JValueOwned::Object(object) => self.capture_object(environment, object),
        }
    }

    fn capture_object(
        &self,
        environment: &mut jni::JNIEnv<'_>,
        object: JObject<'_>,
    ) -> Result<JavaValue, JavaInvocationError> {
        if object.is_null() {
            return Ok(JavaValue::Null);
        }
        let class_name = object_class_name(environment, &object)?;
        if class_name == "java.lang.String" {
            let string = JString::from(object);
            return Ok(JavaValue::String(
                environment.get_string(&string).map_err(jni_error)?.into(),
            ));
        }
        {
            let objects = self.objects.borrow();
            for (handle, metadata, reference) in objects.iter() {
                if environment
                    .is_same_object(&object, reference.as_obj())
                    .map_err(jni_error)?
                {
                    return Ok(JavaValue::Object {
                        handle,
                        class_name: metadata.class_name.clone(),
                    });
                }
            }
        }
        let reference = environment.new_global_ref(object).map_err(jni_error)?;
        let handle = self.objects.borrow_mut().insert(
            JavaObjectMetadata {
                class_name: class_name.clone(),
            },
            reference,
        )?;
        Ok(JavaValue::Object { handle, class_name })
    }
}

fn binary_name(class_name: &str) -> String {
    class_name.replace('.', "/")
}

fn object_class_name(
    environment: &mut jni::JNIEnv<'_>,
    object: &JObject<'_>,
) -> Result<String, JavaInvocationError> {
    let class = environment
        .call_method(object, "getClass", "()Ljava/lang/Class;", &[])
        .and_then(JValueOwned::l)
        .map_err(jni_error)?;
    let name = environment
        .call_method(class, "getName", "()Ljava/lang/String;", &[])
        .and_then(JValueOwned::l)
        .map_err(jni_error)?;
    let name = JString::from(name);
    let value = environment.get_string(&name).map_err(jni_error)?.into();
    Ok(value)
}

fn jni_error(error: jni::errors::Error) -> JavaInvocationError {
    JavaInvocationError::Jni(error.to_string())
}

struct PreparedArguments<'local> {
    prepared: Vec<PreparedValue>,
    objects: Vec<JObject<'local>>,
    null: JObject<'local>,
}

impl PreparedArguments<'_> {
    fn values(&self) -> Vec<JValue<'_, '_>> {
        self.prepared
            .iter()
            .map(|value| match value {
                PreparedValue::Null => JValue::Object(&self.null),
                PreparedValue::Boolean(value) => JValue::Bool(u8::from(*value)),
                PreparedValue::Byte(value) => JValue::Byte(*value),
                PreparedValue::Short(value) => JValue::Short(*value),
                PreparedValue::Int(value) => JValue::Int(*value),
                PreparedValue::Long(value) => JValue::Long(*value),
                PreparedValue::Float(value) => JValue::Float(*value),
                PreparedValue::Double(value) => JValue::Double(*value),
                PreparedValue::Char(value) => JValue::Char(*value),
                PreparedValue::Object(index) => JValue::Object(&self.objects[*index]),
            })
            .collect()
    }
}

enum PreparedValue {
    Null,
    Boolean(bool),
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Char(u16),
    Object(usize),
}
