use super::{error::jni_error, JavaInvocationError, JavaSession, JavaValue};
use crate::{JavaObjectHandle, JavaObjectMetadata};
use jni::objects::{
    JBooleanArray, JByteArray, JCharArray, JDoubleArray, JFloatArray, JIntArray, JLongArray,
    JObject, JObjectArray, JShortArray, JString, JValue, JValueOwned,
};

impl JavaSession {
    pub(super) fn prepare_arguments<'local>(
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
                JavaValue::UnsignedLong(value) => {
                    let text = environment
                        .new_string(value.to_string())
                        .map_err(|error| jni_error(environment, error))?;
                    let text = JObject::from(text);
                    let integer = environment
                        .new_object(
                            "java/math/BigInteger",
                            "(Ljava/lang/String;)V",
                            &[JValue::Object(&text)],
                        )
                        .map_err(|error| jni_error(environment, error))?;
                    objects.push(integer);
                    PreparedValue::Object(objects.len() - 1)
                }
                JavaValue::Float(value) => PreparedValue::Float(*value),
                JavaValue::Double(value) => PreparedValue::Double(*value),
                JavaValue::Char(value) => PreparedValue::Char(*value),
                JavaValue::String(value) => {
                    let string = environment
                        .new_string(value)
                        .map_err(|error| jni_error(environment, error))?;
                    objects.push(JObject::from(string));
                    PreparedValue::Object(objects.len() - 1)
                }
                JavaValue::Object { handle, .. } => {
                    let reference = self.global_reference(*handle)?;
                    objects.push(
                        environment
                            .new_local_ref(reference.as_obj())
                            .map_err(|error| jni_error(environment, error))?,
                    );
                    PreparedValue::Object(objects.len() - 1)
                }
                JavaValue::Array {
                    component,
                    elements,
                } => {
                    objects.push(self.create_array(environment, component, elements)?);
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

    pub(super) fn capture_value(
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

    pub(super) fn capture_object(
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
                environment
                    .get_string(&string)
                    .map_err(|error| jni_error(environment, error))?
                    .into(),
            ));
        }
        if let Some(value) = capture_boxed(environment, &object, &class_name)? {
            return Ok(value);
        }
        if class_name == "java.math.BigInteger" {
            let text = environment
                .call_method(&object, "toString", "()Ljava/lang/String;", &[])
                .and_then(JValueOwned::l)
                .map_err(|error| jni_error(environment, error))?;
            let text = JString::from(text);
            let text: String = environment
                .get_string(&text)
                .map_err(|error| jni_error(environment, error))?
                .into();
            if let Ok(value) = text.parse::<u64>() {
                return Ok(JavaValue::UnsignedLong(value));
            }
        }
        if class_name.starts_with('[') {
            return self.capture_array(environment, object, &class_name);
        }
        self.capture_reference(environment, object, class_name)
    }

    pub(super) fn capture_reference(
        &self,
        environment: &mut jni::JNIEnv<'_>,
        object: JObject<'_>,
        class_name: String,
    ) -> Result<JavaValue, JavaInvocationError> {
        let objects = self.objects.borrow();
        for (handle, metadata, reference) in objects.iter() {
            if environment
                .is_same_object(&object, reference.as_obj())
                .map_err(|error| jni_error(environment, error))?
            {
                return Ok(JavaValue::Object {
                    handle,
                    class_name: metadata.class_name.clone(),
                });
            }
        }
        drop(objects);
        let reference = environment
            .new_global_ref(object)
            .map_err(|error| jni_error(environment, error))?;
        let handle = self.objects.borrow_mut().insert(
            JavaObjectMetadata {
                class_name: class_name.clone(),
            },
            reference,
        )?;
        Ok(JavaValue::Object { handle, class_name })
    }

    fn create_array<'local>(
        &self,
        environment: &mut jni::JNIEnv<'local>,
        component: &crate::JavaParameterType,
        elements: &[JavaValue],
    ) -> Result<JObject<'local>, JavaInvocationError> {
        let length = i32::try_from(elements.len()).map_err(|_| {
            JavaInvocationError::UnsupportedValue("Java array length exceeds i32".into())
        })?;
        macro_rules! primitive_array {
            ($new:ident, $set:ident, $array:ty, $variant:ident, $convert:expr) => {{
                let values = elements
                    .iter()
                    .map(|value| match value {
                        JavaValue::$variant(value) => Ok($convert(*value)),
                        _ => Err(JavaInvocationError::UnsupportedValue(
                            "Java array elements do not match their component type".into(),
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let array: $array = environment
                    .$new(length)
                    .map_err(|error| jni_error(environment, error))?;
                environment
                    .$set(&array, 0, &values)
                    .map_err(|error| jni_error(environment, error))?;
                Ok(JObject::from(array))
            }};
        }
        match component {
            crate::JavaParameterType::Boolean => primitive_array!(
                new_boolean_array,
                set_boolean_array_region,
                JBooleanArray,
                Boolean,
                |value: bool| u8::from(value)
            ),
            crate::JavaParameterType::Byte => primitive_array!(
                new_byte_array,
                set_byte_array_region,
                JByteArray,
                Byte,
                |value: i8| value
            ),
            crate::JavaParameterType::Short => primitive_array!(
                new_short_array,
                set_short_array_region,
                JShortArray,
                Short,
                |value: i16| value
            ),
            crate::JavaParameterType::Int => primitive_array!(
                new_int_array,
                set_int_array_region,
                JIntArray,
                Int,
                |value: i32| value
            ),
            crate::JavaParameterType::Long => primitive_array!(
                new_long_array,
                set_long_array_region,
                JLongArray,
                Long,
                |value: i64| value
            ),
            crate::JavaParameterType::Float => primitive_array!(
                new_float_array,
                set_float_array_region,
                JFloatArray,
                Float,
                |value: f32| value
            ),
            crate::JavaParameterType::Double => primitive_array!(
                new_double_array,
                set_double_array_region,
                JDoubleArray,
                Double,
                |value: f64| value
            ),
            crate::JavaParameterType::Char => primitive_array!(
                new_char_array,
                set_char_array_region,
                JCharArray,
                Char,
                |value: u16| value
            ),
            crate::JavaParameterType::String | crate::JavaParameterType::Object(_) => {
                let class_name = match component {
                    crate::JavaParameterType::String => "java.lang.String",
                    crate::JavaParameterType::Object(name) => name,
                    _ => unreachable!(),
                };
                let array = environment
                    .new_object_array(length, binary_name(class_name), JObject::null())
                    .map_err(|error| jni_error(environment, error))?;
                for (index, element) in elements.iter().enumerate() {
                    let object = self.value_as_object(environment, element)?;
                    environment
                        .set_object_array_element(&array, index as i32, object)
                        .map_err(|error| jni_error(environment, error))?;
                }
                Ok(JObject::from(array))
            }
            crate::JavaParameterType::Array(_) => Err(JavaInvocationError::UnsupportedValue(
                "nested Java array construction is not yet supported".into(),
            )),
        }
    }

    pub fn collection_elements(
        &self,
        handle: JavaObjectHandle,
    ) -> Result<Vec<JavaValue>, JavaInvocationError> {
        let reference = self.global_reference(handle)?;
        self.process.with_attached(|environment| {
            if !environment
                .is_instance_of(reference.as_obj(), "java/util/Collection")
                .map_err(|error| jni_error(environment, error))?
            {
                return Err(JavaInvocationError::UnsupportedValue(
                    "Java object does not implement java.util.Collection".into(),
                ));
            }
            let values = environment
                .call_method(reference.as_obj(), "toArray", "()[Ljava/lang/Object;", &[])
                .and_then(JValueOwned::l)
                .map_err(|error| jni_error(environment, error))?;
            let JavaValue::Array { elements, .. } =
                self.capture_array(environment, values, "[Ljava.lang.Object;")?
            else {
                unreachable!("object-array capture always returns JavaValue::Array")
            };
            Ok(elements)
        })
    }

    fn value_as_object<'local>(
        &self,
        environment: &mut jni::JNIEnv<'local>,
        value: &JavaValue,
    ) -> Result<JObject<'local>, JavaInvocationError> {
        match value {
            JavaValue::Null => Ok(JObject::null()),
            JavaValue::String(value) => Ok(JObject::from(
                environment
                    .new_string(value)
                    .map_err(|error| jni_error(environment, error))?,
            )),
            JavaValue::Object { handle, .. } => environment
                .new_local_ref(self.global_reference(*handle)?.as_obj())
                .map_err(|error| jni_error(environment, error)),
            _ => Err(JavaInvocationError::UnsupportedValue(
                "object arrays currently accept strings, objects, and null".into(),
            )),
        }
    }

    fn capture_array(
        &self,
        environment: &mut jni::JNIEnv<'_>,
        object: JObject<'_>,
        class_name: &str,
    ) -> Result<JavaValue, JavaInvocationError> {
        macro_rules! capture_primitive {
            ($array:ty, $get:ident, $wrap:expr, $component:expr, $default:expr) => {{
                let array = <$array>::from(object);
                let length = environment
                    .get_array_length(&array)
                    .map_err(|error| jni_error(environment, error))? as usize;
                let mut values = vec![$default; length];
                environment
                    .$get(&array, 0, &mut values)
                    .map_err(|error| jni_error(environment, error))?;
                Ok(JavaValue::Array {
                    component: $component,
                    elements: values.into_iter().map($wrap).collect(),
                })
            }};
        }
        match class_name {
            "[Z" => capture_primitive!(
                JBooleanArray,
                get_boolean_array_region,
                |value| JavaValue::Boolean(value != 0),
                crate::JavaParameterType::Boolean,
                0u8
            ),
            "[B" => capture_primitive!(
                JByteArray,
                get_byte_array_region,
                JavaValue::Byte,
                crate::JavaParameterType::Byte,
                0i8
            ),
            "[S" => capture_primitive!(
                JShortArray,
                get_short_array_region,
                JavaValue::Short,
                crate::JavaParameterType::Short,
                0i16
            ),
            "[I" => capture_primitive!(
                JIntArray,
                get_int_array_region,
                JavaValue::Int,
                crate::JavaParameterType::Int,
                0i32
            ),
            "[J" => capture_primitive!(
                JLongArray,
                get_long_array_region,
                JavaValue::Long,
                crate::JavaParameterType::Long,
                0i64
            ),
            "[F" => capture_primitive!(
                JFloatArray,
                get_float_array_region,
                JavaValue::Float,
                crate::JavaParameterType::Float,
                0.0f32
            ),
            "[D" => capture_primitive!(
                JDoubleArray,
                get_double_array_region,
                JavaValue::Double,
                crate::JavaParameterType::Double,
                0.0f64
            ),
            "[C" => capture_primitive!(
                JCharArray,
                get_char_array_region,
                JavaValue::Char,
                crate::JavaParameterType::Char,
                0u16
            ),
            _ => {
                let array = JObjectArray::from(object);
                let length = environment
                    .get_array_length(&array)
                    .map_err(|error| jni_error(environment, error))?;
                let mut elements = Vec::with_capacity(length as usize);
                for index in 0..length {
                    let element = environment
                        .get_object_array_element(&array, index)
                        .map_err(|error| jni_error(environment, error))?;
                    elements.push(self.capture_object(environment, element)?);
                }
                Ok(JavaValue::Array {
                    component: crate::JavaParameterType::Object("java.lang.Object".into()),
                    elements,
                })
            }
        }
    }
}

pub(super) fn binary_name(class_name: &str) -> String {
    class_name.replace('.', "/")
}

fn object_class_name(
    environment: &mut jni::JNIEnv<'_>,
    object: &JObject<'_>,
) -> Result<String, JavaInvocationError> {
    let class = environment
        .call_method(object, "getClass", "()Ljava/lang/Class;", &[])
        .and_then(JValueOwned::l)
        .map_err(|error| jni_error(environment, error))?;
    let name = environment
        .call_method(class, "getName", "()Ljava/lang/String;", &[])
        .and_then(JValueOwned::l)
        .map_err(|error| jni_error(environment, error))?;
    let name = JString::from(name);
    let value = environment
        .get_string(&name)
        .map_err(|error| jni_error(environment, error))?
        .into();
    Ok(value)
}

fn capture_boxed(
    environment: &mut jni::JNIEnv<'_>,
    object: &JObject<'_>,
    class_name: &str,
) -> Result<Option<JavaValue>, JavaInvocationError> {
    let (method, descriptor) = match class_name {
        "java.lang.Boolean" => ("booleanValue", "()Z"),
        "java.lang.Byte" => ("byteValue", "()B"),
        "java.lang.Short" => ("shortValue", "()S"),
        "java.lang.Integer" => ("intValue", "()I"),
        "java.lang.Long" => ("longValue", "()J"),
        "java.lang.Float" => ("floatValue", "()F"),
        "java.lang.Double" => ("doubleValue", "()D"),
        "java.lang.Character" => ("charValue", "()C"),
        _ => return Ok(None),
    };
    let value = environment
        .call_method(object, method, descriptor, &[])
        .map_err(|error| jni_error(environment, error))?;
    Ok(Some(match value {
        JValueOwned::Bool(value) => JavaValue::Boolean(value != 0),
        JValueOwned::Byte(value) => JavaValue::Byte(value),
        JValueOwned::Short(value) => JavaValue::Short(value),
        JValueOwned::Int(value) => JavaValue::Int(value),
        JValueOwned::Long(value) => JavaValue::Long(value),
        JValueOwned::Float(value) => JavaValue::Float(value),
        JValueOwned::Double(value) => JavaValue::Double(value),
        JValueOwned::Char(value) => JavaValue::Char(value),
        _ => {
            return Err(JavaInvocationError::UnsupportedValue(format!(
                "boxed {class_name} returned an unexpected value"
            )))
        }
    }))
}

pub(super) struct PreparedArguments<'local> {
    prepared: Vec<PreparedValue>,
    objects: Vec<JObject<'local>>,
    null: JObject<'local>,
}

impl PreparedArguments<'_> {
    pub(super) fn values(&self) -> Vec<JValue<'_, '_>> {
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
