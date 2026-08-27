use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use jni::objects::{JObject, JObjectArray, JString, JValue, JValueOwned};
use jni::{JNIEnv, NativeMethod};

use super::conversion::{capture_boxed, object_class_name};
use super::{
    error::jni_error, JavaCallbackInvocation, JavaInvocationError, JavaSession, JavaValue,
};

const BRIDGE_CLASS: &str = "org/runmat/bridge/RunMatCallbackHandler";
const INVOKE_SIGNATURE: &str =
    "(Ljava/lang/Object;Ljava/lang/reflect/Method;[Ljava/lang/Object;)Ljava/lang/Object;";

// Java 8 classfile for a final InvocationHandler with one callback-id field,
// one constructor, and one native invoke method. Keeping the bridge this small
// makes its audited JVM surface independent of a build-time JDK.
const BRIDGE_CLASSFILE: &[u8] = &[
    0xca, 0xfe, 0xba, 0xbe, 0x00, 0x00, 0x00, 0x34, 0x00, 0x13, 0x0a, 0x00, 0x02, 0x00, 0x03, 0x07,
    0x00, 0x04, 0x0c, 0x00, 0x05, 0x00, 0x06, 0x01, 0x00, 0x10, 0x6a, 0x61, 0x76, 0x61, 0x2f, 0x6c,
    0x61, 0x6e, 0x67, 0x2f, 0x4f, 0x62, 0x6a, 0x65, 0x63, 0x74, 0x01, 0x00, 0x06, 0x3c, 0x69, 0x6e,
    0x69, 0x74, 0x3e, 0x01, 0x00, 0x03, 0x28, 0x29, 0x56, 0x09, 0x00, 0x08, 0x00, 0x09, 0x07, 0x00,
    0x0a, 0x0c, 0x00, 0x0b, 0x00, 0x0c, 0x01, 0x00, 0x27, 0x6f, 0x72, 0x67, 0x2f, 0x72, 0x75, 0x6e,
    0x6d, 0x61, 0x74, 0x2f, 0x62, 0x72, 0x69, 0x64, 0x67, 0x65, 0x2f, 0x52, 0x75, 0x6e, 0x4d, 0x61,
    0x74, 0x43, 0x61, 0x6c, 0x6c, 0x62, 0x61, 0x63, 0x6b, 0x48, 0x61, 0x6e, 0x64, 0x6c, 0x65, 0x72,
    0x01, 0x00, 0x0a, 0x63, 0x61, 0x6c, 0x6c, 0x62, 0x61, 0x63, 0x6b, 0x49, 0x64, 0x01, 0x00, 0x01,
    0x4a, 0x07, 0x00, 0x0e, 0x01, 0x00, 0x23, 0x6a, 0x61, 0x76, 0x61, 0x2f, 0x6c, 0x61, 0x6e, 0x67,
    0x2f, 0x72, 0x65, 0x66, 0x6c, 0x65, 0x63, 0x74, 0x2f, 0x49, 0x6e, 0x76, 0x6f, 0x63, 0x61, 0x74,
    0x69, 0x6f, 0x6e, 0x48, 0x61, 0x6e, 0x64, 0x6c, 0x65, 0x72, 0x01, 0x00, 0x04, 0x28, 0x4a, 0x29,
    0x56, 0x01, 0x00, 0x04, 0x43, 0x6f, 0x64, 0x65, 0x01, 0x00, 0x06, 0x69, 0x6e, 0x76, 0x6f, 0x6b,
    0x65, 0x01, 0x00, 0x53, 0x28, 0x4c, 0x6a, 0x61, 0x76, 0x61, 0x2f, 0x6c, 0x61, 0x6e, 0x67, 0x2f,
    0x4f, 0x62, 0x6a, 0x65, 0x63, 0x74, 0x3b, 0x4c, 0x6a, 0x61, 0x76, 0x61, 0x2f, 0x6c, 0x61, 0x6e,
    0x67, 0x2f, 0x72, 0x65, 0x66, 0x6c, 0x65, 0x63, 0x74, 0x2f, 0x4d, 0x65, 0x74, 0x68, 0x6f, 0x64,
    0x3b, 0x5b, 0x4c, 0x6a, 0x61, 0x76, 0x61, 0x2f, 0x6c, 0x61, 0x6e, 0x67, 0x2f, 0x4f, 0x62, 0x6a,
    0x65, 0x63, 0x74, 0x3b, 0x29, 0x4c, 0x6a, 0x61, 0x76, 0x61, 0x2f, 0x6c, 0x61, 0x6e, 0x67, 0x2f,
    0x4f, 0x62, 0x6a, 0x65, 0x63, 0x74, 0x3b, 0x00, 0x31, 0x00, 0x08, 0x00, 0x02, 0x00, 0x01, 0x00,
    0x0d, 0x00, 0x01, 0x00, 0x12, 0x00, 0x0b, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0x00,
    0x05, 0x00, 0x0f, 0x00, 0x01, 0x00, 0x10, 0x00, 0x00, 0x00, 0x16, 0x00, 0x03, 0x00, 0x03, 0x00,
    0x00, 0x00, 0x0a, 0x2a, 0xb7, 0x00, 0x01, 0x2a, 0x1f, 0xb5, 0x00, 0x07, 0xb1, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x01, 0x00, 0x11, 0x00, 0x12, 0x00, 0x00, 0x00, 0x00,
];

struct CallbackRegistration {
    session: std::rc::Weak<JavaSession>,
    invoke: Rc<dyn Fn(JavaCallbackInvocation) -> Result<JavaValue, JavaInvocationError>>,
}

thread_local! {
    static CALLBACKS: RefCell<BTreeMap<u64, CallbackRegistration>> = const { RefCell::new(BTreeMap::new()) };
}

static NEXT_CALLBACK_ID: AtomicU64 = AtomicU64::new(1);
static BRIDGE_CLASS_INSTALL: Mutex<()> = Mutex::new(());

impl JavaSession {
    pub fn register_callback(
        self: &Rc<Self>,
        callback: impl Fn(JavaCallbackInvocation) -> Result<JavaValue, JavaInvocationError> + 'static,
    ) -> Result<JavaValue, JavaInvocationError> {
        let id = NEXT_CALLBACK_ID.fetch_add(1, Ordering::Relaxed);
        if id == 0 {
            return Err(JavaInvocationError::Callback(
                "callback identity space is exhausted".into(),
            ));
        }
        CALLBACKS.with(|callbacks| {
            callbacks.borrow_mut().insert(
                id,
                CallbackRegistration {
                    session: Rc::downgrade(self),
                    invoke: Rc::new(callback),
                },
            )
        });
        self.callback_ids.borrow_mut().push(id);
        Ok(JavaValue::Callback(id))
    }

    pub(super) fn create_callback_proxy<'local>(
        &self,
        environment: &mut JNIEnv<'local>,
        callback: u64,
        interface_name: &str,
    ) -> Result<JObject<'local>, JavaInvocationError> {
        let key = (callback, interface_name.to_string());
        if let Some(proxy) = self.callback_proxies.borrow().get(&key) {
            return environment
                .new_local_ref(proxy.as_obj())
                .map_err(|error| jni_error(environment, error));
        }
        let interface = self.load_class(environment, interface_name)?;
        let is_interface = environment
            .call_method(&interface, "isInterface", "()Z", &[])
            .and_then(JValueOwned::z)
            .map_err(|error| jni_error(environment, error))?;
        if !is_interface {
            return Err(JavaInvocationError::UnsupportedValue(format!(
                "Java callback target {interface_name} is not an interface"
            )));
        }
        let bridge = bridge_class(environment)?;
        let handler = environment
            .new_object(bridge, "(J)V", &[JValue::Long(callback as i64)])
            .map_err(|error| jni_error(environment, error))?;
        let interfaces = environment
            .new_object_array(1, "java/lang/Class", JObject::null())
            .map_err(|error| jni_error(environment, error))?;
        environment
            .set_object_array_element(&interfaces, 0, &interface)
            .map_err(|error| jni_error(environment, error))?;
        let loader = environment
            .call_method(
                &interface,
                "getClassLoader",
                "()Ljava/lang/ClassLoader;",
                &[],
            )
            .and_then(JValueOwned::l)
            .map_err(|error| jni_error(environment, error))?;
        let interfaces = JObject::from(interfaces);
        let proxy = environment
            .call_static_method(
                "java/lang/reflect/Proxy",
                "newProxyInstance",
                "(Ljava/lang/ClassLoader;[Ljava/lang/Class;Ljava/lang/reflect/InvocationHandler;)Ljava/lang/Object;",
                &[
                    JValue::Object(&loader),
                    JValue::Object(&interfaces),
                    JValue::Object(&handler),
                ],
            )
            .and_then(JValueOwned::l)
            .map_err(|error| jni_error(environment, error))?;
        let global = environment
            .new_global_ref(&proxy)
            .map_err(|error| jni_error(environment, error))?;
        self.callback_proxies.borrow_mut().insert(key, global);
        Ok(proxy)
    }
}

pub(super) fn remove_callbacks(ids: &[u64]) {
    let _ = CALLBACKS.try_with(|callbacks| {
        let mut callbacks = callbacks.borrow_mut();
        for id in ids {
            callbacks.remove(id);
        }
    });
}

fn bridge_class<'local>(
    environment: &mut JNIEnv<'local>,
) -> Result<jni::objects::JClass<'local>, JavaInvocationError> {
    let _installation = BRIDGE_CLASS_INSTALL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let class = match environment.find_class(BRIDGE_CLASS) {
        Ok(class) => class,
        Err(_) => {
            if environment.exception_check().unwrap_or(false) {
                environment
                    .exception_clear()
                    .map_err(|error| jni_error(environment, error))?;
            }
            environment
                .define_class(BRIDGE_CLASS, &JObject::null(), BRIDGE_CLASSFILE)
                .map_err(|error| jni_error(environment, error))?
        }
    };
    environment
        .register_native_methods(
            &class,
            &[NativeMethod {
                name: "invoke".into(),
                sig: INVOKE_SIGNATURE.into(),
                fn_ptr: callback_invoke as *mut c_void,
            }],
        )
        .map_err(|error| jni_error(environment, error))?;
    Ok(class)
}

extern "system" fn callback_invoke<'local>(
    mut environment: JNIEnv<'local>,
    handler: JObject<'local>,
    proxy: JObject<'local>,
    method: JObject<'local>,
    arguments: JObjectArray<'local>,
) -> jni::sys::jobject {
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        invoke_callback(&mut environment, handler, proxy, method, arguments)
    }));
    match outcome {
        Ok(Ok(value)) => value.into_raw(),
        Ok(Err(error)) => {
            let _ = environment.throw_new("java/lang/RuntimeException", error.to_string());
            JObject::null().into_raw()
        }
        Err(_) => {
            let _ = environment.throw_new("java/lang/RuntimeException", "RunMat callback panicked");
            JObject::null().into_raw()
        }
    }
}

fn invoke_callback<'local>(
    environment: &mut JNIEnv<'local>,
    handler: JObject<'local>,
    proxy: JObject<'local>,
    method: JObject<'local>,
    arguments: JObjectArray<'local>,
) -> Result<JObject<'local>, JavaInvocationError> {
    let id = environment
        .get_field(handler, "callbackId", "J")
        .and_then(JValueOwned::j)
        .map_err(|error| jni_error(environment, error))? as u64;
    let method_name = environment
        .call_method(&method, "getName", "()Ljava/lang/String;", &[])
        .and_then(JValueOwned::l)
        .map_err(|error| jni_error(environment, error))?;
    let method_name: String = environment
        .get_string(&JString::from(method_name))
        .map_err(|error| jni_error(environment, error))?
        .into();
    match method_name.as_str() {
        "toString" => {
            return environment
                .new_string(format!("RunMatCallback({id})"))
                .map(JObject::from)
                .map_err(|error| jni_error(environment, error));
        }
        "hashCode" => {
            return boxed(
                environment,
                "java/lang/Integer",
                "(I)Ljava/lang/Integer;",
                JValue::Int((id ^ (id >> 32)) as i32),
            );
        }
        "equals" => {
            let equal = if arguments.is_null()
                || environment
                    .get_array_length(&arguments)
                    .map_err(|error| jni_error(environment, error))?
                    != 1
            {
                false
            } else {
                let other = environment
                    .get_object_array_element(&arguments, 0)
                    .map_err(|error| jni_error(environment, error))?;
                environment
                    .is_same_object(&proxy, &other)
                    .map_err(|error| jni_error(environment, error))?
            };
            return boxed(
                environment,
                "java/lang/Boolean",
                "(Z)Ljava/lang/Boolean;",
                JValue::Bool(u8::from(equal)),
            );
        }
        _ => {}
    }
    let registration = CALLBACKS
        .with(|callbacks| {
            callbacks.borrow().get(&id).map(|registration| {
                (
                    registration.session.upgrade(),
                    Rc::clone(&registration.invoke),
                )
            })
        })
        .ok_or_else(|| {
            JavaInvocationError::Callback(
                "callback ran outside its originating RunMat thread or session".into(),
            )
        })?;
    let session = registration.0.ok_or_else(|| {
        JavaInvocationError::Callback("callback's originating Java session has ended".into())
    })?;
    let mut values = Vec::new();
    if !arguments.is_null() {
        let length = environment
            .get_array_length(&arguments)
            .map_err(|error| jni_error(environment, error))?;
        values.reserve(length as usize);
        for index in 0..length {
            let value = environment
                .get_object_array_element(&arguments, index)
                .map_err(|error| jni_error(environment, error))?;
            values.push(capture_callback_value(&session, environment, value)?);
        }
    }
    let return_type = environment
        .call_method(&method, "getReturnType", "()Ljava/lang/Class;", &[])
        .and_then(JValueOwned::l)
        .map_err(|error| jni_error(environment, error))?;
    let return_name = environment
        .call_method(return_type, "getName", "()Ljava/lang/String;", &[])
        .and_then(JValueOwned::l)
        .map_err(|error| jni_error(environment, error))?;
    let return_name: String = environment
        .get_string(&JString::from(return_name))
        .map_err(|error| jni_error(environment, error))?
        .into();
    let returns_value = return_name != "void";
    let result = registration.1(JavaCallbackInvocation {
        method_name,
        arguments: values,
        returns_value,
    })?;
    if !returns_value {
        return Ok(JObject::null());
    }
    callback_result_object(environment, &result)
}

fn capture_callback_value(
    session: &JavaSession,
    environment: &mut JNIEnv<'_>,
    object: JObject<'_>,
) -> Result<JavaValue, JavaInvocationError> {
    if object.is_null() {
        return Ok(JavaValue::Null);
    }
    let class_name = object_class_name(environment, &object)?;
    if class_name == "java.lang.String" {
        return Ok(JavaValue::String(
            environment
                .get_string(&JString::from(object))
                .map_err(|error| jni_error(environment, error))?
                .into(),
        ));
    }
    if let Some(value) = capture_boxed(environment, &object, &class_name)? {
        return Ok(value);
    }
    session.capture_object(environment, object)
}

fn callback_result_object<'local>(
    environment: &mut JNIEnv<'local>,
    value: &JavaValue,
) -> Result<JObject<'local>, JavaInvocationError> {
    match value {
        JavaValue::Null => Ok(JObject::null()),
        JavaValue::String(value) => environment
            .new_string(value)
            .map(JObject::from)
            .map_err(|error| jni_error(environment, error)),
        JavaValue::Boolean(value) => boxed(
            environment,
            "java/lang/Boolean",
            "(Z)Ljava/lang/Boolean;",
            JValue::Bool(u8::from(*value)),
        ),
        JavaValue::Byte(value) => boxed(
            environment,
            "java/lang/Byte",
            "(B)Ljava/lang/Byte;",
            JValue::Byte(*value),
        ),
        JavaValue::Short(value) => boxed(
            environment,
            "java/lang/Short",
            "(S)Ljava/lang/Short;",
            JValue::Short(*value),
        ),
        JavaValue::Int(value) => boxed(
            environment,
            "java/lang/Integer",
            "(I)Ljava/lang/Integer;",
            JValue::Int(*value),
        ),
        JavaValue::Long(value) => boxed(
            environment,
            "java/lang/Long",
            "(J)Ljava/lang/Long;",
            JValue::Long(*value),
        ),
        JavaValue::Float(value) => boxed(
            environment,
            "java/lang/Float",
            "(F)Ljava/lang/Float;",
            JValue::Float(*value),
        ),
        JavaValue::Double(value) => boxed(
            environment,
            "java/lang/Double",
            "(D)Ljava/lang/Double;",
            JValue::Double(*value),
        ),
        JavaValue::Char(value) => boxed(
            environment,
            "java/lang/Character",
            "(C)Ljava/lang/Character;",
            JValue::Char(*value),
        ),
        _ => Err(JavaInvocationError::Callback(
            "callback result must be a scalar primitive, string, or null".into(),
        )),
    }
}

fn boxed<'local>(
    environment: &mut JNIEnv<'local>,
    class: &str,
    signature: &str,
    value: JValue<'_, '_>,
) -> Result<JObject<'local>, JavaInvocationError> {
    environment
        .call_static_method(class, "valueOf", signature, &[value])
        .and_then(JValueOwned::l)
        .map_err(|error| jni_error(environment, error))
}
