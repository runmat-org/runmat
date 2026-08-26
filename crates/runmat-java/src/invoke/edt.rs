use std::collections::BTreeMap;
use std::ffi::c_void;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, LazyLock, Mutex};

use jni::objects::{GlobalRef, JClass, JObject, JValue, JValueOwned};
use jni::{JNIEnv, NativeMethod};

use super::conversion::{PreparedArguments, PreparedValue};
use super::{error::jni_error, JavaInvocationError, JavaSession, JavaValue};

const EDT_TASK_CLASS: &str = "org/runmat/bridge/RunMatEdtTask";
const EDT_TASK_CLASSFILE: &[u8] = &[
    0xca, 0xfe, 0xba, 0xbe, 0x00, 0x00, 0x00, 0x34, 0x00, 0x12, 0x0a, 0x00, 0x02, 0x00, 0x03, 0x07,
    0x00, 0x04, 0x0c, 0x00, 0x05, 0x00, 0x06, 0x01, 0x00, 0x10, 0x6a, 0x61, 0x76, 0x61, 0x2f, 0x6c,
    0x61, 0x6e, 0x67, 0x2f, 0x4f, 0x62, 0x6a, 0x65, 0x63, 0x74, 0x01, 0x00, 0x06, 0x3c, 0x69, 0x6e,
    0x69, 0x74, 0x3e, 0x01, 0x00, 0x03, 0x28, 0x29, 0x56, 0x09, 0x00, 0x08, 0x00, 0x09, 0x07, 0x00,
    0x0a, 0x0c, 0x00, 0x0b, 0x00, 0x0c, 0x01, 0x00, 0x1f, 0x6f, 0x72, 0x67, 0x2f, 0x72, 0x75, 0x6e,
    0x6d, 0x61, 0x74, 0x2f, 0x62, 0x72, 0x69, 0x64, 0x67, 0x65, 0x2f, 0x52, 0x75, 0x6e, 0x4d, 0x61,
    0x74, 0x45, 0x64, 0x74, 0x54, 0x61, 0x73, 0x6b, 0x01, 0x00, 0x06, 0x74, 0x61, 0x73, 0x6b, 0x49,
    0x64, 0x01, 0x00, 0x01, 0x4a, 0x07, 0x00, 0x0e, 0x01, 0x00, 0x12, 0x6a, 0x61, 0x76, 0x61, 0x2f,
    0x6c, 0x61, 0x6e, 0x67, 0x2f, 0x52, 0x75, 0x6e, 0x6e, 0x61, 0x62, 0x6c, 0x65, 0x01, 0x00, 0x04,
    0x28, 0x4a, 0x29, 0x56, 0x01, 0x00, 0x04, 0x43, 0x6f, 0x64, 0x65, 0x01, 0x00, 0x03, 0x72, 0x75,
    0x6e, 0x00, 0x31, 0x00, 0x08, 0x00, 0x02, 0x00, 0x01, 0x00, 0x0d, 0x00, 0x01, 0x00, 0x12, 0x00,
    0x0b, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0x00, 0x05, 0x00, 0x0f, 0x00, 0x01, 0x00,
    0x10, 0x00, 0x00, 0x00, 0x16, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0a, 0x2a, 0xb7, 0x00,
    0x01, 0x2a, 0x1f, 0xb5, 0x00, 0x07, 0xb1, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x11, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x00,
];

static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);
static TASKS: LazyLock<Mutex<BTreeMap<u64, EdtTask>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

struct EdtTask {
    invocation: EdtInvocation,
    result: mpsc::SyncSender<Result<EdtResult, JavaInvocationError>>,
}

pub(super) enum EdtTarget {
    Constructor(GlobalRef),
    Static { class: GlobalRef, method: String },
    Instance { receiver: GlobalRef, method: String },
}

pub(super) struct EdtInvocation {
    pub target: EdtTarget,
    pub descriptor: String,
    pub arguments: OwnedArguments,
}

pub(super) enum EdtResult {
    Value(JavaValue),
    Object(GlobalRef),
}

pub(super) struct OwnedArguments {
    prepared: Vec<PreparedValue>,
    objects: Vec<GlobalRef>,
}

impl OwnedArguments {
    pub(super) fn capture(
        environment: &mut JNIEnv<'_>,
        arguments: PreparedArguments<'_>,
    ) -> Result<Self, JavaInvocationError> {
        let objects = arguments
            .objects
            .iter()
            .map(|object| {
                environment
                    .new_global_ref(object)
                    .map_err(|error| jni_error(environment, error))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            prepared: arguments.prepared,
            objects,
        })
    }

    fn with_values<R>(
        &self,
        environment: &mut JNIEnv<'_>,
        operation: impl FnOnce(&mut JNIEnv<'_>, &[JValue<'_, '_>]) -> Result<R, JavaInvocationError>,
    ) -> Result<R, JavaInvocationError> {
        let objects = self
            .objects
            .iter()
            .map(|object| {
                environment
                    .new_local_ref(object.as_obj())
                    .map_err(|error| jni_error(environment, error))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let null = JObject::null();
        let values = self
            .prepared
            .iter()
            .map(|value| match value {
                PreparedValue::Null => JValue::Object(&null),
                PreparedValue::Boolean(value) => JValue::Bool(u8::from(*value)),
                PreparedValue::Byte(value) => JValue::Byte(*value),
                PreparedValue::Short(value) => JValue::Short(*value),
                PreparedValue::Int(value) => JValue::Int(*value),
                PreparedValue::Long(value) => JValue::Long(*value),
                PreparedValue::Float(value) => JValue::Float(*value),
                PreparedValue::Double(value) => JValue::Double(*value),
                PreparedValue::Char(value) => JValue::Char(*value),
                PreparedValue::Object(index) => JValue::Object(&objects[*index]),
            })
            .collect::<Vec<_>>();
        operation(environment, &values)
    }
}

impl EdtInvocation {
    fn execute(self, environment: &mut JNIEnv<'_>) -> Result<EdtResult, JavaInvocationError> {
        self.arguments
            .with_values(environment, |environment, values| {
                let result = match &self.target {
                    EdtTarget::Constructor(class) => {
                        let class = JClass::from(
                            environment
                                .new_local_ref(class.as_obj())
                                .map_err(|error| jni_error(environment, error))?,
                        );
                        JValueOwned::Object(
                            environment
                                .new_object(class, &self.descriptor, values)
                                .map_err(|error| jni_error(environment, error))?,
                        )
                    }
                    EdtTarget::Static { class, method } => {
                        let class = JClass::from(
                            environment
                                .new_local_ref(class.as_obj())
                                .map_err(|error| jni_error(environment, error))?,
                        );
                        environment
                            .call_static_method(class, method, &self.descriptor, values)
                            .map_err(|error| jni_error(environment, error))?
                    }
                    EdtTarget::Instance { receiver, method } => environment
                        .call_method(receiver.as_obj(), method, &self.descriptor, values)
                        .map_err(|error| jni_error(environment, error))?,
                };
                capture_result(environment, result)
            })
    }
}

fn capture_result(
    environment: &mut JNIEnv<'_>,
    value: JValueOwned<'_>,
) -> Result<EdtResult, JavaInvocationError> {
    let scalar = match value {
        JValueOwned::Void => JavaValue::Null,
        JValueOwned::Bool(value) => JavaValue::Boolean(value != 0),
        JValueOwned::Byte(value) => JavaValue::Byte(value),
        JValueOwned::Char(value) => JavaValue::Char(value),
        JValueOwned::Short(value) => JavaValue::Short(value),
        JValueOwned::Int(value) => JavaValue::Int(value),
        JValueOwned::Long(value) => JavaValue::Long(value),
        JValueOwned::Float(value) => JavaValue::Float(value),
        JValueOwned::Double(value) => JavaValue::Double(value),
        JValueOwned::Object(object) => {
            if object.is_null() {
                return Ok(EdtResult::Value(JavaValue::Null));
            }
            return environment
                .new_global_ref(object)
                .map(EdtResult::Object)
                .map_err(|error| jni_error(environment, error));
        }
    };
    Ok(EdtResult::Value(scalar))
}

pub(super) fn dispatch(
    environment: &mut JNIEnv<'_>,
    invocation: EdtInvocation,
) -> Result<EdtResult, JavaInvocationError> {
    let on_edt = environment
        .call_static_method(
            "javax/swing/SwingUtilities",
            "isEventDispatchThread",
            "()Z",
            &[],
        )
        .and_then(JValueOwned::z)
        .map_err(|error| jni_error(environment, error))?;
    if on_edt {
        return invocation.execute(environment);
    }

    let id = NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed);
    if id == 0 {
        return Err(JavaInvocationError::Callback(
            "Java EDT task identity space is exhausted".into(),
        ));
    }
    let (sender, receiver) = mpsc::sync_channel(1);
    TASKS
        .lock()
        .map_err(|_| JavaInvocationError::Callback("Java EDT task registry is poisoned".into()))?
        .insert(
            id,
            EdtTask {
                invocation,
                result: sender,
            },
        );
    let class = edt_task_class(environment)?;
    let task = environment
        .new_object(class, "(J)V", &[JValue::Long(id as i64)])
        .map_err(|error| jni_error(environment, error))?;
    let invoke = environment.call_static_method(
        "javax/swing/SwingUtilities",
        "invokeAndWait",
        "(Ljava/lang/Runnable;)V",
        &[JValue::Object(&task)],
    );
    if let Err(error) = invoke {
        if let Ok(mut tasks) = TASKS.lock() {
            tasks.remove(&id);
        }
        return Err(jni_error(environment, error));
    }
    receiver
        .recv()
        .map_err(|_| JavaInvocationError::Callback("Java EDT task ended without a result".into()))?
}

fn edt_task_class<'local>(
    environment: &mut JNIEnv<'local>,
) -> Result<JClass<'local>, JavaInvocationError> {
    let class = match environment.find_class(EDT_TASK_CLASS) {
        Ok(class) => class,
        Err(_) => {
            if environment.exception_check().unwrap_or(false) {
                environment
                    .exception_clear()
                    .map_err(|error| jni_error(environment, error))?;
            }
            environment
                .define_class(EDT_TASK_CLASS, &JObject::null(), EDT_TASK_CLASSFILE)
                .map_err(|error| jni_error(environment, error))?
        }
    };
    environment
        .register_native_methods(
            &class,
            &[NativeMethod {
                name: "run".into(),
                sig: "()V".into(),
                fn_ptr: run_edt_task as *mut c_void,
            }],
        )
        .map_err(|error| jni_error(environment, error))?;
    Ok(class)
}

extern "system" fn run_edt_task<'local>(mut environment: JNIEnv<'local>, task: JObject<'local>) {
    let id = match environment
        .get_field(task, "taskId", "J")
        .and_then(JValueOwned::j)
    {
        Ok(id) => id as u64,
        Err(_) => return,
    };
    let task = match TASKS.lock() {
        Ok(mut tasks) => tasks.remove(&id),
        Err(_) => {
            let _ = environment.throw_new(
                "java/lang/IllegalStateException",
                "RunMat Java EDT task registry is poisoned",
            );
            return;
        }
    };
    let Some(task) = task else {
        let _ = environment.throw_new(
            "java/lang/IllegalStateException",
            "RunMat Java EDT task is no longer registered",
        );
        return;
    };
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        task.invocation.execute(&mut environment)
    }))
    .unwrap_or_else(|_| {
        Err(JavaInvocationError::Callback(
            "Java EDT task panicked".into(),
        ))
    });
    let _ = task.result.send(result);
}

impl JavaSession {
    pub(super) fn capture_edt_result(
        &self,
        environment: &mut JNIEnv<'_>,
        result: EdtResult,
    ) -> Result<JavaValue, JavaInvocationError> {
        match result {
            EdtResult::Value(value) => Ok(value),
            EdtResult::Object(object) => {
                let local = environment
                    .new_local_ref(object.as_obj())
                    .map_err(|error| jni_error(environment, error))?;
                self.capture_object(environment, local)
            }
        }
    }
}
