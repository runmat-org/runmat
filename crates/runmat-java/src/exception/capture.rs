use jni::objects::{JObject, JObjectArray, JString, JThrowable, JValueOwned};

use super::{JavaException, JavaStackFrame};

const MAX_CAUSE_DEPTH: usize = 16;

pub(crate) fn capture_pending_exception(
    environment: &mut jni::JNIEnv<'_>,
) -> Result<JavaException, String> {
    let throwable = environment
        .exception_occurred()
        .map_err(|error| error.to_string())?;
    environment
        .exception_clear()
        .map_err(|error| error.to_string())?;
    capture_throwable(environment, throwable, 0)
}

fn capture_throwable(
    environment: &mut jni::JNIEnv<'_>,
    throwable: JThrowable<'_>,
    depth: usize,
) -> Result<JavaException, String> {
    let object = JObject::from(throwable);
    let class = environment
        .call_method(&object, "getClass", "()Ljava/lang/Class;", &[])
        .and_then(JValueOwned::l)
        .map_err(|error| error.to_string())?;
    let class_name = string_result(environment, &class, "getName")?
        .unwrap_or_else(|| "java.lang.Throwable".into());
    let message = string_result(environment, &object, "getMessage")?;
    let frames = stack_frames(environment, &object)?;
    let cause = if depth >= MAX_CAUSE_DEPTH {
        None
    } else {
        let cause = environment
            .call_method(&object, "getCause", "()Ljava/lang/Throwable;", &[])
            .and_then(JValueOwned::l)
            .map_err(|error| error.to_string())?;
        if cause.is_null()
            || environment
                .is_same_object(&object, &cause)
                .map_err(|error| error.to_string())?
        {
            None
        } else {
            Some(Box::new(capture_throwable(
                environment,
                JThrowable::from(cause),
                depth + 1,
            )?))
        }
    };
    Ok(JavaException {
        class_name,
        message,
        frames,
        cause,
    })
}

fn stack_frames(
    environment: &mut jni::JNIEnv<'_>,
    throwable: &JObject<'_>,
) -> Result<Vec<JavaStackFrame>, String> {
    let frames = environment
        .call_method(
            throwable,
            "getStackTrace",
            "()[Ljava/lang/StackTraceElement;",
            &[],
        )
        .and_then(JValueOwned::l)
        .map_err(|error| error.to_string())?;
    let frames = JObjectArray::from(frames);
    let length = environment
        .get_array_length(&frames)
        .map_err(|error| error.to_string())?;
    let mut result = Vec::with_capacity(length as usize);
    for index in 0..length {
        let frame = environment
            .get_object_array_element(&frames, index)
            .map_err(|error| error.to_string())?;
        let line = environment
            .call_method(&frame, "getLineNumber", "()I", &[])
            .and_then(JValueOwned::i)
            .map_err(|error| error.to_string())?;
        result.push(JavaStackFrame {
            class_name: string_result(environment, &frame, "getClassName")?.unwrap_or_default(),
            method_name: string_result(environment, &frame, "getMethodName")?.unwrap_or_default(),
            file_name: string_result(environment, &frame, "getFileName")?,
            line: (line >= 0).then_some(line),
        });
    }
    Ok(result)
}

fn string_result(
    environment: &mut jni::JNIEnv<'_>,
    object: &JObject<'_>,
    method: &str,
) -> Result<Option<String>, String> {
    let value = environment
        .call_method(object, method, "()Ljava/lang/String;", &[])
        .and_then(JValueOwned::l)
        .map_err(|error| error.to_string())?;
    if value.is_null() {
        return Ok(None);
    }
    let string = JString::from(value);
    let value = environment
        .get_string(&string)
        .map_err(|error| error.to_string())?
        .into();
    Ok(Some(value))
}
