use std::collections::HashSet;
use std::ffi::CString;

use super::api::{CpythonApi, PyObject};
use crate::{PythonArray, PythonDType, PythonError, PythonFrame};

pub(super) fn array_strides(array: &PythonArray) -> Result<Vec<usize>, PythonError> {
    let mut strides = vec![0; array.shape.len()];
    let mut stride = array.dtype.byte_width();
    if array.column_major {
        for (index, dimension) in array.shape.iter().enumerate() {
            strides[index] = stride;
            stride = stride.checked_mul(*dimension).ok_or_else(|| {
                PythonError::host("PythonOverflowError", "Python array strides overflow usize")
            })?;
        }
    } else {
        for (index, dimension) in array.shape.iter().enumerate().rev() {
            strides[index] = stride;
            stride = stride.checked_mul(*dimension).ok_or_else(|| {
                PythonError::host("PythonOverflowError", "Python array strides overflow usize")
            })?;
        }
    }
    Ok(strides)
}

pub(super) fn parse_numpy_dtype(typestr: &str) -> Result<PythonDType, PythonError> {
    let Some((byte_order, code)) = typestr.split_at_checked(1) else {
        return Err(PythonError::host(
            "PythonArrayConversionError",
            format!("invalid NumPy dtype {typestr:?}"),
        ));
    };
    let native_order = byte_order == "="
        || byte_order == "|"
        || (byte_order == "<" && cfg!(target_endian = "little"))
        || (byte_order == ">" && cfg!(target_endian = "big"));
    if !native_order {
        return Err(PythonError::host(
            "PythonArrayConversionError",
            format!("non-native-endian NumPy dtype {typestr} requires an explicit conversion"),
        ));
    }
    match code {
        "f8" => Ok(PythonDType::Float64),
        "f4" => Ok(PythonDType::Float32),
        "i1" => Ok(PythonDType::Int8),
        "i2" => Ok(PythonDType::Int16),
        "i4" => Ok(PythonDType::Int32),
        "i8" => Ok(PythonDType::Int64),
        "u1" => Ok(PythonDType::Uint8),
        "u2" => Ok(PythonDType::Uint16),
        "u4" => Ok(PythonDType::Uint32),
        "u8" => Ok(PythonDType::Uint64),
        "b1" => Ok(PythonDType::Bool),
        "c8" => Ok(PythonDType::Complex64),
        "c16" => Ok(PythonDType::Complex128),
        code if code.starts_with("M8[") => Ok(PythonDType::DateTime64Micros),
        code if code.starts_with("m8[") => Ok(PythonDType::TimeDelta64Micros),
        _ => Err(PythonError::host(
            "PythonArrayConversionError",
            format!("NumPy dtype {typestr} is not supported"),
        )),
    }
}

pub(super) unsafe fn attr(
    api: &CpythonApi,
    object: *mut PyObject,
    name: &str,
) -> Result<*mut PyObject, PythonError> {
    let name = c_string(name, "Python attribute name")?;
    // SAFETY: object is live and name is NUL-terminated.
    let value = unsafe { (api.py_object_get_attr_string)(object, name.as_ptr()) };
    unsafe { checked(api, value, "read Python attribute") }
}

pub(super) unsafe fn set_dict_string(
    api: &CpythonApi,
    dict: *mut PyObject,
    name: &str,
    value: *mut PyObject,
) -> Result<(), PythonError> {
    let name = c_string(name, "Python dictionary key")?;
    // SAFETY: dictionary and value are live and name is NUL-terminated.
    let status = unsafe { (api.py_dict_set_item_string)(dict, name.as_ptr(), value) };
    if status < 0 {
        Err(unsafe { capture_error(api, "set Python dictionary value") })
    } else {
        Ok(())
    }
}

pub(super) unsafe fn checked(
    api: &CpythonApi,
    object: *mut PyObject,
    operation: &str,
) -> Result<*mut PyObject, PythonError> {
    if object.is_null() {
        Err(unsafe { capture_error(api, operation) })
    } else {
        Ok(object)
    }
}

pub(super) unsafe fn unicode(
    api: &CpythonApi,
    object: *mut PyObject,
) -> Result<String, PythonError> {
    let mut length = 0;
    // SAFETY: object is a live Unicode instance.
    let bytes = unsafe { (api.py_unicode_as_utf8_and_size)(object, &mut length) };
    if bytes.is_null() {
        return Err(unsafe { capture_error(api, "read Python string") });
    }
    let length = usize::try_from(length).map_err(|_| {
        PythonError::host("PythonOverflowError", "Python string has a negative length")
    })?;
    // SAFETY: CPython exposes a valid UTF-8 buffer with the reported length.
    let bytes = unsafe { std::slice::from_raw_parts(bytes.cast::<u8>(), length) };
    String::from_utf8(bytes.to_vec()).map_err(|error| {
        PythonError::host(
            "PythonUnicodeError",
            format!("Python returned invalid UTF-8: {error}"),
        )
    })
}

pub(super) unsafe fn capture_error(api: &CpythonApi, operation: &str) -> PythonError {
    if unsafe { (api.py_err_occurred)() }.is_null() {
        return PythonError::host("PythonHostError", operation);
    }
    let mut type_object = std::ptr::null_mut();
    let mut value = std::ptr::null_mut();
    let mut traceback = std::ptr::null_mut();
    unsafe { (api.py_err_fetch)(&mut type_object, &mut value, &mut traceback) };
    unsafe { (api.py_err_normalize_exception)(&mut type_object, &mut value, &mut traceback) };

    let type_name = if type_object.is_null() {
        "PythonError".to_owned()
    } else {
        unsafe { attr(api, type_object, "__name__") }
            .and_then(|name| {
                let text = unsafe { unicode(api, name) };
                unsafe { (api.py_dec_ref)(name) };
                text
            })
            .unwrap_or_else(|_| "PythonError".to_owned())
    };
    let message = if value.is_null() {
        operation.to_owned()
    } else {
        let rendered = unsafe { (api.py_object_str)(value) };
        if rendered.is_null() {
            unsafe { (api.py_err_clear)() };
            operation.to_owned()
        } else {
            let text = unsafe { unicode(api, rendered) }.unwrap_or_else(|_| operation.to_owned());
            unsafe { (api.py_dec_ref)(rendered) };
            text
        }
    };
    let (frames, formatted_traceback) =
        unsafe { traceback_details(api, type_object, value, traceback) };
    let cause = if value.is_null() {
        None
    } else {
        let mut seen = HashSet::from([value as usize]);
        unsafe { exception_cause(api, value, &mut seen, 0) }
    };
    if !type_object.is_null() {
        unsafe { (api.py_dec_ref)(type_object) };
    }
    if !value.is_null() {
        unsafe { (api.py_dec_ref)(value) };
    }
    if !traceback.is_null() {
        unsafe { (api.py_dec_ref)(traceback) };
    }
    unsafe { (api.py_err_clear)() };
    PythonError {
        type_name,
        message,
        traceback: frames,
        formatted_traceback,
        cause,
    }
}

unsafe fn exception_cause(
    api: &CpythonApi,
    exception: *mut PyObject,
    seen: &mut HashSet<usize>,
    depth: usize,
) -> Option<Box<PythonError>> {
    const MAX_CAUSE_DEPTH: usize = 16;
    if depth >= MAX_CAUSE_DEPTH {
        return None;
    }
    let explicit = unsafe { raw_attr(api, exception, "__cause__") };
    let suppress_context = unsafe { truth_attr(api, exception, "__suppress_context__") };
    let selected = match explicit {
        Some(value) if !unsafe { is_none(api, value) } => Some(value),
        Some(value) => {
            unsafe { (api.py_dec_ref)(value) };
            if suppress_context {
                None
            } else {
                unsafe { raw_attr(api, exception, "__context__") }
            }
        }
        None if !suppress_context => unsafe { raw_attr(api, exception, "__context__") },
        None => None,
    }?;
    if unsafe { is_none(api, selected) } || !seen.insert(selected as usize) {
        unsafe { (api.py_dec_ref)(selected) };
        return None;
    }
    let error = unsafe { error_from_exception_value(api, selected, seen, depth + 1) };
    unsafe { (api.py_dec_ref)(selected) };
    error.map(Box::new)
}

unsafe fn error_from_exception_value(
    api: &CpythonApi,
    exception: *mut PyObject,
    seen: &mut HashSet<usize>,
    depth: usize,
) -> Option<PythonError> {
    let class = unsafe { raw_attr(api, exception, "__class__") }?;
    let type_name =
        unsafe { string_attr(api, class, "__name__") }.unwrap_or_else(|| "PythonError".to_owned());
    let rendered = unsafe { (api.py_object_str)(exception) };
    let message = if rendered.is_null() {
        unsafe { (api.py_err_clear)() };
        String::new()
    } else {
        let message = unsafe { unicode(api, rendered) }.unwrap_or_default();
        unsafe { (api.py_dec_ref)(rendered) };
        message
    };
    let traceback = unsafe { raw_attr(api, exception, "__traceback__") };
    let (frames, formatted_traceback) = traceback.map_or_else(
        || (Vec::new(), String::new()),
        |traceback| {
            let details = if unsafe { is_none(api, traceback) } {
                (Vec::new(), String::new())
            } else {
                unsafe { traceback_details(api, class, exception, traceback) }
            };
            unsafe { (api.py_dec_ref)(traceback) };
            details
        },
    );
    unsafe { (api.py_dec_ref)(class) };
    Some(PythonError {
        type_name,
        message,
        traceback: frames,
        formatted_traceback,
        cause: unsafe { exception_cause(api, exception, seen, depth) },
    })
}

unsafe fn is_none(api: &CpythonApi, value: *mut PyObject) -> bool {
    let Some(class) = (unsafe { raw_attr(api, value, "__class__") }) else {
        return false;
    };
    let is_none = unsafe { string_attr(api, class, "__name__") }.as_deref() == Some("NoneType");
    unsafe { (api.py_dec_ref)(class) };
    is_none
}

unsafe fn truth_attr(api: &CpythonApi, object: *mut PyObject, name: &str) -> bool {
    let Some(value) = (unsafe { raw_attr(api, object, name) }) else {
        return false;
    };
    let truth = unsafe { (api.py_object_is_true)(value) };
    unsafe { (api.py_dec_ref)(value) };
    if truth < 0 {
        unsafe { (api.py_err_clear)() };
        false
    } else {
        truth != 0
    }
}

unsafe fn traceback_details(
    api: &CpythonApi,
    type_object: *mut PyObject,
    value: *mut PyObject,
    traceback: *mut PyObject,
) -> (Vec<PythonFrame>, String) {
    if type_object.is_null() || value.is_null() || traceback.is_null() {
        return (Vec::new(), String::new());
    }
    let module = unsafe { (api.py_import_import_module)(c"traceback".as_ptr()) };
    if module.is_null() {
        unsafe { (api.py_err_clear)() };
        return (Vec::new(), String::new());
    }
    let frames = unsafe { extract_traceback_frames(api, module, traceback) };
    let formatted = unsafe { format_traceback(api, module, type_object, value, traceback) };
    unsafe { (api.py_dec_ref)(module) };
    (frames, formatted)
}

unsafe fn extract_traceback_frames(
    api: &CpythonApi,
    module: *mut PyObject,
    traceback: *mut PyObject,
) -> Vec<PythonFrame> {
    let Some(extract) = (unsafe { raw_attr(api, module, "extract_tb") }) else {
        return Vec::new();
    };
    let Some(arguments) = (unsafe { tuple_from_borrowed(api, &[traceback]) }) else {
        unsafe { (api.py_dec_ref)(extract) };
        return Vec::new();
    };
    let summaries = unsafe { (api.py_object_call)(extract, arguments, std::ptr::null_mut()) };
    unsafe {
        (api.py_dec_ref)(extract);
        (api.py_dec_ref)(arguments);
    }
    if summaries.is_null() {
        unsafe { (api.py_err_clear)() };
        return Vec::new();
    }
    let length = unsafe { (api.py_list_size)(summaries) };
    if length < 0 {
        unsafe {
            (api.py_err_clear)();
            (api.py_dec_ref)(summaries);
        }
        return Vec::new();
    }
    let mut frames = Vec::with_capacity(length as usize);
    for index in 0..length {
        let summary = unsafe { (api.py_list_get_item)(summaries, index) };
        if summary.is_null() {
            unsafe { (api.py_err_clear)() };
            continue;
        }
        let file = unsafe { string_attr(api, summary, "filename") }.unwrap_or_default();
        let function = unsafe { string_attr(api, summary, "name") };
        let source = unsafe { string_attr(api, summary, "line") };
        let line = unsafe { integer_attr(api, summary, "lineno") };
        frames.push(PythonFrame {
            file,
            line,
            function,
            source,
        });
    }
    unsafe { (api.py_dec_ref)(summaries) };
    frames
}

unsafe fn format_traceback(
    api: &CpythonApi,
    module: *mut PyObject,
    type_object: *mut PyObject,
    value: *mut PyObject,
    traceback: *mut PyObject,
) -> String {
    let Some(format_exception) = (unsafe { raw_attr(api, module, "format_exception") }) else {
        return String::new();
    };
    let Some(arguments) = (unsafe { tuple_from_borrowed(api, &[type_object, value, traceback]) })
    else {
        unsafe { (api.py_dec_ref)(format_exception) };
        return String::new();
    };
    let rendered =
        unsafe { (api.py_object_call)(format_exception, arguments, std::ptr::null_mut()) };
    unsafe {
        (api.py_dec_ref)(format_exception);
        (api.py_dec_ref)(arguments);
    }
    if rendered.is_null() {
        unsafe { (api.py_err_clear)() };
        return String::new();
    }
    let length = unsafe { (api.py_list_size)(rendered) };
    if length < 0 {
        unsafe {
            (api.py_err_clear)();
            (api.py_dec_ref)(rendered);
        }
        return String::new();
    }
    let mut output = String::new();
    for index in 0..length {
        let item = unsafe { (api.py_list_get_item)(rendered, index) };
        if !item.is_null() {
            if let Ok(text) = unsafe { unicode(api, item) } {
                output.push_str(&text);
            } else {
                unsafe { (api.py_err_clear)() };
            }
        }
    }
    unsafe { (api.py_dec_ref)(rendered) };
    output.trim_end().to_owned()
}

unsafe fn tuple_from_borrowed(api: &CpythonApi, values: &[*mut PyObject]) -> Option<*mut PyObject> {
    let tuple = unsafe { (api.py_tuple_new)(values.len() as isize) };
    if tuple.is_null() {
        unsafe { (api.py_err_clear)() };
        return None;
    }
    for (index, value) in values.iter().copied().enumerate() {
        unsafe { (api.py_inc_ref)(value) };
        if unsafe { (api.py_tuple_set_item)(tuple, index as isize, value) } != 0 {
            unsafe {
                (api.py_err_clear)();
                (api.py_dec_ref)(tuple);
            }
            return None;
        }
    }
    Some(tuple)
}

unsafe fn raw_attr(api: &CpythonApi, object: *mut PyObject, name: &str) -> Option<*mut PyObject> {
    let name = CString::new(name).ok()?;
    let value = unsafe { (api.py_object_get_attr_string)(object, name.as_ptr()) };
    if value.is_null() {
        unsafe { (api.py_err_clear)() };
        None
    } else {
        Some(value)
    }
}

unsafe fn string_attr(api: &CpythonApi, object: *mut PyObject, name: &str) -> Option<String> {
    let value = unsafe { raw_attr(api, object, name) }?;
    let text = unsafe { unicode(api, value) }.ok();
    unsafe { (api.py_dec_ref)(value) };
    if text.as_deref() == Some("None") {
        None
    } else {
        text
    }
}

unsafe fn integer_attr(api: &CpythonApi, object: *mut PyObject, name: &str) -> Option<u32> {
    let value = unsafe { raw_attr(api, object, name) }?;
    let mut overflow = 0;
    let integer = unsafe { (api.py_long_as_long_long_and_overflow)(value, &mut overflow) };
    unsafe { (api.py_dec_ref)(value) };
    if overflow == 0 {
        u32::try_from(integer).ok()
    } else {
        unsafe { (api.py_err_clear)() };
        None
    }
}

pub(super) fn c_string(value: &str, purpose: &str) -> Result<CString, PythonError> {
    CString::new(value).map_err(|_| {
        PythonError::host(
            "PythonArgumentError",
            format!("{purpose} contains a null byte"),
        )
    })
}
