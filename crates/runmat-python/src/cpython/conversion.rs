use std::sync::Arc;

use super::api::{PyObject, PySsize};
use super::interpreter::Interpreter;
use super::support::{
    array_strides, attr, capture_error, checked, parse_numpy_dtype, set_dict_string, unicode,
};
use crate::{value::OwnedPythonBytes, PythonArray, PythonError, PythonObjectHandle, PythonValue};

impl Interpreter {
    pub(super) fn values_to_tuple(
        &self,
        values: Vec<PythonValue>,
    ) -> Result<*mut PyObject, PythonError> {
        let length = PySsize::try_from(values.len()).map_err(|_| {
            PythonError::host("PythonOverflowError", "too many positional arguments")
        })?;
        // SAFETY: constructor returns a new tuple reference.
        let tuple = unsafe {
            checked(
                &self.api,
                (self.api.py_tuple_new)(length),
                "create argument tuple",
            )?
        };
        for (index, value) in values.into_iter().enumerate() {
            let item = match self.to_owned(value) {
                Ok(item) => item,
                Err(error) => {
                    self.decref(tuple);
                    return Err(error);
                }
            };
            // PyTuple_SetItem steals item on both the success path and the
            // documented in-range failure path.
            let status = unsafe { (self.api.py_tuple_set_item)(tuple, index as PySsize, item) };
            if let Err(error) = self.status(status, "populate Python argument tuple") {
                self.decref(tuple);
                return Err(error);
            }
        }
        Ok(tuple)
    }

    pub(super) fn keywords_to_dict(
        &self,
        values: Vec<(String, PythonValue)>,
    ) -> Result<Option<*mut PyObject>, PythonError> {
        if values.is_empty() {
            return Ok(None);
        }
        // SAFETY: constructor returns a new dictionary reference.
        let dict = unsafe {
            checked(
                &self.api,
                (self.api.py_dict_new)(),
                "create keyword dictionary",
            )?
        };
        for (name, value) in values {
            let value = match self.to_owned(value) {
                Ok(value) => value,
                Err(error) => {
                    self.decref(dict);
                    return Err(error);
                }
            };
            let result = unsafe { set_dict_string(&self.api, dict, &name, value) };
            self.decref(value);
            if let Err(error) = result {
                self.decref(dict);
                return Err(error);
            }
        }
        Ok(Some(dict))
    }

    pub(super) fn to_owned(&self, value: PythonValue) -> Result<*mut PyObject, PythonError> {
        let object = match value {
            PythonValue::None => {
                self.incref(self.builtins.none);
                self.builtins.none
            }
            PythonValue::Bool(value) => unsafe { (self.api.py_bool_from_long)(value.into()) },
            PythonValue::Signed(value) => unsafe { (self.api.py_long_from_long_long)(value) },
            PythonValue::Unsigned(value) => unsafe {
                (self.api.py_long_from_unsigned_long_long)(value)
            },
            PythonValue::Float(value) => unsafe { (self.api.py_float_from_double)(value) },
            PythonValue::Complex { real, imaginary } => unsafe {
                (self.api.py_complex_from_doubles)(real, imaginary)
            },
            PythonValue::String(value) => {
                let length = PySsize::try_from(value.len()).map_err(|_| {
                    PythonError::host("PythonOverflowError", "string is too large for Python")
                })?;
                unsafe { (self.api.py_unicode_from_string_and_size)(value.as_ptr().cast(), length) }
            }
            PythonValue::Bytes(value) => {
                let length = PySsize::try_from(value.len()).map_err(|_| {
                    PythonError::host("PythonOverflowError", "byte array is too large for Python")
                })?;
                unsafe { (self.api.py_bytes_from_string_and_size)(value.as_ptr().cast(), length) }
            }
            PythonValue::List(values) => self.values_to_list(values)?,
            PythonValue::Tuple(values) => self.values_to_tuple(values)?,
            PythonValue::Dict(values) => self.values_to_dict(values)?,
            PythonValue::Object(handle) => {
                let object = self.object(handle)?;
                self.incref(object);
                object
            }
            PythonValue::Callback(callback) => return self.callback_to_python(callback),
            PythonValue::Keywords(_) => {
                return Err(PythonError::host(
                    "PythonArgumentError",
                    "pyargs may appear only as the final function argument",
                ));
            }
            PythonValue::Array(array) => return self.array_to_python(array),
        };
        unsafe { checked(&self.api, object, "convert RunMat value to Python") }
    }

    pub(super) fn values_to_list(
        &self,
        values: Vec<PythonValue>,
    ) -> Result<*mut PyObject, PythonError> {
        let length = PySsize::try_from(values.len())
            .map_err(|_| PythonError::host("PythonOverflowError", "list is too large"))?;
        let list = unsafe {
            checked(
                &self.api,
                (self.api.py_list_new)(length),
                "create Python list",
            )?
        };
        for (index, value) in values.into_iter().enumerate() {
            let item = match self.to_owned(value) {
                Ok(item) => item,
                Err(error) => {
                    self.decref(list);
                    return Err(error);
                }
            };
            let status = unsafe { (self.api.py_list_set_item)(list, index as PySsize, item) };
            if let Err(error) = self.status(status, "populate Python list") {
                self.decref(list);
                return Err(error);
            }
        }
        Ok(list)
    }

    pub(super) fn values_to_dict(
        &self,
        values: Vec<(PythonValue, PythonValue)>,
    ) -> Result<*mut PyObject, PythonError> {
        let dict = unsafe {
            checked(
                &self.api,
                (self.api.py_dict_new)(),
                "create Python dictionary",
            )?
        };
        for (key, value) in values {
            let key = match self.to_owned(key) {
                Ok(key) => key,
                Err(error) => {
                    self.decref(dict);
                    return Err(error);
                }
            };
            let value = match self.to_owned(value) {
                Ok(value) => value,
                Err(error) => {
                    self.decref(key);
                    self.decref(dict);
                    return Err(error);
                }
            };
            let status = unsafe { (self.api.py_dict_set_item)(dict, key, value) };
            self.decref(key);
            self.decref(value);
            if let Err(error) = self.status(status, "populate Python dictionary") {
                self.decref(dict);
                return Err(error);
            }
        }
        Ok(dict)
    }

    pub(super) fn convert_owned(&self, object: *mut PyObject) -> Result<PythonValue, PythonError> {
        let converted = self.convert_borrowed(object);
        if !matches!(converted, Ok(PythonValue::Object(_))) {
            self.decref(object);
        }
        converted
    }

    pub(super) fn convert_borrowed(
        &self,
        object: *mut PyObject,
    ) -> Result<PythonValue, PythonError> {
        if object == self.builtins.none {
            return Ok(PythonValue::None);
        }
        if self.is_instance(object, self.builtins.bool_)? {
            let truth = unsafe { (self.api.py_object_is_true)(object) };
            self.status(truth.min(0), "read Python boolean")?;
            return Ok(PythonValue::Bool(truth != 0));
        }
        if self.is_instance(object, self.builtins.int)? {
            let mut overflow = 0;
            let signed =
                unsafe { (self.api.py_long_as_long_long_and_overflow)(object, &mut overflow) };
            if overflow == 0 {
                return Ok(PythonValue::Signed(signed));
            }
            unsafe { (self.api.py_err_clear)() };
            let unsigned = unsafe { (self.api.py_long_as_unsigned_long_long)(object) };
            if unsafe { (self.api.py_err_occurred)() }.is_null() {
                return Ok(PythonValue::Unsigned(unsigned));
            }
            return Err(unsafe { capture_error(&self.api, "convert Python integer") });
        }
        if self.is_instance(object, self.builtins.float)? {
            let value = unsafe { (self.api.py_float_as_double)(object) };
            if !unsafe { (self.api.py_err_occurred)() }.is_null() {
                return Err(unsafe { capture_error(&self.api, "convert Python float") });
            }
            return Ok(PythonValue::Float(value));
        }
        if self.is_instance(object, self.builtins.complex)? {
            let real = unsafe { (self.api.py_complex_real_as_double)(object) };
            let imaginary = unsafe { (self.api.py_complex_imag_as_double)(object) };
            if !unsafe { (self.api.py_err_occurred)() }.is_null() {
                return Err(unsafe { capture_error(&self.api, "convert Python complex") });
            }
            return Ok(PythonValue::Complex { real, imaginary });
        }
        if self.is_instance(object, self.builtins.str_)? {
            return unsafe { unicode(&self.api, object).map(PythonValue::String) };
        }
        if self.is_instance(object, self.builtins.bytes)? {
            let mut data = std::ptr::null_mut();
            let mut length = 0;
            let status =
                unsafe { (self.api.py_bytes_as_string_and_size)(object, &mut data, &mut length) };
            self.status(status, "read Python bytes")?;
            let length = usize::try_from(length).map_err(|_| {
                PythonError::host("PythonOverflowError", "Python bytes have a negative length")
            })?;
            // SAFETY: CPython returned a live byte buffer of exactly length.
            let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), length) };
            return Ok(PythonValue::Bytes(bytes.to_vec()));
        }
        if let Some(array) = self.array_from_python(object)? {
            return Ok(PythonValue::Array(array));
        }
        let handle = PythonObjectHandle(self.next_handle.get());
        self.next_handle
            .set(self.next_handle.get().checked_add(1).ok_or_else(|| {
                PythonError::host("PythonObjectLimit", "Python object handle space exhausted")
            })?);
        self.objects.borrow_mut().insert(handle, object);
        Ok(PythonValue::Object(handle))
    }

    pub(super) fn tuple_values(
        &self,
        tuple: *mut PyObject,
        operation: &str,
    ) -> Result<Vec<PythonValue>, PythonError> {
        let length = unsafe { (self.api.py_tuple_size)(tuple) };
        let length =
            usize::try_from(length).map_err(|_| unsafe { capture_error(&self.api, operation) })?;
        let mut values = Vec::with_capacity(length);
        for index in 0..length {
            let item = unsafe { (self.api.py_tuple_get_item)(tuple, index as PySsize) };
            let item = unsafe { checked(&self.api, item, operation)? };
            self.incref(item);
            values.push(self.convert_owned(item)?);
        }
        Ok(values)
    }

    pub(super) fn array_to_python(&self, array: PythonArray) -> Result<*mut PyObject, PythonError> {
        let elements = array
            .shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
        let expected_bytes = elements.and_then(|count| count.checked_mul(array.dtype.byte_width()));
        if expected_bytes != Some(array.owner.byte_length()) {
            return Err(PythonError::host(
                "PythonArrayConversionError",
                format!(
                    "{:?} array shape {:?} does not match its {}-byte owner",
                    array.dtype,
                    array.shape,
                    array.owner.byte_length()
                ),
            ));
        }
        let stride_values = array_strides(&array)?;
        let interface = unsafe {
            checked(
                &self.api,
                (self.api.py_dict_new)(),
                "create Python array interface",
            )?
        };
        let shape = PythonValue::Tuple(
            array
                .shape
                .iter()
                .map(|dimension| PythonValue::Unsigned(*dimension as u64))
                .collect(),
        );
        let strides = PythonValue::Tuple(
            stride_values
                .into_iter()
                .map(|stride| PythonValue::Unsigned(stride as u64))
                .collect(),
        );
        let data = PythonValue::Tuple(vec![
            PythonValue::Unsigned(array.owner.address() as u64),
            PythonValue::Bool(array.read_only),
        ]);
        for (name, value) in [
            ("shape", shape),
            ("strides", strides),
            (
                "typestr",
                PythonValue::String(array.dtype.numpy_typestr().into()),
            ),
            ("data", data),
            ("version", PythonValue::Signed(3)),
        ] {
            let value = match self.to_owned(value) {
                Ok(value) => value,
                Err(error) => {
                    self.decref(interface);
                    return Err(error);
                }
            };
            let result = unsafe { set_dict_string(&self.api, interface, name, value) };
            self.decref(value);
            if let Err(error) = result {
                self.decref(interface);
                return Err(error);
            }
        }
        let capsule = match self.api.owner_capsule(Arc::clone(&array.owner)) {
            Ok(capsule) => capsule,
            Err(error) => {
                self.decref(interface);
                return Err(error);
            }
        };
        let arguments = match unsafe {
            checked(
                &self.api,
                (self.api.py_tuple_new)(2),
                "create Python buffer-view arguments",
            )
        } {
            Ok(arguments) => arguments,
            Err(error) => {
                self.decref(capsule);
                self.decref(interface);
                return Err(error);
            }
        };
        let first = unsafe { (self.api.py_tuple_set_item)(arguments, 0, interface) };
        if let Err(error) = self.status(first, "attach Python array interface") {
            self.decref(capsule);
            self.decref(arguments);
            return Err(error);
        }
        let second = unsafe { (self.api.py_tuple_set_item)(arguments, 1, capsule) };
        if let Err(error) = self.status(second, "attach Python buffer owner") {
            self.decref(arguments);
            return Err(error);
        }
        let wrapper = unsafe {
            (self.api.py_object_call)(self.builtins.buffer_view, arguments, std::ptr::null_mut())
        };
        self.decref(arguments);
        let wrapper = unsafe { checked(&self.api, wrapper, "create Python buffer view")? };

        let numpy = unsafe { (self.api.py_import_import_module)(c"numpy".as_ptr()) };
        if numpy.is_null() {
            unsafe { (self.api.py_err_clear)() };
            return Ok(wrapper);
        }
        let asarray = match unsafe { attr(&self.api, numpy, "asarray") } {
            Ok(asarray) => asarray,
            Err(error) => {
                self.decref(numpy);
                self.decref(wrapper);
                return Err(error);
            }
        };
        self.decref(numpy);
        let arguments = unsafe {
            checked(
                &self.api,
                (self.api.py_tuple_new)(1),
                "create NumPy conversion arguments",
            )?
        };
        let status = unsafe { (self.api.py_tuple_set_item)(arguments, 0, wrapper) };
        self.status(status, "attach RunMat buffer view")?;
        let converted =
            unsafe { (self.api.py_object_call)(asarray, arguments, std::ptr::null_mut()) };
        self.decref(arguments);
        self.decref(asarray);
        unsafe { checked(&self.api, converted, "create NumPy array view") }
    }

    pub(super) fn array_from_python(
        &self,
        object: *mut PyObject,
    ) -> Result<Option<PythonArray>, PythonError> {
        let interface = match unsafe { attr(&self.api, object, "__array_interface__") } {
            Ok(interface) => interface,
            Err(_) => {
                unsafe { (self.api.py_err_clear)() };
                return Ok(None);
            }
        };
        self.decref(interface);
        let dtype = unsafe { attr(&self.api, object, "dtype")? };
        let typestr = unsafe { attr(&self.api, dtype, "str")? };
        self.decref(dtype);
        let typestr_text = unsafe { unicode(&self.api, typestr) };
        self.decref(typestr);
        let dtype = parse_numpy_dtype(&typestr_text?)?;

        let shape_object = unsafe { attr(&self.api, object, "shape")? };
        let shape = self.tuple_values(shape_object, "read Python array shape");
        self.decref(shape_object);
        let shape = shape?;
        let shape = shape
            .into_iter()
            .map(|dimension| match dimension {
                PythonValue::Signed(value) => usize::try_from(value).ok(),
                PythonValue::Unsigned(value) => usize::try_from(value).ok(),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                PythonError::host(
                    "PythonArrayConversionError",
                    "Python array shape contains an invalid dimension",
                )
            })?;

        let flags = unsafe { attr(&self.api, object, "flags")? };
        let fortran = unsafe { attr(&self.api, flags, "f_contiguous")? };
        self.decref(flags);
        let column_major = unsafe { (self.api.py_object_is_true)(fortran) };
        self.decref(fortran);
        if column_major < 0 {
            return Err(unsafe { capture_error(&self.api, "read Python array layout") });
        }

        let tobytes = unsafe { attr(&self.api, object, "tobytes")? };
        // RunMat stores dense arrays in column-major order. NumPy performs an
        // explicit reorder here when the source is C-contiguous.
        let arguments = self.values_to_tuple(vec![PythonValue::String("F".into())])?;
        let bytes = unsafe { (self.api.py_object_call)(tobytes, arguments, std::ptr::null_mut()) };
        self.decref(arguments);
        self.decref(tobytes);
        let bytes = unsafe { checked(&self.api, bytes, "copy Python array payload")? };
        let bytes = self.convert_owned(bytes)?;
        let PythonValue::Bytes(bytes) = bytes else {
            return Err(PythonError::host(
                "PythonArrayConversionError",
                "Python array tobytes returned a non-bytes value",
            ));
        };
        Ok(Some(PythonArray {
            dtype,
            shape,
            column_major: true,
            read_only: false,
            owner: Arc::new(OwnedPythonBytes(bytes)),
        }))
    }
}
