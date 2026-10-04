use super::*;

pub(super) fn read_dataset_value(
    builtin: &'static str,
    ds: &Dataset,
    selection: Option<&MatlabSelection>,
) -> BuiltinResult<Value> {
    let shape = ds.shape();
    let dtype = ds
        .dtype()
        .and_then(|dt| dt.to_descriptor())
        .map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to inspect dataset type"),
                err,
            )
        })?;
    match dtype {
        TypeDescriptor::Float(FloatSize::U4) => {
            let (raw, out_shape) = read_raw_or_slice::<f32>(builtin, ds, &shape, selection)?;
            let data = row_major_to_col_major(&raw, &out_shape);
            Tensor::from_f32(data, out_shape)
                .map(Value::Tensor)
                .map_err(|error| hdf5_error(builtin, &ERR_IO, format!("{builtin}: {error}")))
        }
        TypeDescriptor::Float(FloatSize::U8) => {
            let (raw, out_shape) = read_raw_or_slice::<f64>(builtin, ds, &shape, selection)?;
            matlab_value_from_f64(row_major_to_col_major(&raw, &out_shape), out_shape)
        }
        TypeDescriptor::Integer(IntSize::U1) => {
            read_integer_dataset::<i8, _>(builtin, ds, &shape, selection, IntegerStorage::I8)
        }
        TypeDescriptor::Integer(IntSize::U2) => {
            read_integer_dataset::<i16, _>(builtin, ds, &shape, selection, IntegerStorage::I16)
        }
        TypeDescriptor::Integer(IntSize::U4) => {
            read_integer_dataset::<i32, _>(builtin, ds, &shape, selection, IntegerStorage::I32)
        }
        TypeDescriptor::Integer(IntSize::U8) => {
            read_integer_dataset::<i64, _>(builtin, ds, &shape, selection, IntegerStorage::I64)
        }
        TypeDescriptor::Unsigned(IntSize::U1) => {
            read_integer_dataset::<u8, _>(builtin, ds, &shape, selection, IntegerStorage::U8)
        }
        TypeDescriptor::Unsigned(IntSize::U2) => {
            read_integer_dataset::<u16, _>(builtin, ds, &shape, selection, IntegerStorage::U16)
        }
        TypeDescriptor::Unsigned(IntSize::U4) => {
            read_integer_dataset::<u32, _>(builtin, ds, &shape, selection, IntegerStorage::U32)
        }
        TypeDescriptor::Unsigned(IntSize::U8) => {
            read_integer_dataset::<u64, _>(builtin, ds, &shape, selection, IntegerStorage::U64)
        }
        TypeDescriptor::Boolean => {
            let (raw, out_shape) = read_raw_or_slice::<bool>(builtin, ds, &shape, selection)?;
            matlab_value_from_bool(row_major_to_col_major(&raw, &out_shape), out_shape)
        }
        TypeDescriptor::VarLenUnicode => {
            let (raw, out_shape) =
                read_raw_or_slice::<VarLenUnicode>(builtin, ds, &shape, selection)?;
            let strings: Vec<String> = raw.into_iter().map(|s| s.as_str().to_string()).collect();
            matlab_value_from_strings(row_major_to_col_major(&strings, &out_shape), out_shape)
        }
        TypeDescriptor::VarLenAscii => {
            let (raw, out_shape) =
                read_raw_or_slice::<VarLenAscii>(builtin, ds, &shape, selection)?;
            let strings: Vec<String> = raw.into_iter().map(|s| s.as_str().to_string()).collect();
            matlab_value_from_strings(row_major_to_col_major(&strings, &out_shape), out_shape)
        }
        other => Err(hdf5_error(
            builtin,
            &ERR_UNSUPPORTED,
            format!(
                "{builtin}: unsupported HDF5 dataset type {}",
                type_name(&other)
            ),
        )),
    }
}

fn read_integer_dataset<T, F>(
    builtin: &'static str,
    ds: &Dataset,
    shape: &[usize],
    selection: Option<&MatlabSelection>,
    convert: F,
) -> BuiltinResult<Value>
where
    T: hdf5::H5Type + Copy,
    F: Fn(Vec<T>) -> IntegerStorage,
{
    let (raw, out_shape) = read_raw_or_slice::<T>(builtin, ds, shape, selection)?;
    matlab_value_from_integer(convert(row_major_to_col_major(&raw, &out_shape)), out_shape)
}

fn read_raw_or_slice<T>(
    builtin: &'static str,
    ds: &Dataset,
    shape: &[usize],
    selection: Option<&MatlabSelection>,
) -> BuiltinResult<(Vec<T>, Vec<usize>)>
where
    T: hdf5::H5Type,
{
    if let Some(selection) = selection {
        let hyperslab = selection.to_hdf5_hyperslab();
        let array = ds
            .as_reader()
            .read_slice::<T, _, ndarray::IxDyn>(hdf5::Selection::Hyperslab(hyperslab))
            .map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to read HDF5 hyperslab"),
                    err,
                )
            })?;
        // An ndarray slice may retain a buffer that starts before the first
        // selected element; its logical index [0, 0, ...] can therefore be at
        // a non-zero element offset in that buffer. Consume the array in
        // logical row-major order so the MATLAB conversion below receives
        // exactly the selected hyperslab, never unrelated buffer elements.
        Ok((array.into_iter().collect(), selection.count.clone()))
    } else {
        let raw = ds.read_raw::<T>().map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to read HDF5 dataset"),
                err,
            )
        })?;
        Ok((raw, shape.to_vec()))
    }
}
