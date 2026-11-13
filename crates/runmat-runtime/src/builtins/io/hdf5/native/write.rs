use super::*;

pub(super) enum WritableData {
    Numeric {
        data: Vec<f64>,
        shape: Vec<usize>,
    },
    Single {
        data: Vec<f32>,
        shape: Vec<usize>,
    },
    Integer {
        storage: IntegerStorage,
        shape: Vec<usize>,
    },
    Logical {
        data: Vec<bool>,
        shape: Vec<usize>,
    },
    Strings {
        data: Vec<VarLenUnicode>,
        shape: Vec<usize>,
    },
}

impl WritableData {
    pub(super) fn from_value(builtin: &'static str, value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(v) => Ok(Self::Numeric {
                data: vec![v],
                shape: vec![1, 1],
            }),
            Value::Int(v) => Ok(Self::Integer {
                storage: integer_storage_from_scalar(v),
                shape: vec![1, 1],
            }),
            Value::Tensor(t) => {
                let shape = t.shape().to_vec();
                let storage = t.into_numeric_storage().map_err(|error| {
                    hdf5_error(builtin, &ERR_UNSUPPORTED, format!("{builtin}: {error}"))
                })?;
                match storage {
                    NumericStorage::F64(data) => Ok(Self::Numeric { data, shape }),
                    NumericStorage::F32(data) => Ok(Self::Single { data, shape }),
                    storage => Ok(Self::Integer {
                        storage: storage
                            .into_integer_storage()
                            .expect("non-floating numeric storage is integer"),
                        shape,
                    }),
                }
            }
            Value::Bool(v) => Ok(Self::Logical {
                data: vec![v],
                shape: vec![1, 1],
            }),
            Value::LogicalArray(array) => Ok(Self::Logical {
                data: array.data.into_iter().map(|v| v != 0).collect(),
                shape: array.shape.to_vec(),
            }),
            Value::String(runmat_value::StringElement::Text(text)) => Ok(Self::Strings {
                data: encode_strings(builtin, &[text])?,
                shape: vec![1, 1],
            }),
            Value::StringArray(array) => Ok(Self::Strings {
                data: encode_strings(builtin, &array.data().iter().map(|value| value.as_str().ok_or_else(|| hdf5_error(builtin, &ERR_UNSUPPORTED, format!("{builtin}: HDF5 string data cannot represent missing strings")))).collect::<BuiltinResult<Vec<_>>>()?)?,
                shape: array.shape.to_vec(),
            }),
            Value::CharArray(chars) => Ok(Self::Strings {
                data: encode_strings(builtin, &char_rows(builtin, &chars)?)?,
                shape: vec![chars.rows.max(1), 1],
            }),
            _ => Err(hdf5_error(
                builtin,
                &ERR_UNSUPPORTED,
                format!("{builtin}: only dense numeric, logical, string, and char data can be written to HDF5"),
            )),
        }
    }
}

pub(super) fn create_integer_dataset(
    builtin: &'static str,
    file: &File,
    dataset_path: &str,
    storage: IntegerStorage,
    shape: &[usize],
) -> BuiltinResult<()> {
    macro_rules! create {
        ($values:expr, $ty:ty) => {
            create_typed_dataset::<$ty>(builtin, file, dataset_path, $values, shape)
        };
    }
    match storage {
        IntegerStorage::I8(values) => create!(values, i8),
        IntegerStorage::I16(values) => create!(values, i16),
        IntegerStorage::I32(values) => create!(values, i32),
        IntegerStorage::I64(values) => create!(values, i64),
        IntegerStorage::U8(values) => create!(values, u8),
        IntegerStorage::U16(values) => create!(values, u16),
        IntegerStorage::U32(values) => create!(values, u32),
        IntegerStorage::U64(values) => create!(values, u64),
    }
}

pub(super) fn create_typed_dataset<T>(
    builtin: &'static str,
    file: &File,
    dataset_path: &str,
    data: Vec<T>,
    shape: &[usize],
) -> BuiltinResult<()>
where
    T: hdf5::H5Type + Clone,
{
    let row_major = col_major_to_row_major(&data, shape);
    let ds = file
        .new_dataset::<T>()
        .shape(shape)
        .create_intermediate_group(true)
        .create(dataset_path)
        .map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to create dataset {dataset_path}"),
                err,
            )
        })?;
    ds.write_raw(&row_major).map_err(|err| {
        io_error(
            builtin,
            format!("{builtin}: unable to write dataset {dataset_path}"),
            err,
        )
    })
}

pub(super) fn write_existing_integer_dataset(
    builtin: &'static str,
    ds: &Dataset,
    dataset_path: &str,
    storage: IntegerStorage,
    shape: &[usize],
) -> BuiltinResult<()> {
    macro_rules! write {
        ($values:expr) => {{
            let row_major = col_major_to_row_major(&$values, shape);
            ds.write_raw(&row_major).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write existing dataset {dataset_path}"),
                    err,
                )
            })
        }};
    }
    match storage {
        IntegerStorage::I8(values) => write!(values),
        IntegerStorage::I16(values) => write!(values),
        IntegerStorage::I32(values) => write!(values),
        IntegerStorage::I64(values) => write!(values),
        IntegerStorage::U8(values) => write!(values),
        IntegerStorage::U16(values) => write!(values),
        IntegerStorage::U32(values) => write!(values),
        IntegerStorage::U64(values) => write!(values),
    }
}

pub(super) fn write_integer_hyperslab(
    builtin: &'static str,
    ds: &Dataset,
    selection: &MatlabSelection,
    storage: IntegerStorage,
    shape: &[usize],
) -> BuiltinResult<()> {
    macro_rules! write {
        ($values:expr, $ty:ty) => {{
            let row_major = col_major_to_row_major(&$values, shape);
            write_hyperslab::<$ty>(builtin, ds, selection, row_major)
        }};
    }
    match storage {
        IntegerStorage::I8(values) => write!(values, i8),
        IntegerStorage::I16(values) => write!(values, i16),
        IntegerStorage::I32(values) => write!(values, i32),
        IntegerStorage::I64(values) => write!(values, i64),
        IntegerStorage::U8(values) => write!(values, u8),
        IntegerStorage::U16(values) => write!(values, u16),
        IntegerStorage::U32(values) => write!(values, u32),
        IntegerStorage::U64(values) => write!(values, u64),
    }
}

pub(super) fn validate_write_selection_shape(
    builtin: &'static str,
    data_shape: &[usize],
    count: &[usize],
) -> BuiltinResult<()> {
    if data_shape == count {
        return Ok(());
    }
    if checked_shape_len(data_shape, builtin)? == 1 && checked_shape_len(count, builtin)? == 1 {
        return Ok(());
    }
    Err(hdf5_error(
        builtin,
        &ERR_ARGUMENT,
        format!("{builtin}: data shape must match the selected HDF5 count"),
    ))
}

pub(super) fn validate_full_write_shape(
    builtin: &'static str,
    data_shape: &[usize],
    dataset_shape: &[usize],
) -> BuiltinResult<()> {
    if data_shape == dataset_shape {
        return Ok(());
    }
    Err(hdf5_error(
        builtin,
        &ERR_ARGUMENT,
        format!(
            "{builtin}: data shape {data_shape:?} must match existing HDF5 dataset shape {dataset_shape:?}"
        ),
    ))
}

pub(super) fn write_hyperslab<T>(
    builtin: &'static str,
    ds: &Dataset,
    selection: &MatlabSelection,
    data: Vec<T>,
) -> BuiltinResult<()>
where
    T: hdf5::H5Type,
{
    let array =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&selection.count), data).map_err(|err| {
            hdf5_error(
                builtin,
                &ERR_ARGUMENT,
                format!("{builtin}: unable to shape selected write data: {err}"),
            )
        })?;
    ds.as_writer()
        .write_slice(
            array.view(),
            hdf5::Selection::Hyperslab(selection.to_hdf5_hyperslab()),
        )
        .map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to write HDF5 hyperslab"),
                err,
            )
        })
}

pub(super) fn write_attr_to_location(
    builtin: &'static str,
    location: &Location,
    name: &str,
    data: WritableData,
) -> BuiltinResult<()> {
    if location
        .attr_names()
        .map(|names| names.iter().any(|item| item == name))
        .unwrap_or(false)
    {
        location.delete_attr(name).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to replace attribute {name}"),
                err,
            )
        })?;
    }
    match data {
        WritableData::Numeric { data, shape } => {
            let row_major = col_major_to_row_major(&data, &shape);
            let attr = location
                .new_attr::<f64>()
                .shape(shape.as_slice())
                .create(name)
                .map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to create attribute {name}"),
                        err,
                    )
                })?;
            attr.write_raw(&row_major).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write attribute {name}"),
                    err,
                )
            })?;
        }
        WritableData::Single { data, shape } => {
            let row_major = col_major_to_row_major(&data, &shape);
            let attr = location
                .new_attr::<f32>()
                .shape(shape.as_slice())
                .create(name)
                .map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to create attribute {name}"),
                        err,
                    )
                })?;
            attr.write_raw(&row_major).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write attribute {name}"),
                    err,
                )
            })?;
        }
        WritableData::Integer { storage, shape } => {
            write_integer_attribute(builtin, location, name, storage, &shape)?;
        }
        WritableData::Logical { data, shape } => {
            let row_major = col_major_to_row_major(&data, &shape);
            let attr = location
                .new_attr::<bool>()
                .shape(shape.as_slice())
                .create(name)
                .map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to create attribute {name}"),
                        err,
                    )
                })?;
            attr.write_raw(&row_major).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write attribute {name}"),
                    err,
                )
            })?;
        }
        WritableData::Strings { data, shape } => {
            let encoded = col_major_to_row_major(&data, &shape);
            let attr = location
                .new_attr::<VarLenUnicode>()
                .shape(shape.as_slice())
                .create(name)
                .map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to create attribute {name}"),
                        err,
                    )
                })?;
            attr.write_raw(&encoded).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write attribute {name}"),
                    err,
                )
            })?;
        }
    }
    Ok(())
}

fn write_integer_attribute(
    builtin: &'static str,
    location: &Location,
    name: &str,
    storage: IntegerStorage,
    shape: &[usize],
) -> BuiltinResult<()> {
    macro_rules! write {
        ($values:expr, $ty:ty) => {{
            let row_major = col_major_to_row_major(&$values, shape);
            let attr = location
                .new_attr::<$ty>()
                .shape(shape)
                .create(name)
                .map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to create attribute {name}"),
                        err,
                    )
                })?;
            attr.write_raw(&row_major).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write attribute {name}"),
                    err,
                )
            })
        }};
    }
    match storage {
        IntegerStorage::I8(values) => write!(values, i8),
        IntegerStorage::I16(values) => write!(values, i16),
        IntegerStorage::I32(values) => write!(values, i32),
        IntegerStorage::I64(values) => write!(values, i64),
        IntegerStorage::U8(values) => write!(values, u8),
        IntegerStorage::U16(values) => write!(values, u16),
        IntegerStorage::U32(values) => write!(values, u32),
        IntegerStorage::U64(values) => write!(values, u64),
    }
}
