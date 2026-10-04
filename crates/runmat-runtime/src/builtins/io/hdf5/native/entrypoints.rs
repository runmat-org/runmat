use super::*;

pub fn read_dataset(
    builtin: &'static str,
    path: PathBuf,
    dataset_path: String,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let file = open_file_read(builtin, &path)?;
    let ds = file.dataset(&dataset_path).map_err(|err| {
        io_error(
            builtin,
            format!("{builtin}: unable to open dataset {dataset_path}"),
            err,
        )
    })?;
    let shape = ds.shape();
    let selection = parse_selection(&rest, builtin, &shape)?;
    read_dataset_value(builtin, &ds, selection.as_ref())
}

pub fn write_dataset(
    builtin: &'static str,
    path: PathBuf,
    dataset_path: String,
    data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<()> {
    let file = File::append(&path).map_err(|err| {
        io_error(
            builtin,
            format!("{builtin}: unable to open {}", path.display()),
            err,
        )
    })?;
    if !rest.is_empty() {
        return write_dataset_selection(builtin, &file, &dataset_path, data, rest);
    }
    let data = WritableData::from_value(builtin, data)?;
    if file.link_exists(&dataset_path) {
        let ds = file.dataset(&dataset_path).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to open existing dataset {dataset_path}"),
                err,
            )
        })?;
        return write_existing_dataset(builtin, &ds, &dataset_path, data);
    }
    match data {
        WritableData::Numeric { data, shape } => {
            let row_major = col_major_to_row_major(&data, &shape);
            let ds = file
                .new_dataset::<f64>()
                .shape(shape.as_slice())
                .create_intermediate_group(true)
                .create(dataset_path.as_str())
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
            })?;
        }
        WritableData::Single { data, shape } => {
            create_typed_dataset::<f32>(builtin, &file, &dataset_path, data, &shape)?;
        }
        WritableData::Integer { storage, shape } => {
            create_integer_dataset(builtin, &file, &dataset_path, storage, &shape)?;
        }
        WritableData::Logical { data, shape } => {
            let row_major = col_major_to_row_major(&data, &shape);
            let ds = file
                .new_dataset::<bool>()
                .shape(shape.as_slice())
                .create_intermediate_group(true)
                .create(dataset_path.as_str())
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
            })?;
        }
        WritableData::Strings { data, shape } => {
            let row_major = col_major_to_row_major(&data, &shape);
            let encoded = encode_strings(builtin, &row_major)?;
            let ds = file
                .new_dataset::<VarLenUnicode>()
                .shape(shape.as_slice())
                .create_intermediate_group(true)
                .create(dataset_path.as_str())
                .map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to create dataset {dataset_path}"),
                        err,
                    )
                })?;
            ds.write_raw(&encoded).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write dataset {dataset_path}"),
                    err,
                )
            })?;
        }
    }
    Ok(())
}

fn write_existing_dataset(
    builtin: &'static str,
    ds: &Dataset,
    dataset_path: &str,
    data: WritableData,
) -> BuiltinResult<()> {
    match data {
        WritableData::Numeric { data, shape } => {
            validate_full_write_shape(builtin, &shape, &ds.shape())?;
            let row_major = col_major_to_row_major(&data, &shape);
            ds.write_raw(&row_major).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write existing dataset {dataset_path}"),
                    err,
                )
            })
        }
        WritableData::Single { data, shape } => {
            validate_full_write_shape(builtin, &shape, &ds.shape())?;
            let row_major = col_major_to_row_major(&data, &shape);
            ds.write_raw(&row_major).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write existing dataset {dataset_path}"),
                    err,
                )
            })
        }
        WritableData::Integer { storage, shape } => {
            validate_full_write_shape(builtin, &shape, &ds.shape())?;
            write_existing_integer_dataset(builtin, ds, dataset_path, storage, &shape)
        }
        WritableData::Logical { data, shape } => {
            validate_full_write_shape(builtin, &shape, &ds.shape())?;
            let row_major = col_major_to_row_major(&data, &shape);
            ds.write_raw(&row_major).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write existing dataset {dataset_path}"),
                    err,
                )
            })
        }
        WritableData::Strings { data, shape } => {
            validate_full_write_shape(builtin, &shape, &ds.shape())?;
            let row_major = col_major_to_row_major(&data, &shape);
            let encoded = encode_strings(builtin, &row_major)?;
            ds.write_raw(&encoded).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write existing dataset {dataset_path}"),
                    err,
                )
            })
        }
    }
}

fn write_dataset_selection(
    builtin: &'static str,
    file: &File,
    dataset_path: &str,
    data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<()> {
    let ds = file.dataset(dataset_path).map_err(|err| {
        io_error(
            builtin,
            format!("{builtin}: unable to open dataset {dataset_path}"),
            err,
        )
    })?;
    let shape = ds.shape();
    let selection = parse_selection(&rest, builtin, &shape)?.ok_or_else(|| {
        hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: expected start/count or start/count/stride selection"),
        )
    })?;
    match WritableData::from_value(builtin, data)? {
        WritableData::Numeric { data, shape } => {
            validate_write_selection_shape(builtin, &shape, &selection.count)?;
            let row_major = col_major_to_row_major(&data, &shape);
            write_hyperslab::<f64>(builtin, &ds, &selection, row_major)
        }
        WritableData::Single { data, shape } => {
            validate_write_selection_shape(builtin, &shape, &selection.count)?;
            let row_major = col_major_to_row_major(&data, &shape);
            write_hyperslab::<f32>(builtin, &ds, &selection, row_major)
        }
        WritableData::Integer { storage, shape } => {
            validate_write_selection_shape(builtin, &shape, &selection.count)?;
            write_integer_hyperslab(builtin, &ds, &selection, storage, &shape)
        }
        WritableData::Logical { data, shape } => {
            validate_write_selection_shape(builtin, &shape, &selection.count)?;
            let row_major = col_major_to_row_major(&data, &shape);
            write_hyperslab::<bool>(builtin, &ds, &selection, row_major)
        }
        WritableData::Strings { data, shape } => {
            validate_write_selection_shape(builtin, &shape, &selection.count)?;
            let row_major = col_major_to_row_major(&data, &shape);
            let encoded = encode_strings(builtin, &row_major)?;
            write_hyperslab::<VarLenUnicode>(builtin, &ds, &selection, encoded)
        }
    }
}

pub fn write_attribute(
    builtin: &'static str,
    path: PathBuf,
    location: String,
    attribute: String,
    data: Value,
) -> BuiltinResult<()> {
    let file = File::append(&path).map_err(|err| {
        io_error(
            builtin,
            format!("{builtin}: unable to open {}", path.display()),
            err,
        )
    })?;
    if location == "/" {
        write_attr_to_location(builtin, &file, &attribute, data)
    } else if let Ok(ds) = file.dataset(&location) {
        write_attr_to_location(builtin, &ds, &attribute, data)
    } else if let Ok(group) = file.group(&location) {
        write_attr_to_location(builtin, &group, &attribute, data)
    } else {
        Err(hdf5_error(
            builtin,
            &ERR_IO,
            format!("{builtin}: location {location} does not exist"),
        ))
    }
}

pub fn info(builtin: &'static str, path: PathBuf, location: String) -> BuiltinResult<Value> {
    let file = open_file_read(builtin, &path)?;
    let info = if location == "/" {
        let group = file.group("/").map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to open root group"),
                err,
            )
        })?;
        group_info(builtin, &group, "/", Some(path.as_path()))?
    } else if let Ok(ds) = file.dataset(&location) {
        dataset_info(builtin, &ds, &location)?
    } else {
        let group = file.group(&location).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: location {location} does not exist"),
                err,
            )
        })?;
        group_info(builtin, &group, &location, Some(path.as_path()))?
    };
    Ok(Value::Struct(info))
}

pub fn disp(builtin: &'static str, path: PathBuf, location: String) -> BuiltinResult<String> {
    let file = open_file_read(builtin, &path)?;
    let mut out = String::new();
    out.push_str(&format!("HDF5 {location}\n"));
    if location == "/" {
        let group = file.group("/").map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to open root group"),
                err,
            )
        })?;
        render_group(builtin, &group, "/", 0, &mut out)?;
    } else if let Ok(ds) = file.dataset(&location) {
        render_dataset(builtin, &ds, &location, 0, &mut out)?;
    } else {
        let group = file.group(&location).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: location {location} does not exist"),
                err,
            )
        })?;
        render_group(builtin, &group, &location, 0, &mut out)?;
    }
    Ok(out)
}
