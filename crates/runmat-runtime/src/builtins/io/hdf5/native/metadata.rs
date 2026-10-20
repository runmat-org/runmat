use super::*;

pub(super) fn group_info(
    builtin: &'static str,
    group: &Group,
    name: &str,
    filename: Option<&Path>,
) -> BuiltinResult<StructValue> {
    let mut info = runmat_value::UnboundedStructValueBuilder::new();
    if let Some(path) = filename {
        info.insert("Filename", Value::string(path.display().to_string()));
    }
    info.insert("Name", Value::string(name.to_string()));
    info.insert("Attributes", attributes_info(builtin, group)?);

    let mut groups = Vec::new();
    let mut datasets = Vec::new();
    for member in group.member_names().map_err(|err| {
        io_error(
            builtin,
            format!("{builtin}: unable to list group {name}"),
            err,
        )
    })? {
        let child_path = join_hdf5_path(name, &member);
        if let Ok(child) = group.group(&member) {
            groups.push(Value::Struct(group_info(
                builtin,
                &child,
                &child_path,
                None,
            )?));
        } else if let Ok(ds) = group.dataset(&member) {
            datasets.push(Value::Struct(dataset_info(builtin, &ds, &child_path)?));
        }
    }
    info.insert("Groups", cell_column(groups, builtin)?);
    info.insert("Datasets", cell_column(datasets, builtin)?);
    info.insert("Datatypes", cell_column(Vec::new(), builtin)?);
    info.insert("Links", cell_column(Vec::new(), builtin)?);
    Ok(info.finish())
}

pub(super) fn dataset_info(
    builtin: &'static str,
    ds: &Dataset,
    name: &str,
) -> BuiltinResult<StructValue> {
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
    let mut info = runmat_value::UnboundedStructValueBuilder::new();
    info.insert("Name", Value::string(name.to_string()));
    info.insert("Datatype", Value::string(type_name(&dtype)));
    info.insert("Class", Value::string(matlab_class_for_type(&dtype)));
    info.insert(
        "Dataspace",
        Value::string(if shape.is_empty() { "scalar" } else { "simple" }.to_string()),
    );
    info.insert("Size", shape_value(shape)?);
    info.insert("Attributes", attributes_info(builtin, ds)?);
    Ok(info.finish())
}

pub(super) fn attributes_info(builtin: &'static str, location: &Location) -> BuiltinResult<Value> {
    let mut values = Vec::new();
    for name in location.attr_names().map_err(|err| {
        io_error(
            builtin,
            format!("{builtin}: unable to list attributes"),
            err,
        )
    })? {
        let attr = location.attr(&name).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to open attribute {name}"),
                err,
            )
        })?;
        values.push(Value::Struct(attribute_info(builtin, &attr, &name)?));
    }
    cell_column(values, builtin)
}

pub(super) fn attribute_info(
    builtin: &'static str,
    attr: &Attribute,
    name: &str,
) -> BuiltinResult<StructValue> {
    let dtype = attr
        .dtype()
        .and_then(|dt| dt.to_descriptor())
        .map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to inspect attribute type"),
                err,
            )
        })?;
    let mut info = runmat_value::UnboundedStructValueBuilder::new();
    info.insert("Name", Value::string(name.to_string()));
    info.insert("Datatype", Value::string(type_name(&dtype)));
    info.insert("Class", Value::string(matlab_class_for_type(&dtype)));
    info.insert("Size", shape_value(attr.shape())?);
    Ok(info.finish())
}

pub(super) fn render_group(
    builtin: &'static str,
    group: &Group,
    name: &str,
    indent: usize,
    mode: h5disp_controls::H5DispMode,
    out: &mut String,
) -> BuiltinResult<()> {
    out.push_str(&format!("{}Group: {name}\n", " ".repeat(indent)));
    if mode == h5disp_controls::H5DispMode::Simple {
        render_attributes(builtin, group, indent + 2, out)?;
    }
    for member in group.member_names().map_err(|err| {
        io_error(
            builtin,
            format!("{builtin}: unable to list group {name}"),
            err,
        )
    })? {
        let child_path = join_hdf5_path(name, &member);
        if let Ok(child) = group.group(&member) {
            render_group(builtin, &child, &child_path, indent + 2, mode, out)?;
        } else if let Ok(ds) = group.dataset(&member) {
            render_dataset(builtin, &ds, &child_path, indent + 2, mode, out)?;
        }
    }
    Ok(())
}

pub(super) fn render_dataset(
    builtin: &'static str,
    ds: &Dataset,
    name: &str,
    indent: usize,
    mode: h5disp_controls::H5DispMode,
    out: &mut String,
) -> BuiltinResult<()> {
    if mode == h5disp_controls::H5DispMode::Min {
        out.push_str(&format!("{}Dataset: {name}\n", " ".repeat(indent)));
        return Ok(());
    }
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
    out.push_str(&format!(
        "{}Dataset: {name} [{}] {}\n",
        " ".repeat(indent),
        ds.shape()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("x"),
        type_name(&dtype)
    ));
    render_attributes(builtin, ds, indent + 2, out)
}

pub(super) fn render_attributes(
    builtin: &'static str,
    location: &Location,
    indent: usize,
    out: &mut String,
) -> BuiltinResult<()> {
    for name in location.attr_names().map_err(|err| {
        io_error(
            builtin,
            format!("{builtin}: unable to list attributes"),
            err,
        )
    })? {
        let attr = location.attr(&name).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to open attribute {name}"),
                err,
            )
        })?;
        let dtype = attr
            .dtype()
            .and_then(|dt| dt.to_descriptor())
            .map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to inspect attribute type"),
                    err,
                )
            })?;
        out.push_str(&format!(
            "{}Attribute: {name} [{}] {}",
            " ".repeat(indent),
            attr.shape()
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            type_name(&dtype)
        ));
        if let Some(value) = scalar_attribute_text(builtin, &attr, &dtype)? {
            out.push_str(&format!(" Value: {value}"));
        }
        out.push('\n');
    }
    Ok(())
}
