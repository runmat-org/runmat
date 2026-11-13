use std::str::FromStr;

use hdf5::types::{FloatSize, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::{Attribute, Dataset, File, Group, Hyperslab, Location, SliceOrIndex};
use runmat_value::{CharArray, StructValue};

use super::*;

impl MatlabSelection {
    fn to_hdf5_hyperslab(&self) -> Hyperslab {
        Hyperslab::new(
            self.start
                .iter()
                .zip(&self.count)
                .zip(&self.stride)
                .map(|((&start, &count), &step)| SliceOrIndex::SliceCount {
                    start,
                    step,
                    count,
                    block: 1,
                })
                .collect::<Vec<_>>(),
        )
    }
}

mod entrypoints;
pub(super) use entrypoints::{disp, info, read_dataset, write_attribute, write_dataset};

mod read;
mod write;
use read::read_dataset_value;
use write::{
    create_integer_dataset, create_typed_dataset, validate_full_write_shape,
    validate_write_selection_shape, write_attr_to_location, write_existing_integer_dataset,
    write_hyperslab, write_integer_hyperslab, WritableData,
};

fn open_file_read(builtin: &'static str, path: &Path) -> BuiltinResult<File> {
    File::open(path).map_err(|err| {
        io_error(
            builtin,
            format!("{builtin}: unable to open {}", path.display()),
            err,
        )
    })
}

mod metadata;
use metadata::{dataset_info, group_info, render_dataset, render_group};
mod attribute_values;
use attribute_values::scalar_attribute_text;

fn char_rows(builtin: &'static str, chars: &CharArray) -> BuiltinResult<Vec<String>> {
    if chars.rows == 0 || chars.cols == 0 {
        return Ok(vec![String::new()]);
    }
    (0..chars.rows)
        .map(|row| {
            let start = row * chars.cols;
            crate::builtins::io::utf16::decode_utf16_with_error(
                &chars.data()[start..start + chars.cols],
                || {
                    hdf5_error(
                        builtin,
                        &ERR_ARGUMENT,
                        format!("{builtin}: character input contains malformed UTF-16 text"),
                    )
                },
            )
        })
        .collect()
}

fn encode_strings(
    builtin: &'static str,
    values: &[impl AsRef<str>],
) -> BuiltinResult<Vec<VarLenUnicode>> {
    values
        .iter()
        .map(|value| {
            VarLenUnicode::from_str(value.as_ref()).map_err(|err| {
                hdf5_error(
                    builtin,
                    &ERR_ARGUMENT,
                    format!("{builtin}: string value cannot be encoded as HDF5 UTF-8: {err}"),
                )
            })
        })
        .collect()
}

fn shape_value(shape: Vec<usize>) -> BuiltinResult<Value> {
    if shape.is_empty() {
        Ok(Value::Num(1.0))
    } else {
        let rank = shape.len();
        Tensor::new(shape.into_iter().map(|v| v as f64).collect(), vec![1, rank])
            .map(Value::Tensor)
            .map_err(|msg| hdf5_error(H5INFO_NAME, &ERR_IO, msg))
    }
}

fn cell_column(values: Vec<Value>, builtin: &'static str) -> BuiltinResult<Value> {
    let rows = values.len();
    crate::make_cell(values, rows, 1).map_err(|msg| hdf5_error(builtin, &ERR_IO, msg))
}

fn join_hdf5_path(parent: &str, child: &str) -> String {
    if parent == "/" {
        format!("/{child}")
    } else {
        format!("{parent}/{child}")
    }
}

pub(super) fn type_name(dtype: &TypeDescriptor) -> String {
    match dtype {
        TypeDescriptor::Float(FloatSize::U4) => "single".to_string(),
        TypeDescriptor::Float(FloatSize::U8) => "double".to_string(),
        TypeDescriptor::Integer(IntSize::U1) => "int8".to_string(),
        TypeDescriptor::Integer(IntSize::U2) => "int16".to_string(),
        TypeDescriptor::Integer(IntSize::U4) => "int32".to_string(),
        TypeDescriptor::Integer(IntSize::U8) => "int64".to_string(),
        TypeDescriptor::Unsigned(IntSize::U1) => "uint8".to_string(),
        TypeDescriptor::Unsigned(IntSize::U2) => "uint16".to_string(),
        TypeDescriptor::Unsigned(IntSize::U4) => "uint32".to_string(),
        TypeDescriptor::Unsigned(IntSize::U8) => "uint64".to_string(),
        TypeDescriptor::Boolean => "logical".to_string(),
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => "string".to_string(),
        TypeDescriptor::FixedUnicode(_) | TypeDescriptor::FixedAscii(_) => "char".to_string(),
        TypeDescriptor::Compound(_) => "compound".to_string(),
        TypeDescriptor::Enum(_) => "enum".to_string(),
        TypeDescriptor::FixedArray(_, _) => "fixed-array".to_string(),
        TypeDescriptor::VarLenArray(_) => "variable-length-array".to_string(),
        TypeDescriptor::Reference(_) => "reference".to_string(),
    }
}

fn matlab_class_for_type(dtype: &TypeDescriptor) -> String {
    match dtype {
        TypeDescriptor::Integer(IntSize::U1) => "int8",
        TypeDescriptor::Integer(IntSize::U2) => "int16",
        TypeDescriptor::Integer(IntSize::U4) => "int32",
        TypeDescriptor::Integer(IntSize::U8) => "int64",
        TypeDescriptor::Unsigned(IntSize::U1) => "uint8",
        TypeDescriptor::Unsigned(IntSize::U2) => "uint16",
        TypeDescriptor::Unsigned(IntSize::U4) => "uint32",
        TypeDescriptor::Unsigned(IntSize::U8) => "uint64",
        TypeDescriptor::Boolean => "logical",
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => "string",
        TypeDescriptor::FixedUnicode(_) | TypeDescriptor::FixedAscii(_) => "char",
        _ => "double",
    }
    .to_string()
}

fn io_error<E>(builtin: &'static str, message: impl Into<String>, source: E) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(builtin)
        .with_source(source);
    if let Some(identifier) = ERR_IO.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
