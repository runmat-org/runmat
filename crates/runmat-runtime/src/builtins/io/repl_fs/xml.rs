//! XML file compatibility helpers.

use runmat_builtins::{ReadStructOption, XML_DOCUMENT_CLASS, XML_ELEMENT_CLASS};
use runmat_macros::runtime_builtin;
use runmat_value::{ObjectInstance, StructValue, Value};

use crate::builtins::common::fs::path_to_string;
use crate::builtins::io::repl_fs::compat::{char_value, compat_error, gather_args, value_to_path};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[path = "xml/source.rs"]
mod source;
use source::XmlFsPath;

fn xml_io_error(name: &str, err: std::io::Error) -> RuntimeError {
    build_runtime_error(format!("{name}: {err}"))
        .with_builtin(name)
        .with_source(err)
        .build()
}

fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
        compat_error(
            name,
            format!("{name}: {arg} must be a string scalar or character vector"),
        )
    })
}

fn known_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<()> {
    if !matches!(value, Value::GpuTensor(_)) {
        scalar_text(value, name, arg)?;
    }
    Ok(())
}

fn requested_output_preflight(name: &str) -> BuiltinResult<()> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(compat_error(
            name,
            format!("{name}: at most one output is supported"),
        ));
    }
    Ok(())
}

fn known_document_preflight(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::GpuTensor(_) | Value::Struct(_) => Ok(()),
        Value::Object(object)
            if object.class_name.is(XML_DOCUMENT_CLASS)
                || object.class_name.is(XML_ELEMENT_CLASS) =>
        {
            Ok(())
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            scalar_text(value, "xmlwrite", "document").map(|_| ())
        }
        _ => Err(compat_error(
            "xmlwrite",
            "xmlwrite: expected XML document, XML element, XML text, or scalar struct",
        )),
    }
}

#[runtime_builtin(
    name = "readstruct",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::xml"
)]
async fn readstruct_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    requested_output_preflight("readstruct")?;
    if args.is_empty() {
        return Err(compat_error(
            "readstruct",
            "readstruct: filename is required",
        ));
    }
    if args.len() > 1 && !(args.len() - 1).is_multiple_of(2) {
        return Err(compat_error(
            "readstruct",
            "readstruct: name-value options must be paired",
        ));
    }
    known_text(&args[0], "readstruct", "filename")?;
    ReadStructOptions::preflight(&args[1..])?;
    let args = gather_args("readstruct", &args).await?;
    scalar_text(&args[0], "readstruct", "filename")?;
    let path = value_to_path(&args[0], "readstruct", "filename")?;
    let options = ReadStructOptions::parse(&args[1..])?;
    let source = XmlFsPath::capture(path).map_err(|err| xml_io_error("readstruct", err))?;
    let text = source
        .read_to_string()
        .await
        .map_err(|err| xml_io_error("readstruct", err))?;
    let root = parse_xml_document_with_context(&text, "readstruct")?;
    let value = Value::Struct(xml_node_to_struct(&root, &options)?);
    source
        .check_current()
        .map_err(|err| xml_io_error("readstruct", err))?;
    Ok(value)
}

#[runtime_builtin(
    name = "xmlread",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::xml"
)]
async fn xmlread_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    requested_output_preflight("xmlread")?;
    if args
        .iter()
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        return Err(compat_error("xmlread", "xmlread: filename must be text"));
    }
    if args.len() != 1 {
        return Err(compat_error(
            "xmlread",
            "xmlread: expected exactly one filename",
        ));
    }
    known_text(&args[0], "xmlread", "filename")?;
    let args = gather_args("xmlread", &args).await?;
    scalar_text(&args[0], "xmlread", "filename")?;
    let path = value_to_path(&args[0], "xmlread", "filename")?;
    let source = XmlFsPath::capture(path).map_err(|err| xml_io_error("xmlread", err))?;
    let text = source
        .read_to_string()
        .await
        .map_err(|err| xml_io_error("xmlread", err))?;
    let root = parse_xml_document_with_context(&text, "xmlread")?;
    let mut object = ObjectInstance::new(XML_DOCUMENT_CLASS.display_name().to_string());
    object
        .properties
        .insert("Filename".to_string(), char_value(&path_to_string(&path)));
    object
        .properties
        .insert("Text".to_string(), Value::String(text.clone()));
    object
        .properties
        .insert("DocumentElementName".to_string(), char_value(&root.name));
    object.properties.insert(
        "DocumentElement".to_string(),
        Value::Object(root.into_object()?),
    );
    source
        .check_current()
        .map_err(|err| xml_io_error("xmlread", err))?;
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "xmlwrite",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::xml"
)]
async fn xmlwrite_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    requested_output_preflight("xmlwrite")?;
    if !matches!(args.len(), 1 | 2) {
        return Err(compat_error(
            "xmlwrite",
            "xmlwrite: expected document or filename and document",
        ));
    }
    if args.len() == 2 {
        known_text(&args[0], "xmlwrite", "filename")?;
    }
    let document = if args.len() == 2 { &args[1] } else { &args[0] };
    known_document_preflight(document)?;
    let args = gather_args("xmlwrite", &args).await?;
    match args.len() {
        1 => {
            let xml = serialize_xml_value(&args[0])?;
            Ok(char_value(&xml))
        }
        2 => {
            scalar_text(&args[0], "xmlwrite", "filename")?;
            let path = value_to_path(&args[0], "xmlwrite", "filename")?;
            let xml = serialize_xml_value(&args[1])?;
            let destination =
                XmlFsPath::capture(path).map_err(|err| xml_io_error("xmlwrite", err))?;
            destination
                .write(xml.as_bytes())
                .await
                .map_err(|err| xml_io_error("xmlwrite", err))?;
            Ok(Value::Num(xml.len() as f64))
        }
        _ => Err(compat_error(
            "xmlwrite",
            "xmlwrite: expected document or filename and document",
        )),
    }
}

#[derive(Debug)]
#[path = "xml/parser.rs"]
mod parser;
use parser::{parse_xml_document_with_context, xml_node_to_struct, ReadStructOptions};

#[path = "xml/serialize.rs"]
mod serialize;
#[cfg(test)]
use serialize::scalar_xml_text;
use serialize::serialize_xml_value;

#[cfg(test)]
#[path = "xml/tests.rs"]
mod tests;
