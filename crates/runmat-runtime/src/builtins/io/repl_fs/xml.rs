//! XML file compatibility helpers.

use runmat_builtins::{
    ReadStructOption, XmlDiagnosticKind, XmlOperation, XML_DOCUMENT_CLASS, XML_ELEMENT_CLASS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ObjectInstance, StructValue, Value};

use crate::builtins::common::fs::path_to_string;
use crate::builtins::io::repl_fs::compat::{char_value, compat_error, value_to_path};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

async fn gather_args(
    name: &str,
    args: &crate::call::BuiltinArguments,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    args.try_map_preserving_roots_async(args.len(), runtime, |index| async move {
        crate::gather_if_needed_async(&args[index])
            .await
            .map_err(|error| crate::builtins::io::repl_fs::compat::map_control_flow(name, error))
    })
    .await
}

#[path = "xml/cooperation.rs"]
mod cooperation;
use cooperation::XmlCooperation;

fn xml_error(
    operation: XmlOperation,
    kind: XmlDiagnosticKind,
    message: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(operation.identifier(kind))
        .with_builtin(operation.name())
        .build()
}

fn xml_error_with_source(
    operation: XmlOperation,
    kind: XmlDiagnosticKind,
    source: RuntimeError,
) -> RuntimeError {
    let message = source.message().to_string();
    build_runtime_error(message)
        .with_identifier(operation.identifier(kind))
        .with_builtin(operation.name())
        .with_source(source)
        .build()
}

#[path = "xml/source.rs"]
mod source;
use source::{XmlFsPath, XmlPathOwner};

enum CapturedXmlPath {
    Exact(XmlFsPath),
    AwaitingFilename(XmlPathOwner),
}

impl CapturedXmlPath {
    fn capture(filename: &Value, operation: XmlOperation) -> BuiltinResult<Self> {
        match filename {
            Value::GpuTensor(_) => Ok(Self::AwaitingFilename(
                XmlPathOwner::capture().map_err(|error| xml_io_error(operation, error))?,
            )),
            known => {
                let path = value_to_path(known, operation.name(), "filename").map_err(|error| {
                    xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, error)
                })?;
                Ok(Self::Exact(
                    XmlFsPath::capture(path).map_err(|error| xml_io_error(operation, error))?,
                ))
            }
        }
    }

    fn resolve(self, filename: &Value, operation: XmlOperation) -> BuiltinResult<XmlFsPath> {
        match self {
            Self::Exact(path) => Ok(path),
            Self::AwaitingFilename(owner) => {
                let path = value_to_path(filename, operation.name(), "filename")?;
                Ok(owner.resolve(path))
            }
        }
    }
}

fn xml_io_error(operation: XmlOperation, err: std::io::Error) -> RuntimeError {
    let name = operation.name();
    build_runtime_error(format!("{name}: {err}"))
        .with_identifier(operation.identifier(XmlDiagnosticKind::Io))
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
            if object.class_name().is(XML_DOCUMENT_CLASS)
                || object.class_name().is(XML_ELEMENT_CLASS) =>
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
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::xml"
)]
async fn readstruct_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let operation = XmlOperation::ReadStruct;
    let mut cooperation = XmlCooperation::new(runtime.as_ref(), operation);
    requested_output_preflight("readstruct")
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err))?;
    if args.is_empty() {
        return Err(xml_error(
            operation,
            XmlDiagnosticKind::InvalidArgument,
            "readstruct: filename is required",
        ));
    }
    if args.len() > 1 && !(args.len() - 1).is_multiple_of(2) {
        return Err(xml_error(
            operation,
            XmlDiagnosticKind::InvalidArgument,
            "readstruct: name-value options must be paired",
        ));
    }
    known_text(&args[0], "readstruct", "filename")
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err))?;
    ReadStructOptions::preflight(args.view().range(1..)?)
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidOption, err))?;
    let target = CapturedXmlPath::capture(&args[0], operation)?;
    cooperation.completed(0).await?;
    let args = cooperation
        .await_io(gather_args("readstruct", &args, runtime.as_ref()))
        .await??;
    scalar_text(&args[0], "readstruct", "filename")
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err))?;
    let options = ReadStructOptions::parse(args.view().range(1..)?)
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidOption, err))?;
    let source = target
        .resolve(&args[0], operation)
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err))?;
    let bytes = cooperation
        .await_io(source.read_bytes())
        .await?
        .map_err(|err| xml_io_error(operation, err))?;
    let text = decode_document_bytes(&bytes, &mut cooperation)
        .await
        .map_err(|err| {
            if cooperation.is_execution_error(&err) {
                err
            } else {
                xml_error_with_source(operation, XmlDiagnosticKind::MalformedXml, err)
            }
        })?;
    let root = parse_xml_document_with_context(&text, &mut cooperation).await?;
    let value = Value::Struct(
        xml_node_to_struct(&root, &options, &mut cooperation)
            .await
            .map_err(|err| {
                if cooperation.is_execution_error(&err) {
                    err
                } else {
                    xml_error_with_source(operation, XmlDiagnosticKind::Projection, err)
                }
            })?,
    );
    source
        .check_current()
        .map_err(|err| xml_io_error(operation, err))?;
    Ok(value)
}

#[runtime_builtin(
    name = "xmlread",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::xml"
)]
async fn xmlread_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let operation = XmlOperation::XmlRead;
    let mut cooperation = XmlCooperation::new(runtime.as_ref(), operation);
    requested_output_preflight("xmlread")
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err))?;
    if args
        .iter()
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        return Err(xml_error(
            operation,
            XmlDiagnosticKind::InvalidArgument,
            "xmlread: filename must be text",
        ));
    }
    if args.len() != 1 {
        return Err(xml_error(
            operation,
            XmlDiagnosticKind::InvalidArgument,
            "xmlread: expected exactly one filename",
        ));
    }
    known_text(&args[0], "xmlread", "filename")
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err))?;
    let target = CapturedXmlPath::capture(&args[0], operation)?;
    cooperation.completed(0).await?;
    let args = cooperation
        .await_io(gather_args("xmlread", &args, runtime.as_ref()))
        .await??;
    scalar_text(&args[0], "xmlread", "filename")
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err))?;
    let path = value_to_path(&args[0], "xmlread", "filename")
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err))?;
    let filename = path_to_string(&path);
    let source = target
        .resolve(&args[0], operation)
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err))?;
    let bytes = cooperation
        .await_io(source.read_bytes())
        .await?
        .map_err(|err| xml_io_error(operation, err))?;
    let text = decode_document_bytes(&bytes, &mut cooperation)
        .await
        .map_err(|err| {
            if cooperation.is_execution_error(&err) {
                err
            } else {
                xml_error_with_source(operation, XmlDiagnosticKind::MalformedXml, err)
            }
        })?;
    let root = parse_xml_document_with_context(&text, &mut cooperation).await?;
    let root_name = char_value(&root.name);
    let element = root
        .into_object(&mut cooperation, runtime.as_ref())
        .await
        .map_err(|err| {
            if cooperation.is_execution_error(&err) {
                err
            } else {
                xml_error_with_source(operation, XmlDiagnosticKind::Projection, err)
            }
        })?;
    source
        .check_current()
        .map_err(|err| xml_io_error(operation, err))?;
    let fields = [
        ("Filename", char_value(&filename)),
        ("Text", Value::string(text)),
        ("DocumentElementName", root_name),
        ("DocumentElement", element.value().clone()),
    ];
    crate::execution::object::construct_object_from_borrowed_fields(
        runtime.as_ref(),
        XML_DOCUMENT_CLASS.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )
}

#[runtime_builtin(
    name = "xmlwrite",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::xml"
)]
async fn xmlwrite_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let operation = XmlOperation::XmlWrite;
    let mut cooperation = XmlCooperation::new(runtime.as_ref(), operation);
    requested_output_preflight("xmlwrite")
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err))?;
    if !matches!(args.len(), 1 | 2) {
        return Err(xml_error(
            operation,
            XmlDiagnosticKind::InvalidArgument,
            "xmlwrite: expected document or filename and document",
        ));
    }
    if args.len() == 2 {
        known_text(&args[0], "xmlwrite", "filename").map_err(|err| {
            xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err)
        })?;
    }
    let document = if args.len() == 2 { &args[1] } else { &args[0] };
    known_document_preflight(document)
        .map_err(|err| xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err))?;
    let target = if args.len() == 2 {
        Some(CapturedXmlPath::capture(&args[0], operation)?)
    } else {
        None
    };
    cooperation.completed(0).await?;
    let args = cooperation
        .await_io(gather_args("xmlwrite", &args, runtime.as_ref()))
        .await??;
    match args.len() {
        1 => {
            let xml = serialize_xml_value(&args[0], &mut cooperation)
                .await
                .map_err(|err| {
                    if cooperation.is_execution_error(&err) {
                        err
                    } else {
                        xml_error_with_source(operation, XmlDiagnosticKind::Projection, err)
                    }
                })?;
            let xml = normalize_utf8_output(xml, &mut cooperation)
                .await
                .map_err(|err| {
                    if cooperation.is_execution_error(&err) {
                        err
                    } else {
                        xml_error_with_source(operation, XmlDiagnosticKind::Projection, err)
                    }
                })?;
            cooperation.completed(xml.len()).await?;
            Ok(char_value(&xml))
        }
        2 => {
            scalar_text(&args[0], "xmlwrite", "filename").map_err(|err| {
                xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err)
            })?;
            let xml = serialize_xml_value(&args[1], &mut cooperation)
                .await
                .map_err(|err| {
                    if cooperation.is_execution_error(&err) {
                        err
                    } else {
                        xml_error_with_source(operation, XmlDiagnosticKind::Projection, err)
                    }
                })?;
            let xml = normalize_utf8_output(xml, &mut cooperation)
                .await
                .map_err(|err| {
                    if cooperation.is_execution_error(&err) {
                        err
                    } else {
                        xml_error_with_source(operation, XmlDiagnosticKind::Projection, err)
                    }
                })?;
            cooperation.completed(xml.len()).await?;
            let destination = target
                .expect("two-argument XML write has a captured destination")
                .resolve(&args[0], operation)
                .map_err(|err| {
                    xml_error_with_source(operation, XmlDiagnosticKind::InvalidArgument, err)
                })?;
            cooperation
                .await_io(destination.write(xml.as_bytes()))
                .await?
                .map_err(|err| xml_io_error(operation, err))?;
            cooperation.completed(0).await?;
            Ok(Value::Num(xml.len() as f64))
        }
        _ => Err(compat_error(
            "xmlwrite",
            "xmlwrite: expected document or filename and document",
        )),
    }
}

#[path = "xml/parser.rs"]
mod parser;
use parser::{
    decode_document_bytes, normalize_utf8_output, parse_xml_document_with_context,
    xml_node_to_struct, ReadStructOptions,
};

#[path = "xml/serialize.rs"]
mod serialize;
#[cfg(test)]
use serialize::scalar_xml_text;
use serialize::serialize_xml_value;

#[path = "xml/names.rs"]
mod names;

#[cfg(test)]
#[path = "xml/tests.rs"]
mod tests;
