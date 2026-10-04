//! XML file compatibility helpers.

use runmat_builtins::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{ObjectInstance, StructValue, Value};

use crate::builtins::common::fs::path_to_string;
use crate::builtins::io::repl_fs::compat::{
    char_value, compat_error, gather_args, scalar_text, value_to_path,
};
use crate::BuiltinResult;

const XML_DOCUMENT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("org.w3c.dom.Document");
const XML_ELEMENT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("org.w3c.dom.Element");

const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "input",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input argument.",
}];
const INPUTS_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "input1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input argument.",
    },
    BuiltinParamDescriptor {
        name: "input2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Second input argument.",
    },
];
const INPUTS_THREE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "input1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input argument.",
    },
    BuiltinParamDescriptor {
        name: "input2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional second input argument.",
    },
    BuiltinParamDescriptor {
        name: "input3",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional third input argument.",
    },
];
const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result value.",
}];

macro_rules! xml_descriptor {
    ($sig:ident, $desc:ident, $label:expr, $inputs:expr, $outputs:expr) => {
        const $sig: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: $outputs,
        }];
        pub const $desc: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$sig,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &[],
        };
    };
}

xml_descriptor!(
    READSTRUCT_SIGNATURES,
    READSTRUCT_DESCRIPTOR,
    "s = readstruct(filename, Name, Value)",
    &INPUTS_THREE,
    &OUTPUT_VALUE
);
pub const XMLREAD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "xmlread accepts a text filename only; integer and resident numeric inputs reject before file or provider access.",
};
xml_descriptor!(
    XMLREAD_SIGNATURES,
    XMLREAD_DESCRIPTOR,
    "document = xmlread(filename)",
    &INPUTS_ONE,
    &OUTPUT_VALUE
);
xml_descriptor!(
    XMLWRITE_SIGNATURES,
    XMLWRITE_DESCRIPTOR,
    "xml = xmlwrite(document)",
    &INPUTS_TWO,
    &OUTPUT_VALUE
);

#[runtime_builtin(
    name = "readstruct",
    category = "io/repl_fs",
    summary = "Read XML data into a scalar structure.",
    keywords = "readstruct,xml,struct,file",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::io::repl_fs::xml::READSTRUCT_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::xml"
)]
async fn readstruct_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("readstruct", &args).await?;
    if args.is_empty() {
        return Err(compat_error(
            "readstruct",
            "readstruct: filename is required",
        ));
    }
    if args.len() > 1 && (args.len() - 1) % 2 != 0 {
        return Err(compat_error(
            "readstruct",
            "readstruct: name-value options must be paired",
        ));
    }
    let path = value_to_path(&args[0], "readstruct", "filename")?;
    let options = ReadStructOptions::parse(&args[1..])?;
    let text = vfs::read_to_string_async(&path)
        .await
        .map_err(|err| compat_error("readstruct", format!("readstruct: {err}")))?;
    let root = parse_xml_document_with_context(&text, "readstruct")?;
    Ok(Value::Struct(xml_node_to_struct(&root, &options)?))
}

#[runtime_builtin(
    name = "xmlread",
    category = "io/repl_fs",
    summary = "Read an XML file into a lightweight DOM-compatible object.",
    keywords = "xmlread,xml,file,dom",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::io::repl_fs::xml::XMLREAD_DESCRIPTOR),
    integer_audit(crate::builtins::io::repl_fs::xml::XMLREAD_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::xml"
)]
async fn xmlread_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args
        .iter()
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        return Err(compat_error("xmlread", "xmlread: filename must be text"));
    }
    let args = gather_args("xmlread", &args).await?;
    if args.len() != 1 {
        return Err(compat_error(
            "xmlread",
            "xmlread: expected exactly one filename",
        ));
    }
    let path = value_to_path(&args[0], "xmlread", "filename")?;
    let text = vfs::read_to_string_async(&path)
        .await
        .map_err(|err| compat_error("xmlread", format!("xmlread: {err}")))?;
    let root = parse_xml_document_with_context(&text, "xmlread")?;
    let mut object = ObjectInstance::new("org.w3c.dom.Document".to_string());
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
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "xmlwrite",
    category = "io/repl_fs",
    summary = "Serialize a lightweight DOM-compatible XML object.",
    keywords = "xmlwrite,xml,file,dom",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::string_type),
    descriptor(crate::builtins::io::repl_fs::xml::XMLWRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::xml"
)]
async fn xmlwrite_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("xmlwrite", &args).await?;
    match args.len() {
        1 => {
            let xml = serialize_xml_value(&args[0])?;
            Ok(char_value(&xml))
        }
        2 => {
            let path = value_to_path(&args[0], "xmlwrite", "filename")?;
            let xml = serialize_xml_value(&args[1])?;
            vfs::write_async(&path, xml.as_bytes())
                .await
                .map_err(|err| compat_error("xmlwrite", format!("xmlwrite: {err}")))?;
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
use serialize::serialize_xml_value;

#[cfg(test)]
#[path = "xml/tests.rs"]
mod tests;
