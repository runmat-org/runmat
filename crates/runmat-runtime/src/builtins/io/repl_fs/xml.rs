//! XML file compatibility helpers.

use std::collections::BTreeMap;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, CellArray,
    ObjectInstance, StructValue, Value,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::path_to_string;
use crate::builtins::common::identifiers::is_valid_varname;
use crate::builtins::io::repl_fs::compat::{
    char_value, compat_error, gather_args, scalar_text, value_to_path,
};
use crate::BuiltinResult;

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
    builtin_path = "crate::builtins::io::repl_fs::xml"
)]
async fn xmlread_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
struct ReadStructOptions {
    attribute_suffix: String,
    text_node_name: String,
}

impl Default for ReadStructOptions {
    fn default() -> Self {
        Self {
            attribute_suffix: "Attribute".to_string(),
            text_node_name: "Text".to_string(),
        }
    }
}

impl ReadStructOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut options = Self::default();
        let mut idx = 0usize;
        while idx < args.len() {
            let name = scalar_text(&args[idx], "readstruct", "option")?;
            let value = &args[idx + 1];
            match name.to_ascii_lowercase().as_str() {
                "filetype" => {
                    let file_type = scalar_text(value, "readstruct", "FileType")?;
                    if !file_type.eq_ignore_ascii_case("xml") {
                        return Err(compat_error(
                            "readstruct",
                            "readstruct: only XML FileType is supported",
                        ));
                    }
                }
                "attributesuffix" => {
                    options.attribute_suffix = scalar_text(value, "readstruct", "AttributeSuffix")?;
                }
                "textnodename" => {
                    options.text_node_name = scalar_text(value, "readstruct", "TextNodeName")?;
                }
                other => {
                    return Err(compat_error(
                        "readstruct",
                        format!("readstruct: unsupported option '{other}'"),
                    ));
                }
            }
            idx += 2;
        }
        Ok(options)
    }
}

#[derive(Clone, Debug)]
struct XmlNode {
    name: String,
    attributes: BTreeMap<String, String>,
    children: Vec<XmlNode>,
    text: String,
}

impl XmlNode {
    fn new(name: String, attributes: BTreeMap<String, String>) -> Self {
        Self {
            name,
            attributes,
            children: Vec::new(),
            text: String::new(),
        }
    }

    fn into_object(self) -> BuiltinResult<ObjectInstance> {
        let mut object = ObjectInstance::new("org.w3c.dom.Element".to_string());
        object
            .properties
            .insert("NodeName".to_string(), char_value(&self.name));
        object
            .properties
            .insert("TagName".to_string(), char_value(&self.name));
        object
            .properties
            .insert("TextContent".to_string(), char_value(self.text.trim()));
        object.properties.insert(
            "Attributes".to_string(),
            Value::Struct(attributes_struct(&self)),
        );
        let children = self
            .children
            .into_iter()
            .map(|child| child.into_object().map(Value::Object))
            .collect::<BuiltinResult<Vec<_>>>()?;
        object.properties.insert(
            "ChildNodes".to_string(),
            Value::Cell(
                CellArray::new(children.clone(), children.len(), 1)
                    .map_err(|err| compat_error("xmlread", err))?,
            ),
        );
        Ok(object)
    }
}

fn attributes_struct(node: &XmlNode) -> StructValue {
    let mut st = StructValue::new();
    for (name, value) in &node.attributes {
        st.insert(name.clone(), char_value(value));
    }
    st
}

fn xml_node_to_struct(node: &XmlNode, options: &ReadStructOptions) -> BuiltinResult<StructValue> {
    let mut out = StructValue::new();
    for (name, value) in &node.attributes {
        out.insert(
            xml_struct_field_name(&format!("{name}{}", options.attribute_suffix)),
            char_value(value),
        );
    }
    let text = node.text.trim();
    if !text.is_empty() {
        out.insert(
            xml_struct_field_name(&options.text_node_name),
            char_value(text),
        );
    }

    let mut groups: Vec<(String, Vec<Value>)> = Vec::new();
    for child in &node.children {
        let field = xml_struct_field_name(&child.name);
        let value = xml_node_to_struct_value(child, options)?;
        if let Some((_, values)) = groups.iter_mut().find(|(name, _)| *name == field) {
            values.push(value);
        } else {
            groups.push((field, vec![value]));
        }
    }
    for (field, values) in groups {
        let value = if values.len() == 1 {
            values.into_iter().next().unwrap()
        } else {
            Value::Cell(
                CellArray::new(values.clone(), values.len(), 1)
                    .map_err(|err| compat_error("readstruct", err))?,
            )
        };
        out.insert(field, value);
    }
    Ok(out)
}

fn xml_node_to_struct_value(node: &XmlNode, options: &ReadStructOptions) -> BuiltinResult<Value> {
    if node.attributes.is_empty() && node.children.is_empty() {
        return Ok(char_value(node.text.trim()));
    }
    xml_node_to_struct(node, options).map(Value::Struct)
}

fn xml_struct_field_name(name: &str) -> String {
    let mut out = String::new();
    for (idx, ch) in name.chars().enumerate() {
        if (idx == 0 && (ch.is_ascii_alphabetic() || ch == '_'))
            || (idx > 0 && (ch.is_ascii_alphanumeric() || ch == '_'))
        {
            out.push(ch);
        } else if idx == 0 && ch.is_ascii_digit() {
            out.push('x');
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('x');
    }
    if !is_valid_varname(&out) {
        out.insert_str(0, "x_");
    }
    out
}

fn parse_xml_document(text: &str) -> BuiltinResult<XmlNode> {
    let mut stack: Vec<XmlNode> = Vec::new();
    let mut root: Option<XmlNode> = None;
    let mut cursor = 0usize;
    while let Some(open_rel) = text[cursor..].find('<') {
        let open = cursor + open_rel;
        let preceding = &text[cursor..open];
        if !preceding.trim().is_empty() {
            if let Some(node) = stack.last_mut() {
                if !node.text.is_empty() {
                    node.text.push(' ');
                }
                node.text.push_str(preceding.trim());
            }
        }
        let close = text[open..]
            .find('>')
            .map(|idx| open + idx)
            .ok_or_else(|| compat_error("xmlread", "xmlread: malformed XML (unterminated tag)"))?;
        let raw_tag = text[open + 1..close].trim();
        cursor = close + 1;
        if raw_tag.is_empty()
            || raw_tag.starts_with('?')
            || raw_tag.starts_with("!--")
            || raw_tag.starts_with("![CDATA[")
            || raw_tag.starts_with('!')
        {
            continue;
        }
        if let Some(closing) = raw_tag.strip_prefix('/') {
            let closing_name = closing.split_whitespace().next().unwrap_or_default();
            let node = stack.pop().ok_or_else(|| {
                compat_error("xmlread", "xmlread: malformed XML (unexpected closing tag)")
            })?;
            if node.name != closing_name {
                return Err(compat_error(
                    "xmlread",
                    format!(
                        "xmlread: malformed XML (expected closing tag </{}> but found </{}>)",
                        node.name, closing_name
                    ),
                ));
            }
            if let Some(parent) = stack.last_mut() {
                parent.children.push(node);
            } else if root.replace(node).is_some() {
                return Err(compat_error(
                    "xmlread",
                    "xmlread: malformed XML (multiple root elements)",
                ));
            }
            continue;
        }
        let self_closing = raw_tag.ends_with('/');
        let tag = raw_tag.trim_end_matches('/').trim();
        let (name, attributes) = parse_xml_tag(tag)?;
        let node = XmlNode::new(name, attributes);
        if self_closing {
            if let Some(parent) = stack.last_mut() {
                parent.children.push(node);
            } else if root.replace(node).is_some() {
                return Err(compat_error(
                    "xmlread",
                    "xmlread: malformed XML (multiple root elements)",
                ));
            }
        } else {
            stack.push(node);
        }
    }
    let trailing = text[cursor..].trim();
    if !trailing.is_empty() {
        if let Some(node) = stack.last_mut() {
            if !node.text.is_empty() {
                node.text.push(' ');
            }
            node.text.push_str(trailing);
        }
    }
    if let Some(node) = stack.last() {
        return Err(compat_error(
            "xmlread",
            format!("xmlread: malformed XML (unclosed tag <{}>)", node.name),
        ));
    }
    root.ok_or_else(|| compat_error("xmlread", "xmlread: malformed XML (no root element)"))
}

fn parse_xml_document_with_context(text: &str, context: &'static str) -> BuiltinResult<XmlNode> {
    parse_xml_document(text).map_err(|err| {
        let message = err.message().replace("xmlread:", &format!("{context}:"));
        compat_error(context, message)
    })
}

fn parse_xml_tag(tag: &str) -> BuiltinResult<(String, BTreeMap<String, String>)> {
    let mut parts = tag.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or_default().to_string();
    if name.is_empty() {
        return Err(compat_error(
            "xmlread",
            "xmlread: malformed XML (empty tag)",
        ));
    }
    let mut attrs = BTreeMap::new();
    let mut rest = parts.next().unwrap_or_default().trim();
    while !rest.is_empty() {
        let Some(eq) = rest.find('=') else {
            break;
        };
        let key = rest[..eq].trim();
        rest = rest[eq + 1..].trim_start();
        let quote = rest.chars().next().ok_or_else(|| {
            compat_error(
                "xmlread",
                "xmlread: malformed XML (attribute missing value)",
            )
        })?;
        if quote != '"' && quote != '\'' {
            return Err(compat_error(
                "xmlread",
                "xmlread: malformed XML (attribute values must be quoted)",
            ));
        }
        rest = &rest[quote.len_utf8()..];
        let end = rest.find(quote).ok_or_else(|| {
            compat_error("xmlread", "xmlread: malformed XML (unterminated attribute)")
        })?;
        attrs.insert(key.to_string(), xml_unescape(&rest[..end]));
        rest = rest[end + quote.len_utf8()..].trim_start();
    }
    Ok((name, attrs))
}

fn xml_unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn serialize_xml_value(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::Object(object) if object.class_name == "org.w3c.dom.Document" => {
            if let Some(Value::String(text)) = object.properties.get("Text") {
                return Ok(text.clone());
            }
            if let Some(element) = object.properties.get("DocumentElement") {
                return serialize_xml_value(element);
            }
            Err(compat_error(
                "xmlwrite",
                "xmlwrite: document object has no serializable DocumentElement",
            ))
        }
        Value::Object(object) if object.class_name == "org.w3c.dom.Element" => {
            serialize_xml_element_object(object)
        }
        Value::Struct(st) => serialize_xml_struct("root", st),
        _ => Err(compat_error(
            "xmlwrite",
            "xmlwrite: expected XML document, XML element, XML text, or scalar struct",
        )),
    }
}

fn serialize_xml_element_object(object: &ObjectInstance) -> BuiltinResult<String> {
    let name = object
        .properties
        .get("TagName")
        .or_else(|| object.properties.get("NodeName"))
        .ok_or_else(|| compat_error("xmlwrite", "xmlwrite: element object is missing TagName"))
        .and_then(|value| scalar_text(value, "xmlwrite", "TagName"))?;
    let attrs = object
        .properties
        .get("Attributes")
        .and_then(|value| match value {
            Value::Struct(st) => Some(st),
            _ => None,
        });
    let text = object
        .properties
        .get("TextContent")
        .and_then(|value| scalar_text(value, "xmlwrite", "TextContent").ok())
        .unwrap_or_default();
    let children = object
        .properties
        .get("ChildNodes")
        .and_then(|value| match value {
            Value::Cell(cell) => Some(cell.data.as_slice()),
            _ => None,
        })
        .unwrap_or(&[]);
    let mut out = format!("<{}", xml_name(&name)?);
    if let Some(attrs) = attrs {
        for (name, value) in &attrs.fields {
            let value = scalar_text(value, "xmlwrite", "attribute value")?;
            out.push_str(&format!(
                " {}=\"{}\"",
                xml_name(name)?,
                xml_attr_escape(&value)
            ));
        }
    }
    if text.trim().is_empty() && children.is_empty() {
        out.push_str("/>");
        return Ok(out);
    }
    out.push('>');
    out.push_str(&xml_text_escape(text.trim()));
    for child in children {
        out.push_str(&serialize_xml_value(child)?);
    }
    out.push_str(&format!("</{}>", xml_name(&name)?));
    Ok(out)
}

fn serialize_xml_struct(root_name: &str, st: &StructValue) -> BuiltinResult<String> {
    let root = xml_name(root_name)?;
    let mut out = format!("<{root}>");
    for (name, value) in &st.fields {
        out.push_str(&serialize_xml_struct_field(name, value)?);
    }
    out.push_str(&format!("</{root}>"));
    Ok(out)
}

fn serialize_xml_struct_field(name: &str, value: &Value) -> BuiltinResult<String> {
    let name = xml_name(name)?;
    match value {
        Value::Struct(st) => {
            let mut out = format!("<{name}>");
            for (child_name, child_value) in &st.fields {
                out.push_str(&serialize_xml_struct_field(child_name, child_value)?);
            }
            out.push_str(&format!("</{name}>"));
            Ok(out)
        }
        Value::Cell(cell) => {
            let mut out = String::new();
            for value in &cell.data {
                out.push_str(&serialize_xml_struct_field(&name, value)?);
            }
            Ok(out)
        }
        _ => Ok(format!(
            "<{name}>{}</{name}>",
            xml_text_escape(&scalar_xml_text(value)?)
        )),
    }
}

fn scalar_xml_text(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::Num(number) => Ok(number.to_string()),
        Value::Int(number) => Ok(number.decimal_string()),
        Value::Bool(flag) => Ok(flag.to_string()),
        _ => Err(compat_error(
            "xmlwrite",
            "xmlwrite: struct XML fields must contain scalar text or nested structs",
        )),
    }
}

fn xml_name(name: &str) -> BuiltinResult<String> {
    if name.is_empty() || name.chars().any(char::is_whitespace) {
        return Err(compat_error("xmlwrite", "xmlwrite: invalid XML name"));
    }
    Ok(name.to_string())
}

fn xml_text_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn xml_attr_escape(value: &str) -> String {
    xml_text_escape(value)
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::fs::path_to_string;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

    #[test]
    fn xml_scalar_text_preserves_exact_uint64() {
        assert_eq!(
            scalar_xml_text(&Value::Int(runmat_builtins::IntValue::U64(u64::MAX)))
                .expect("xml text"),
            "18446744073709551615"
        );
    }

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn xmlread_returns_lightweight_document_object() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_xmlread_test.xml");
        std::fs::write(
            &path,
            "<?xml version=\"1.0\"?><root id=\"r\"><a>text</a></root>",
        )
        .unwrap();
        let value = run(xmlread_builtin(vec![Value::String(path_to_string(&path))])).unwrap();
        match value {
            Value::Object(object) => {
                assert_eq!(object.class_name, "org.w3c.dom.Document");
                assert_eq!(
                    object.properties.get("DocumentElementName"),
                    Some(&char_value("root"))
                );
                let Some(Value::Object(root)) = object.properties.get("DocumentElement") else {
                    panic!("expected DocumentElement object");
                };
                assert_eq!(root.properties.get("TagName"), Some(&char_value("root")));
                let Some(Value::Struct(attrs)) = root.properties.get("Attributes") else {
                    panic!("expected attributes struct");
                };
                assert_eq!(attrs.fields.get("id"), Some(&char_value("r")));
            }
            other => panic!("unexpected value {other:?}"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn readstruct_imports_xml_children_attributes_and_repeats() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_readstruct_test.xml");
        std::fs::write(
            &path,
            r#"<root id="r1"><item code="a">one</item><item code="b">two</item><label>done</label></root>"#,
        )
        .unwrap();
        let value = run(readstruct_builtin(vec![Value::String(path_to_string(
            &path,
        ))]))
        .unwrap();
        let Value::Struct(st) = value else {
            panic!("expected readstruct result");
        };
        assert_eq!(st.fields.get("idAttribute"), Some(&char_value("r1")));
        assert_eq!(st.fields.get("label"), Some(&char_value("done")));
        let Some(Value::Cell(items)) = st.fields.get("item") else {
            panic!("expected repeated item cell");
        };
        assert_eq!(items.rows, 2);
        let Value::Struct(first) = &items.data[0] else {
            panic!("expected first item struct");
        };
        assert_eq!(first.fields.get("codeAttribute"), Some(&char_value("a")));
        assert_eq!(first.fields.get("Text"), Some(&char_value("one")));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn readstruct_supports_xml_option_aliases() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_readstruct_options_test.xml");
        std::fs::write(&path, r#"<root id="r1">body</root>"#).unwrap();
        let value = run(readstruct_builtin(vec![
            Value::String(path_to_string(&path)),
            char_value("FileType"),
            char_value("xml"),
            char_value("AttributeSuffix"),
            char_value("_attr"),
            char_value("TextNodeName"),
            char_value("Body"),
        ]))
        .unwrap();
        let Value::Struct(st) = value else {
            panic!("expected readstruct result");
        };
        assert_eq!(st.fields.get("id_attr"), Some(&char_value("r1")));
        assert_eq!(st.fields.get("Body"), Some(&char_value("body")));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn xmlwrite_serializes_dom_to_string_or_file() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let mut root = ObjectInstance::new("org.w3c.dom.Element".to_string());
        root.properties
            .insert("TagName".to_string(), char_value("root"));
        let mut attrs = StructValue::new();
        attrs.insert("id", char_value("r1"));
        root.properties
            .insert("Attributes".to_string(), Value::Struct(attrs));
        root.properties
            .insert("TextContent".to_string(), char_value("hello & goodbye"));
        let xml = run(xmlwrite_builtin(vec![Value::Object(root.clone())])).unwrap();
        assert_eq!(
            xml,
            char_value("<root id=\"r1\">hello &amp; goodbye</root>")
        );

        let path = std::env::temp_dir().join("runmat_xmlwrite_test.xml");
        let bytes = run(xmlwrite_builtin(vec![
            Value::String(path_to_string(&path)),
            Value::Object(root),
        ]))
        .unwrap();
        assert!(matches!(bytes, Value::Num(n) if n > 0.0));
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, "<root id=\"r1\">hello &amp; goodbye</root>");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn xmlread_rejects_mismatched_tags() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_xmlread_bad_test.xml");
        std::fs::write(&path, "<root><a></root>").unwrap();
        let err = run(xmlread_builtin(vec![Value::String(path_to_string(&path))])).unwrap_err();
        assert!(err.message().contains("malformed XML"));
        let _ = std::fs::remove_file(path);
    }
}
