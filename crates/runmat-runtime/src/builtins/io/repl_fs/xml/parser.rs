use super::*;
use crate::builtins::common::identifiers::is_valid_varname;
use runmat_value::CellArray;
use std::collections::BTreeMap;
pub(super) struct ReadStructOptions {
    attribute_suffix: String,
    text_node_name: String,
}

impl Default for ReadStructOptions {
    fn default() -> Self {
        Self {
            attribute_suffix: "Attribute".to_string(),
            text_node_name: "Text".into(),
        }
    }
}

impl ReadStructOptions {
    pub(super) fn preflight(args: &[Value]) -> BuiltinResult<()> {
        for pair in args.chunks_exact(2) {
            known_text(&pair[0], "readstruct", "option")?;
            known_text(&pair[1], "readstruct", "option value")?;
            if !matches!(&pair[0], Value::GpuTensor(_)) {
                let name = scalar_text(&pair[0], "readstruct", "option")?;
                let Some(option) = ReadStructOption::from_name(&name) else {
                    return Err(compat_error(
                        "readstruct",
                        format!(
                            "readstruct: unsupported option '{}'",
                            name.to_ascii_lowercase()
                        ),
                    ));
                };
                if option == ReadStructOption::FileType && !matches!(&pair[1], Value::GpuTensor(_))
                {
                    let file_type = scalar_text(&pair[1], "readstruct", "FileType")?;
                    if !file_type.eq_ignore_ascii_case("xml") {
                        return Err(compat_error(
                            "readstruct",
                            "readstruct: only XML FileType is supported",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut options = Self::default();
        let mut idx = 0usize;
        while idx < args.len() {
            let name = scalar_text(&args[idx], "readstruct", "option")?;
            let value = &args[idx + 1];
            match ReadStructOption::from_name(&name) {
                Some(ReadStructOption::FileType) => {
                    let file_type = scalar_text(value, "readstruct", "FileType")?;
                    if !file_type.eq_ignore_ascii_case("xml") {
                        return Err(compat_error(
                            "readstruct",
                            "readstruct: only XML FileType is supported",
                        ));
                    }
                }
                Some(ReadStructOption::AttributeSuffix) => {
                    options.attribute_suffix = scalar_text(value, "readstruct", "AttributeSuffix")?;
                }
                Some(ReadStructOption::TextNodeName) => {
                    options.text_node_name = scalar_text(value, "readstruct", "TextNodeName")?;
                }
                None => {
                    return Err(compat_error(
                        "readstruct",
                        format!(
                            "readstruct: unsupported option '{}'",
                            name.to_ascii_lowercase()
                        ),
                    ));
                }
            }
            idx += 2;
        }
        Ok(options)
    }
}

#[derive(Clone, Debug)]
pub(super) struct XmlNode {
    pub(super) name: String,
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

    #[async_recursion::async_recursion(?Send)]
    pub(super) async fn into_object(
        self,
        cooperation: &mut XmlCooperation,
    ) -> BuiltinResult<ObjectInstance> {
        cooperation
            .completed(self.name.len() + self.text.len())
            .await?;
        let mut object = ObjectInstance::new(XML_ELEMENT_CLASS);
        object
            .properties
            .insert("NodeName".to_string(), char_value(&self.name));
        object
            .properties
            .insert("TagName".to_string(), char_value(&self.name));
        object
            .properties
            .insert("TextContent".to_string(), char_value(self.text.trim()));
        let mut attributes = StructValue::new();
        for (name, value) in &self.attributes {
            cooperation.completed(name.len() + value.len()).await?;
            attributes.insert(name.clone(), char_value(value));
        }
        object
            .properties
            .insert("Attributes".to_string(), Value::Struct(attributes));
        let mut children = Vec::new();
        children.try_reserve(self.children.len()).map_err(|err| {
            build_runtime_error("xmlread: cannot allocate child-node collection")
                .with_builtin("xmlread")
                .with_source(err)
                .build()
        })?;
        for child in self.children {
            children.push(Value::Object(child.into_object(cooperation).await?));
        }
        let child_count = children.len();
        object.properties.insert(
            "ChildNodes".to_string(),
            Value::Cell(
                CellArray::new(children, child_count, 1)
                    .map_err(|err| compat_error("xmlread", err))?,
            ),
        );
        Ok(object)
    }
}

#[async_recursion::async_recursion(?Send)]
pub(super) async fn xml_node_to_struct(
    node: &XmlNode,
    options: &ReadStructOptions,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<StructValue> {
    cooperation
        .completed(node.name.len() + node.text.len())
        .await?;
    let mut out = StructValue::new();
    for (name, value) in &node.attributes {
        cooperation.completed(name.len() + value.len()).await?;
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
        let value = xml_node_to_struct_value(child, options, cooperation).await?;
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
            let value_count = values.len();
            Value::Cell(
                CellArray::new(values, value_count, 1)
                    .map_err(|err| compat_error("readstruct", err))?,
            )
        };
        out.insert(field, value);
    }
    Ok(out)
}

async fn xml_node_to_struct_value(
    node: &XmlNode,
    options: &ReadStructOptions,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<Value> {
    if node.attributes.is_empty() && node.children.is_empty() {
        return Ok(char_value(node.text.trim()));
    }
    xml_node_to_struct(node, options, cooperation)
        .await
        .map(Value::Struct)
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

#[path = "parser/events.rs"]
mod events;
#[path = "parser/decode.rs"]
mod decode;
#[path = "parser/attributes.rs"]
mod attributes;
#[path = "parser/declaration.rs"]
mod declaration;
#[path = "parser/encoding.rs"]
mod encoding;
#[path = "parser/reader.rs"]
mod reader;
pub(super) use declaration::{document_declaration_prefix, normalize_utf8_output};
pub(super) use encoding::decode_document_bytes;
pub(super) use events::parse_xml_document_with_context;
