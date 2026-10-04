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
    pub(super) fn parse(args: &[Value]) -> BuiltinResult<Self> {
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

    pub(super) fn into_object(self) -> BuiltinResult<ObjectInstance> {
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

pub(super) fn xml_node_to_struct(
    node: &XmlNode,
    options: &ReadStructOptions,
) -> BuiltinResult<StructValue> {
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

pub(super) fn parse_xml_document_with_context(
    text: &str,
    context: &'static str,
) -> BuiltinResult<XmlNode> {
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
