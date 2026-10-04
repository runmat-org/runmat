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
        let mut object = ObjectInstance::new(XML_ELEMENT_CLASS.display_name().to_string());
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

async fn parse_xml_document(
    text: &str,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<XmlNode> {
    let mut stack: Vec<XmlNode> = Vec::new();
    let mut root: Option<XmlNode> = None;
    let mut cursor = 0usize;
    while let Some(open) = next_xml_open(text, cursor, cooperation).await? {
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
        let (name, attributes) = parse_xml_tag(tag, cooperation).await?;
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

async fn next_xml_open(
    text: &str,
    cursor: usize,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<Option<usize>> {
    let bytes = text.as_bytes();
    let mut offset = cursor;
    while offset < bytes.len() {
        let end = offset.saturating_add(4096).min(bytes.len());
        let chunk = &bytes[offset..end];
        let position = chunk.iter().position(|byte| *byte == b'<');
        cooperation
            .completed(position.map_or(chunk.len(), |relative| relative + 1))
            .await?;
        if let Some(relative) = position {
            return Ok(Some(offset + relative));
        }
        offset = end;
    }
    Ok(None)
}

pub(super) async fn parse_xml_document_with_context(
    text: &str,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<XmlNode> {
    let context = cooperation.operation().name();
    parse_xml_document(text, cooperation).await.map_err(|err| {
        if cooperation.is_cancelled_error(&err) {
            return err;
        }
        let message = err.message().replace("xmlread:", &format!("{context}:"));
        build_runtime_error(message)
            .with_identifier(
                cooperation
                    .operation()
                    .identifier(XmlDiagnosticKind::MalformedXml),
            )
            .with_builtin(context)
            .with_source(err)
            .build()
    })
}

async fn parse_xml_tag(
    tag: &str,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<(String, BTreeMap<String, String>)> {
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
        let Some(eq) = find_ascii_delimiter(rest, b'=', cooperation).await? else {
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
        let end = find_ascii_delimiter(rest, quote as u8, cooperation)
            .await?
            .ok_or_else(|| {
                compat_error("xmlread", "xmlread: malformed XML (unterminated attribute)")
            })?;
        attrs.insert(
            key.to_string(),
            xml_unescape(&rest[..end], cooperation).await?,
        );
        rest = rest[end + quote.len_utf8()..].trim_start();
    }
    Ok((name, attrs))
}

async fn find_ascii_delimiter(
    text: &str,
    delimiter: u8,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<Option<usize>> {
    for (chunk_idx, chunk) in text.as_bytes().chunks(4096).enumerate() {
        let position = chunk.iter().position(|byte| *byte == delimiter);
        cooperation
            .completed(position.map_or(chunk.len(), |relative| relative + 1))
            .await?;
        if let Some(relative) = position {
            return Ok(Some(chunk_idx * 4096 + relative));
        }
    }
    Ok(None)
}

async fn xml_unescape(text: &str, cooperation: &mut XmlCooperation) -> BuiltinResult<String> {
    let mut out = String::new();
    let mut rest = text;
    let mut completed = 0usize;
    while !rest.is_empty() {
        let entity = [
            ("&lt;", '<'),
            ("&gt;", '>'),
            ("&quot;", '"'),
            ("&apos;", '\''),
            ("&amp;", '&'),
        ]
        .into_iter()
        .find(|(prefix, _)| rest.starts_with(*prefix));
        if let Some((prefix, decoded)) = entity {
            out.push(decoded);
            rest = &rest[prefix.len()..];
            completed += prefix.len();
        } else {
            let ch = rest.chars().next().expect("nonempty XML source");
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
            completed += ch.len_utf8();
        }
        if completed >= 4096 {
            cooperation.completed(completed).await?;
            completed = 0;
        }
    }
    cooperation.completed(completed).await?;
    Ok(out)
}
