use super::*;
pub(super) fn serialize_xml_value(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(_) | Value::CharArray(_) => scalar_text(value, "xmlwrite", "document"),
        Value::Object(object) if object.class_name.is(XML_DOCUMENT_CLASS) => {
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
        Value::Object(object) if object.class_name.is(XML_ELEMENT_CLASS) => {
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

pub(super) fn scalar_xml_text(value: &Value) -> BuiltinResult<String> {
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
