use super::names::XmlName;
use super::*;

#[async_recursion::async_recursion(?Send)]
pub(super) async fn serialize_xml_value(
    value: &Value,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<String> {
    cooperation.completed(1).await?;
    match value {
        Value::String(text) => {
            cooperation.completed(text.len()).await?;
            Ok(text.clone())
        }
        Value::StringArray(_) | Value::CharArray(_) => {
            let text = scalar_text(value, "xmlwrite", "document")?;
            cooperation.completed(text.len()).await?;
            Ok(text)
        }
        Value::Object(object) if object.class_name.is(XML_DOCUMENT_CLASS) => {
            if let Some(Value::String(text)) = object.properties.get("Text") {
                cooperation.completed(text.len()).await?;
                return Ok(text.clone());
            }
            if let Some(element) = object.properties.get("DocumentElement") {
                return serialize_xml_value(element, cooperation).await;
            }
            Err(compat_error(
                "xmlwrite",
                "xmlwrite: document object has no serializable DocumentElement",
            ))
        }
        Value::Object(object) if object.class_name.is(XML_ELEMENT_CLASS) => {
            serialize_xml_element_object(object, cooperation).await
        }
        Value::Struct(st) => serialize_xml_struct("root", st, cooperation).await,
        _ => Err(compat_error(
            "xmlwrite",
            "xmlwrite: expected XML document, XML element, XML text, or scalar struct",
        )),
    }
}

async fn serialize_xml_element_object(
    object: &ObjectInstance,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<String> {
    let name = object
        .properties
        .get("TagName")
        .or_else(|| object.properties.get("NodeName"))
        .ok_or_else(|| compat_error("xmlwrite", "xmlwrite: element object is missing TagName"))
        .and_then(|value| scalar_text(value, "xmlwrite", "TagName"))?;
    let attrs = match object.properties.get("Attributes") {
        None => None,
        Some(Value::Struct(st)) => Some(st),
        Some(_) => {
            return Err(compat_error(
                "xmlwrite",
                "xmlwrite: Attributes must be a scalar struct",
            ))
        }
    };
    let text = match object.properties.get("TextContent") {
        None => String::new(),
        Some(value) => scalar_text(value, "xmlwrite", "TextContent")?,
    };
    let children = match object.properties.get("ChildNodes") {
        None => &[][..],
        Some(Value::Cell(cell)) => cell.data.as_slice(),
        Some(_) => {
            return Err(compat_error(
                "xmlwrite",
                "xmlwrite: ChildNodes must be a cell array",
            ))
        }
    };
    let name = XmlName::parse(&name)?;
    let mut out = format!("<{name}");
    if let Some(attrs) = attrs {
        for (name, value) in &attrs.fields {
            cooperation.completed(name.len()).await?;
            let value = scalar_text(value, "xmlwrite", "attribute value")?;
            out.push_str(&format!(
                " {}=\"{}\"",
                XmlName::parse(name)?,
                xml_attr_escape(&value, cooperation).await?
            ));
        }
    }
    if text.trim().is_empty() && children.is_empty() {
        out.push_str("/>");
        return Ok(out);
    }
    out.push('>');
    out.push_str(&xml_text_escape(text.trim(), cooperation).await?);
    for child in children {
        out.push_str(&serialize_xml_value(child, cooperation).await?);
    }
    out.push_str(&format!("</{name}>"));
    Ok(out)
}

async fn serialize_xml_struct(
    root_name: &str,
    st: &StructValue,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<String> {
    let root = XmlName::parse(root_name)?;
    let mut out = format!("<{root}>");
    for (name, value) in &st.fields {
        out.push_str(&serialize_xml_struct_field(name, value, cooperation).await?);
    }
    out.push_str(&format!("</{root}>"));
    Ok(out)
}

#[async_recursion::async_recursion(?Send)]
async fn serialize_xml_struct_field(
    name: &str,
    value: &Value,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<String> {
    cooperation.completed(name.len()).await?;
    let name = XmlName::parse(name)?;
    match value {
        Value::Struct(st) => {
            let mut out = format!("<{name}>");
            for (child_name, child_value) in &st.fields {
                out.push_str(
                    &serialize_xml_struct_field(child_name, child_value, cooperation).await?,
                );
            }
            out.push_str(&format!("</{name}>"));
            Ok(out)
        }
        Value::Cell(cell) => {
            let mut out = String::new();
            for value in &cell.data {
                out.push_str(&serialize_xml_struct_field(name.as_str(), value, cooperation).await?);
            }
            Ok(out)
        }
        _ => Ok(format!(
            "<{name}>{}</{name}>",
            xml_text_escape(&scalar_xml_text(value)?, cooperation).await?
        )),
    }
}

pub(super) fn scalar_xml_text(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            scalar_text(value, "xmlwrite", "struct field")
        }
        Value::Num(number) => Ok(number.to_string()),
        Value::Int(number) => Ok(number.decimal_string()),
        Value::Bool(flag) => Ok(flag.to_string()),
        _ => Err(compat_error(
            "xmlwrite",
            "xmlwrite: struct XML fields must contain scalar text or nested structs",
        )),
    }
}

async fn xml_text_escape(value: &str, cooperation: &mut XmlCooperation) -> BuiltinResult<String> {
    let mut out = String::new();
    let mut completed = 0usize;
    for ch in value.chars() {
        completed += ch.len_utf8();
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
        if completed >= 4096 {
            cooperation.completed(completed).await?;
            completed = 0;
        }
    }
    cooperation.completed(completed).await?;
    Ok(out)
}

async fn xml_attr_escape(value: &str, cooperation: &mut XmlCooperation) -> BuiltinResult<String> {
    let mut out = String::new();
    let mut completed = 0usize;
    for ch in value.chars() {
        completed += ch.len_utf8();
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
        if completed >= 4096 {
            cooperation.completed(completed).await?;
            completed = 0;
        }
    }
    cooperation.completed(completed).await?;
    Ok(out)
}

#[cfg(test)]
#[path = "serialize/tests.rs"]
mod tests;
