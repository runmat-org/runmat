//! MATLAB-compatible HTML text extraction helpers.

use std::collections::BTreeMap;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, ObjectInstance, ResolveContext, StringArray, StructValue, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::strings::core::compat::{scalar_text, text_items};
use crate::{build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult};

const HTML_TREE_CLASS: &str = "htmlTree";
const MISSING: &str = "<missing>";

const OUT_TREE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tree",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Parsed HTML tree.",
}];

const OUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Extracted text.",
}];

const IN_CODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "code",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HTML code or htmlTree object.",
}];

const IN_CODE_METHOD: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "code",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HTML code or htmlTree object.",
    },
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("ExtractionMethod"),
        description: "Extraction method option name.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("tree"),
        description: "Extraction method: tree, article, or all-text.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HTML.INVALID_INPUT",
    identifier: Some("RunMat:html:InvalidInput"),
    when: "Inputs are not a supported htmlTree or extractHTMLText form.",
    message: "HTML Text Analytics helper received invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const HTML_TREE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "tree = htmlTree(code)",
        inputs: &IN_CODE,
        outputs: &OUT_TREE,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const EXTRACT_HTML_TEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "str = extractHTMLText(code)",
            inputs: &IN_CODE,
            outputs: &OUT_TEXT,
        },
        BuiltinSignatureDescriptor {
            label: "str = extractHTMLText(___, 'ExtractionMethod', method)",
            inputs: &IN_CODE_METHOD,
            outputs: &OUT_TEXT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn string_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::String
}

fn html_error(fn_name: &str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin(fn_name)
        .with_identifier("RunMat:html:InvalidInput")
        .build()
}

#[runtime_builtin(
    name = "htmlTree",
    category = "strings/text_analytics",
    summary = "Parse HTML code into a lightweight htmlTree object.",
    keywords = "htmlTree,HTML,text analytics,DOM",
    accel = "metadata",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::html::HTML_TREE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::html"
)]
async fn html_tree_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() != 1 {
        return Err(html_error(
            "htmlTree",
            "htmlTree: expected exactly one input",
        ));
    }
    let value = gather_if_needed_async(&args[0]).await.map_err(|err| {
        html_error(
            "htmlTree",
            format!("htmlTree: failed to gather input: {err}"),
        )
    })?;
    html_tree_value(value)
}

#[runtime_builtin(
    name = "extractHTMLText",
    category = "strings/text_analytics",
    summary = "Extract visible text from HTML code or htmlTree objects.",
    keywords = "extractHTMLText,HTML,text analytics,parse",
    accel = "sink",
    type_resolver(string_type),
    descriptor(crate::builtins::strings::text_analytics::html::EXTRACT_HTML_TEXT_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::html"
)]
async fn extract_html_text_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (source, method) = parse_extract_args(args).await?;
    extract_html_text_value(source, method)
}

async fn parse_extract_args(args: Vec<Value>) -> BuiltinResult<(Value, ExtractionMethod)> {
    match args.len() {
        1 => {
            let source = gather_if_needed_async(&args[0]).await.map_err(|err| {
                html_error(
                    "extractHTMLText",
                    format!("extractHTMLText: failed to gather input: {err}"),
                )
            })?;
            Ok((source, ExtractionMethod::Tree))
        }
        3 => {
            let source = gather_if_needed_async(&args[0]).await.map_err(|err| {
                html_error(
                    "extractHTMLText",
                    format!("extractHTMLText: failed to gather input: {err}"),
                )
            })?;
            let name = gather_if_needed_async(&args[1]).await.map_err(|err| {
                html_error(
                    "extractHTMLText",
                    format!("extractHTMLText: failed to gather option name: {err}"),
                )
            })?;
            let method = gather_if_needed_async(&args[2]).await.map_err(|err| {
                html_error(
                    "extractHTMLText",
                    format!("extractHTMLText: failed to gather option value: {err}"),
                )
            })?;
            let option = scalar_text(&name, "extractHTMLText")?.to_ascii_lowercase();
            if option != "extractionmethod" {
                return Err(html_error(
                    "extractHTMLText",
                    format!("extractHTMLText: unsupported option '{option}'"),
                ));
            }
            Ok((
                source,
                ExtractionMethod::parse(&scalar_text(&method, "extractHTMLText")?)?,
            ))
        }
        _ => Err(html_error(
            "extractHTMLText",
            "extractHTMLText: expected input or input with 'ExtractionMethod', method",
        )),
    }
}

fn html_tree_value(value: Value) -> BuiltinResult<Value> {
    if let Value::Object(object) = value {
        if object.is_class(HTML_TREE_CLASS) {
            return Ok(Value::Object(object));
        }
        return Err(html_error(
            "htmlTree",
            format!(
                "htmlTree: expected HTML text, got object {}",
                object.class_name
            ),
        ));
    }

    let list = text_items(value, "htmlTree")?;
    let mut objects = Vec::with_capacity(list.items.len());
    for item in list.items {
        let html = item.unwrap_or_else(|| MISSING.to_string());
        objects.push(Value::Object(parse_html_tree(&html)?.into_object(None)?));
    }
    if objects.len() == 1 {
        Ok(objects.remove(0))
    } else {
        make_cell_with_shape(objects, list.shape).map_err(|err| html_error("htmlTree", err))
    }
}

fn extract_html_text_value(value: Value, method: ExtractionMethod) -> BuiltinResult<Value> {
    match value {
        Value::Object(object) if object.is_class(HTML_TREE_CLASS) => {
            Ok(Value::String(extract_from_object(&object, method)?))
        }
        Value::Object(object) => Err(html_error(
            "extractHTMLText",
            format!(
                "extractHTMLText: expected HTML text or htmlTree, got object {}",
                object.class_name
            ),
        )),
        Value::Cell(cell) if cell.data.iter().any(is_html_tree_value) => {
            extract_html_text_from_cell(cell, method)
        }
        other => {
            let list = text_items(other, "extractHTMLText")?;
            let shape = list.shape.clone();
            let mut texts = Vec::with_capacity(list.items.len());
            for item in list.items {
                match item {
                    Some(html) => {
                        let tree = parse_html_tree(&html)?;
                        texts.push(tree.extract_text(method));
                    }
                    None => texts.push(MISSING.to_string()),
                }
            }
            string_output(texts, shape, "extractHTMLText")
        }
    }
}

fn extract_html_text_from_cell(cell: CellArray, method: ExtractionMethod) -> BuiltinResult<Value> {
    let shape = cell.shape.clone();
    let mut texts = Vec::with_capacity(cell.data.len());
    for value in cell.data {
        match value {
            Value::Object(object) if object.is_class(HTML_TREE_CLASS) => {
                texts.push(extract_from_object(&object, method)?);
            }
            Value::Object(object) => {
                return Err(html_error(
                    "extractHTMLText",
                    format!(
                        "extractHTMLText: expected HTML text or htmlTree, got object {}",
                        object.class_name
                    ),
                ));
            }
            other => {
                let html = scalar_text(&other, "extractHTMLText")?;
                let tree = parse_html_tree(&html)?;
                texts.push(tree.extract_text(method));
            }
        }
    }
    string_output(texts, shape, "extractHTMLText")
}

fn is_html_tree_value(value: &Value) -> bool {
    matches!(value, Value::Object(object) if object.is_class(HTML_TREE_CLASS))
}

fn string_output(data: Vec<String>, shape: Vec<usize>, fn_name: &str) -> BuiltinResult<Value> {
    if data.len() == 1 {
        Ok(Value::String(data.into_iter().next().unwrap_or_default()))
    } else {
        StringArray::new(data, shape)
            .map(Value::StringArray)
            .map_err(|err| html_error(fn_name, err))
    }
}

fn extract_from_object(object: &ObjectInstance, method: ExtractionMethod) -> BuiltinResult<String> {
    match object.properties.get("RawHTML") {
        Some(Value::String(html)) => Ok(parse_html_tree(html)?.extract_text(method)),
        Some(Value::CharArray(_)) | Some(Value::StringArray(_)) => {
            let html = scalar_text(
                object
                    .properties
                    .get("RawHTML")
                    .expect("RawHTML checked above"),
                "extractHTMLText",
            )?;
            Ok(parse_html_tree(&html)?.extract_text(method))
        }
        _ => Err(html_error(
            "extractHTMLText",
            "extractHTMLText: invalid htmlTree object",
        )),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtractionMethod {
    Tree,
    Article,
    AllText,
}

impl ExtractionMethod {
    fn parse(value: &str) -> BuiltinResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "tree" => Ok(Self::Tree),
            "article" => Ok(Self::Article),
            "all-text" => Ok(Self::AllText),
            other => Err(html_error(
                "extractHTMLText",
                format!(
                    "extractHTMLText: ExtractionMethod must be 'tree', 'article', or 'all-text', got '{other}'"
                ),
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum HtmlNode {
    Element(HtmlElement),
    Text(String),
}

#[derive(Clone, Debug, PartialEq)]
struct HtmlElement {
    name: String,
    attrs: BTreeMap<String, String>,
    children: Vec<HtmlNode>,
    raw: String,
}

#[derive(Debug)]
struct OpenElement {
    name: String,
    attrs: BTreeMap<String, String>,
    children: Vec<HtmlNode>,
    start: usize,
}

fn parse_html_tree(html: &str) -> BuiltinResult<HtmlNode> {
    let mut stack = vec![OpenElement {
        name: "document".to_string(),
        attrs: BTreeMap::new(),
        children: Vec::new(),
        start: 0,
    }];
    let mut cursor = 0;

    while let Some(rel_start) = html[cursor..].find('<') {
        let tag_start = cursor + rel_start;
        push_text(&mut stack, &html[cursor..tag_start]);
        let Some(tag_end) = find_tag_end(html, tag_start) else {
            push_text(&mut stack, &html[tag_start..]);
            cursor = html.len();
            break;
        };
        let token = &html[tag_start + 1..tag_end];
        cursor = tag_end + 1;

        if token.starts_with("!--") {
            if let Some(close) = html[tag_start + 4..].find("-->") {
                cursor = tag_start + 4 + close + 3;
            }
            continue;
        }

        let trimmed = token.trim();
        if trimmed.is_empty() || trimmed.starts_with('!') || trimmed.starts_with('?') {
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix('/') {
            let name = tag_name(rest).to_ascii_lowercase();
            close_element(&mut stack, &name, html, cursor);
            continue;
        }

        let self_closing = trimmed.ends_with('/') || is_void_tag(tag_name(trimmed));
        let (name, attrs) = parse_open_tag(trimmed);
        if name.is_empty() {
            continue;
        }
        stack.push(OpenElement {
            name: name.to_ascii_lowercase(),
            attrs,
            children: Vec::new(),
            start: tag_start,
        });
        if self_closing {
            let lower = stack
                .last()
                .map(|open| open.name.clone())
                .unwrap_or_default();
            close_element(&mut stack, &lower, html, cursor);
        }
    }

    if cursor < html.len() {
        push_text(&mut stack, &html[cursor..]);
    }

    while stack.len() > 1 {
        close_top(&mut stack, html, html.len());
    }

    let root = stack.pop().expect("root element");
    let element_children = root
        .children
        .iter()
        .filter(|child| matches!(child, HtmlNode::Element(_)))
        .count();
    if element_children == 1 && root.children.iter().all(|child| !is_nonblank_text(child)) {
        Ok(root
            .children
            .into_iter()
            .find(|child| matches!(child, HtmlNode::Element(_)))
            .expect("one element child"))
    } else {
        let raw = html.to_string();
        Ok(HtmlNode::Element(HtmlElement {
            name: "html".to_string(),
            attrs: BTreeMap::new(),
            children: root.children,
            raw,
        }))
    }
}

fn push_text(stack: &mut [OpenElement], text: &str) {
    if !text.is_empty() {
        if let Some(current) = stack.last_mut() {
            current.children.push(HtmlNode::Text(text.to_string()));
        }
    }
}

fn find_tag_end(html: &str, tag_start: usize) -> Option<usize> {
    let mut quote: Option<char> = None;
    for (offset, ch) in html[tag_start + 1..].char_indices() {
        match (quote, ch) {
            (Some(active), current) if current == active => quote = None,
            (None, '"' | '\'') => quote = Some(ch),
            (None, '>') => return Some(tag_start + 1 + offset),
            _ => {}
        }
    }
    None
}

fn close_element(stack: &mut Vec<OpenElement>, name: &str, html: &str, end: usize) {
    let Some(pos) = stack.iter().rposition(|open| open.name == name) else {
        return;
    };
    while stack.len() > pos && stack.len() > 1 {
        close_top(stack, html, end);
    }
}

fn close_top(stack: &mut Vec<OpenElement>, html: &str, end: usize) {
    let Some(open) = stack.pop() else {
        return;
    };
    let safe_end = end.min(html.len()).max(open.start.min(html.len()));
    let raw = html[open.start.min(html.len())..safe_end].to_string();
    let node = HtmlNode::Element(HtmlElement {
        name: open.name,
        attrs: open.attrs,
        children: open.children,
        raw,
    });
    if let Some(parent) = stack.last_mut() {
        parent.children.push(node);
    }
}

fn tag_name(token: &str) -> &str {
    token
        .trim_start()
        .split(|ch: char| ch.is_whitespace() || ch == '/' || ch == '>')
        .next()
        .unwrap_or("")
}

fn parse_open_tag(token: &str) -> (String, BTreeMap<String, String>) {
    let mut rest = token.trim().trim_end_matches('/').trim();
    let name = tag_name(rest).to_string();
    rest = rest.get(name.len()..).unwrap_or("").trim();
    (name, parse_attrs(rest))
}

fn parse_attrs(mut rest: &str) -> BTreeMap<String, String> {
    let mut attrs = BTreeMap::new();
    while !rest.trim_start().is_empty() {
        rest = rest.trim_start();
        let name_end = rest
            .find(|ch: char| ch.is_whitespace() || ch == '=' || ch == '/')
            .unwrap_or(rest.len());
        if name_end == 0 {
            break;
        }
        let name = rest[..name_end].to_ascii_lowercase();
        rest = rest[name_end..].trim_start();
        let mut value = String::new();
        if let Some(after_eq) = rest.strip_prefix('=') {
            rest = after_eq.trim_start();
            if let Some(quote) = rest.chars().next().filter(|ch| *ch == '"' || *ch == '\'') {
                rest = &rest[quote.len_utf8()..];
                if let Some(end) = rest.find(quote) {
                    value = decode_html_entities(&rest[..end]);
                    rest = &rest[end + quote.len_utf8()..];
                } else {
                    value = decode_html_entities(rest);
                    rest = "";
                }
            } else {
                let value_end = rest
                    .find(|ch: char| ch.is_whitespace() || ch == '/')
                    .unwrap_or(rest.len());
                value = decode_html_entities(&rest[..value_end]);
                rest = &rest[value_end..];
            }
        }
        attrs.insert(name, value);
    }
    attrs
}

fn is_void_tag(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "area"
            | "base"
            | "br"
            | "col"
            | "embed"
            | "hr"
            | "img"
            | "input"
            | "link"
            | "meta"
            | "param"
            | "source"
            | "track"
            | "wbr"
    )
}

fn is_nonblank_text(node: &HtmlNode) -> bool {
    matches!(node, HtmlNode::Text(text) if !text.trim().is_empty())
}

impl HtmlNode {
    fn into_object(self, parent_name: Option<&str>) -> BuiltinResult<ObjectInstance> {
        let mut object = ObjectInstance::new(HTML_TREE_CLASS.to_string());
        match self {
            HtmlNode::Text(text) => {
                let decoded = normalize_space(&decode_html_entities(&text));
                object
                    .properties
                    .insert("Name".to_string(), Value::String("#text".to_string()));
                object
                    .properties
                    .insert("RawHTML".to_string(), Value::String(text));
                object
                    .properties
                    .insert("Text".to_string(), Value::String(decoded));
                object
                    .properties
                    .insert("Attributes".to_string(), Value::Struct(StructValue::new()));
                object.properties.insert(
                    "Children".to_string(),
                    Value::Cell(
                        CellArray::new(Vec::new(), 0, 1)
                            .map_err(|err| html_error("htmlTree", format!("htmlTree: {err}")))?,
                    ),
                );
            }
            HtmlNode::Element(element) => {
                let name = element.name.to_ascii_uppercase();
                let text = HtmlNode::Element(element.clone()).extract_text(ExtractionMethod::Tree);
                let mut attr_struct = StructValue::new();
                for (key, value) in &element.attrs {
                    attr_struct.insert(key.clone(), Value::String(value.clone()));
                }
                let children = element
                    .children
                    .into_iter()
                    .map(|child| child.into_object(Some(&name)).map(Value::Object))
                    .collect::<BuiltinResult<Vec<_>>>()?;
                object
                    .properties
                    .insert("Name".to_string(), Value::String(name));
                object
                    .properties
                    .insert("RawHTML".to_string(), Value::String(element.raw));
                object
                    .properties
                    .insert("Text".to_string(), Value::String(text));
                object
                    .properties
                    .insert("Attributes".to_string(), Value::Struct(attr_struct));
                object.properties.insert(
                    "Children".to_string(),
                    Value::Cell(
                        CellArray::new(children.clone(), children.len(), 1)
                            .map_err(|err| html_error("htmlTree", format!("htmlTree: {err}")))?,
                    ),
                );
            }
        }
        object.properties.insert(
            "Parent".to_string(),
            Value::String(parent_name.unwrap_or(MISSING).to_string()),
        );
        Ok(object)
    }

    fn extract_text(&self, method: ExtractionMethod) -> String {
        let body = find_body(self).unwrap_or(self);
        match method {
            ExtractionMethod::Tree | ExtractionMethod::Article => paragraph_text(body),
            ExtractionMethod::AllText => all_text(body),
        }
    }
}

fn find_body(node: &HtmlNode) -> Option<&HtmlNode> {
    match node {
        HtmlNode::Element(element) if element.name.eq_ignore_ascii_case("body") => Some(node),
        HtmlNode::Element(element) => element.children.iter().find_map(find_body),
        HtmlNode::Text(_) => None,
    }
}

fn paragraph_text(node: &HtmlNode) -> String {
    let mut blocks = Vec::new();
    collect_blocks(node, &mut blocks);
    if blocks.is_empty() {
        normalize_space(&all_text(node))
    } else {
        blocks.join("\n\n")
    }
}

fn collect_blocks(node: &HtmlNode, blocks: &mut Vec<String>) {
    match node {
        HtmlNode::Text(_) => {}
        HtmlNode::Element(element) => {
            if should_skip_text(&element.name) {
                return;
            }
            if is_block_tag(&element.name) {
                let inline = inline_text(element);
                if !inline.is_empty() {
                    blocks.push(inline);
                }
            }
            for child in &element.children {
                collect_blocks(child, blocks);
            }
        }
    }
}

fn inline_text(element: &HtmlElement) -> String {
    let mut text = String::new();
    for child in &element.children {
        match child {
            HtmlNode::Text(raw) => push_decoded_text(&mut text, raw),
            HtmlNode::Element(child_element) => {
                if should_skip_text(&child_element.name) || is_block_tag(&child_element.name) {
                    continue;
                }
                let nested = all_text(child);
                if !nested.is_empty() {
                    if !text.ends_with(char::is_whitespace) && !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&nested);
                }
            }
        }
    }
    normalize_space(&text)
}

fn all_text(node: &HtmlNode) -> String {
    let mut text = String::new();
    collect_all_text(node, &mut text);
    normalize_space(&text)
}

fn collect_all_text(node: &HtmlNode, out: &mut String) {
    match node {
        HtmlNode::Text(raw) => push_decoded_text(out, raw),
        HtmlNode::Element(element) => {
            if should_skip_text(&element.name) {
                return;
            }
            if element.name.eq_ignore_ascii_case("br") {
                out.push(' ');
                return;
            }
            for child in &element.children {
                collect_all_text(child, out);
                if matches!(child, HtmlNode::Element(child_element) if is_block_tag(&child_element.name))
                {
                    out.push(' ');
                }
            }
        }
    }
}

fn push_decoded_text(out: &mut String, raw: &str) {
    let decoded = decode_html_entities(raw);
    if !decoded.is_empty() {
        if !out.ends_with(char::is_whitespace) && !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&decoded);
    }
}

fn normalize_space(text: &str) -> String {
    let mut out = String::new();
    let mut in_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            in_space = true;
        } else {
            if in_space && !out.is_empty() {
                out.push(' ');
            }
            out.push(ch);
            in_space = false;
        }
    }
    out
}

fn should_skip_text(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "script" | "style" | "noscript" | "template" | "head"
    )
}

fn is_block_tag(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "address"
            | "article"
            | "aside"
            | "blockquote"
            | "body"
            | "dd"
            | "div"
            | "dl"
            | "dt"
            | "fieldset"
            | "figcaption"
            | "figure"
            | "footer"
            | "form"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "header"
            | "hr"
            | "li"
            | "main"
            | "nav"
            | "ol"
            | "p"
            | "pre"
            | "section"
            | "table"
            | "tbody"
            | "td"
            | "tfoot"
            | "th"
            | "thead"
            | "tr"
            | "ul"
    )
}

fn decode_html_entities(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(pos) = rest.find('&') {
        out.push_str(&rest[..pos]);
        let after_amp = &rest[pos + 1..];
        let Some(semi) = after_amp.find(';') else {
            out.push('&');
            rest = after_amp;
            continue;
        };
        let entity = &after_amp[..semi];
        if let Some(decoded) = decode_entity(entity) {
            out.push(decoded);
        } else {
            out.push('&');
            out.push_str(entity);
            out.push(';');
        }
        rest = &after_amp[semi + 1..];
    }
    out.push_str(rest);
    out
}

fn decode_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "nbsp" => Some(' '),
        "copy" => Some('\u{00A9}'),
        "reg" => Some('\u{00AE}'),
        "trade" => Some('\u{2122}'),
        value if value.starts_with("#x") || value.starts_with("#X") => {
            u32::from_str_radix(&value[2..], 16)
                .ok()
                .and_then(char::from_u32)
        }
        value if value.starts_with('#') => value[1..].parse::<u32>().ok().and_then(char::from_u32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string_value(value: Value) -> String {
        match value {
            Value::String(text) => text,
            other => panic!("expected string, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extracts_text_from_scalar_html() {
        let html = Value::String(
            "<html><body><h1>THE SONNETS</h1><p>by William Shakespeare</p></body></html>"
                .to_string(),
        );
        let out =
            futures::executor::block_on(extract_html_text_builtin(vec![html])).expect("extract");
        assert_eq!(string_value(out), "THE SONNETS\n\nby William Shakespeare");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extraction_preserves_string_array_shape() {
        let input = StringArray::new(
            vec![
                "<p>alpha</p>".to_string(),
                "<div>beta&nbsp;two</div>".to_string(),
            ],
            vec![1, 2],
        )
        .unwrap();
        let out =
            futures::executor::block_on(extract_html_text_builtin(vec![Value::StringArray(input)]))
                .expect("extract");
        let Value::StringArray(array) = out else {
            panic!("expected string array");
        };
        assert_eq!(array.shape, vec![1, 2]);
        assert_eq!(array.data, vec!["alpha", "beta two"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_text_skips_scripts_and_styles() {
        let html = Value::String(
            "<body><style>.x{}</style><p>visible</p><script>hidden()</script><span>tail</span></body>"
                .to_string(),
        );
        let out = futures::executor::block_on(extract_html_text_builtin(vec![
            html,
            Value::String("ExtractionMethod".to_string()),
            Value::String("all-text".to_string()),
        ]))
        .expect("extract");
        assert_eq!(string_value(out), "visible tail");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn quoted_attribute_gt_does_not_end_tag() {
        let html = Value::String("<p title=\"1 > 0\">ok &amp; done</p>".to_string());
        let out =
            futures::executor::block_on(extract_html_text_builtin(vec![html])).expect("extract");
        assert_eq!(string_value(out), "ok & done");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extraction_accepts_cell_array_of_text() {
        let cell = CellArray::new(
            vec![
                Value::String("<p>first</p>".to_string()),
                Value::String("<p>second</p>".to_string()),
            ],
            2,
            1,
        )
        .unwrap();
        let out = futures::executor::block_on(extract_html_text_builtin(vec![Value::Cell(cell)]))
            .expect("extract");
        let Value::StringArray(array) = out else {
            panic!("expected string array");
        };
        assert_eq!(array.shape, vec![2, 1]);
        assert_eq!(array.data, vec!["first", "second"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn html_tree_returns_object_with_core_properties() {
        let tree = futures::executor::block_on(html_tree_builtin(vec![Value::String(
            "<html><body><p class='lead'>RunMat</p></body></html>".to_string(),
        )]))
        .expect("tree");
        let Value::Object(object) = tree else {
            panic!("expected object");
        };
        assert_eq!(object.class_name, HTML_TREE_CLASS);
        assert_eq!(
            object.properties.get("Name"),
            Some(&Value::String("HTML".to_string()))
        );
        assert_eq!(
            object.properties.get("Text"),
            Some(&Value::String("RunMat".to_string()))
        );
        assert!(matches!(
            object.properties.get("Children"),
            Some(Value::Cell(_))
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn html_tree_preserves_array_shape_as_cell_of_objects() {
        let input = StringArray::new(
            vec!["<p>one</p>".to_string(), "<p>two</p>".to_string()],
            vec![1, 2],
        )
        .unwrap();
        let out = futures::executor::block_on(html_tree_builtin(vec![Value::StringArray(input)]))
            .expect("tree");
        let Value::Cell(cell) = out else {
            panic!("expected cell");
        };
        assert_eq!(cell.shape, vec![1, 2]);
        assert!(cell.data.iter().all(
            |value| matches!(value, Value::Object(object) if object.is_class(HTML_TREE_CLASS))
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extracts_from_html_tree_object() {
        let tree = futures::executor::block_on(html_tree_builtin(vec![Value::String(
            "<article><h2>Title</h2><p>Body &amp; tail</p></article>".to_string(),
        )]))
        .expect("tree");
        let out =
            futures::executor::block_on(extract_html_text_builtin(vec![tree])).expect("extract");
        assert_eq!(string_value(out), "Title\n\nBody & tail");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_extraction_method_errors() {
        let err = futures::executor::block_on(extract_html_text_builtin(vec![
            Value::String("<p>x</p>".to_string()),
            Value::String("ExtractionMethod".to_string()),
            Value::String("summary".to_string()),
        ]))
        .expect_err("invalid method");
        assert!(err.to_string().contains("ExtractionMethod"));
    }
}
