//! MATLAB-compatible HTML text extraction helpers.

use std::collections::{BTreeMap, BTreeSet};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CellArray, ObjectInstance, ResolveContext, StringArray,
    StructValue, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::strings::core::compat::{scalar_text, text_items};
use crate::{build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult};

const HTML_TREE_CLASS: &str = "htmlTree";
const MISSING: &str = "<missing>";

const EXTRACT_HTML_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "extracthtmltext-char-matrix",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "extractHTMLText row-wise character-matrix input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExtractHTMLTextCharMatrixExtension"),
};
const EXTRACT_HTML_BROAD_CELL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "extracthtmltext-broad-cell",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "extractHTMLText with string-valued or mixed htmlTree/text cells is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExtractHTMLTextBroadCellExtension"),
};
const EXTRACT_HTML_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    EXTRACT_HTML_CHAR_MATRIX_EXTENSION,
    EXTRACT_HTML_BROAD_CELL_EXTENSION,
];
pub const EXTRACT_HTML_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "extractHTMLText accepts host text or htmlTree input and textual ExtractionMethod values only. All eight integer classes, logical and complex values, and resident numeric handles reject without numeric-to-text conversion or provider access.",
    };
pub const FIND_ELEMENT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "findElement accepts a scalar htmlTree object and a textual CSS selector. All eight integer classes, logical and complex values, and resident numeric handles reject without numeric-to-object conversion or provider access.",
    };
pub const GET_ATTRIBUTE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "htmlTree.getAttribute accepts host htmlTree objects and a host text attribute name only. All eight integer classes and provider-resident numeric values reject without implicit conversion, gather, or provider access.",
    };

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

const OUT_SUBTREES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "subtrees",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matching htmlTree subtrees.",
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

const IN_TREE_SELECTOR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tree",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "htmlTree object or cell array of htmlTree objects.",
    },
    BuiltinParamDescriptor {
        name: "selector",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "CSS selector.",
    },
];

const IN_TREE_ATTR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tree",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "htmlTree object or cell array of htmlTree objects.",
    },
    BuiltinParamDescriptor {
        name: "attr",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Attribute name.",
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

pub const FIND_ELEMENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "subtrees = findElement(tree, selector)",
        inputs: &IN_TREE_SELECTOR,
        outputs: &OUT_SUBTREES,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const GET_ATTRIBUTE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "str = getAttribute(tree, attr)",
        inputs: &IN_TREE_ATTR,
        outputs: &OUT_TEXT,
    }],
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
    let mut builder = build_runtime_error(message).with_builtin(fn_name);
    if let Some(identifier) = ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
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
    extensions(EXTRACT_HTML_EXTENSIONS),
    integer_audit(crate::builtins::strings::text_analytics::html::EXTRACT_HTML_INTEGER_AUDIT),
    type_resolver(string_type),
    descriptor(crate::builtins::strings::text_analytics::html::EXTRACT_HTML_TEXT_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::html"
)]
async fn extract_html_text_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    preflight_extract_html_args(&args)?;
    let (source, method) = parse_extract_args(args).await?;
    extract_html_text_value(source, method)
}

fn preflight_extract_html_args(args: &[Value]) -> BuiltinResult<()> {
    if let Some(source) = args.first() {
        if numeric_or_resident(source) || contains_numeric_or_resident(source) {
            return Err(html_error(
                "extractHTMLText",
                "extractHTMLText: expected HTML text or htmlTree input",
            ));
        }
        if matches!(source, Value::CharArray(array) if array.rows > 1) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &EXTRACT_HTML_CHAR_MATRIX_EXTENSION,
                "extractHTMLText",
            )?;
        }
        if html_cell_is_broad(source) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &EXTRACT_HTML_BROAD_CELL_EXTENSION,
                "extractHTMLText",
            )?;
        }
    }
    for value in args.iter().skip(1) {
        if numeric_or_resident(value) || contains_numeric_or_resident(value) {
            return Err(html_error(
                "extractHTMLText",
                "extractHTMLText: option names and values must be text scalars",
            ));
        }
    }
    Ok(())
}

fn numeric_or_resident(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::LogicalArray(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
            | Value::Symbolic(_)
            | Value::GpuTensor(_)
    )
}

fn contains_numeric_or_resident(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .any(|value| numeric_or_resident(value) || contains_numeric_or_resident(value)),
        _ => false,
    }
}

fn html_cell_is_broad(value: &Value) -> bool {
    let Value::Cell(cell) = value else {
        return false;
    };
    let contains_tree = cell.data.iter().any(is_html_tree_value);
    if contains_tree {
        cell.data.iter().any(|value| !is_html_tree_value(value))
    } else {
        cell.data
            .iter()
            .any(|value| !matches!(value, Value::CharArray(array) if array.rows <= 1))
    }
}

#[runtime_builtin(
    name = "findElement",
    category = "strings/text_analytics",
    summary = "Find elements in an htmlTree using common CSS selectors.",
    keywords = "findElement,htmlTree,HTML,CSS selector,text analytics",
    accel = "metadata",
    integer_audit(crate::builtins::strings::text_analytics::html::FIND_ELEMENT_INTEGER_AUDIT),
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::html::FIND_ELEMENT_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::html"
)]
async fn find_element_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() != 2 {
        return Err(html_error(
            "findElement",
            "findElement: expected htmlTree and selector",
        ));
    }
    if args
        .iter()
        .any(|value| numeric_or_resident(value) || contains_numeric_or_resident(value))
    {
        return Err(html_error(
            "findElement",
            "findElement: expected htmlTree and textual CSS selector",
        ));
    }
    let tree = gather_if_needed_async(&args[0]).await.map_err(|err| {
        html_error(
            "findElement",
            format!("findElement: failed to gather tree: {err}"),
        )
    })?;
    let selector = gather_if_needed_async(&args[1]).await.map_err(|err| {
        html_error(
            "findElement",
            format!("findElement: failed to gather selector: {err}"),
        )
    })?;
    let selector = scalar_text(&selector, "findElement")?;
    find_element_value(tree, &selector)
}

#[runtime_builtin(
    name = "getAttribute",
    category = "strings/text_analytics",
    summary = "Read an HTML attribute from htmlTree root nodes.",
    keywords = "getAttribute,htmlTree,HTML,attribute,text analytics",
    accel = "metadata",
    integer_audit(crate::builtins::strings::text_analytics::html::GET_ATTRIBUTE_INTEGER_AUDIT),
    type_resolver(string_type),
    descriptor(crate::builtins::strings::text_analytics::html::GET_ATTRIBUTE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::html"
)]
async fn get_attribute_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() != 2 {
        return Err(html_error(
            "getAttribute",
            "getAttribute: expected htmlTree and attribute name",
        ));
    }
    if args
        .iter()
        .any(|value| numeric_or_resident(value) || contains_numeric_or_resident(value))
    {
        return Err(html_error(
            "getAttribute",
            "getAttribute: expected htmlTree and textual attribute name",
        ));
    }
    let attr = attribute_name_text(&args[1])?;
    get_attribute_value(args[0].clone(), &attr)
}

fn attribute_name_text(value: &Value) -> BuiltinResult<String> {
    if let Value::Cell(cell) = value {
        if cell.data.len() == 1 {
            if let Value::CharArray(array) = &cell.data[0] {
                if array.rows <= 1 {
                    return scalar_text(&cell.data[0], "getAttribute");
                }
            }
        }
        return Err(html_error(
            "getAttribute",
            "getAttribute: attribute name cell must contain one character vector",
        ));
    }
    scalar_text(value, "getAttribute")
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

pub(in crate::builtins::strings::text_analytics) fn extract_html_text_value(
    value: Value,
    method: ExtractionMethod,
) -> BuiltinResult<Value> {
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
    Ok(node_from_object(object, "extractHTMLText")?.extract_text(method))
}

fn find_element_value(value: Value, selector: &str) -> BuiltinResult<Value> {
    let selector = Selector::parse(selector)?;
    let mut matches = Vec::new();
    match value {
        Value::Object(object) if object.is_class(HTML_TREE_CLASS) => {
            let root = node_from_object(&object, "findElement")?;
            collect_selector_matches(&root, &selector, &mut matches)?;
        }
        Value::Cell(cell) => {
            for item in cell.data {
                match item {
                    Value::Object(object) if object.is_class(HTML_TREE_CLASS) => {
                        let root = node_from_object(&object, "findElement")?;
                        collect_selector_matches(&root, &selector, &mut matches)?;
                    }
                    other => {
                        return Err(html_error(
                            "findElement",
                            format!("findElement: expected htmlTree object, got {other:?}"),
                        ));
                    }
                }
            }
        }
        Value::Object(object) => {
            return Err(html_error(
                "findElement",
                format!(
                    "findElement: expected htmlTree, got object {}",
                    object.class_name
                ),
            ));
        }
        other => {
            return Err(html_error(
                "findElement",
                format!("findElement: expected htmlTree object, got {other:?}"),
            ));
        }
    }

    let objects = matches
        .into_iter()
        .map(|matched| {
            matched
                .node
                .into_object(matched.parent.as_deref())
                .map(Value::Object)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let rows = objects.len();
    Ok(Value::Cell(CellArray::new(objects, rows, 1).map_err(
        |err| html_error("findElement", format!("findElement: {err}")),
    )?))
}

fn get_attribute_value(value: Value, attr: &str) -> BuiltinResult<Value> {
    let attr = attr.trim().to_ascii_lowercase();
    if attr.is_empty() {
        return Err(html_error(
            "getAttribute",
            "getAttribute: attribute name must be nonempty",
        ));
    }
    match value {
        Value::Object(object) if object.is_class(HTML_TREE_CLASS) => {
            Ok(Value::String(attribute_from_object(&object, &attr)?))
        }
        Value::Object(object) => Err(html_error(
            "getAttribute",
            format!(
                "getAttribute: expected htmlTree, got object {}",
                object.class_name
            ),
        )),
        Value::Cell(cell) => {
            let shape = cell.shape.clone();
            let mut values = Vec::with_capacity(cell.data.len());
            for item in cell.data {
                match item {
                    Value::Object(object) if object.is_class(HTML_TREE_CLASS) => {
                        values.push(attribute_from_object(&object, &attr)?);
                    }
                    other => {
                        return Err(html_error(
                            "getAttribute",
                            format!("getAttribute: expected htmlTree object, got {other:?}"),
                        ));
                    }
                }
            }
            string_output(values, shape, "getAttribute")
        }
        other => Err(html_error(
            "getAttribute",
            format!("getAttribute: expected htmlTree object, got {other:?}"),
        )),
    }
}

fn attribute_from_object(object: &ObjectInstance, attr: &str) -> BuiltinResult<String> {
    let node = node_from_object(object, "getAttribute")?;
    match node {
        HtmlNode::Element(element) => Ok(element
            .attrs
            .get(attr)
            .cloned()
            .unwrap_or_else(|| MISSING.to_string())),
        HtmlNode::Text(_) => Ok(MISSING.to_string()),
    }
}

fn node_from_object(object: &ObjectInstance, fn_name: &str) -> BuiltinResult<HtmlNode> {
    match object.properties.get("RawHTML") {
        Some(Value::String(html)) => parse_html_tree(html),
        Some(Value::CharArray(_)) | Some(Value::StringArray(_)) => {
            let html = scalar_text(
                object
                    .properties
                    .get("RawHTML")
                    .expect("RawHTML checked above"),
                fn_name,
            )?;
            parse_html_tree(&html)
        }
        _ => Err(html_error(
            fn_name,
            format!("{fn_name}: invalid htmlTree object"),
        )),
    }
}

fn collect_selector_matches(
    root: &HtmlNode,
    selector: &Selector,
    out: &mut Vec<MatchedNode>,
) -> BuiltinResult<()> {
    let mut seen = BTreeSet::<Vec<usize>>::new();
    for chain in &selector.chains {
        for path in execute_selector_chain(root, chain)? {
            seen.insert(path);
        }
    }
    for path in seen {
        out.push(MatchedNode {
            node: node_at(root, &path)
                .ok_or_else(|| html_error("findElement", "findElement: invalid match"))?
                .clone(),
            parent: parent_name(root, &path).map(str::to_ascii_uppercase),
        });
    }
    Ok(())
}

fn execute_selector_chain(
    root: &HtmlNode,
    chain: &SelectorChain,
) -> BuiltinResult<Vec<Vec<usize>>> {
    let Some(first) = chain.parts.first() else {
        return Ok(Vec::new());
    };
    let mut candidates = all_paths(root)
        .into_iter()
        .filter(|path| matches_simple_selector(root, path, &first.simple))
        .collect::<Vec<_>>();

    for part in chain.parts.iter().skip(1) {
        let mut next = BTreeSet::<Vec<usize>>::new();
        for path in &candidates {
            match part.combinator.unwrap_or(Combinator::Descendant) {
                Combinator::Descendant => {
                    for descendant in descendant_paths(root, path) {
                        if matches_simple_selector(root, &descendant, &part.simple) {
                            next.insert(descendant);
                        }
                    }
                }
                Combinator::Child => {
                    for child in child_paths(root, path) {
                        if matches_simple_selector(root, &child, &part.simple) {
                            next.insert(child);
                        }
                    }
                }
                Combinator::AdjacentSibling => {
                    if let Some(sibling) = next_sibling_path(root, path) {
                        if matches_simple_selector(root, &sibling, &part.simple) {
                            next.insert(sibling);
                        }
                    }
                }
                Combinator::GeneralSibling => {
                    for sibling in following_sibling_paths(root, path) {
                        if matches_simple_selector(root, &sibling, &part.simple) {
                            next.insert(sibling);
                        }
                    }
                }
            }
        }
        candidates = next.into_iter().collect();
    }
    Ok(candidates)
}

fn all_paths(root: &HtmlNode) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    collect_paths(root, &mut Vec::new(), &mut out);
    out
}

fn collect_paths(node: &HtmlNode, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    out.push(path.clone());
    if let HtmlNode::Element(element) = node {
        for (idx, child) in element.children.iter().enumerate() {
            path.push(idx);
            collect_paths(child, path, out);
            path.pop();
        }
    }
}

fn descendant_paths(root: &HtmlNode, path: &[usize]) -> Vec<Vec<usize>> {
    let Some(node) = node_at(root, path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut current = path.to_vec();
    if let HtmlNode::Element(element) = node {
        for (idx, child) in element.children.iter().enumerate() {
            current.push(idx);
            collect_paths(child, &mut current, &mut out);
            current.pop();
        }
    }
    out
}

fn child_paths(root: &HtmlNode, path: &[usize]) -> Vec<Vec<usize>> {
    match node_at(root, path) {
        Some(HtmlNode::Element(element)) => (0..element.children.len())
            .map(|idx| {
                let mut child = path.to_vec();
                child.push(idx);
                child
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn next_sibling_path(root: &HtmlNode, path: &[usize]) -> Option<Vec<usize>> {
    let (last, parent_path) = path.split_last()?;
    let parent = node_at(root, parent_path)?;
    let HtmlNode::Element(element) = parent else {
        return None;
    };
    let mut next = *last + 1;
    while next < element.children.len() {
        if matches!(element.children.get(next), Some(HtmlNode::Element(_))) {
            let mut sibling = parent_path.to_vec();
            sibling.push(next);
            return Some(sibling);
        }
        next += 1;
    }
    None
}

fn following_sibling_paths(root: &HtmlNode, path: &[usize]) -> Vec<Vec<usize>> {
    let Some((last, parent_path)) = path.split_last() else {
        return Vec::new();
    };
    let Some(HtmlNode::Element(parent)) = node_at(root, parent_path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for index in (*last + 1)..parent.children.len() {
        if matches!(parent.children.get(index), Some(HtmlNode::Element(_))) {
            let mut sibling = parent_path.to_vec();
            sibling.push(index);
            out.push(sibling);
        }
    }
    out
}

fn node_at<'a>(root: &'a HtmlNode, path: &[usize]) -> Option<&'a HtmlNode> {
    let mut node = root;
    for index in path {
        let HtmlNode::Element(element) = node else {
            return None;
        };
        node = element.children.get(*index)?;
    }
    Some(node)
}

fn matches_simple_selector(root: &HtmlNode, path: &[usize], selector: &SimpleSelector) -> bool {
    node_at(root, path)
        .map(|node| selector.matches(root, path, node))
        .unwrap_or(false)
}

fn parent_name<'a>(root: &'a HtmlNode, path: &[usize]) -> Option<&'a str> {
    let (_, parent_path) = path.split_last()?;
    match node_at(root, parent_path) {
        Some(HtmlNode::Element(element)) => Some(element.name.as_str()),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::strings::text_analytics) enum ExtractionMethod {
    Tree,
    Article,
    AllText,
}

impl ExtractionMethod {
    pub(in crate::builtins::strings::text_analytics) fn parse(value: &str) -> BuiltinResult<Self> {
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

#[derive(Clone, Debug)]
struct Selector {
    chains: Vec<SelectorChain>,
}

#[derive(Clone, Debug)]
struct SelectorChain {
    parts: Vec<SelectorPart>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Combinator {
    Descendant,
    Child,
    AdjacentSibling,
    GeneralSibling,
}

#[derive(Clone, Debug)]
struct SelectorPart {
    combinator: Option<Combinator>,
    simple: SimpleSelector,
}

#[derive(Clone, Debug, Default)]
struct SimpleSelector {
    tag: Option<String>,
    id: Option<String>,
    classes: Vec<String>,
    attrs: Vec<AttrSelector>,
    empty: Option<bool>,
    first_child: Option<bool>,
    first_of_type: Option<bool>,
}

#[derive(Clone, Debug)]
struct AttrSelector {
    name: String,
    op: AttrSelectorOp,
    value: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttrSelectorOp {
    Exists,
    Exact,
    WhitespaceListContains,
    DashPrefix,
    Prefix,
    Suffix,
    Contains,
}

#[derive(Clone, Debug)]
struct MatchedNode {
    node: HtmlNode,
    parent: Option<String>,
}

impl Selector {
    fn parse(input: &str) -> BuiltinResult<Self> {
        let input = input.trim();
        if input.is_empty() {
            return Err(html_error(
                "findElement",
                "findElement: selector must be nonempty",
            ));
        }
        let mut chains = Vec::new();
        for group in split_selector_groups(input)? {
            let chain = SelectorChain::parse(group.trim())?;
            if !chain.parts.is_empty() {
                chains.push(chain);
            }
        }
        if chains.is_empty() {
            return Err(html_error(
                "findElement",
                "findElement: selector must contain at least one simple selector",
            ));
        }
        Ok(Self { chains })
    }
}

impl SelectorChain {
    fn parse(input: &str) -> BuiltinResult<Self> {
        let mut parts = Vec::new();
        let mut index = 0;
        let mut pending = None;

        while index < input.len() {
            let (after_space, had_space) = skip_selector_space(input, index);
            index = after_space;
            if index >= input.len() {
                break;
            }
            let ch = input[index..].chars().next().expect("in bounds");
            match ch {
                '>' | '+' | '~' => {
                    if parts.is_empty() {
                        return Err(html_error(
                            "findElement",
                            "findElement: selector cannot start with a combinator",
                        ));
                    }
                    if pending.is_some() {
                        return Err(html_error(
                            "findElement",
                            "findElement: selector cannot contain repeated combinators",
                        ));
                    }
                    pending = Some(match ch {
                        '>' => Combinator::Child,
                        '+' => Combinator::AdjacentSibling,
                        '~' => Combinator::GeneralSibling,
                        _ => unreachable!("matched selector combinator"),
                    });
                    index += ch.len_utf8();
                    continue;
                }
                _ => {}
            }
            if had_space && !parts.is_empty() && pending.is_none() {
                pending = Some(Combinator::Descendant);
            }

            let end = simple_selector_end(input, index)?;
            if end == index {
                return Err(html_error(
                    "findElement",
                    format!("findElement: invalid selector near '{}'", &input[index..]),
                ));
            }
            let simple = SimpleSelector::parse(&input[index..end])?;
            let combinator = if parts.is_empty() {
                if pending.is_some() {
                    return Err(html_error(
                        "findElement",
                        "findElement: selector cannot start with a combinator",
                    ));
                }
                None
            } else {
                pending.take().or(Some(Combinator::Descendant))
            };
            parts.push(SelectorPart { combinator, simple });
            index = end;
        }

        if pending.is_some() {
            return Err(html_error(
                "findElement",
                "findElement: selector cannot end with a combinator",
            ));
        }
        Ok(Self { parts })
    }
}

impl SimpleSelector {
    fn parse(input: &str) -> BuiltinResult<Self> {
        let mut selector = Self::default();
        let mut index = 0;

        if let Some(ch) = input.chars().next() {
            if ch == '*' {
                index += ch.len_utf8();
            } else if is_selector_ident_start(ch) {
                let end = consume_selector_ident(input, index);
                selector.tag = Some(input[index..end].to_ascii_lowercase());
                index = end;
            }
        }

        while index < input.len() {
            let ch = input[index..].chars().next().expect("in bounds");
            match ch {
                '.' => {
                    index += ch.len_utf8();
                    let end = consume_selector_ident(input, index);
                    if end == index {
                        return Err(html_error(
                            "findElement",
                            "findElement: class selector requires a name",
                        ));
                    }
                    selector.classes.push(input[index..end].to_string());
                    index = end;
                }
                '#' => {
                    index += ch.len_utf8();
                    let end = consume_selector_ident(input, index);
                    if end == index {
                        return Err(html_error(
                            "findElement",
                            "findElement: id selector requires a name",
                        ));
                    }
                    selector.id = Some(input[index..end].to_string());
                    index = end;
                }
                '[' => {
                    let Some(end) = matching_selector_bracket(input, index, '[', ']')? else {
                        return Err(html_error(
                            "findElement",
                            "findElement: attribute selector is missing ']'",
                        ));
                    };
                    selector
                        .attrs
                        .push(AttrSelector::parse(&input[index + 1..end])?);
                    index = end + 1;
                }
                ':' => {
                    let end = parse_pseudo_selector(input, index, &mut selector)?;
                    index = end;
                }
                _ => {
                    return Err(html_error(
                        "findElement",
                        format!("findElement: unsupported selector token '{ch}'"),
                    ));
                }
            }
        }

        Ok(selector)
    }

    fn matches(&self, root: &HtmlNode, path: &[usize], node: &HtmlNode) -> bool {
        let HtmlNode::Element(element) = node else {
            return false;
        };
        if let Some(tag) = &self.tag {
            if !element.name.eq_ignore_ascii_case(tag) {
                return false;
            }
        }
        if let Some(id) = &self.id {
            if element.attrs.get("id") != Some(id) {
                return false;
            }
        }
        if !self.classes.is_empty() {
            let classes = element
                .attrs
                .get("class")
                .map(|value| value.split_whitespace().collect::<Vec<_>>())
                .unwrap_or_default();
            if self
                .classes
                .iter()
                .any(|class| !classes.iter().any(|existing| existing == class))
            {
                return false;
            }
        }
        for attr in &self.attrs {
            let Some(value) = element.attrs.get(&attr.name) else {
                return false;
            };
            if let Some(expected) = &attr.value {
                if !attr.op.matches(value, expected) {
                    return false;
                }
            }
        }
        if let Some(expect_empty) = self.empty {
            let is_empty = element.children.iter().all(|child| match child {
                HtmlNode::Element(_) => false,
                HtmlNode::Text(text) => text.trim().is_empty(),
            });
            if is_empty != expect_empty {
                return false;
            }
        }
        if let Some(expect_first_child) = self.first_child {
            if is_first_element_child(root, path) != expect_first_child {
                return false;
            }
        }
        if let Some(expect_first_of_type) = self.first_of_type {
            if is_first_element_of_type(root, path, &element.name) != expect_first_of_type {
                return false;
            }
        }
        true
    }
}

impl AttrSelector {
    fn parse(input: &str) -> BuiltinResult<Self> {
        let input = input.trim();
        if input.is_empty() {
            return Err(html_error(
                "findElement",
                "findElement: attribute selector requires a name",
            ));
        }
        let Some((name, op, raw_value)) = split_attr_selector(input) else {
            return Ok(Self {
                name: input.to_ascii_lowercase(),
                op: AttrSelectorOp::Exists,
                value: None,
            });
        };
        let name = name.trim();
        if name.is_empty() {
            return Err(html_error(
                "findElement",
                "findElement: attribute selector requires a name",
            ));
        }
        let value = strip_selector_quotes(raw_value.trim())?;
        Ok(Self {
            name: name.to_ascii_lowercase(),
            op,
            value: Some(decode_html_entities(value)),
        })
    }
}

impl AttrSelectorOp {
    fn matches(self, actual: &str, expected: &str) -> bool {
        match self {
            Self::Exists => true,
            Self::Exact => actual == expected,
            Self::WhitespaceListContains => actual.split_whitespace().any(|part| part == expected),
            Self::DashPrefix => actual == expected || actual.starts_with(&format!("{expected}-")),
            Self::Prefix => actual.starts_with(expected),
            Self::Suffix => actual.ends_with(expected),
            Self::Contains => actual.contains(expected),
        }
    }
}

fn split_attr_selector(input: &str) -> Option<(&str, AttrSelectorOp, &str)> {
    let mut quote = None;
    for (idx, ch) in input.char_indices() {
        match (quote, ch) {
            (Some(active), current) if current == active => quote = None,
            (Some(_), _) => {}
            (None, '"' | '\'') => quote = Some(ch),
            (None, '=') => {
                let before = input[..idx].trim_end();
                let Some((op, name)) = before
                    .strip_suffix('~')
                    .map(|name| (AttrSelectorOp::WhitespaceListContains, name))
                    .or_else(|| {
                        before
                            .strip_suffix('|')
                            .map(|name| (AttrSelectorOp::DashPrefix, name))
                    })
                    .or_else(|| {
                        before
                            .strip_suffix('^')
                            .map(|name| (AttrSelectorOp::Prefix, name))
                    })
                    .or_else(|| {
                        before
                            .strip_suffix('$')
                            .map(|name| (AttrSelectorOp::Suffix, name))
                    })
                    .or_else(|| {
                        before
                            .strip_suffix('*')
                            .map(|name| (AttrSelectorOp::Contains, name))
                    })
                else {
                    return Some((before, AttrSelectorOp::Exact, &input[idx + ch.len_utf8()..]));
                };
                return Some((name, op, &input[idx + ch.len_utf8()..]));
            }
            _ => {}
        }
    }
    None
}

fn split_selector_groups(input: &str) -> BuiltinResult<Vec<&str>> {
    let mut groups = Vec::new();
    let mut start = 0;
    let mut quote = None;
    let mut bracket_depth = 0usize;
    let mut paren_depth = 0usize;
    for (idx, ch) in input.char_indices() {
        match (quote, ch) {
            (Some(active), current) if current == active => quote = None,
            (Some(_), _) => {}
            (None, '"' | '\'') => quote = Some(ch),
            (None, '[') => bracket_depth += 1,
            (None, ']') => bracket_depth = bracket_depth.saturating_sub(1),
            (None, '(') => paren_depth += 1,
            (None, ')') => paren_depth = paren_depth.saturating_sub(1),
            (None, ',') if bracket_depth == 0 && paren_depth == 0 => {
                groups.push(&input[start..idx]);
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    groups.push(&input[start..]);
    if groups.iter().any(|group| group.trim().is_empty()) {
        return Err(html_error(
            "findElement",
            "findElement: selector group must be nonempty",
        ));
    }
    Ok(groups)
}

fn skip_selector_space(input: &str, mut index: usize) -> (usize, bool) {
    let mut skipped = false;
    while index < input.len() {
        let ch = input[index..].chars().next().expect("in bounds");
        if !ch.is_whitespace() {
            break;
        }
        skipped = true;
        index += ch.len_utf8();
    }
    (index, skipped)
}

fn simple_selector_end(input: &str, start: usize) -> BuiltinResult<usize> {
    let mut quote = None;
    let mut bracket_depth = 0usize;
    let mut paren_depth = 0usize;
    for (rel, ch) in input[start..].char_indices() {
        let idx = start + rel;
        match (quote, ch) {
            (Some(active), current) if current == active => quote = None,
            (Some(_), _) => {}
            (None, '"' | '\'') => quote = Some(ch),
            (None, '[') => bracket_depth += 1,
            (None, ']') => {
                bracket_depth = bracket_depth.checked_sub(1).ok_or_else(|| {
                    html_error("findElement", "findElement: unmatched ']' in selector")
                })?;
            }
            (None, '(') => paren_depth += 1,
            (None, ')') => {
                paren_depth = paren_depth.checked_sub(1).ok_or_else(|| {
                    html_error("findElement", "findElement: unmatched ')' in selector")
                })?;
            }
            (None, '>' | '+' | '~') if bracket_depth == 0 && paren_depth == 0 => return Ok(idx),
            (None, current)
                if current.is_whitespace() && bracket_depth == 0 && paren_depth == 0 =>
            {
                return Ok(idx);
            }
            _ => {}
        }
    }
    if quote.is_some() || bracket_depth != 0 || paren_depth != 0 {
        return Err(html_error(
            "findElement",
            "findElement: selector has unclosed quote, bracket, or parenthesis",
        ));
    }
    Ok(input.len())
}

fn matching_selector_bracket(
    input: &str,
    start: usize,
    open: char,
    close: char,
) -> BuiltinResult<Option<usize>> {
    let mut quote = None;
    let mut depth = 0usize;
    for (rel, ch) in input[start..].char_indices() {
        let idx = start + rel;
        match (quote, ch) {
            (Some(active), current) if current == active => quote = None,
            (Some(_), _) => {}
            (None, '"' | '\'') => quote = Some(ch),
            (None, current) if current == open => depth += 1,
            (None, current) if current == close => {
                depth = depth.checked_sub(1).ok_or_else(|| {
                    html_error("findElement", "findElement: unmatched selector delimiter")
                })?;
                if depth == 0 {
                    return Ok(Some(idx));
                }
            }
            _ => {}
        }
    }
    Ok(None)
}

fn parse_pseudo_selector(
    input: &str,
    start: usize,
    selector: &mut SimpleSelector,
) -> BuiltinResult<usize> {
    let name_start = start + 1;
    let name_end = consume_selector_ident(input, name_start);
    if name_end == name_start {
        return Err(html_error(
            "findElement",
            "findElement: pseudo-class selector requires a name",
        ));
    }
    let name = &input[name_start..name_end];
    if name.eq_ignore_ascii_case("empty") {
        selector.empty = Some(true);
        return Ok(name_end);
    }
    if name.eq_ignore_ascii_case("first-child") {
        selector.first_child = Some(true);
        return Ok(name_end);
    }
    if name.eq_ignore_ascii_case("first-of-type") {
        selector.first_of_type = Some(true);
        return Ok(name_end);
    }
    if name.eq_ignore_ascii_case("not") {
        if !input[name_end..].starts_with('(') {
            return Err(html_error(
                "findElement",
                "findElement: :not selector requires parentheses",
            ));
        }
        let Some(close) = matching_selector_bracket(input, name_end, '(', ')')? else {
            return Err(html_error(
                "findElement",
                "findElement: :not selector is missing ')'",
            ));
        };
        let inner = input[name_end + 1..close].trim();
        apply_negated_simple_pseudo(inner, selector)?;
        return Ok(close + 1);
    }
    Err(html_error(
        "findElement",
        format!("findElement: unsupported pseudo-class ':{name}'"),
    ))
}

fn apply_negated_simple_pseudo(input: &str, selector: &mut SimpleSelector) -> BuiltinResult<()> {
    let Some(name) = input.trim().strip_prefix(':') else {
        return Err(html_error(
            "findElement",
            "findElement: :not selector supports simple pseudo-classes only",
        ));
    };
    if name.eq_ignore_ascii_case("empty") {
        selector.empty = Some(false);
        return Ok(());
    }
    if name.eq_ignore_ascii_case("first-child") {
        selector.first_child = Some(false);
        return Ok(());
    }
    if name.eq_ignore_ascii_case("first-of-type") {
        selector.first_of_type = Some(false);
        return Ok(());
    }
    Err(html_error(
        "findElement",
        "findElement: :not selector supports :empty, :first-child, or :first-of-type",
    ))
}

fn is_first_element_child(root: &HtmlNode, path: &[usize]) -> bool {
    let Some((last, parent_path)) = path.split_last() else {
        return false;
    };
    let Some(HtmlNode::Element(parent)) = node_at(root, parent_path) else {
        return false;
    };
    parent
        .children
        .iter()
        .position(|child| matches!(child, HtmlNode::Element(_)))
        == Some(*last)
}

fn is_first_element_of_type(root: &HtmlNode, path: &[usize], name: &str) -> bool {
    let Some((last, parent_path)) = path.split_last() else {
        return false;
    };
    let Some(HtmlNode::Element(parent)) = node_at(root, parent_path) else {
        return false;
    };
    parent.children.iter().enumerate().find_map(|(idx, child)| {
        if matches!(child, HtmlNode::Element(element) if element.name.eq_ignore_ascii_case(name)) {
            Some(idx)
        } else {
            None
        }
    }) == Some(*last)
}

fn strip_selector_quotes(input: &str) -> BuiltinResult<&str> {
    let Some(first) = input.chars().next() else {
        return Ok(input);
    };
    if first != '"' && first != '\'' {
        return Ok(input);
    }
    let Some(last) = input.chars().last() else {
        return Ok(input);
    };
    if first != last || input.len() < first.len_utf8() + last.len_utf8() {
        return Err(html_error(
            "findElement",
            "findElement: attribute selector has an unterminated quoted value",
        ));
    }
    Ok(&input[first.len_utf8()..input.len() - last.len_utf8()])
}

fn consume_selector_ident(input: &str, start: usize) -> usize {
    let mut end = start;
    for (rel, ch) in input[start..].char_indices() {
        if rel == 0 {
            if !is_selector_ident_start(ch) {
                break;
            }
        } else if !is_selector_ident_continue(ch) {
            break;
        }
        end = start + rel + ch.len_utf8();
    }
    end
}

fn is_selector_ident_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_' || ch == '-'
}

fn is_selector_ident_continue(ch: char) -> bool {
    is_selector_ident_start(ch) || ch.is_ascii_digit()
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
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
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
    fn find_element_supports_common_selectors_and_attribute_reads() {
        let tree = futures::executor::block_on(html_tree_builtin(vec![Value::String(
            "<div><a id='home' class='nav primary' href='/home'>Home</a><a class='nav'>Docs</a><p data-kind='lead'>Lead</p></div>"
                .to_string(),
        )]))
        .expect("tree");
        let links = futures::executor::block_on(find_element_builtin(vec![
            tree.clone(),
            Value::String("a.nav".to_string()),
        ]))
        .expect("find links");
        let Value::Cell(link_cell) = links else {
            panic!("expected cell of htmlTree objects");
        };
        assert_eq!(link_cell.shape, vec![2, 1]);

        let link_text = futures::executor::block_on(extract_html_text_builtin(vec![Value::Cell(
            link_cell.clone(),
        )]))
        .expect("extract link text");
        let Value::StringArray(link_text) = link_text else {
            panic!("expected string array");
        };
        assert_eq!(link_text.shape, vec![2, 1]);
        assert_eq!(link_text.data, vec!["Home", "Docs"]);

        let hrefs = futures::executor::block_on(get_attribute_builtin(vec![
            Value::Cell(link_cell),
            Value::String("href".to_string()),
        ]))
        .expect("hrefs");
        let Value::StringArray(hrefs) = hrefs else {
            panic!("expected string array");
        };
        assert_eq!(hrefs.shape, vec![2, 1]);
        assert_eq!(hrefs.data, vec!["/home", MISSING]);

        let lead = futures::executor::block_on(find_element_builtin(vec![
            tree.clone(),
            Value::String("[data-kind=lead]".to_string()),
        ]))
        .expect("find attr");
        let Value::Cell(lead) = lead else {
            panic!("expected cell");
        };
        assert_eq!(lead.shape, vec![1, 1]);

        let home = futures::executor::block_on(find_element_builtin(vec![
            tree,
            Value::String("#home".to_string()),
        ]))
        .expect("find id");
        let Value::Cell(home) = home else {
            panic!("expected cell");
        };
        assert_eq!(home.shape, vec![1, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_element_supports_combinators_empty_not_empty_and_groups() {
        let tree = futures::executor::block_on(html_tree_builtin(vec![Value::String(
            "<section><h1>Title</h1>\n<p class='lead'>Lead</p><label></label><label>Name</label></section>"
                .to_string(),
        )]))
        .expect("tree");
        let adjacent = futures::executor::block_on(find_element_builtin(vec![
            tree.clone(),
            Value::String("h1 + p".to_string()),
        ]))
        .expect("adjacent");
        let Value::Cell(adjacent) = adjacent else {
            panic!("expected cell");
        };
        assert_eq!(adjacent.shape, vec![1, 1]);

        let child = futures::executor::block_on(find_element_builtin(vec![
            tree.clone(),
            Value::String("section > p.lead".to_string()),
        ]))
        .expect("child");
        let Value::Cell(child) = child else {
            panic!("expected cell");
        };
        assert_eq!(child.shape, vec![1, 1]);

        let empty = futures::executor::block_on(find_element_builtin(vec![
            tree.clone(),
            Value::String("label:empty".to_string()),
        ]))
        .expect("empty");
        let Value::Cell(empty) = empty else {
            panic!("expected cell");
        };
        assert_eq!(empty.shape, vec![1, 1]);

        let not_empty = futures::executor::block_on(find_element_builtin(vec![
            tree,
            Value::String("label:not(:empty), h1".to_string()),
        ]))
        .expect("not empty");
        let Value::Cell(not_empty) = not_empty else {
            panic!("expected cell");
        };
        assert_eq!(not_empty.shape, vec![2, 1]);
        let text =
            futures::executor::block_on(extract_html_text_builtin(vec![Value::Cell(not_empty)]))
                .expect("extract group order");
        let Value::StringArray(text) = text else {
            panic!("expected string array");
        };
        assert_eq!(text.data, vec!["Title", "Name"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn get_attribute_returns_missing_for_absent_scalar_attribute() {
        let tree = futures::executor::block_on(html_tree_builtin(vec![Value::String(
            "<a href='/home'>Home</a>".to_string(),
        )]))
        .expect("tree");
        let out = futures::executor::block_on(get_attribute_builtin(vec![
            tree,
            Value::String("title".to_string()),
        ]))
        .expect("attribute");
        assert_eq!(string_value(out), MISSING);
    }

    #[test]
    fn get_attribute_accepts_scalar_cell_character_attribute_name() {
        let tree = futures::executor::block_on(html_tree_builtin(vec![Value::String(
            "<a href='/home'>Home</a>".to_string(),
        )]))
        .expect("tree");
        let attr = Value::Cell(
            CellArray::new(
                vec![Value::CharArray(runmat_builtins::CharArray::new_row(
                    "href",
                ))],
                1,
                1,
            )
            .expect("scalar cell attribute"),
        );
        let out = futures::executor::block_on(get_attribute_builtin(vec![tree, attr]))
            .expect("attribute");
        assert_eq!(string_value(out), "/home");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_element_handles_quoted_attribute_selectors_and_empty_results() {
        let tree = futures::executor::block_on(html_tree_builtin(vec![Value::String(
            "<div><p data-kind=\"lead item\">Lead</p></div>".to_string(),
        )]))
        .expect("tree");
        let quoted = futures::executor::block_on(find_element_builtin(vec![
            tree.clone(),
            Value::String("p[data-kind=\"lead item\"]".to_string()),
        ]))
        .expect("quoted attribute selector");
        let Value::Cell(quoted) = quoted else {
            panic!("expected cell");
        };
        assert_eq!(quoted.shape, vec![1, 1]);

        let missing = futures::executor::block_on(find_element_builtin(vec![
            tree,
            Value::String("span.unknown".to_string()),
        ]))
        .expect("empty result");
        let Value::Cell(missing) = missing else {
            panic!("expected cell");
        };
        assert_eq!(missing.shape, vec![0, 1]);
        assert!(missing.data.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_element_supports_css_attribute_operator_selectors() {
        let tree = futures::executor::block_on(html_tree_builtin(vec![Value::String(
            "<div><a href='manual.pdf' hreflang='en-US' rel='tag external' data-code='abc-123'>Manual</a><a href='guide.html' hreflang='en' rel='external' data-code='xyz'>Guide</a><a href='notes.txt' data-code='a$=b'>Notes</a></div>"
                .to_string(),
        )]))
        .expect("tree");

        for (selector, expected) in [
            ("a[href$='.pdf']", vec!["Manual"]),
            ("a[href^='guide']", vec!["Guide"]),
            ("a[rel~='tag']", vec!["Manual"]),
            ("a[hreflang|='en']", vec!["Manual", "Guide"]),
            ("a[data-code*='bc-']", vec!["Manual"]),
            ("a[data-code='a$=b']", vec!["Notes"]),
        ] {
            let matches = futures::executor::block_on(find_element_builtin(vec![
                tree.clone(),
                Value::String(selector.to_string()),
            ]))
            .unwrap_or_else(|err| panic!("{selector} failed: {err}"));
            let text = futures::executor::block_on(extract_html_text_builtin(vec![matches]))
                .unwrap_or_else(|err| panic!("{selector} text extraction failed: {err}"));
            match text {
                Value::String(text) => assert_eq!(vec![text], expected),
                Value::StringArray(array) => assert_eq!(array.data, expected),
                other => panic!("expected extracted text for {selector}, got {other:?}"),
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_element_supports_first_child_first_of_type_and_general_sibling() {
        let tree = futures::executor::block_on(html_tree_builtin(vec![Value::String(
            "<section><p>Intro</p><span>Aside</span><a>First link</a><a>Second link</a></section>"
                .to_string(),
        )]))
        .expect("tree");

        let first_child = futures::executor::block_on(find_element_builtin(vec![
            tree.clone(),
            Value::String("p:first-child".to_string()),
        ]))
        .expect("first child");
        let text = futures::executor::block_on(extract_html_text_builtin(vec![first_child]))
            .expect("first child text");
        assert_eq!(string_value(text), "Intro");

        let first_of_type = futures::executor::block_on(find_element_builtin(vec![
            tree.clone(),
            Value::String("a:first-of-type".to_string()),
        ]))
        .expect("first of type");
        let text = futures::executor::block_on(extract_html_text_builtin(vec![first_of_type]))
            .expect("first of type text");
        assert_eq!(string_value(text), "First link");

        let following_links = futures::executor::block_on(find_element_builtin(vec![
            tree,
            Value::String("p ~ a".to_string()),
        ]))
        .expect("general sibling");
        let text = futures::executor::block_on(extract_html_text_builtin(vec![following_links]))
            .expect("general sibling text");
        let Value::StringArray(text) = text else {
            panic!("expected string array");
        };
        assert_eq!(text.data, vec!["First link", "Second link"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_element_rejects_unsupported_selectors() {
        let tree =
            futures::executor::block_on(html_tree_builtin(vec![Value::String("<p>x</p>".into())]))
                .expect("tree");
        let err = futures::executor::block_on(find_element_builtin(vec![
            tree,
            Value::String("p:hover".to_string()),
        ]))
        .expect_err("unsupported selector");
        assert!(err.to_string().contains("unsupported pseudo-class"));

        let tree =
            futures::executor::block_on(html_tree_builtin(vec![Value::String("<p>x</p>".into())]))
                .expect("tree");
        let err = futures::executor::block_on(find_element_builtin(vec![
            tree,
            Value::String("div > + p".to_string()),
        ]))
        .expect_err("repeated combinator");
        assert!(err.to_string().contains("repeated combinators"));
    }

    #[test]
    fn find_element_integer_audit_rejects_numeric_inputs_before_provider_access() {
        assert_eq!(
            FIND_ELEMENT_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        for value in [
            Value::Int(runmat_builtins::IntValue::I8(1)),
            Value::Int(runmat_builtins::IntValue::I16(1)),
            Value::Int(runmat_builtins::IntValue::I32(1)),
            Value::Int(runmat_builtins::IntValue::I64(1)),
            Value::Int(runmat_builtins::IntValue::U8(1)),
            Value::Int(runmat_builtins::IntValue::U16(1)),
            Value::Int(runmat_builtins::IntValue::U32(1)),
            Value::Int(runmat_builtins::IntValue::U64(1)),
        ] {
            let error = futures::executor::block_on(find_element_builtin(vec![
                value,
                Value::String("p".into()),
            ]))
            .expect_err("integer tree input");
            assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
        }
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let error = futures::executor::block_on(find_element_builtin(vec![
            resident,
            Value::String("p".into()),
        ]))
        .expect_err("resident tree input");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn get_attribute_integer_audit_rejects_all_numeric_roles_before_provider_access() {
        assert_eq!(
            GET_ATTRIBUTE_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        for value in [
            Value::Int(runmat_builtins::IntValue::I8(1)),
            Value::Int(runmat_builtins::IntValue::I16(1)),
            Value::Int(runmat_builtins::IntValue::I32(1)),
            Value::Int(runmat_builtins::IntValue::I64(1)),
            Value::Int(runmat_builtins::IntValue::U8(1)),
            Value::Int(runmat_builtins::IntValue::U16(1)),
            Value::Int(runmat_builtins::IntValue::U32(1)),
            Value::Int(runmat_builtins::IntValue::U64(1)),
        ] {
            let error = futures::executor::block_on(get_attribute_builtin(vec![
                value.clone(),
                Value::String("href".into()),
            ]))
            .expect_err("integer tree input");
            assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
            let tree = futures::executor::block_on(html_tree_builtin(vec![Value::String(
                "<a href='x'>x</a>".into(),
            )]))
            .expect("tree");
            let error = futures::executor::block_on(get_attribute_builtin(vec![tree, value]))
                .expect_err("integer attribute input");
            assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
        }

        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let error = futures::executor::block_on(get_attribute_builtin(vec![
            resident,
            Value::String("href".into()),
        ]))
        .expect_err("resident tree input");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn get_attribute_remains_scoped_to_html_tree_objects() {
        let object = ObjectInstance::new("OtherClass".to_string());
        let error = futures::executor::block_on(get_attribute_builtin(vec![
            Value::Object(object),
            Value::String("href".into()),
        ]))
        .expect_err("non-html object");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
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

    #[test]
    fn extract_html_text_integer_audit_is_not_applicable() {
        assert_eq!(
            EXTRACT_HTML_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert!(EXTRACT_HTML_INTEGER_AUDIT
            .notes
            .contains("All eight integer"));
    }

    #[test]
    fn extract_html_text_rejects_integer_and_resident_numeric_input_before_gather() {
        for value in [
            Value::Int(runmat_builtins::IntValue::I8(1)),
            Value::Int(runmat_builtins::IntValue::I16(1)),
            Value::Int(runmat_builtins::IntValue::I32(1)),
            Value::Int(runmat_builtins::IntValue::I64(1)),
            Value::Int(runmat_builtins::IntValue::U8(1)),
            Value::Int(runmat_builtins::IntValue::U16(1)),
            Value::Int(runmat_builtins::IntValue::U32(1)),
            Value::Int(runmat_builtins::IntValue::U64(1)),
        ] {
            let error = futures::executor::block_on(extract_html_text_builtin(vec![value]))
                .expect_err("integer HTML input");
            assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
        }
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let error = futures::executor::block_on(extract_html_text_builtin(vec![resident]))
            .expect_err("resident HTML input");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn extract_html_text_strict_mode_gates_broad_text_containers() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let matrix = Value::CharArray(
            runmat_builtins::CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap(),
        );
        let error = futures::executor::block_on(extract_html_text_builtin(vec![matrix]))
            .expect_err("strict char matrix gate");
        assert_eq!(
            error.identifier(),
            EXTRACT_HTML_CHAR_MATRIX_EXTENSION.error_identifier
        );
        let broad_cell =
            Value::Cell(CellArray::new(vec![Value::String("<p>x</p>".into())], 1, 1).unwrap());
        let error = futures::executor::block_on(extract_html_text_builtin(vec![broad_cell]))
            .expect_err("strict broad cell gate");
        assert_eq!(
            error.identifier(),
            EXTRACT_HTML_BROAD_CELL_EXTENSION.error_identifier
        );
    }
}
