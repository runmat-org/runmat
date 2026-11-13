use super::*;
use crate::builtins::common::identifiers::is_valid_varname;
use runmat_value::CellArray;
use std::collections::{BTreeMap, HashMap};
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
    pub(super) fn preflight(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
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

    pub(super) fn parse(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Self> {
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
    content: Vec<XmlContent>,
}

impl XmlNode {
    fn new(name: String, attributes: BTreeMap<String, String>) -> Self {
        Self {
            name,
            attributes,
            content: Vec::new(),
        }
    }

    #[async_recursion::async_recursion(?Send)]
    pub(super) async fn into_object(
        self,
        cooperation: &mut XmlCooperation,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        let text = self.direct_text(cooperation).await?;
        cooperation.completed(self.name.len()).await?;
        for (name, value) in &self.attributes {
            cooperation.completed(name.len() + value.len()).await?;
        }
        let attributes = if let Some(runtime) = runtime {
            let mut source = self.attributes.values();
            let values = crate::execution::RootedValues::try_initialize_in_context(
                self.attributes.len(),
                runtime,
                |_| {
                    Ok::<_, crate::RuntimeError>(char_value(
                        source.next().expect("one attribute value per slot"),
                    ))
                },
            )?;
            crate::execution::structure::construct_struct_in_context(
                runtime,
                self.attributes.keys().map(String::as_str),
                values,
            )?
        } else {
            // This is the explicit ordinary parser route, not a Managed
            // fallback. Check policy before creating its projection backing.
            use crate::context::host_allocation::{
                host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
                HostAllocationReceipt,
            };
            let allocation = |source| {
                host_allocation_runtime_error(
                    None,
                    "RunMat:xmlread:Projection",
                    "XML attribute projection admission failed",
                    source,
                )
            };
            if HostAllocationReceipt::policy(None).map_err(allocation)?
                != HostAllocationPolicy::Unbounded
            {
                return Err(allocation(HostAllocationError::MissingManagedOwner));
            }
            let mut attributes = runmat_value::UnboundedStructValueBuilder::new();
            for (name, value) in &self.attributes {
                attributes.insert(name.clone(), char_value(value));
            }
            crate::execution::RootedValue::in_optional_context(
                Value::Struct(attributes.finish()),
                None,
            )?
        };
        let mut children = Vec::new();
        for item in self.content {
            cooperation.completed(1).await?;
            if let XmlContent::Element(child) = item {
                children.try_reserve(1).map_err(|err| {
                    build_runtime_error("xmlread: cannot allocate child-node collection")
                        .with_builtin("xmlread")
                        .with_source(err)
                        .build()
                })?;
                children.push(child.into_object(cooperation, runtime).await?);
            }
        }
        let child_count = children.len();
        let child_nodes = crate::execution::structure::cell::from_borrowed_values(
            runtime,
            &[child_count, 1],
            children.iter().map(|child| child.value()),
        )?;
        let fields = [
            ("NodeName", char_value(&self.name)),
            ("TagName", char_value(&self.name)),
            ("TextContent", char_value(text.trim())),
            ("Attributes", attributes.value().clone()),
            ("ChildNodes", child_nodes.value().clone()),
        ];
        crate::execution::object::construct_object_from_borrowed_fields(
            runtime,
            XML_ELEMENT_CLASS.display_name(),
            fields.iter().map(|(name, value)| (*name, value)),
        )
    }
}

#[async_recursion::async_recursion(?Send)]
pub(super) async fn xml_node_to_struct(
    node: &XmlNode,
    options: &ReadStructOptions,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<StructValue> {
    cooperation.completed(node.name.len()).await?;
    let mut out = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in &node.attributes {
        cooperation.completed(name.len() + value.len()).await?;
        out.insert(
            xml_struct_field_name(&format!("{name}{}", options.attribute_suffix)),
            char_value(value),
        );
    }
    let direct_text = node.direct_text(cooperation).await?;
    let text = direct_text.trim();
    if !text.is_empty() {
        out.insert(
            xml_struct_field_name(&options.text_node_name),
            char_value(text),
        );
    }

    // The vector owns first-seen field order. This lookup is only a local
    // projection index; it is not another XML content or field authority.
    let mut groups: Vec<(String, Vec<Value>)> = Vec::new();
    let mut group_index = HashMap::<String, usize>::new();
    for item in &node.content {
        cooperation.completed(1).await?;
        let XmlContent::Element(child) = item else {
            continue;
        };
        let field = xml_struct_field_name(&child.name);
        let value = xml_node_to_struct_value(child, options, cooperation).await?;
        if let Some(index) = group_index.get(field.as_str()).copied() {
            let values = &mut groups[index].1;
            values.try_reserve(1).map_err(|error| {
                build_runtime_error("readstruct: cannot reserve repeated XML field")
                    .with_builtin("readstruct")
                    .with_source(error)
                    .build()
            })?;
            values.push(value);
        } else {
            groups.try_reserve(1).map_err(|error| {
                build_runtime_error("readstruct: cannot reserve XML field groups")
                    .with_builtin("readstruct")
                    .with_source(error)
                    .build()
            })?;
            group_index.try_reserve(1).map_err(|error| {
                build_runtime_error("readstruct: cannot reserve XML field index")
                    .with_builtin("readstruct")
                    .with_source(error)
                    .build()
            })?;
            let mut index_key = String::new();
            index_key.try_reserve(field.len()).map_err(|error| {
                build_runtime_error("readstruct: cannot reserve XML field name")
                    .with_builtin("readstruct")
                    .with_source(error)
                    .build()
            })?;
            index_key.push_str(&field);
            let mut values = Vec::new();
            values.try_reserve(1).map_err(|error| {
                build_runtime_error("readstruct: cannot reserve XML field value")
                    .with_builtin("readstruct")
                    .with_source(error)
                    .build()
            })?;
            values.push(value);
            group_index.insert(index_key, groups.len());
            groups.push((field, values));
        }
    }
    for (field, values) in groups {
        cooperation.completed(1).await?;
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
    Ok(out.finish())
}

async fn xml_node_to_struct_value(
    node: &XmlNode,
    options: &ReadStructOptions,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<Value> {
    if node.attributes.is_empty() && !node.has_element(cooperation).await? {
        return Ok(char_value(node.direct_text(cooperation).await?.trim()));
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
#[path = "parser/content.rs"]
mod content;
use content::XmlContent;
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
