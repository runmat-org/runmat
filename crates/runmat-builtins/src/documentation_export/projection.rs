use crate::{
    validate_builtin_catalog, BuiltinCatalogEntry, BuiltinConstantCatalogEntry,
    BuiltinDocumentation, BuiltinDocumentationLinkTarget,
};
use serde_json::{json, Map, Value};

pub(super) fn validate_catalog_documentation(
    entry: &'static BuiltinCatalogEntry,
) -> Result<(), String> {
    let name = entry.identity.name;
    let errors = validate_builtin_catalog(&[entry]);
    if !errors.is_empty() {
        return Err(format!(
            "{name} has invalid canonical catalog data: {errors:#?}"
        ));
    }
    Ok(())
}

pub(super) fn catalog_document(entry: &BuiltinCatalogEntry) -> Value {
    let mut document =
        documentation_document(entry.identity.name, entry.category, &entry.documentation);
    document["catalog"] = json!({
        "descriptor": entry.descriptor,
        "contract": entry.contract,
        "placement": entry.placement,
        "link": entry.link,
        "bindings": entry.bindings,
        "extensions": entry.extensions,
        "integer_capabilities": entry.integer_capabilities,
        "integer_audit": entry.integer_audit,
    });
    document
}

pub(super) fn constant_document(
    entry: &BuiltinConstantCatalogEntry,
    documentation: &BuiltinDocumentation,
) -> Value {
    let mut document = documentation_document(entry.name, "constants", documentation);
    document["catalog"] = json!({ "constant": { "name": entry.name, "kind": entry.kind } });
    document
}

fn documentation_document(
    name: &str,
    category: &str,
    documentation: &BuiltinDocumentation,
) -> Value {
    let title = documentation.title.unwrap_or(name);
    let slug = documentation.slug.unwrap_or(name).to_ascii_lowercase();
    let examples = documentation
        .examples
        .iter()
        .map(|example| {
            json!({
                "id": example.id,
                "description": example.title,
                "input": example.program,
                "output": example.display_output,
                "compatibility": example.compatibility,
                "harness": example.harness,
                "fixture": example.fixture,
                "requirements": example.requirements,
                "verification": example.verification,
            })
        })
        .collect::<Vec<_>>();
    let related = documentation
        .related
        .iter()
        .map(|name| json!({ "label": name, "url": format!("./{name}") }))
        .collect::<Vec<_>>();
    let mut object = Map::new();
    object.insert("key".into(), Value::String(name.to_ascii_lowercase()));
    object.insert("module_stem".into(), Value::String(name.into()));
    object.insert("authority".into(), Value::String("catalog".into()));
    object.insert("title".into(), Value::String(title.into()));
    object.insert("slug".into(), Value::String(slug));
    object.insert("category".into(), Value::String(category.into()));
    object.insert(
        "summary".into(),
        Value::String(documentation.summary.into()),
    );
    object.insert(
        "description".into(),
        Value::String(documentation.description.into()),
    );
    object.insert("keywords".into(), json!(documentation.keywords));
    object.insert("sections".into(), json!(documentation.sections));
    object.insert("examples".into(), Value::Array(examples));
    object.insert(
        "example_exemption".into(),
        json!(documentation.example_exemption),
    );
    object.insert("faqs".into(), json!(documentation.faqs));
    object.insert(
        "links".into(),
        Value::Array(
            documentation
                .links
                .iter()
                .map(|link| {
                    let url = match link.target {
                        BuiltinDocumentationLinkTarget::Builtin(name) => format!("./{name}"),
                        BuiltinDocumentationLinkTarget::Documentation(url)
                        | BuiltinDocumentationLinkTarget::Source(url)
                        | BuiltinDocumentationLinkTarget::External(url) => url.to_string(),
                    };
                    json!({ "label": link.label, "url": url })
                })
                .collect(),
        ),
    );
    object.insert("related".into(), Value::Array(related));
    object.insert("media".into(), json!(documentation.media));
    object.insert("evidence".into(), json!(documentation.evidence));
    if let Some(source) = documentation
        .evidence
        .implementation
        .iter()
        .find_map(|link| {
            let BuiltinDocumentationLinkTarget::Source(url) = link.target else {
                return None;
            };
            Some(json!({ "label": link.label, "url": url }))
        })
    {
        object.insert("source".into(), source);
    }
    object.insert("introduced".into(), json!(documentation.introduced));
    object.insert("status".into(), json!(documentation.status));
    Value::Object(object)
}
