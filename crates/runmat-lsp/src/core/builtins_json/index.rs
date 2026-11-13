//! Closed presentation envelope admission; no filesystem or sidecar fallback.
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema_version: u32,
    inventory: Inventory,
    builtins: Vec<Value>,
    presentation: Presentation,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Inventory {
    documents: usize,
    catalog_identities: usize,
    legacy_sidecars: usize,
    missing_catalog_documentation: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Presentation {
    kind: PresentationKind,
    executable_qualification: bool,
}
#[derive(Deserialize)]
enum PresentationKind {
    #[serde(rename = "editor_prose")]
    EditorProse,
}
#[derive(Deserialize)]
enum Authority {
    #[serde(rename = "catalog")]
    Catalog,
    #[serde(rename = "legacy_sidecar")]
    LegacySidecar,
}

pub(super) fn index_presentation(source: &str) -> Result<BTreeMap<String, String>, String> {
    let envelope: Envelope = serde_json::from_str(source).map_err(|error| error.to_string())?;
    if envelope.schema_version != runmat_builtins::documentation_export::EXPORT_SCHEMA_VERSION
        || envelope.presentation.executable_qualification
    {
        return Err("unsupported documentation presentation schema or qualification claim".into());
    }
    let PresentationKind::EditorProse = envelope.presentation.kind;
    if envelope.inventory.documents != envelope.builtins.len() {
        return Err("documentation presentation inventory differs from its rows".into());
    }
    // Counts describe the shared export's source inventory; it can include
    // legacy-only public identities as well as migrated canonical identities.
    let Inventory {
        catalog_identities: _catalog_identities,
        legacy_sidecars: _legacy_sidecars,
        missing_catalog_documentation: _missing_catalog_documentation,
        ..
    } = envelope.inventory;
    let mut documents = BTreeMap::new();
    for document in envelope.builtins {
        let key = document
            .get("key")
            .and_then(Value::as_str)
            .filter(|key| !key.is_empty())
            .ok_or("documentation row has no identity")?;
        let _: Authority = serde_json::from_value(
            document
                .get("authority")
                .cloned()
                .ok_or("documentation row has no authority")?,
        )
        .map_err(|error| error.to_string())?;
        let key = key.to_ascii_lowercase();
        if documents
            .insert(key.clone(), document.to_string())
            .is_some()
        {
            return Err(format!(
                "duplicate documentation presentation identity: {key}"
            ));
        }
    }
    Ok(documents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn envelope(rows: Vec<Value>) -> Value {
        json!({
            "schema_version": 2,
            "inventory": {"documents": rows.len(), "catalog_identities": 1, "legacy_sidecars": 0, "missing_catalog_documentation": []},
            "builtins": rows,
            "presentation": {"kind": "editor_prose", "executable_qualification": false},
        })
    }

    #[test]
    fn presentation_index_preserves_rows_and_rejects_wrong_authority_or_envelope() {
        let row = json!({"key":"example", "authority":"catalog", "sections":[{"heading":"Details","paragraphs":["Complete prose."]}]});
        let indexed = index_presentation(&envelope(vec![row.clone()]).to_string()).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&indexed["example"]).unwrap(),
            row
        );
        let mut invalid = envelope(vec![row.clone()]);
        invalid.as_object_mut().unwrap().remove("presentation");
        assert!(index_presentation(&invalid.to_string()).is_err());
        let mut invalid = envelope(vec![row.clone()]);
        invalid["preview"] = json!({"kind":"catalog", "final_acceptance":false});
        assert!(index_presentation(&invalid.to_string()).is_err());
        let mut invalid = envelope(vec![row.clone()]);
        invalid["schema_version"] = json!(1);
        assert!(index_presentation(&invalid.to_string()).is_err());
        let mut invalid = envelope(vec![row.clone()]);
        invalid["presentation"]["executable_qualification"] = json!(true);
        assert!(index_presentation(&invalid.to_string()).is_err());
        let mut duplicate = row.clone();
        duplicate["key"] = json!("EXAMPLE");
        assert!(index_presentation(&envelope(vec![row, duplicate]).to_string()).is_err());
        let bad = json!({"key":"bad", "authority":"preview"});
        assert!(index_presentation(&envelope(vec![bad]).to_string()).is_err());
    }
}
