use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

pub(super) fn annotate_legacy_document(key: &str, document: &mut Value) -> Result<(), String> {
    let object = document
        .as_object_mut()
        .ok_or_else(|| format!("legacy documentation for {key} is not an object"))?;
    object.insert("key".into(), Value::String(key.into()));
    object.insert("authority".into(), Value::String("legacy_sidecar".into()));
    if let Some(examples) = object.get_mut("examples").and_then(Value::as_array_mut) {
        for example in examples {
            match example {
                Value::Object(example) => annotate_legacy_example(example),
                Value::String(description) if !description.trim().is_empty() => {
                    let description = description.clone();
                    *example = json!({
                        "id": legacy_presentation_example_id(key, &description),
                        "description": description,
                        "fixture": "None",
                        "requirements": no_example_requirements(),
                    });
                }
                _ => {
                    return Err(format!(
                        "legacy documentation for {key} has an invalid example"
                    ))
                }
            }
        }
    }
    Ok(())
}

fn annotate_legacy_example(example: &mut Map<String, Value>) {
    example
        .entry("fixture")
        .or_insert_with(|| Value::String("None".into()));
    example
        .entry("requirements")
        .or_insert_with(no_example_requirements);
}

pub(super) fn no_example_requirements() -> Value {
    json!({
        "host": "Any",
        "engine": "Default",
        "compiler": [],
        "runtime": [],
        "toolchain": []
    })
}

pub(super) fn legacy_presentation_example_id(key: &str, description: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"runmat.builtin-example.legacy-presentation.v1\0");
    digest.update(key.as_bytes());
    digest.update(b"\0");
    digest.update(description.as_bytes());
    format!("legacy-presentation-{}", hex::encode(digest.finalize()))
}
