//! Properties and configuration commands on a part:
//!
//! - `part.set-property`: one field of a selected feature, as Properties edits it. A value
//!   field's edit carries its configuration scope (`all`, `this`, `specify`); `this` and
//!   `specify` write overrides into those configurations. `suppressed` suppresses the feature in
//!   the active configuration (`values.configuration`), or in the part for the default.
//! - `manage.configurations`: adds a configuration (`values.name`).
//! - `part.configuration-part-number`, `part.configuration-description`: a configuration's
//!   own part number and description (`values.configuration`, `values.value`).

use super::{model, selected_features, EditRequest};
use runmat_cad::model::PropertyValue;
use runmat_cad::syntax::configuration_edit::{
    add_configuration, set_configuration_part_number, set_configuration_property,
    set_feature_value, suppress_features,
};
use runmat_cad::syntax::edit::{ConfigurationScope, EditFailure, SourceChange};
use runmat_cad::syntax::parse_part;
use serde_json::{json, Map, Value};

/// The active configuration a command ran in (`None` for the default).
fn active(values: &Map<String, Value>) -> Option<&str> {
    values
        .get("configuration")
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty() && *c != "default")
}

/// Where a Properties field of a feature lives in its definition. Fields of the command that
/// makes a feature (`part.extrude`'s `distance`, `draft_angle` …) map onto the definition's
/// fields; `distance` is the extent's depth when the feature has an extent.
fn field_path(kind: &Value, field: &str) -> Option<Vec<&'static str>> {
    let has = |key: &str| kind.get(key).is_some_and(|v| !v.is_null());
    Some(match field {
        "distance" if has("extent") => vec!["extent", "distance"],
        "end" => vec!["extent", "extent"],
        "second_distance" => vec!["second", "distance"],
        "second_end" => vec!["second", "extent"],
        "draft_angle" => vec!["draft", "angle"],
        "draft_outward" => vec!["draft", "outward"],
        "thin_thickness" => vec!["thin", "thickness"],
        "flip" => vec!["reverse"],
        "distance" => vec!["distance"],
        "radius" => vec!["radius"],
        "thickness" => vec!["thickness"],
        "diameter" => vec!["diameter"],
        "angle" => vec!["angle"],
        _ => return None,
    })
}

/// A Properties value as the definition's JSON and its scope: a number in the field's catalog
/// unit (millimetres, degrees for angles), a flag or word, or a value edit with its scope.
fn field_value(field: &str, value: &Value) -> Result<(Value, ConfigurationScope), EditFailure> {
    if let Some(edit) = value
        .as_object()
        .filter(|o| o.get("kind") == Some(&json!("value")))
    {
        let scope = edit
            .get("scope")
            .map(|s| serde_json::from_value::<ConfigurationScope>(s.clone()))
            .transpose()
            .map_err(|e| model(format!("the value's scope: {e}")))?
            .unwrap_or(ConfigurationScope::All);
        let expr = edit
            .get("value")
            .cloned()
            .ok_or_else(|| model("the value edit has no value"))?;
        return Ok((expr, scope));
    }
    let plain = match value {
        Value::Number(n) => {
            let unit = if field.contains("angle") { "deg" } else { "mm" };
            json!({ "form": "quantity", "value": n.as_f64().unwrap_or(0.0), "unit": unit })
        }
        other => other.clone(),
    };
    Ok((plain, ConfigurationScope::All))
}

pub(super) fn edit(
    path: &str,
    text: &str,
    request: &EditRequest,
) -> Option<Result<(SourceChange, String), EditFailure>> {
    Some(match request.command.as_str() {
        "part.set-property" => set_property(path, text, request),
        "manage.configurations" => {
            let name = request
                .values
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            add_configuration(text, path, name).map(|c| (c, format!("Added configuration {name}")))
        }
        "part.configuration-part-number" | "part.configuration-description" => {
            let Some(configuration) = active(&request.values) else {
                return Some(Err(model(
                    "activate a configuration first: the default is the part as written",
                )));
            };
            let value = request
                .values
                .get("value")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let result = if request.command == "part.configuration-part-number" {
                set_configuration_part_number(text, path, configuration, Some(value))
            } else {
                set_configuration_property(
                    text,
                    path,
                    configuration,
                    "description",
                    (!value.is_empty()).then(|| PropertyValue::Text(value.to_owned())),
                )
            };
            result.map(|c| (c, format!("Set {configuration}")))
        }
        _ => return None,
    })
}

fn set_property(
    path: &str,
    text: &str,
    request: &EditRequest,
) -> Result<(SourceChange, String), EditFailure> {
    let selected = selected_features(request);
    let [feature] = selected.as_slice() else {
        return Err(model("select one feature to set its properties"));
    };
    let (field, value) = request
        .values
        .iter()
        .find(|(k, _)| k.as_str() != "configuration")
        .ok_or_else(|| model("no property to set"))?;
    if field == "suppressed" {
        let on = value.as_bool().unwrap_or(false);
        let change = suppress_features(text, path, &selected, active(&request.values), on)?;
        let verb = if on { "Suppressed" } else { "Unsuppressed" };
        return Ok((change, format!("{verb} {feature}")));
    }
    let parsed = parse_part(path, text);
    let document = parsed
        .document
        .ok_or(EditFailure::Invalid(parsed.diagnostics))?;
    let kind = document
        .features
        .iter()
        .find(|f| f.id.0 == *feature)
        .map(|f| serde_json::to_value(&f.kind).unwrap_or(Value::Null))
        .ok_or_else(|| model(format!("the part has no feature `{feature}`")))?;
    let at = field_path(&kind, field).ok_or_else(|| {
        model(format!(
            "`{field}` is set in the source editor in this build"
        ))
    })?;
    let (json, scope) = field_value(field, value)?;
    let change = set_feature_value(text, path, feature, &at, json, &scope)?;
    let whom = match &scope {
        ConfigurationScope::All => String::new(),
        ConfigurationScope::This { configuration } => format!(" in {configuration}"),
        ConfigurationScope::Specify { configurations } => {
            format!(" in {}", configurations.join(", "))
        }
    };
    Ok((change, format!("Set {feature} {field}{whom}")))
}

#[cfg(test)]
mod tests {
    use crate::edit::edit_json;
    use crate::runtime::{compile_part_in, Files};
    use serde_json::{json, Value};

    const BRACKET: &str = "runmat 0.7\nstd 1\n\npart bracket {\n    input width : length = 80 mm\n    input depth : length = 40 mm\n\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])\n        l2 = line(start = [input.width, 0 mm], end = [input.width, input.depth])\n        l3 = line(start = [input.width, input.depth], end = [0 mm, input.depth])\n        l4 = line(start = [0 mm, input.depth], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature round = fillet(edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end)], radius = 1.5 mm)\n\n    output bracket = { body = feature.body.body, part_number = \"BR-100\" }\n\n    configuration long = { width = 120 mm }\n    configuration analysis = { suppress = [feature.round] }\n    configuration left = { features = { body = { extent = { distance = 12 mm } } }, part_number = \"BR-100-L\", description = \"Bracket, left hand\" }\n}\n";
    const PATH: &str = "/p/bracket.part";

    fn run(text: &str, request: Value) -> Value {
        serde_json::from_str(&edit_json(PATH, text, &Files::new(), &request.to_string())).unwrap()
    }

    fn text_of(out: &Value) -> String {
        assert_eq!(out["ok"], true, "{out}");
        out["files"][0]["text"].as_str().unwrap().to_owned()
    }

    /// The body's extrude depth (mm) in a configuration.
    fn depth(text: &str, configuration: Option<&str>) -> f64 {
        let part = compile_part_in(PATH, text, configuration).unwrap();
        let kind = serde_json::to_value(&part.document.features[1].kind).unwrap();
        kind["extent"]["distance"]["value"].as_f64().unwrap()
    }

    fn feature_status(text: &str, configuration: Option<&str>, feature: &str) -> String {
        let part = compile_part_in(PATH, text, configuration).unwrap();
        let row = part.projection["index"]["features"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == feature)
            .cloned()
            .unwrap();
        row["status"]["status"].as_str().unwrap_or("").to_owned()
    }

    #[test]
    fn configurations_override_inputs_suppress_features_and_set_parameters() {
        // The written forms read back as written.
        let parsed = runmat_cad::syntax::parse_part(PATH, BRACKET);
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
        let doc = parsed.document.unwrap();
        let printed = runmat_cad::syntax::print_part(&doc, &[]).unwrap();
        assert!(
            printed.contains("configuration long = { width = 120 mm }"),
            "{printed}"
        );
        assert!(
            printed.contains("configuration analysis = { suppress = [feature.round] }"),
            "{printed}"
        );
        assert!(printed.contains("part_number = \"BR-100-L\""), "{printed}");
        // Suppression applies in its configuration only.
        assert_eq!(
            feature_status(BRACKET, Some("analysis"), "round"),
            "suppressed_in_configuration"
        );
        assert_ne!(
            feature_status(BRACKET, None, "round"),
            "suppressed_in_configuration"
        );
        // The left hand: its own depth and part number.
        let left = compile_part_in(PATH, BRACKET, Some("left")).unwrap();
        assert_eq!(left.index.configuration.as_deref(), Some("left"));
        assert_eq!(
            left.document.parts[0].part_number.as_deref(),
            Some("BR-100-L")
        );
        let kind = serde_json::to_value(&left.document.features[1].kind).unwrap();
        assert_eq!(kind["extent"]["distance"]["value"], 12.0, "{kind}");
        let default = compile_part_in(PATH, BRACKET, None).unwrap();
        assert_eq!(
            default.document.parts[0].part_number.as_deref(),
            Some("BR-100")
        );
        let props = crate::drawing::props::output_properties(&left.document);
        assert_eq!(props["description"], "Bracket, left hand");
        assert_eq!(props["part_number"], "BR-100-L");
        // Its geometry differs from the default's.
        assert_ne!(left.display.bodies.len(), 0);
    }

    #[test]
    fn a_feature_override_the_part_lacks_is_an_error_on_the_configuration() {
        let text = BRACKET.replace("features = { body =", "features = { missing =");
        let part = compile_part_in(PATH, &text, Some("left")).unwrap();
        assert!(
            part.diagnostics
                .iter()
                .any(|d| d.severity == "error" && d.message.contains("`missing`")),
            "{:?}",
            part.diagnostics
        );
    }

    #[test]
    fn properties_edits_write_into_the_scope_they_name() {
        // "This configuration": the override goes into `long`, the feature keeps 10 mm.
        let edit = json!({ "kind": "value", "value": { "form": "quantity", "value": 14, "unit": "mm" }, "text": "14 mm", "scope": { "scope": "this", "configuration": "long" } });
        let out = run(
            BRACKET,
            json!({ "command": "part.set-property", "selection": ["feature:body"], "values": { "distance": edit } }),
        );
        let text = text_of(&out);
        assert_eq!(depth(&text, Some("long")), 14.0, "{text}");
        assert_eq!(depth(&text, None), 10.0, "{text}");
        // "All configurations": the feature itself, and the override cleared.
        let out = run(
            &text,
            json!({ "command": "part.set-property", "selection": ["feature:body"], "values": { "distance": 16 } }),
        );
        let text = text_of(&out);
        assert!(
            text.contains("configuration long = { width = 120 mm }"),
            "{text}"
        );
        assert_eq!(depth(&text, Some("long")), 16.0, "{text}");
        // Suppress in the active configuration only.
        let out = run(
            &text,
            json!({ "command": "part.suppress", "selection": ["feature:round"], "values": { "configuration": "long" } }),
        );
        let text = text_of(&out);
        assert!(
            text.contains("configuration long = { width = 120 mm, suppress = [feature.round] }"),
            "{text}"
        );
        assert!(!text.contains("suppressed = true"), "{text}");
        // A new configuration, and its own part number.
        let out = run(
            &text,
            json!({ "command": "manage.configurations", "values": { "name": "right" } }),
        );
        let text = text_of(&out);
        assert!(text.contains("configuration right = {}"), "{text}");
        let out = run(
            &text,
            json!({ "command": "part.configuration-part-number", "values": { "configuration": "right", "value": "BR-100-R" } }),
        );
        let text = text_of(&out);
        assert!(
            text.contains("configuration right = { part_number = \"BR-100-R\" }"),
            "{text}"
        );
        // A configuration that already exists, or `default`, is refused.
        let out = run(
            &text,
            json!({ "command": "manage.configurations", "values": { "name": "default" } }),
        );
        assert_eq!(out["ok"], false);
    }
}
