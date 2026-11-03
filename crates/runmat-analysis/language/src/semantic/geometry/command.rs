use crate::{CheckedFeaStatement, CheckedFeaStatementKind};
use runmat_geometry_core::{
    CadParameterAccess, CadParameterSelector, GeometryFeatureRef, GeometryRegionCardinality,
    GeometryRegionSelector,
};

/// Selector words that choose entities by a geometric test (normal, plane, box, proximity)
/// rather than by an authored name or canonical id. Planned: geometry.select.predicate.
const GEOMETRIC_PREDICATES: &[&str] = &[
    "where",
    "normal",
    "by_normal",
    "planar",
    "plane",
    "within",
    "in_box",
    "box",
    "inside",
    "near",
    "nearest",
    "at",
    "coordinate",
    "position",
];

pub fn region_selector(
    statements: &[CheckedFeaStatement],
) -> Result<GeometryRegionSelector, String> {
    let commands = commands(statements, "select");
    if commands.len() != 1 {
        return Err("region requires exactly one `select` command".to_string());
    }
    let words = words(commands[0])?;
    if words
        .get(1)
        .is_some_and(|kind| GEOMETRIC_PREDICATES.contains(&kind.as_str()))
    {
        return Err(format!(
            "geometric-predicate selector `select {} ...` is not accepted yet; select a CAD-authored \
             name, a tag or a persistent id{}",
            words[1],
            crate::semantic::pending::marker(crate::semantic::pending::GEOMETRY_SELECT_PREDICATE)
        ));
    }
    match words.as_slice() {
        [select, result] if select == "select" && result == "result" => {
            Ok(GeometryRegionSelector::Result)
        }
        [select, kind, value] if select == "select" => match kind.as_str() {
            "name" => Ok(GeometryRegionSelector::Name(value.clone())),
            "tag" => Ok(GeometryRegionSelector::Tag(value.clone())),
            // Ids are canonical `persistent-entity:sha256:` analysis selection ids only;
            // source aliases are rejected by the validation below.
            "id" | "persistent_id" => {
                runmat_geometry_core::PersistentEntityId::validate_analysis_selection_id(value)
                    .map_err(|error| error.to_string())?;
                Ok(GeometryRegionSelector::PersistentId(value.clone()))
            }
            "boundary_introduced_by" => Ok(GeometryRegionSelector::BoundaryIntroducedBy(
                feature_reference(value)?,
            )),
            _ => Err("unsupported region selector".to_string()),
        },
        [select, value] if select == "select" && value.starts_with("feature.") => Ok(
            GeometryRegionSelector::FeatureResult(feature_reference(value)?),
        ),
        [select, value] if select == "select" && value.starts_with("output.") => {
            Ok(GeometryRegionSelector::GeneratedOutput(value.clone()))
        }
        _ => Err("unsupported region selector syntax".to_string()),
    }
}

pub fn region_cardinality(
    statements: &[CheckedFeaStatement],
) -> Result<GeometryRegionCardinality, String> {
    let commands = commands(statements, "expect");
    if commands.len() != 1 {
        return Err("region requires exactly one `expect` command".to_string());
    }
    match words(commands[0])?.as_slice() {
        [expect, value] if expect == "expect" && value == "exactly_one" => {
            Ok(GeometryRegionCardinality::ExactlyOne)
        }
        [expect, value] if expect == "expect" && value == "one_or_more" => {
            Ok(GeometryRegionCardinality::OneOrMore)
        }
        _ => Err("region cardinality must be `exactly_one` or `one_or_more`".to_string()),
    }
}

pub fn parameter_selector(
    statements: &[CheckedFeaStatement],
) -> Result<CadParameterSelector, String> {
    let commands = commands(statements, "select");
    if commands.len() != 1 {
        return Err("CAD parameter requires exactly one `select` command".to_string());
    }
    match words(commands[0])?.as_slice() {
        [select, kind, value] if select == "select" && kind == "name" => {
            Ok(CadParameterSelector::Name(value.clone()))
        }
        [select, kind, value]
            if select == "select" && matches!(kind.as_str(), "id" | "persistent_id") =>
        {
            Ok(CadParameterSelector::PersistentId(value.clone()))
        }
        _ => Err("CAD parameter selector must use `name` or `persistent_id`".to_string()),
    }
}

pub fn parameter_access(statements: &[CheckedFeaStatement]) -> Result<CadParameterAccess, String> {
    let commands = commands(statements, "require");
    if commands.len() != 1 {
        return Err("CAD parameter requires exactly one access declaration".to_string());
    }
    match words(commands[0])?.as_slice() {
        [require, access] if require == "require" && access == "readable" => {
            Ok(CadParameterAccess::Readable)
        }
        [require, access] if require == "require" && access == "writable" => {
            Ok(CadParameterAccess::Writable)
        }
        _ => Err("CAD parameter access must be `readable` or `writable`".to_string()),
    }
}

fn commands<'a>(statements: &'a [CheckedFeaStatement], prefix: &str) -> Vec<&'a str> {
    statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::Command(command) = &statement.kind else {
                return None;
            };
            command
                .trim()
                .starts_with(prefix)
                .then_some(command.as_str())
        })
        .collect()
}

fn words(source: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    for character in source.trim().chars() {
        match (quote, character) {
            (Some(expected), value) if value == expected => quote = None,
            (Some(_), value) => current.push(value),
            (None, '"' | '\'') => quote = Some(character),
            (None, value) if value.is_whitespace() => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            }
            (None, value) => current.push(value),
        }
    }
    if quote.is_some() {
        return Err("unterminated quoted selector".to_string());
    }
    if !current.is_empty() {
        words.push(current);
    }
    Ok(words)
}

fn feature_reference(value: &str) -> Result<GeometryFeatureRef, String> {
    value
        .strip_prefix("feature.")
        .filter(|name| !name.is_empty())
        .map(|name| GeometryFeatureRef(name.to_string()))
        .ok_or_else(|| "feature selector requires `feature.<name>`".to_string())
}
