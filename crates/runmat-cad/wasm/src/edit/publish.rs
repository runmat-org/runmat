//! `simulate.publish`: a `publish` entry, the part's interface for assemblies and studies.
//!
//! - Region: a region the part names (`region:<name>`).
//! - Mate connector: a frame feature (`feature:<id>`), which an assembly attaches to.
//! - Value: an input (`input:<name>`).
//! - Path: picked edges, joined end to end into one path: a cam's closed profile or a path
//!   mate's track (`publish rail = entities(kind = edge, references = […])`).

use super::model;
use crate::features::picks_of;
use crate::picks::Picks;
use runmat_cad::ids::{FeatureId, InputName, PublishName, RegionName};
use runmat_cad::model::datum::DatumRef;
use runmat_cad::model::items::{PublishEntry, PublishTarget};
use runmat_cad::refs::{Reference, TopoKind};
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use serde_json::{Map, Value};

pub fn publish(
    path: &str,
    text: &str,
    values: &Map<String, Value>,
    picks: &Picks,
) -> Result<(SourceChange, String), EditFailure> {
    let name = values
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    if name.is_empty() {
        return Err(model("name what the part publishes"));
    }
    let target = values
        .get("target")
        .and_then(Value::as_str)
        .unwrap_or("region");
    let target = if target == "path" {
        let edges = picks_of(values, "edges");
        if edges.is_empty() {
            return Err(model("pick the edges the path runs along"));
        }
        let references = picks
            .references(&edges)?
            .into_iter()
            .map(|r| {
                serde_json::from_value::<Reference>(r)
                    .map_err(|e| model(format!("an edge pick: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        PublishTarget::Entities {
            kind: TopoKind::Edge,
            references,
        }
    } else {
        let item = picks_of(values, "item")
            .into_iter()
            .next()
            .ok_or_else(|| model("pick what to publish"))?;
        match (target, item.split_once(':')) {
            ("region", Some(("region", region))) => PublishTarget::Region {
                region: RegionName::new(region),
            },
            ("frame", Some(("feature", feature))) => PublishTarget::Datum {
                datum: DatumRef::Feature {
                    feature: FeatureId(feature.to_owned()),
                },
            },
            ("value", Some(("input", input))) => PublishTarget::Value {
                input: InputName::new(input),
            },
            ("region", _) => return Err(model("pick a region")),
            ("frame", _) => return Err(model("pick a mate connector")),
            ("value", _) => return Err(model("pick an input")),
            (other, _) => return Err(model(format!("a part publishes no `{other}`"))),
        }
    };
    let change = edit_part(text, path, |document| {
        if document.publishes.iter().any(|p| p.name.as_str() == name) {
            return Err(model(format!("the part already publishes `{name}`")));
        }
        document.publishes.push(PublishEntry {
            name: PublishName::new(name),
            target: target.clone(),
            auto: false,
        });
        Ok(())
    })?;
    Ok((change, format!("Published {name}")))
}
