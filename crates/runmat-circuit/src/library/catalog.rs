//! Catalog parts: the record a manufacturer part search returns and a `.circuit` file's `part`
//! declarations hold, and its mapping onto pool items.
//!
//! A catalog part names its package by a footprint-generator string (`SOT23-6`, `R0402`,
//! `QFN32_5mm_P0.5_EP3.1`) and lists its pins with the pads they land on and their electrical
//! types. [`add_to_pool`] makes the unit, entity, package and part, so the elaborator, the
//! board, the 3D view and the simulator treat it like a built-in part. Supply data (stock,
//! price, lifecycle) is not part of the record: it comes from the server's part search with
//! each [`CatalogHit`].

use super::footprint;
use super::{make, Entity, Gate, Lifecycle, PadMapEntry, Part, PinType, Pool, SpiceModel, SupplyRecord, Unit, UnitPin};
use crate::ids::Uuid;
use crate::ir::{Param, Tolerance};
use runmat_rules::{parse_quantity, BuiltinUnits};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One pin of a catalog part: its name, the pads it lands on and its electrical type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogPin {
    pub name: String,
    pub pads: Vec<String>,
    /// The ERC pin type by its written word (`input`, `power_in`, `passive` …).
    #[serde(rename = "type", default = "passive")]
    pub kind: String,
}

fn passive() -> String {
    "passive".into()
}

/// A part's SPICE model: a model or subcircuit name, the library file that defines it, and the
/// subcircuit's node order as pin names.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogSpice {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library: Option<String>,
    /// The subcircuit's nodes in order, each a pin name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<String>,
    /// A primitive device letter (`D`, `Q`, `M`) when the model is a `.model` card.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

/// An orderable part as a catalog describes it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogPart {
    pub mpn: String,
    pub manufacturer: String,
    #[serde(default)]
    pub description: String,
    /// What kind of part (`regulator`, `resistor`, `mcu` …), for search and the designator.
    #[serde(default)]
    pub category: String,
    /// Designator prefix (`U`, `R`); empty takes one from the category.
    #[serde(default)]
    pub prefix: String,
    /// A footprint-generator string.
    pub package: String,
    pub pins: Vec<CatalogPin>,
    /// Parametrics as written quantities (`vin_max = "36 V"`) or text.
    #[serde(default)]
    pub params: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub datasheet: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spice: Option<CatalogSpice>,
}

/// One distributor's offer for a part.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogOffer {
    pub supplier: String,
    pub sku: String,
    #[serde(default)]
    pub stock: Option<u64>,
    /// `[quantity, unit price]` breaks, ascending by quantity.
    #[serde(default)]
    pub price_breaks: Vec<(u64, f64)>,
    #[serde(default)]
    pub currency: String,
}

/// One part search result: the part, its lifecycle and the offers for it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogHit {
    pub part: CatalogPart,
    #[serde(default)]
    pub lifecycle: String,
    #[serde(default)]
    pub offers: Vec<CatalogOffer>,
}

/// The designator prefix a category takes when the part names none.
pub fn prefix_for(category: &str) -> &'static str {
    match category {
        "resistor" => "R",
        "capacitor" => "C",
        "inductor" | "ferrite" => "L",
        "diode" | "led" | "tvs" => "D",
        "transistor" | "mosfet" => "Q",
        "connector" => "J",
        "crystal" | "oscillator" => "Y",
        "fuse" => "F",
        _ => "U",
    }
}

/// Lifecycle by its catalog word.
pub fn lifecycle_of(word: &str) -> Lifecycle {
    match word.to_ascii_lowercase().replace([' ', '-'], "_").as_str() {
        "preview" | "new" => Lifecycle::Preview,
        "active" | "production" => Lifecycle::Active,
        "nrnd" | "not_recommended" => Lifecycle::Nrnd,
        "last_time_buy" | "ltb" => Lifecycle::LastTimeBuy,
        "obsolete" | "discontinued" => Lifecycle::Obsolete,
        _ => Lifecycle::Unknown,
    }
}

impl CatalogPart {
    /// What is wrong with the record, if anything: the package, the pin types, the pads.
    pub fn check(&self) -> Result<(), String> {
        if self.mpn.trim().is_empty() {
            return Err("a part needs its `mpn`".into());
        }
        let package = footprint::from_string(&self.package)
            .ok_or_else(|| format!("part `{}`: `{}` is not a package the footprint generator makes", self.mpn, self.package))?;
        if self.pins.is_empty() {
            return Err(format!("part `{}` has no pins", self.mpn));
        }
        let mut seen = std::collections::BTreeSet::new();
        for pin in &self.pins {
            if PinType::from_name(&pin.kind).is_none() {
                return Err(format!("part `{}`: pin `{}` has type `{}`, which is not a pin type", self.mpn, pin.name, pin.kind));
            }
            if !seen.insert(pin.name.as_str()) {
                return Err(format!("part `{}` lists pin `{}` twice", self.mpn, pin.name));
            }
            for pad in &pin.pads {
                if package.pad(pad).is_none() {
                    return Err(format!("part `{}`: pin `{}` lands on pad `{pad}`, which {} has no pad of", self.mpn, pin.name, self.package));
                }
            }
        }
        Ok(())
    }

    fn entity_name(&self) -> String {
        format!("catalog:{}", self.mpn)
    }
}

/// Adds `part` to `pool` as a unit, an entity, a package and a part keyed by its MPN, replacing
/// a part of the same MPN. Returns the part's UUID.
pub fn add_to_pool(pool: &mut Pool, part: &CatalogPart) -> Result<Uuid, String> {
    part.check()?;
    let entity_name = part.entity_name();
    let unit = pool.add_unit(Unit {
        uuid: make::uuid("unit", &entity_name),
        name: entity_name.clone(),
        pins: part
            .pins
            .iter()
            .map(|p| UnitPin {
                name: p.name.clone(),
                kind: PinType::from_name(&p.kind).unwrap_or(PinType::Passive),
                swap_group: 0,
                alternates: vec![],
            })
            .collect(),
    });
    let prefix = if part.prefix.is_empty() { prefix_for(&part.category).to_string() } else { part.prefix.clone() };
    let entity = pool.add_entity(Entity {
        uuid: make::uuid("entity", &entity_name),
        name: entity_name,
        prefix,
        gates: vec![Gate { name: "A".into(), unit, swap_group: 0 }],
        tags: if part.category.is_empty() { vec![] } else { vec![part.category.clone()] },
    });
    let package = match pool.package(&part.package).map(|p| p.uuid) {
        Some(uuid) => uuid,
        None => pool.add_package(footprint::from_string(&part.package).expect("checked")),
    };
    let mut params = BTreeMap::new();
    for (k, v) in &part.params {
        let param = match parse_quantity(v, &BuiltinUnits) {
            Ok(q) => Param::quantity(q, Tolerance::Exact),
            Err(_) => Param::text(v.clone()),
        };
        params.insert(k.clone(), param);
    }
    let spice = part.spice.as_ref().map(|s| SpiceModel {
        model: s.model.clone(),
        library: s.library.clone(),
        pin_order: s.nodes.iter().map(|n| ("A".to_string(), n.clone())).collect(),
        device: s.device.as_ref().and_then(|d| d.chars().next()),
    });
    Ok(pool.add_part(Part {
        uuid: make::uuid("part", &part.mpn),
        mpn: part.mpn.clone(),
        manufacturer: part.manufacturer.clone(),
        description: part.description.clone(),
        entity,
        package,
        pad_map: part
            .pins
            .iter()
            .map(|p| PadMapEntry { gate: "A".into(), pin: p.name.clone(), pads: p.pads.clone() })
            .collect(),
        params,
        lifecycle: Lifecycle::Unknown,
        datasheet: part.datasheet.clone(),
        supply: vec![],
        thermal: Default::default(),
        spice,
        base: None,
    }))
}

/// The pool's supply records for a hit's offers.
pub fn supply_of(hit: &CatalogHit) -> Vec<SupplyRecord> {
    hit.offers
        .iter()
        .map(|o| SupplyRecord {
            supplier: o.supplier.clone(),
            sku: o.sku.clone(),
            stock: o.stock,
            price_breaks: o.price_breaks.clone(),
            currency: o.currency.clone(),
        })
        .collect()
}

/// The built-in pool plus a design's declared parts (a declared part replaces a built-in one of
/// the same MPN).
pub fn pool_with(mut pool: Pool, parts: &BTreeMap<String, CatalogPart>) -> Result<Pool, String> {
    for part in parts.values() {
        if let Some(old) = pool.part(&part.mpn).map(|p| p.uuid) {
            pool.parts.remove(&old);
        }
        add_to_pool(&mut pool, part)?;
    }
    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn buck() -> CatalogPart {
        let pin = |name: &str, pad: &str, kind: &str| CatalogPin { name: name.into(), pads: vec![pad.into()], kind: kind.into() };
        CatalogPart {
            mpn: "BUCK-60V-3A".into(),
            manufacturer: "Sample Catalog".into(),
            description: "60 V 3 A synchronous buck".into(),
            category: "regulator".into(),
            prefix: String::new(),
            package: "SOT23-6".into(),
            pins: vec![
                pin("BST", "1", "passive"),
                pin("GND", "2", "power_in"),
                pin("FB", "3", "input"),
                pin("EN", "4", "input"),
                pin("VIN", "5", "power_in"),
                pin("SW", "6", "power_out"),
            ],
            params: [("vin_max".to_string(), "60 V".to_string()), ("topology".to_string(), "synchronous buck".to_string())].into(),
            datasheet: Some("https://example.com/buck.pdf".into()),
            spice: None,
        }
    }

    #[test]
    fn a_catalog_part_becomes_a_pool_part_on_its_generated_package() {
        let mut pool = Pool::new();
        let id = add_to_pool(&mut pool, &buck()).unwrap();
        let part = &pool.parts[&id];
        assert_eq!(pool.part("BUCK-60V-3A").unwrap().uuid, id);
        let package = &pool.packages[&part.package];
        assert_eq!(package.pads.len(), 6);
        let entity = &pool.entities[&part.entity];
        assert_eq!(entity.prefix, "U");
        let pins = pool.pins(entity, Some(part));
        let vin = pins.iter().find(|p| p.pin == "VIN").unwrap();
        assert_eq!((vin.kind, vin.pads.clone()), (PinType::PowerIn, vec!["5".to_string()]));
        assert!(part.params["vin_max"].as_quantity().is_some(), "a written quantity becomes a quantity");
        assert!(part.params["topology"].as_quantity().is_none());
    }

    #[test]
    fn a_record_with_a_pad_the_package_lacks_is_refused() {
        let mut bad = buck();
        bad.pins[0].pads = vec!["9".into()];
        let err = add_to_pool(&mut Pool::new(), &bad).unwrap_err();
        assert!(err.contains("pad `9`"), "{err}");
        let mut bad = buck();
        bad.package = "NOT-A-PACKAGE".into();
        assert!(add_to_pool(&mut Pool::new(), &bad).unwrap_err().contains("footprint generator"));
        let mut bad = buck();
        bad.pins[1].kind = "power".into();
        assert!(add_to_pool(&mut Pool::new(), &bad).unwrap_err().contains("not a pin type"));
    }

    #[test]
    fn a_declared_part_replaces_the_built_in_part_of_its_mpn() {
        let mut part = buck();
        part.mpn = "BUCK-36V-2A".into();
        part.description = "replaced".into();
        let pool = pool_with(crate::fixture::standard_pool(), &[(part.mpn.clone(), part)].into()).unwrap();
        let found: Vec<_> = pool.parts.values().filter(|p| p.mpn == "BUCK-36V-2A").collect();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].description, "replaced");
    }

    #[test]
    fn a_hit_reads_from_the_server_json() {
        let json = r#"{"part":{"mpn":"X","manufacturer":"M","package":"R0402","pins":[{"name":"1","pads":["1"]},{"name":"2","pads":["2"]}]},
            "lifecycle":"active","offers":[{"supplier":"S","sku":"S-1","stock":1200,"priceBreaks":[[1,0.1],[100,0.02]],"currency":"USD"}]}"#;
        let hit: CatalogHit = serde_json::from_str(json).unwrap();
        assert_eq!(hit.part.pins[0].kind, "passive");
        assert_eq!(lifecycle_of(&hit.lifecycle), Lifecycle::Active);
        assert_eq!(supply_of(&hit)[0].price_breaks[1], (100, 0.02));
        add_to_pool(&mut Pool::new(), &hit.part).unwrap();
    }
}
