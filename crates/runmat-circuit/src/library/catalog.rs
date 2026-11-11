//! Catalog parts: the mapping of a [`PartRecord`] (the record a manufacturer part search returns
//! and a `.circuit` file's `part` declarations hold, owned by `runmat-part-record`) onto pool
//! items.
//!
//! A catalog part names its package by a footprint-generator string (`SOT23-6`, `R0402`,
//! `QFN32_5mm_P0.5_EP3.1`) and lists its pins with the pads they land on and their electrical
//! types. [`add_to_pool`] makes the unit, entity, package and part, so the elaborator, the
//! board, the 3D view and the simulator treat it like a built-in part. Supply data (stock,
//! price, lifecycle) is not part of the record: it comes from the server's part search with
//! each [`PartHit`].

use super::footprint;
use super::{
    make, Entity, Gate, Lifecycle, Package, PadMapEntry, Part, PinType, Pool, SpiceModel,
    SupplyRecord, Unit, UnitPin,
};
use crate::ids::Uuid;
use crate::ir::{Param, Tolerance};
use runmat_rules::{parse_quantity, BuiltinUnits};
pub use runmat_part_record::{PartHit, PartOffer, PartPin, PartRecord, PartSpice};
use std::collections::BTreeMap;

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

/// What is wrong with `part`, if anything: the package, the pin types, the pads. Footprint files
/// are not read; [`check_on`] checks the pads against one.
pub fn check(part: &PartRecord) -> Result<(), String> {
    if part.mpn.trim().is_empty() {
        return Err("a part needs its `mpn`".into());
    }
    if part.footprint.is_some() || is_footprint_file(&part.package) || has_no_package(part) {
        return check_pins(part, None);
    }
    let package = footprint::from_string(&part.package)
        .ok_or_else(|| format!("part `{}`: `{}` is not a package the footprint generator makes, nor a footprint file", part.mpn, part.package))?;
    check_pins(part, Some(&package))
}

/// [`check`] with the pads checked against `package`.
pub fn check_on(part: &PartRecord, package: &Package) -> Result<(), String> {
    check_pins(part, Some(package))
}

fn check_pins(part: &PartRecord, package: Option<&Package>) -> Result<(), String> {
    if part.pins.is_empty() {
        return Err(format!("part `{}` has no pins", part.mpn));
    }
    let mut seen = std::collections::BTreeSet::new();
    for pin in &part.pins {
        if PinType::from_name(&pin.kind).is_none() {
            return Err(format!(
                "part `{}`: pin `{}` has type `{}`, which is not a pin type",
                part.mpn, pin.name, pin.kind
            ));
        }
        if !seen.insert(pin.name.as_str()) {
            return Err(format!(
                "part `{}` lists pin `{}` twice",
                part.mpn, pin.name
            ));
        }
        let Some(package) = package else { continue };
        for pad in &pin.pads {
            if package.pad(pad).is_none() {
                return Err(format!(
                    "part `{}`: pin `{}` lands on pad `{pad}`, which {} has no pad of",
                    part.mpn, pin.name, part.package
                ));
            }
        }
    }
    Ok(())
}

fn entity_name(part: &PartRecord) -> String {
    format!("catalog:{}", part.mpn)
}

/// The `package` of a part with no footprint (`package = none`): a device wired in the field,
/// such as a breaker, a receptacle or a light fixture, drawn on a schematic-only design.
pub const NO_PACKAGE: &str = "none";

/// Whether `part` has no footprint (`package = none`). Its pins take no pads; it has an empty
/// package, so it shows on schematics, netlists and the BOM, and a board has nothing to place.
pub fn has_no_package(part: &PartRecord) -> bool {
    part.footprint.is_none() && part.package == NO_PACKAGE
}

/// Whether a part's `package` names a footprint file (`libs/Power.pretty/SOT-23-6.kicad_mod`)
/// rather than a footprint-generator string.
pub fn is_footprint_file(package: &str) -> bool {
    package.to_ascii_lowercase().ends_with(".kicad_mod")
}

/// The `footprint` and `symbol` declarations in scope where parts are declared: the file's own
/// and those of the libraries it uses, by name.
#[derive(Clone, Copy, Default)]
pub struct Declared<'a> {
    pub footprints: Option<&'a BTreeMap<String, Package>>,
    pub symbols: Option<&'a BTreeMap<String, super::Symbol>>,
}

/// A part's package: generated from its string, or read from its footprint file through `file`
/// (the file's text by the path as written).
pub fn package_of(
    part: &PartRecord,
    file: &dyn Fn(&str) -> Option<String>,
) -> Result<Package, String> {
    package_in(part, Declared::default(), file)
}

/// [`package_of`], with a footprint named by identifier found among `declared`, and the part's
/// own `body` over the footprint's model.
pub fn package_in(
    part: &PartRecord,
    declared: Declared<'_>,
    file: &dyn Fn(&str) -> Option<String>,
) -> Result<Package, String> {
    let mut package = base_package(part, declared, file)?;
    if let Some(body) = &part.body {
        package.model = Some(super::Model3d {
            path: body.clone(),
            offset: (0.0, 0.0, 0.0),
            rotation: (0.0, 0.0, 0.0),
            scale: (1.0, 1.0, 1.0),
            origin: "footprint".into(),
        });
        package.uuid = make::uuid("package", &format!("{}|{body}", package.name));
    }
    Ok(package)
}

fn base_package(
    part: &PartRecord,
    declared: Declared<'_>,
    file: &dyn Fn(&str) -> Option<String>,
) -> Result<Package, String> {
    if let Some(name) = &part.footprint {
        return declared
            .footprints
            .and_then(|f| f.get(name))
            .cloned()
            .ok_or_else(|| format!("part `{}`: no footprint `{name}` is declared here or in a library this file uses", part.mpn));
    }
    if has_no_package(part) {
        return Ok(Package {
            uuid: make::uuid("package", NO_PACKAGE),
            name: NO_PACKAGE.into(),
            description: "No footprint: wired in the field".into(),
            pads: vec![],
            courtyard: vec![],
            silk: vec![],
            fab: vec![],
            model: None,
            body: None,
            smd: false,
        });
    }
    if !is_footprint_file(&part.package) {
        return footprint::from_string(&part.package)
            .ok_or_else(|| format!("part `{}`: `{}` is not a package the footprint generator makes, nor a footprint file", part.mpn, part.package));
    }
    let text = file(&part.package).ok_or_else(|| {
        format!(
            "part `{}`: the footprint file \"{}\" cannot be read",
            part.mpn, part.package
        )
    })?;
    let mut package = crate::kicad::footprint::read_mod(&text)
        .map_err(|e| format!("part `{}`: \"{}\": {e}", part.mpn, part.package))?;
    package.uuid = make::uuid("package", &part.package);
    Ok(package)
}

/// Adds `part` to `pool` as a unit, an entity, a package and a part keyed by its MPN, replacing
/// a part of the same MPN. Returns the part's UUID. A footprint-file package fails here; use
/// [`add_to_pool_with`].
pub fn add_to_pool(pool: &mut Pool, part: &PartRecord) -> Result<Uuid, String> {
    add_to_pool_with(pool, part, &|_| None)
}

/// [`add_to_pool`], reading footprint files through `file`.
pub fn add_to_pool_with(
    pool: &mut Pool,
    part: &PartRecord,
    file: &dyn Fn(&str) -> Option<String>,
) -> Result<Uuid, String> {
    add_to_pool_in(pool, part, Declared::default(), file)
}

/// [`add_to_pool_with`], with footprints and symbols named by identifier found among `declared`.
pub fn add_to_pool_in(
    pool: &mut Pool,
    part: &PartRecord,
    declared: Declared<'_>,
    file: &dyn Fn(&str) -> Option<String>,
) -> Result<Uuid, String> {
    check(part)?;
    let package = package_in(part, declared, file)?;
    check_on(part, &package)?;
    let symbol = match &part.symbol {
        Some(name) => {
            let symbol = declared.symbols.and_then(|s| s.get(name)).ok_or_else(|| {
                format!(
                    "part `{}`: no symbol `{name}` is declared here or in a library this file uses",
                    part.mpn
                )
            })?;
            super::checks::pins_match_symbol(part, name, symbol)?;
            Some(symbol.clone())
        }
        None => None,
    };
    let entity_name = entity_name(part);
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
    let prefix = if part.prefix.is_empty() {
        prefix_for(&part.category).to_string()
    } else {
        part.prefix.clone()
    };
    let entity = pool.add_entity(Entity {
        uuid: make::uuid("entity", &entity_name),
        name: entity_name,
        prefix,
        gates: vec![Gate {
            name: "A".into(),
            unit,
            swap_group: 0,
        }],
        tags: if part.category.is_empty() {
            vec![]
        } else {
            vec![part.category.clone()]
        },
    });
    if let Some(mut symbol) = symbol {
        symbol.uuid = make::uuid("symbol", &part.mpn);
        symbol.unit = unit;
        pool.symbols.insert(symbol.uuid, symbol);
    }
    let package = pool.add_package(package);
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
        pin_order: s
            .nodes
            .iter()
            .map(|n| ("A".to_string(), n.clone()))
            .collect(),
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
            .map(|p| PadMapEntry {
                gate: "A".into(),
                pin: p.name.clone(),
                pads: p.pads.clone(),
            })
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
pub fn supply_of(hit: &PartHit) -> Vec<SupplyRecord> {
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
pub fn pool_with(
    pool: Pool,
    parts: &BTreeMap<String, PartRecord>,
    file: &dyn Fn(&str) -> Option<String>,
) -> Result<Pool, String> {
    pool_in(pool, parts, Declared::default(), file)
}

/// [`pool_with`], with footprints and symbols named by identifier found among `declared`.
pub fn pool_in(
    mut pool: Pool,
    parts: &BTreeMap<String, PartRecord>,
    declared: Declared<'_>,
    file: &dyn Fn(&str) -> Option<String>,
) -> Result<Pool, String> {
    for part in parts.values() {
        if let Some(old) = pool.part(&part.mpn).map(|p| p.uuid) {
            pool.parts.remove(&old);
        }
        add_to_pool_in(&mut pool, part, declared, file)?;
    }
    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn buck() -> PartRecord {
        let pin = |name: &str, pad: &str, kind: &str| PartPin {
            name: name.into(),
            pads: vec![pad.into()],
            kind: kind.into(),
        };
        PartRecord {
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
            params: [
                ("vin_max".to_string(), "60 V".to_string()),
                ("topology".to_string(), "synchronous buck".to_string()),
            ]
            .into(),
            datasheet: Some("https://example.com/buck.pdf".into()),
            spice: None,
            ..Default::default()
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
        assert_eq!(
            (vin.kind, vin.pads.clone()),
            (PinType::PowerIn, vec!["5".to_string()])
        );
        assert!(
            part.params["vin_max"].as_quantity().is_some(),
            "a written quantity becomes a quantity"
        );
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
        assert!(add_to_pool(&mut Pool::new(), &bad)
            .unwrap_err()
            .contains("footprint generator"));
        let mut bad = buck();
        bad.pins[1].kind = "power".into();
        assert!(add_to_pool(&mut Pool::new(), &bad)
            .unwrap_err()
            .contains("not a pin type"));
    }

    #[test]
    fn a_declared_part_replaces_the_built_in_part_of_its_mpn() {
        let mut part = buck();
        part.mpn = "BUCK-36V-2A".into();
        part.description = "replaced".into();
        let pool = pool_with(
            crate::fixture::standard_pool(),
            &[(part.mpn.clone(), part)].into(),
            &|_| None,
        )
        .unwrap();
        let found: Vec<_> = pool
            .parts
            .values()
            .filter(|p| p.mpn == "BUCK-36V-2A")
            .collect();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].description, "replaced");
    }

    #[test]
    fn a_hit_reads_from_the_server_json() {
        let json = r#"{"part":{"mpn":"X","manufacturer":"M","package":"R0402","pins":[{"name":"1","pads":["1"]},{"name":"2","pads":["2"]}]},
            "lifecycle":"active","offers":[{"supplier":"S","sku":"S-1","stock":1200,"priceBreaks":[[1,0.1],[100,0.02]],"currency":"USD"}]}"#;
        let hit: PartHit = serde_json::from_str(json).unwrap();
        assert_eq!(lifecycle_of(&hit.lifecycle), Lifecycle::Active);
        assert_eq!(supply_of(&hit)[0].price_breaks[1], (100, 0.02));
        add_to_pool(&mut Pool::new(), &hit.part).unwrap();
    }
}
