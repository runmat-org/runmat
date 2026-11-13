//! Part displays for the configurations an assembly uses.
//!
//! An assembly can place one part file in several configurations (a named configuration, or
//! input values on top of one: `configuration = { select = inputs, values = { chord = 2 m } }`).
//! Each distinct configuration is regenerated once on the installed kernel and tessellated, so
//! every occurrence draws its own configuration's geometry.

use crate::runtime::part_cache;
use runmat_cad::assembly::model::DefinitionKind;
use runmat_cad::assembly::resolve::ResolvedAssembly;
use runmat_cad::assembly::source::ConfigKey;
use runmat_cad::display::{configuration_key, part_display, PartDisplay};
use runmat_cad::model::PartDocument;
use runmat_cad::regen::{regenerate, RegenCache, RegenContext, RegenRequest, RegenResult};
use runmat_cad::sketch_solver::ConstraintSketchSolver;
use runmat_cad::units::{StubUnits, UnitAuthority};
use runmat_cad::value::{NoDynamicValues, ValueExpr};
use std::collections::BTreeMap;

/// The document with `key`'s input values as its input defaults.
fn configured_document(document: &PartDocument, key: &ConfigKey) -> PartDocument {
    let mut doc = document.clone();
    for input in doc.inputs.iter_mut() {
        let Some(si) = key.inputs.get(&input.name) else {
            continue;
        };
        input.default = match StubUnits.kind(&input.kind) {
            Some(kind) if kind.si_symbol != "1" => ValueExpr::qty(*si, &kind.si_symbol),
            _ => ValueExpr::number(*si),
        };
    }
    doc
}

/// The part `document` regenerated in configuration `key` (a named configuration and input
/// values on top of it) on a fresh kernel, which holds the result's bodies.
pub fn regenerate_configured(
    document: &PartDocument,
    key: &ConfigKey,
) -> (RegenResult, Box<dyn crate::kernel::DesignKernel>) {
    regenerate_configured_watched(document, key, &std::cell::Cell::new(false))
}

/// [`regenerate_configured`], noting in `used` whether the regeneration asked for a dynamic value.
fn regenerate_configured_watched(
    document: &PartDocument,
    key: &ConfigKey,
    used: &std::cell::Cell<bool>,
) -> (RegenResult, Box<dyn crate::kernel::DesignKernel>) {
    let doc = configured_document(document, key);
    let kernel = crate::kernel::make();
    let watched = crate::runtime::part_cache::Watched::new(&NoDynamicValues, used);
    let ctx = RegenContext {
        kernel: kernel.as_ref(),
        solver: &ConstraintSketchSolver,
        units: &StubUnits,
        dynamic: &watched,
        lock: None,
        catalog: None,
        policy: Default::default(),
        slots: None,
    };
    let request = RegenRequest {
        configuration: key.name.clone(),
    };
    let result = regenerate(&doc, &ctx, &request, &mut RegenCache::default());
    (result, kernel)
}

/// Displays of every non-default part configuration `resolved` places, keyed as
/// `assembly_display_configured` reads them. `documents` holds each part file's document.
pub fn configured_displays(
    resolved: &ResolvedAssembly,
    documents: &BTreeMap<String, PartDocument>,
) -> BTreeMap<(String, String), PartDisplay> {
    let mut out = BTreeMap::new();
    for node in &resolved.nodes {
        if node.definition.kind != DefinitionKind::Part || node.represented_by_ancestor {
            continue;
        }
        if node.configuration == ConfigKey::default() {
            continue;
        }
        let key = configuration_key(node);
        if out.contains_key(&key) {
            continue;
        }
        let Some(document) = documents.get(&key.0) else {
            continue;
        };
        // Kept across compiles by the part's document and the configuration ([`part_cache`]).
        let cache_key = part_cache::configured_key(&key.0, document, &node.configuration);
        if part_cache::skip(&cache_key) {
            continue;
        }
        let part = part_cache::configured_part(cache_key, || {
            let used = std::cell::Cell::new(false);
            let (result, kernel) =
                regenerate_configured_watched(document, &node.configuration, &used);
            let mut display = part_display(
                &result,
                kernel.as_ref(),
                &crate::runtime::ASSEMBLY_TESSELLATION,
            );
            display.levels = runmat_cad::display::display_levels(
                &result,
                kernel.as_ref(),
                &crate::runtime::ASSEMBLY_TESSELLATION,
            );
            // A configured occurrence draws in the part's own colors, as its default does.
            runmat_cad::appearance::apply_document_appearance(
                &configured_document(document, &node.configuration),
                &result,
                &mut display,
            );
            (part_cache::ConfiguredPart { display }, used.get())
        });
        out.insert(key, part.display.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::runtime::{compile_json, Files};
    use serde_json::Value;

    /// An occurrence configured by input draws in the part's own color, as the default one does
    /// (it had drawn in its material's appearance, so a glass-filled nylon cowl drew see-through).
    #[test]
    fn a_configured_occurrence_draws_in_the_parts_color() {
        let block = "runmat 0.7\nstd 1\n\npart block {\n    input width : length = 20 mm\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])\n        l2 = line(start = l1.end, end = [input.width, 10 mm])\n        l3 = line(start = l2.end, end = [0 mm, 10 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature box = extrude(profile = sketch.s, distance = 5 mm)\n    output block = { body = feature.box.body, material = \"glass_filled_nylon\", color = \"#4A4F57\" }\n}\n";
        let rig = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part block as plain\n    use part block as wide {\n        configuration = { select = inputs, values = { width = 40 mm } }\n        placement = { placement = offset, offset = { translation = [0 mm, 50 mm, 0 mm] } }\n    }\n}\n";
        let files: Files = [
            ("/p/block.part".to_owned(), block.to_owned()),
            ("/p/rig.assembly".to_owned(), rig.to_owned()),
        ]
        .into();
        let out: Value =
            serde_json::from_str(&compile_json("/p/rig.assembly", rig, &files)).unwrap();
        let protos = out["display"]["data"]["prototypes"]
            .as_array()
            .expect("prototypes");
        assert_eq!(protos.len(), 2, "{}", out["diagnostics"]);
        let want = runmat_cad::appearance::parse_srgb_hex("#4A4F57").unwrap();
        for p in protos {
            let got: Vec<f32> = p["appearance"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            assert_eq!(got, want.to_vec(), "{}", p["configuration"]);
        }
    }
}
