//! Part displays for the configurations an assembly uses.
//!
//! An assembly can place one part file in several configurations (a named configuration, or
//! input values on top of one: `configuration = { select = inputs, values = { chord = 2 m } }`).
//! Each distinct configuration is regenerated once on the installed kernel and tessellated, so
//! every occurrence draws its own configuration's geometry.

use runmat_cad::assembly::model::DefinitionKind;
use runmat_cad::assembly::resolve::ResolvedAssembly;
use runmat_cad::assembly::source::ConfigKey;
use runmat_cad::display::{configuration_key, part_display, PartDisplay};
use runmat_cad::model::PartDocument;
use runmat_cad::regen::kernel::TessellationOptions;
use runmat_cad::regen::{regenerate, RegenCache, RegenContext, RegenRequest};
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
        let doc = configured_document(document, &node.configuration);
        let kernel = crate::kernel::make();
        let ctx = RegenContext {
            kernel: kernel.as_ref(),
            solver: &ConstraintSketchSolver,
            units: &StubUnits,
            dynamic: &NoDynamicValues,
            lock: None,
            catalog: None,
            policy: Default::default(),
            slots: None,
        };
        let request = RegenRequest {
            configuration: node.configuration.name.clone(),
        };
        let result = regenerate(&doc, &ctx, &request, &mut RegenCache::default());
        let display = part_display(&result, kernel.as_ref(), &TessellationOptions::default());
        out.insert(key, display);
    }
    out
}
