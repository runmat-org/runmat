//! Top-level `table <name> : <quantity> over <domain> { ... }` declarations. An impedance
//! table over frequency is a porous-layer model (`model = delany_bazley`) or tabulated
//! resistance and reactance samples; it lowers into the boundaries that reference it.
use super::{diagnostic, properties::*};
use crate::{
    semantic::expression::{CheckedExpression, CheckedExpressionKind},
    BoundFeaProgram, CheckedFeaDeclaration, FeaDiagnostic, FeaSymbolKey, FeaSymbolKind,
    LogicalSourcePath, PhysicalDimension,
};
use runmat_analysis_core::AcousticImpedanceLaw;
use runmat_design_syntax::{FeaDeclarationKind, FeaModuleItem};
use std::collections::BTreeMap;

pub(in crate::semantic) type ImpedanceTables =
    BTreeMap<(LogicalSourcePath, String), AcousticImpedanceLaw>;

pub(in crate::semantic) fn lower_impedance_tables(
    bound: &BoundFeaProgram,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> ImpedanceTables {
    let mut tables = BTreeMap::new();
    for (path, module) in &bound.graph.modules {
        let Some(syntax) = &module.syntax else {
            continue;
        };
        for item in &syntax.items {
            let FeaModuleItem::Declaration(table) = item else {
                continue;
            };
            if table.kind != FeaDeclarationKind::Table {
                continue;
            }
            let key = FeaSymbolKey {
                kind: FeaSymbolKind::Table,
                name: table.name.text.clone(),
            };
            let Some(declaration) = declarations.get(&(path.clone(), key)) else {
                continue;
            };
            let header = syntax
                .source()
                .get(table.header.span.start..table.header.span.end)
                .unwrap_or_default()
                .split_whitespace()
                .collect::<Vec<_>>();
            if header != [":", "impedance", "over", "frequency"] {
                diagnostic(
                    declaration,
                    table.header.span,
                    "tables are `table <name> : impedance over frequency`",
                    diagnostics,
                );
                continue;
            }
            if let Some(law) = impedance_law(declaration, diagnostics) {
                tables.insert((path.clone(), table.name.text.clone()), law);
            }
        }
    }
    tables
}

fn impedance_law(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AcousticImpedanceLaw> {
    let body = &declaration.statements;
    let law = if optional_property_assignment(body, "samples").is_some() {
        samples(declaration, diagnostics)?
    } else {
        match required_enum(declaration, body, "model", diagnostics)? {
            "delany_bazley" => AcousticImpedanceLaw::DelanyBazley {
                flow_resistivity_pa_s_per_m2: required_quantity(
                    declaration,
                    body,
                    "flow_resistivity",
                    flow_resistivity()?,
                    diagnostics,
                )?,
                thickness_m: required_quantity(
                    declaration,
                    body,
                    "thickness",
                    PhysicalDimension::LENGTH,
                    diagnostics,
                )?,
            },
            other => {
                diagnostic(
                    declaration,
                    declaration.span,
                    format!("unknown impedance model `{other}`; the porous-layer model is `delany_bazley`"),
                    diagnostics,
                );
                return None;
            }
        }
    };
    if !law.is_valid() {
        diagnostic(
            declaration,
            declaration.span,
            "impedance table needs ascending positive frequencies, nonnegative resistance and \
             a nonzero impedance, or a positive flow resistivity and thickness",
            diagnostics,
        );
        return None;
    }
    Some(law)
}

fn samples(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AcousticImpedanceLaw> {
    let list = required_assignment(declaration, &declaration.statements, "samples", diagnostics)?;
    let CheckedExpressionKind::List(entries) = &list.kind else {
        diagnostic(
            declaration,
            list.span,
            "impedance samples are a list of `{ at, resistance, reactance }` records",
            diagnostics,
        );
        return None;
    };
    let impedance = specific_acoustic_impedance()?;
    let mut frequencies = Vec::with_capacity(entries.len());
    let mut resistance = Vec::with_capacity(entries.len());
    let mut reactance = Vec::with_capacity(entries.len());
    for entry in entries {
        let CheckedExpressionKind::Record(members) = &entry.kind else {
            diagnostic(
                declaration,
                entry.span,
                "impedance sample must be a record",
                diagnostics,
            );
            return None;
        };
        if members.len() != 3 {
            diagnostic(
                declaration,
                entry.span,
                "impedance sample requires exactly `at`, `resistance`, `reactance`",
                diagnostics,
            );
            return None;
        }
        let member =
            |name: &str, dimension: PhysicalDimension, diagnostics: &mut Vec<FeaDiagnostic>| {
                let value = members.get(name);
                match value.and_then(|value| typed(value, dimension)) {
                    Some(value) => Some(value),
                    None => {
                        diagnostic(
                            declaration,
                            entry.span,
                            format!(
                            "impedance sample `{name}` needs a constant quantity of its dimension"
                        ),
                            diagnostics,
                        );
                        None
                    }
                }
            };
        frequencies.push(member("at", PhysicalDimension::FREQUENCY, diagnostics)?);
        resistance.push(member("resistance", impedance, diagnostics)?);
        reactance.push(member("reactance", impedance, diagnostics)?);
    }
    Some(AcousticImpedanceLaw::Table {
        frequencies_hz: frequencies,
        resistance_pa_s_per_m: resistance,
        reactance_pa_s_per_m: reactance,
    })
}

fn typed(value: &CheckedExpression, dimension: PhysicalDimension) -> Option<f64> {
    value
        .value_type
        .is_quantity(dimension.into())
        .then(|| scalar(value))
        .flatten()
}

fn specific_acoustic_impedance() -> Option<PhysicalDimension> {
    PhysicalDimension::PRESSURE
        .multiply(PhysicalDimension::TIME)?
        .divide(PhysicalDimension::LENGTH)
}

fn flow_resistivity() -> Option<PhysicalDimension> {
    specific_acoustic_impedance()?.divide(PhysicalDimension::LENGTH)
}
