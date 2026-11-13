//! Property schemas, check packs over the resolved assembly, typed links and the gate.

use runmat_cad::assembly::resolve::*;
use runmat_cad::assembly::source::*;
use runmat_cad::assembly::*;
use runmat_cad::diagnostics::DiagCode;
use runmat_cad::ids::{FilePath, PropertyName};
use runmat_cad::model::items::{Link, LinkKind, LocalItem, Property, PropertyScope, PropertyValue};
use runmat_cad::props::*;
use runmat_cad::units::StubUnits;
use runmat_cad::value::{Evaluator, NoDynamicValues, ValueExpr};
use runmat_rules as rules;

fn prop(name: &str, v: PropertyValue, scope: PropertyScope) -> Property {
    Property {
        name: PropertyName::new(name),
        value: v,
        scope,
    }
}

fn tree() -> (MemorySource, ResolvedAssembly) {
    let mut src = MemorySource::default();
    for (f, pn) in [("bracket.part", "BR-1"), ("bolt.part", "BT-6")] {
        let mut s = DefinitionSummary::new(DefinitionRef::part(f, None));
        s.part_number = Some(pn.into());
        src.add_part(
            s,
            ConfiguredSummary {
                bbox: Some(Aabb::new([0.0; 3], [0.1; 3])),
                ..Default::default()
            },
        );
    }
    let mut top = AssemblyDocument::new("rig.assembly", "rig");
    let mut b = Occurrence::new("bracket", DefinitionRef::part("bracket.part", None));
    b.properties.push(prop(
        "class",
        PropertyValue::Text("structural primary".into()),
        PropertyScope::Occurrence,
    ));
    b.properties.push(prop(
        "weight_budget",
        PropertyValue::Quantity(ValueExpr::qty(0.5, "kg")),
        PropertyScope::Occurrence,
    ));
    top.occurrences.push(b);
    let mut bolt = Occurrence::new("bolt", DefinitionRef::part("bolt.part", None));
    bolt.properties.push(prop(
        "supplier",
        PropertyValue::Text("acme".into()),
        PropertyScope::Occurrence,
    ));
    top.occurrences.push(bolt);
    src.add_assembly(top);
    let ctx = ResolveContext {
        source: &src,
        units: &StubUnits,
        dynamic: &NoDynamicValues,
        lock: None,
        catalog: None,
        tables: None,
        gauge: None,
    };
    let r = resolve(
        &FilePath::new("rig.assembly"),
        &ResolveRequest::default(),
        &ctx,
    );
    (src, r)
}

struct Masses;
impl Measurements for Masses {
    fn measure(&self, tree: &ResolvedAssembly, node: u32, name: &str) -> Option<rules::Quantity> {
        (name == "mass").then(|| {
            let kg = if tree.node(node).segment.name == "bracket" {
                0.62
            } else {
                0.01
            };
            rules::Quantity::new(kg, rules::Dim::MASS)
        })
    }
}

fn ev() -> Evaluator<'static> {
    Evaluator {
        units: &StubUnits,
        dynamic: &NoDynamicValues,
    }
}

#[test]
fn p01_schemas_type_scope_and_release_requirements() {
    let schema = PropertySchema {
        fields: vec![
            PropertyField {
                name: PropertyName::new("weight_budget"),
                ty: PropertyType::Quantity {
                    kind: runmat_cad::units::QuantityKindId::new("mass"),
                },
                scopes: [PropertyScope::Occurrence, PropertyScope::Definition]
                    .into_iter()
                    .collect(),
                applies_to: Default::default(),
                required_at_release: false,
            },
            PropertyField {
                name: PropertyName::new("export_class"),
                ty: PropertyType::Enum {
                    values: vec!["EAR99".into(), "ITAR".into()],
                },
                scopes: [PropertyScope::Definition].into_iter().collect(),
                applies_to: ["part".to_string()].into_iter().collect(),
                required_at_release: true,
            },
        ],
    };
    let good = [prop(
        "weight_budget",
        PropertyValue::Quantity(ValueExpr::qty(2.0, "kg")),
        PropertyScope::Occurrence,
    )];
    let wrong_unit = [prop(
        "weight_budget",
        PropertyValue::Quantity(ValueExpr::mm(2.0)),
        PropertyScope::Occurrence,
    )];
    let wrong_scope = [prop(
        "export_class",
        PropertyValue::Text("ITAR".into()),
        PropertyScope::Occurrence,
    )];
    let bad_enum = [prop(
        "export_class",
        PropertyValue::Text("secret".into()),
        PropertyScope::Definition,
    )];
    let unknown = [prop(
        "colour",
        PropertyValue::Text("red".into()),
        PropertyScope::Definition,
    )];
    let run = |props: &[Property], release: bool| {
        let sites = vec![PropertySite {
            label: "x".into(),
            kind: "part",
            properties: props,
        }];
        validate_properties(&schema, &sites, &StubUnits, &ev(), release)
    };
    let r = run(&good, false);
    assert!(r.errors().next().is_none(), "{:?}", r);
    for bad in [
        &wrong_unit[..],
        &wrong_scope[..],
        &bad_enum[..],
        &unknown[..],
    ] {
        let r = run(bad, false);
        assert_eq!(r.with_code(DiagCode::PropertyInvalid).len(), 1, "{:?}", bad);
    }
    let r = run(&good, true);
    assert_eq!(
        r.with_code(DiagCode::PropertyMissing).len(),
        1,
        "release needs export_class"
    );
}

#[test]
fn p02_check_packs_select_measure_and_gate_release() {
    let (_src, r) = tree();
    let g = AssemblyGraph::new(&r, &Masses, ev());
    let pack = CheckPack {
        name: "company".into(),
        version: "1.0".into(),
        checks: vec![
            CheckDef {
                name: "primary_mass".into(),
                select: "part.primary".into(),
                attribute: "mass".into(),
                op: CompareOp::Le,
                expected: Expected::Quantity {
                    text: "0.5 kg".into(),
                },
                gate: Gate::Release,
                missing_fails: true,
            },
            CheckDef {
                name: "bolts_have_supplier".into(),
                select: "part#bolt".into(),
                attribute: "supplier".into(),
                op: CompareOp::Present,
                expected: Expected::None,
                gate: Gate::Publish,
                missing_fails: true,
            },
            CheckDef {
                name: "pn_set".into(),
                select: "part".into(),
                attribute: "part_number".into(),
                op: CompareOp::Present,
                expected: Expected::None,
                gate: Gate::Release,
                missing_fails: true,
            },
        ],
    };
    let (results, diags) = run_checks(&[pack], &g);
    assert_eq!(results.len(), 3);
    assert!(!results[0].passed(), "0.62 kg over the 0.5 kg target");
    assert_eq!(results[0].items.len(), 1);
    assert!(results[1].passed());
    assert!(results[2].passed());
    assert_eq!(diags.with_code(DiagCode::AsmCheckFailed).len(), 1);
    let links = vec![
        Link {
            kind: LinkKind::Verifies,
            from: LocalItem::Part {
                part: "bracket".into(),
            },
            to: "check.company.primary_mass".into(),
        },
        Link {
            kind: LinkKind::DerivedFrom,
            from: LocalItem::Part {
                part: "bracket".into(),
            },
            to: "legacy/bracket.step".into(),
        },
    ];
    let lp = check_links(&links, &|t| t == "legacy/bracket.step", &results, true);
    assert_eq!(lp.len(), 1);
    assert!(matches!(lp[0].1, LinkProblem::VerifiesFailing { .. }));
    let empty = Default::default();
    let publish = release_gate(Gate::Publish, &empty, &results, &[]);
    assert!(publish.open, "{:?}", publish.reasons);
    let release = release_gate(Gate::Release, &empty, &results, &lp);
    assert!(!release.open);
    assert_eq!(release.reasons.len(), 2, "{:?}", release.reasons);
}
