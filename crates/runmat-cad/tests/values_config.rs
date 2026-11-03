//! The value model (report 12) and configurations at scale: `as <unit>`, the determinism
//! contract, cache keys, volatile snapshots, references; columnar design tables, change
//! impact and affected-only verify.

mod common;
use common::*;
use runmat_cad::assembly::fasteners::{Column, ColumnData, ColumnarTable, MemoryTables};
use runmat_cad::config::*;
use runmat_cad::ids::{ConfigurationName, InputName};
use runmat_cad::model::*;
use runmat_cad::units::{Dimension, StubUnits};
use runmat_cad::value::*;
use std::collections::BTreeMap;

struct Fns;

impl DynamicValueEvaluator for Fns {
    fn evaluate(&self, call: &DynamicCall, _: &[(String, Quantity)]) -> Result<Quantity, String> {
        Err(format!("use evaluate_output for {}", call.function))
    }
    fn evaluate_output(
        &self,
        call: &DynamicCall,
        args: &[(String, Quantity)],
    ) -> Result<DynamicOutput, String> {
        match call.function.as_str() {
            "plainThickness" => Ok(DynamicOutput::Double { value: 0.004 }),
            "typedThickness" => Ok(DynamicOutput::Quantity {
                value: Quantity::new(0.004, Dimension::LENGTH),
            }),
            "liveWind" => Ok(DynamicOutput::Quantity {
                value: Quantity::new(7.0, Dimension::LENGTH),
            }),
            "monteCarlo" => Ok(DynamicOutput::Double {
                value: args[0].1.si * 2.0,
            }),
            other => Err(format!("no function {}", other)),
        }
    }
    fn digest(&self, call: &DynamicCall) -> String {
        format!("digest:{}", call.function)
    }
    fn contract(&self, call: &DynamicCall) -> DynamicContract {
        DynamicContract {
            function_digest: self.digest(call),
            callees: vec![],
            reads: if call.function == "typedThickness" {
                vec![("profile.csv".into(), "sha256:aa".into())]
            } else {
                vec![]
            },
            runtime_version: "0.7.1".into(),
            effect: if call.function == "liveWind" {
                Some("reads the network".into())
            } else {
                None
            },
            randomness: if call.function == "monteCarlo" {
                Randomness::Seeded {
                    argument: "seed".into(),
                }
            } else {
                Randomness::None
            },
        }
    }
    fn snapshot(&self, key: &str) -> Option<Quantity> {
        (key == "replay").then(|| Quantity::new(1.0, Dimension::LENGTH))
    }
    fn reference(&self, path: &[String]) -> Option<Result<Quantity, String>> {
        match path.join(".").as_str() {
            "part.bracket.mass" => Some(Ok(Quantity::new(0.3, runmat_cad::units::Dimension::MASS))),
            _ => None,
        }
    }
}

fn call(f: &str, volatile: bool) -> ValueExpr {
    ValueExpr::Dynamic {
        call: DynamicCall {
            function: f.into(),
            arguments: vec![],
            volatile,
        },
    }
}

fn input_env(expr: ValueExpr) -> ValueEnv {
    let ev = Evaluator {
        units: &StubUnits,
        dynamic: &Fns,
    };
    let mut overrides = BTreeMap::new();
    overrides.insert(InputName::new("t"), expr);
    ev.evaluate_env(&[InputDecl::new("t", "length", mm(1.0))], &[], &overrides)
}

#[test]
fn v01_bare_doubles_need_as_unit() {
    let env = input_env(call("plainThickness", false));
    match env.failed.get("input.t") {
        Some(ValueError::UnitlessResult { si, .. }) => assert_eq!(si, "m"),
        other => panic!("{:?}", other),
    }
    let env = input_env(ValueExpr::coerce(call("plainThickness", false), "mm"));
    assert!((env.inputs[&InputName::new("t")].si - 0.000004).abs() < 1e-15);
    let env = input_env(call("typedThickness", false));
    assert!((env.inputs[&InputName::new("t")].si - 0.004).abs() < 1e-15);
    // `as` on a dimensioned value of another dimension is an error, not a conversion.
    let env = input_env(ValueExpr::coerce(call("typedThickness", false), "deg"));
    assert!(matches!(
        env.failed.get("input.t"),
        Some(ValueError::CoerceDimensioned { .. })
    ));
    assert_eq!(VALUE_CHOICES.comment, "//");
}

#[test]
fn v02_effects_need_volatile_and_volatile_values_are_recorded() {
    let env = input_env(call("liveWind", false));
    assert!(matches!(
        env.failed.get("input.t"),
        Some(ValueError::NotDeterministic { .. })
    ));
    let env = input_env(call("liveWind", true));
    assert!(env.failed.is_empty(), "{:?}", env.failed);
    let v = env.volatile.borrow();
    assert_eq!(v.len(), 1);
    assert!(!v.values().next().unwrap().from_snapshot);
}

#[test]
fn v03_cache_keys_chain_the_contract_and_seeded_randomness_is_deterministic() {
    let ev = Evaluator {
        units: &StubUnits,
        dynamic: &Fns,
    };
    let mc = |seed: f64| ValueExpr::Dynamic {
        call: DynamicCall {
            function: "monteCarlo".into(),
            arguments: vec![("seed".into(), ValueExpr::number(seed))],
            volatile: false,
        },
    };
    assert_eq!(ev.eval(&mc(3.0), &ValueEnv::default()).unwrap().si, 6.0);
    let c = DynamicCall {
        function: "typedThickness".into(),
        arguments: vec![],
        volatile: false,
    };
    let mut k = Fns.contract(&c);
    let a = dynamic_cache_key(&c, &[], &k);
    k.reads[0].1 = "sha256:bb".into();
    let b = dynamic_cache_key(&c, &[], &k);
    assert_ne!(a, b, "editing a declared read invalidates the value");
    k.runtime_version = "0.7.2".into();
    assert_ne!(dynamic_cache_key(&c, &[], &k), b);
    assert_eq!(
        ev.eval(
            &ValueExpr::Reference {
                path: vec!["part".into(), "bracket".into(), "mass".into()]
            },
            &ValueEnv::default()
        )
        .unwrap()
        .si,
        0.3
    );
    assert!(matches!(
        ev.eval(
            &ValueExpr::Reference {
                path: vec!["part".into(), "nope".into()]
            },
            &ValueEnv::default()
        ),
        Err(ValueError::UnresolvedReference { .. })
    ));
}

fn table(rows: &[(&str, f64, f64)]) -> ColumnarTable {
    ColumnarTable {
        columns: vec![
            Column {
                name: "name".into(),
                data: ColumnData::Text {
                    values: rows.iter().map(|r| r.0.to_string()).collect(),
                },
            },
            Column {
                name: "thickness".into(),
                data: ColumnData::F64 {
                    unit: "mm".into(),
                    values: rows.iter().map(|r| r.1).collect(),
                },
            },
            Column {
                name: "w".into(),
                data: ColumnData::F64 {
                    unit: "mm".into(),
                    values: rows.iter().map(|r| r.2).collect(),
                },
            },
        ],
    }
}

fn spec(digest: &str) -> ColumnarConfigurations {
    ColumnarConfigurations {
        name: "sizes".into(),
        table: digest.into(),
        key: "name".into(),
        inputs: [
            ("thickness".to_string(), InputName::new("thickness")),
            ("w".to_string(), InputName::new("width")),
        ]
        .into_iter()
        .collect(),
    }
}

#[test]
fn c01_columnar_design_tables_expand_typed() {
    let doc = block();
    let mut tables = MemoryTables::default();
    let d = tables.put(&table(&[
        ("std", 8.0, 80.0),
        ("light", 5.0, 70.0),
        ("xl", 12.0, 120.0),
    ]));
    let (m, problems) = materialize(&doc, &[spec(&d)], &tables, &StubUnits);
    assert!(problems.is_empty(), "{:?}", problems);
    let names: Vec<String> = m
        .all_configurations()
        .into_iter()
        .map(|c| c.name.0)
        .collect();
    assert_eq!(names, vec!["sizes.std", "sizes.light", "sizes.xl"]);
    let env = Env::new();
    let r = env.regen_config(&m, "sizes.light");
    assert!(
        r.diagnostics.errors().next().is_none(),
        "{:?}",
        r.diagnostics
    );
    assert!((r.values.inputs[&InputName::new("thickness")].si - 0.005).abs() < 1e-12);
    // A column of the wrong kind is refused.
    let mut bad = table(&[("std", 8.0, 80.0)]);
    if let ColumnData::F64 { unit, .. } = &mut bad.columns[1].data {
        *unit = "deg".into();
    }
    let d = tables.put(&bad);
    let (_, problems) = materialize(&doc, &[spec(&d)], &tables, &StubUnits);
    assert!(
        problems
            .iter()
            .any(|(_, p)| matches!(p, TableProblem::KindMismatch { .. })),
        "{:?}",
        problems
    );
}

#[test]
fn c02_impact_and_verify_touch_only_affected_configurations() {
    let mut old = block();
    old.configurations.push(Configuration {
        name: ConfigurationName::new("narrow"),
        overrides: [(InputName::new("width"), mm(40.0))].into_iter().collect(),
        suppress: Default::default(),
    });
    old.configurations.push(Configuration {
        name: ConfigurationName::new("thin"),
        overrides: [(InputName::new("thickness"), mm(4.0))]
            .into_iter()
            .collect(),
        suppress: Default::default(),
    });
    // Change the default width: `narrow` overrides it, so it cannot change.
    let mut new = old.clone();
    set_input(&mut new, "width", mm(90.0));
    let imp = affected_configurations(&old, &new);
    assert!(imp.affected.contains_key(&None));
    assert!(imp
        .affected
        .contains_key(&Some(ConfigurationName::new("thin"))));
    assert!(imp.unaffected.contains(&ConfigurationName::new("narrow")));
    let env = Env::new();
    let rep = verify(&new, &env.ctx(), Some(&imp));
    assert!(rep.ok(), "{:?}", rep);
    assert_eq!(rep.rows.len(), 2);
    assert!(rep.skipped.contains(&ConfigurationName::new("narrow")));
    // Changing only `thin`'s row reaches `thin` alone.
    let mut new2 = old.clone();
    new2.configurations[1]
        .overrides
        .insert(InputName::new("thickness"), mm(3.0));
    let imp = affected_configurations(&old, &new2);
    assert_eq!(
        imp.affected.keys().cloned().collect::<Vec<_>>(),
        vec![Some(ConfigurationName::new("thin"))]
    );
}
