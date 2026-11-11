//! Design values from `.m` functions (report 12 §2.3–2.6): resolution through the folder scope,
//! named arguments in declared order, the determinism checks and the cache.

use runmat_cad::ids::{InputName, LetName};
use runmat_cad::modules::ModuleScope;
use runmat_cad::syntax::parse_part;
use runmat_cad::units::StubUnits;
use runmat_cad::value::m_functions::{effect_of, parameters_of, MFunctions, MRunner};
use runmat_cad::value::{Evaluator, ValueError};
use std::cell::Cell;
use std::collections::BTreeMap;

/// Runs the few functions these tests define, by name, and counts calls.
struct Fake {
    calls: Cell<usize>,
    drift: bool,
}

impl MRunner for Fake {
    fn call(
        &self,
        sources: &[(String, String)],
        function: &str,
        args: &[f64],
    ) -> Result<Vec<f64>, String> {
        self.calls.set(self.calls.get() + 1);
        assert!(
            sources[0].1.contains(&format!("= {function}(")),
            "{function} is defined first"
        );
        match function {
            // A cam's base radius from its lift, rise angle and largest pressure angle.
            "camBaseRadius" => {
                let (lift, rise, max_pressure) = (args[0], args[1], args[2]);
                let base =
                    std::f64::consts::PI * lift / (2.0 * rise * max_pressure.tan()) - lift / 2.0;
                Ok(vec![
                    base + if self.drift {
                        self.calls.get() as f64
                    } else {
                        0.0
                    },
                ])
            }
            "lift" => Ok(vec![args[0] * 2.0]),
            "profile" => Ok(vec![1.0, 2.0]),
            "liveWind" => Ok(vec![7.0]),
            other => Err(format!("no {other}")),
        }
    }
    fn runtime_version(&self) -> String {
        "test".into()
    }
}

fn files() -> BTreeMap<String, String> {
    [
        ("runmat.toml", "[project]\n"),
        (
            "cam/cam.part",
            "part cam {\n    input lift : length = 8 mm\n    let base = camBaseRadius(rise = 120 deg, lift = input.lift, max_pressure = 30 deg) as m\n    let tip = cams.lift(2 mm) as m\n}\n",
        ),
        (
            "cam/camBaseRadius.m",
            "function r = camBaseRadius(lift, rise, max_pressure)\n% Base radius that keeps the pressure angle under max_pressure (harmonic rise).\nr = pi * lift / (2 * rise * tan(max_pressure)) - lift / 2;\nend\n",
        ),
        ("cam/helper.m", "function y = helper(x)\ny = x;\nend\n"),
        ("+cams/lift.m", "function s = lift(x)\ns = 2 * x;\nend\n"),
        ("cam/profile.m", "function p = profile()\np = [1 2];\nend\n"),
        ("cam/liveWind.m", "function v = liveWind()\nv = webread('https://example.invalid/wind');\nend\n"),
    ]
    .into_iter()
    .map(|(p, t)| (p.to_string(), t.to_string()))
    .collect()
}

fn scope(files: &BTreeMap<String, String>) -> ModuleScope {
    ModuleScope::from_files(files.iter().map(|(p, t)| (p.as_str(), t.as_str())))
}

#[test]
fn a_part_dimension_comes_from_a_same_folder_function_with_named_arguments() {
    let files = files();
    let scope = scope(&files);
    let runner = Fake {
        calls: Cell::new(0),
        drift: false,
    };
    let m = MFunctions::new(&scope, &files, "cam/cam.part", &runner);
    let f = m.resolve("camBaseRadius").unwrap();
    assert_eq!(f.file, "cam/camBaseRadius.m");
    assert_eq!(f.parameters, ["lift", "rise", "max_pressure"]);

    let doc = parse_part("cam/cam.part", &files["cam/cam.part"])
        .document
        .unwrap();
    let ev = Evaluator {
        units: &StubUnits,
        dynamic: &m,
    };
    let env = ev.evaluate_env(&doc.inputs, &doc.lets, &BTreeMap::new());
    assert!(env.failed.is_empty(), "{:?}", env.failed);
    let (lift, rise, pa) = (0.008_f64, 120_f64.to_radians(), 30_f64.to_radians());
    let expected = std::f64::consts::PI * lift / (2.0 * rise * pa.tan()) - lift / 2.0;
    assert!((env.lets[&LetName::new("base")].si - expected).abs() < 1e-15);
    // `cams.lift` is `+cams/lift.m` in a folder above the part.
    assert!((env.lets[&LetName::new("tip")].si - 0.004).abs() < 1e-15);
    assert_eq!(env.inputs[&InputName::new("lift")].si, 0.008);

    // Each deterministic call ran twice (the bit-for-bit audit); a second evaluation is cached.
    let ran = runner.calls.get();
    assert_eq!(ran, 4);
    ev.evaluate_env(&doc.inputs, &doc.lets, &BTreeMap::new());
    assert_eq!(runner.calls.get(), ran);
}

#[test]
fn a_function_that_differs_between_runs_is_refused() {
    let files = files();
    let scope = scope(&files);
    let runner = Fake {
        calls: Cell::new(0),
        drift: true,
    };
    let m = MFunctions::new(&scope, &files, "cam/cam.part", &runner);
    let doc = parse_part("cam/cam.part", &files["cam/cam.part"])
        .document
        .unwrap();
    let env = Evaluator {
        units: &StubUnits,
        dynamic: &m,
    }
    .evaluate_env(&doc.inputs, &doc.lets, &BTreeMap::new());
    match env
        .failed
        .get("let.base")
        .or_else(|| env.failed.get("base"))
    {
        Some(ValueError::DynamicFailed { message, .. }) => {
            assert!(message.contains("nondeterministic"), "{message}")
        }
        other => panic!("{other:?} in {:?}", env.failed),
    }
}

#[test]
fn effects_need_volatile_and_wrong_calls_say_what_the_function_takes() {
    let mut files = files();
    files.insert(
        "cam/cam.part".into(),
        "part cam {\n    let wind = liveWind() as m\n    let live = volatile liveWind() as m\n    let bad = camBaseRadius(lift = 1 mm, pitch = 2 deg) as m\n    let two = profile() as m\n    let none = missing(1) as m\n}\n".into(),
    );
    let scope = scope(&files);
    let runner = Fake {
        calls: Cell::new(0),
        drift: false,
    };
    let m = MFunctions::new(&scope, &files, "cam/cam.part", &runner);
    let doc = parse_part("cam/cam.part", &files["cam/cam.part"])
        .document
        .unwrap();
    let env = Evaluator {
        units: &StubUnits,
        dynamic: &m,
    }
    .evaluate_env(&doc.inputs, &doc.lets, &BTreeMap::new());
    let failed = |name: &str| {
        env.failed
            .iter()
            .find(|(k, _)| k.ends_with(name))
            .map(|(_, e)| e.to_string())
            .unwrap_or_default()
    };
    assert!(failed("wind").contains("webread"), "{}", failed("wind"));
    assert_eq!(env.lets[&LetName::new("live")].si, 7.0);
    assert!(
        failed("bad").contains("takes (lift, rise, max_pressure), not `pitch`"),
        "{}",
        failed("bad")
    );
    assert!(
        failed("two").contains("returned 2 numbers"),
        "{}",
        failed("two")
    );
    assert!(failed("none").contains("missing"), "{}", failed("none"));
}

#[test]
fn signatures_and_effects_read_from_the_text() {
    assert_eq!(
        parameters_of("function [a, b] = f(x, ~, y)\n", "f").unwrap(),
        ["x", "y"]
    );
    assert_eq!(
        parameters_of("function g\n", "g").unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        effect_of("x = rand(3); % noise").as_deref(),
        Some("`rand` (unseeded randomness)")
    );
    assert_eq!(effect_of("s = 'tic toc'; % rand"), None);
    assert_eq!(effect_of("y = opts.load + x';"), None);
}
