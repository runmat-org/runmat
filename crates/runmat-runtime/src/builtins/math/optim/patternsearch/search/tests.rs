use super::*;
use crate::builtins::common::random;
use crate::builtins::math::optim::common::ScalarCooperation;
use std::collections::BTreeSet;

#[test]
fn search_handle_selection_uses_original_namespace_and_routes_real_user_shadowing() {
    let original = crate::user_functions::test_support::exact_fixture_runtime();
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &foreign,
        &[("searchlhs", 29224)],
    )
    .unwrap();
    let _ambient = foreign.enter();
    let target = runmat_value::Value::FunctionHandle("searchlhs".into());
    let builtin = spec::prepare(&target, Some(&original)).unwrap();
    spec::metadata(&builtin, super::super::OPTIONS, Some(&original)).unwrap();
    assert!(matches!(
        spec::parse(&builtin, 1).unwrap(),
        Search::Latin { .. }
    ));
    let user = spec::prepare(&target, Some(&foreign)).unwrap();
    assert!(
        matches!(&user, runmat_value::Value::BoundFunctionHandle(handle) if handle.function == 29224)
    );
    spec::metadata(&user, super::super::OPTIONS, Some(&foreign)).unwrap();
    assert_eq!(spec::parse(&user, 1).unwrap(), Search::Custom);
}
#[tokio::test]
async fn original_stream_latin_strata_replay_and_stable_extreme_windows() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
        Some(&runtime),
    );
    let window = Window::new(
        vec![f64::NEG_INFINITY, 2.0, -f64::MAX, 4.0, f64::MAX],
        vec![
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            4.0,
            f64::INFINITY,
        ],
        &mut cooperation,
    )
    .await
    .unwrap();
    assert_eq!((window.lower[0], window.upper[0]), (-1.0, 1.0));
    assert_eq!((window.lower[1], window.upper[1]), (2.0, 4.0));
    assert!(window.upper[2] > window.lower[2] && window.upper[2].is_finite());
    assert_eq!((window.lower[3], window.upper[3]), (4.0, 4.0));
    assert_eq!((window.lower[4], window.upper[4]), (f64::MAX, f64::MAX));
    let mut prior = None;
    for _ in 0..2 {
        let mut turn = random::acquire_turn(Some(&runtime), "seed").await.unwrap();
        random::set_seed_in_turn(&mut turn, 29).unwrap();
        drop(turn);
        let samples = latin::generate(&[0.0, 0.0, 4.0], &[1.0, 1.0, 4.0], 7, &mut cooperation)
            .await
            .unwrap();
        for coordinate in 0..2 {
            let strata = samples
                .iter()
                .map(|point| (point[coordinate] * 7.0).floor() as usize)
                .collect::<BTreeSet<_>>();
            assert_eq!(strata, (0..7).collect());
        }
        assert!(samples.iter().all(|point| point[2] == 4.0));
        let turn = random::try_acquire_turn(Some(&runtime), "callback").unwrap();
        drop(turn);
        if let Some(expected) = &prior {
            assert_eq!(&samples, expected);
        } else {
            prior = Some(samples);
        }
    }
}
