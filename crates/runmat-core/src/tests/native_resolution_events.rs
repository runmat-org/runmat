//! Executable request-level witnesses for class event admission and name observation.
use super::*;

fn text_value(outcome: &abi::ExecutionOutcome, name: &str) -> String {
    match outcome_named_upsert_value(outcome, name).expect("named output") {
        runmat_value::Value::String(text) => text.as_str().expect("nonmissing text").to_owned(),
        runmat_value::Value::CharArray(chars) => chars.row_string().expect("row text"),
        other => panic!("{name} was not text: {other:?}"),
    }
}

#[test]
fn inherited_declared_event_dispatches_event_data_and_rejects_before_callback() {
    run_deep_semantic_test(|| {
        let _serial = cwd_lock();
        let temporary = tempfile::TempDir::new().expect("class sources");
        std::fs::write(
            temporary.path().join("ObservedEventBase.m"),
            r#"classdef ObservedEventBase < handle
  events
    Tick
  end
  events(NotifyAccess=protected)
    Guarded
  end
end
"#,
        )
        .expect("base source");
        std::fs::write(
            temporary.path().join("ObservedEventChild.m"),
            r#"classdef ObservedEventChild < ObservedEventBase
  methods
    function fire(obj)
      notify(obj, 'Tick');
    end
    function fire_guarded(obj)
      notify(obj, 'Guarded');
    end
  end
end
"#,
        )
        .expect("child source");
        std::fs::write(
            temporary.path().join("ObservedEventCallback.m"),
            r#"classdef ObservedEventCallback
  properties(Static)
    Count = 0;
    LastName = '';
  end
  methods(Static)
    function on_tick(src, data)
      ObservedEventCallback.Count = ObservedEventCallback.Count + 1;
      ObservedEventCallback.LastName = data.EventName;
    end
  end
end
"#,
        )
        .expect("callback source");
        std::fs::write(
            temporary.path().join("main.m"),
            concat!(
                "h = ObservedEventChild(); ",
                "listener = addlistener(h, 'Tick', @ObservedEventCallback.on_tick); ",
                "h.fire(); count_after_fire = ObservedEventCallback.Count; ",
                "event_name = ObservedEventCallback.LastName; ",
                "try, addlistener(h, 'Missing', @ObservedEventCallback.on_tick); ",
                "unknown_id = 'BAD'; catch e, unknown_id = e.identifier; end; ",
                "try, notify(h, 'Missing'); unknown_notify_id = 'BAD'; ",
                "catch e, unknown_notify_id = e.identifier; end; ",
                "try, notify(h, 'Guarded'); access_id = 'BAD'; ",
                "catch e, access_id = e.identifier; end; ",
                "count_after_rejections = ObservedEventCallback.Count; ",
                "h.fire_guarded(); count_after_guarded = ObservedEventCallback.Count;"
            ),
        )
        .expect("request source");
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let source = temporary.path().join("main.m");
        let outcome = execute_path_request(&mut session, source.to_string_lossy().as_ref())
            .expect("class request");
        assert!(outcome_has_named_upsert(
            &outcome,
            "count_after_fire",
            &runmat_value::Value::Num(1.0)
        ));
        assert_eq!(text_value(&outcome, "event_name"), "Tick");
        assert_eq!(
            text_value(&outcome, "unknown_id"),
            "RunMat:AddListenerEventUnknown"
        );
        assert_eq!(
            text_value(&outcome, "unknown_notify_id"),
            "RunMat:NotifyEventUnknown"
        );
        assert_eq!(
            text_value(&outcome, "access_id"),
            "RunMat:NotifyEventAccess"
        );
        assert!(outcome_has_named_upsert(
            &outcome,
            "count_after_rejections",
            &runmat_value::Value::Num(1.0)
        ));
        assert!(outcome_has_named_upsert(
            &outcome,
            "count_after_guarded",
            &runmat_value::Value::Num(1.0)
        ));
    });
}

#[test]
fn which_tracks_executable_source_private_scope_and_later_path_revision() {
    let _serial = cwd_lock();
    let temporary = tempfile::TempDir::new().expect("source roots");
    for (folder, result) in [("first", 11), ("second", 22)] {
        let root = temporary.path().join(folder);
        std::fs::create_dir_all(&root).expect("source directory");
        std::fs::write(
            root.join("observed_path_value.m"),
            format!("function y = observed_path_value()\n y = {result};\nend\n"),
        )
        .expect("function source");
    }
    let first_root = temporary.path().join("first");
    std::fs::create_dir_all(first_root.join("private")).expect("private directory");
    std::fs::write(
        first_root.join("observed_private_entry.m"),
        "function [value, located] = observed_private_entry()\n located = which('observed_private_value');\n value = observed_private_value();\nend\n",
    )
    .expect("private caller");
    std::fs::write(
        first_root.join("private/observed_private_value.m"),
        "function y = observed_private_value()\n y = 33;\nend\n",
    )
    .expect("private companion");
    std::fs::write(
        first_root.join("observed_private_probe.m"),
        "function [located, value] = observed_private_probe()\n located = which('observed_private_value');\n value = feval('observed_private_value');\nend\n",
    )
    .expect("observation-only private caller");
    let _cwd = push_cwd(temporary.path());
    let _path = push_path_state("");
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let first = execute_text_request(
        &mut session,
        concat!(
            "addpath('first'); addpath('second'); ",
            "first_location = which('observed_path_value'); ",
            "first_value = observed_path_value(); ",
            "[private_value, private_location] = observed_private_entry(); ",
            "[private_probe_location, private_probe_value] = observed_private_probe(); ",
            "session_variable = 7; variable_location = which('session_variable');"
        ),
    )
    .expect("first request");
    assert!(outcome_has_named_upsert(
        &first,
        "first_value",
        &runmat_value::Value::Num(22.0)
    ));
    assert_eq!(
        text_value(&first, "first_location"),
        std::fs::canonicalize(temporary.path().join("second/observed_path_value.m"))
            .expect("canonical second source")
            .to_string_lossy()
    );
    assert!(outcome_has_named_upsert(
        &first,
        "private_value",
        &runmat_value::Value::Num(33.0)
    ));
    assert_eq!(
        text_value(&first, "private_location"),
        std::fs::canonicalize(first_root.join("private/observed_private_value.m"))
            .expect("canonical private source")
            .to_string_lossy()
    );
    assert_eq!(
        text_value(&first, "private_probe_location"),
        text_value(&first, "private_location"),
        "a private source remains observable without a static call edge"
    );
    assert!(outcome_has_named_upsert(
        &first,
        "private_probe_value",
        &runmat_value::Value::Num(33.0)
    ));
    assert_eq!(
        text_value(&first, "variable_location"),
        "'session_variable' is a variable."
    );
    let later = execute_text_request(
        &mut session,
        "rmpath('second'); later_location = which('observed_path_value'); later_value = observed_path_value();",
    )
    .expect("later request");
    assert!(outcome_has_named_upsert(
        &later,
        "later_value",
        &runmat_value::Value::Num(11.0)
    ));
    assert_eq!(
        text_value(&later, "later_location"),
        std::fs::canonicalize(first_root.join("observed_path_value.m"))
            .expect("canonical first source")
            .to_string_lossy()
    );
}
