//! A study's step spans reach the trace feed with their `step` field at B and their `result`
//! at E, which is what the Desktop Trace pane reads. Runs in its own process because the
//! subscriber is process-global.

use std::sync::{Arc, Mutex};

use runmat_analysis_fea::progress::trace::{self, StudyStep};
use runmat_logging::{init_logging, set_trace_hook, LoggingOptions, TraceEvent};

fn arg<'a>(event: &'a TraceEvent, key: &str) -> Option<&'a str> {
    event.args.as_ref()?.get(key)?.as_str()
}

#[test]
fn step_and_result_fields_reach_the_trace_feed() {
    let traces: Arc<Mutex<Vec<TraceEvent>>> = Arc::default();
    {
        let traces = traces.clone();
        set_trace_hook(move |events| traces.lock().unwrap().extend_from_slice(events));
    }
    let _guard = init_logging(LoggingOptions {
        enable_otlp: false,
        enable_traces: true,
        pid: 1,
        default_filter: Some("info".into()),
    });

    {
        let study = trace::study_span("/project/bracket.fea");
        trace::record_run_id(&study, "run-7");
        study.in_scope(|| {
            let case = trace::case_span("load=1 kN");
            case.in_scope(|| {
                let mesh = StudyStep::Mesh.span();
                mesh.in_scope(|| tracing::info_span!("meshing.native_session").in_scope(|| {}));
                trace::finish(&mesh, "12,180 tet10");
            });
        });
    }

    let events = traces.lock().unwrap().clone();
    let find = |name: &str, ph: &str| {
        events
            .iter()
            .find(|e| e.name == name && e.ph == ph)
            .unwrap_or_else(|| panic!("{name} {ph} missing from {events:#?}"))
    };
    let study = find("fea.study", "B");
    assert_eq!(arg(study, "study"), Some("/project/bracket.fea"));
    assert_eq!(arg(study, "step"), None);
    let case = find("fea.study.case", "B");
    assert_eq!(arg(case, "step"), Some("Case load=1 kN"));
    assert_eq!(case.parent_span_id, study.span_id);
    let mesh = find("fea.study.mesh", "B");
    assert_eq!(arg(mesh, "step"), Some("Mesh"));
    assert_eq!(mesh.parent_span_id, case.span_id);
    assert_eq!(arg(find("fea.study.mesh", "E"), "result"), Some("12,180 tet10"));
    assert_eq!(find("meshing.native_session", "B").parent_span_id, mesh.span_id);
}
