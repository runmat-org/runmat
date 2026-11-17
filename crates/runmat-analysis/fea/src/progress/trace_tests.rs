use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

use super::*;

/// One span as a test layer saw it: name, parent name and fields, in creation order.
#[derive(Debug, Clone)]
pub(crate) struct CapturedSpan {
    pub name: &'static str,
    pub parent: Option<&'static str>,
    pub fields: BTreeMap<String, String>,
    pub closed: bool,
}

/// Captures span names, parents and fields (including fields recorded later).
#[derive(Clone, Default)]
pub(crate) struct SpanCapture(pub Arc<Mutex<Vec<CapturedSpan>>>);

struct Fields<'a>(&'a mut BTreeMap<String, String>);

impl Visit for Fields<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().into(), value.into());
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().into(), format!("{value:?}"));
    }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for SpanCapture {
    fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, context: Context<'_, S>) {
        let span = context.span(id).expect("new span");
        let mut fields = BTreeMap::new();
        attributes.record(&mut Fields(&mut fields));
        let mut spans = self.0.lock().unwrap();
        span.extensions_mut().insert(spans.len());
        spans.push(CapturedSpan {
            name: attributes.metadata().name(),
            parent: span.parent().map(|parent| parent.name()),
            fields,
            closed: false,
        });
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, context: Context<'_, S>) {
        let index = *context.span(id).unwrap().extensions().get::<usize>().unwrap();
        values.record(&mut Fields(&mut self.0.lock().unwrap()[index].fields));
    }

    fn on_close(&self, id: Id, context: Context<'_, S>) {
        let index = *context.span(&id).unwrap().extensions().get::<usize>().unwrap();
        self.0.lock().unwrap()[index].closed = true;
    }
}

/// A study run as the hosts trace it: check, then a two-case sweep whose cases mesh (with the
/// meshing spans inside), assemble, solve (with a factorization inside), recover and assess.
#[test]
fn a_study_run_traces_its_steps_in_order_with_labels_and_results() {
    let capture = SpanCapture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());
    tracing::subscriber::with_default(subscriber, || {
        let study = study_span("/project/bracket.fea");
        let _study = study.enter();
        let check = StudyStep::Check.span();
        check.in_scope(|| ());
        finish(&check, &check_result(0));
        drop(check);
        record_run_id(&study, "run-7");
        for case in ["load=1 kN", "load=2 kN"] {
            let case = case_span(case);
            let _case = case.enter();
            for step in &StudyStep::ALL[1..] {
                let span = step.span();
                span.in_scope(|| match step {
                    StudyStep::Mesh => {
                        tracing::info_span!("meshing.native_session")
                            .in_scope(|| tracing::info_span!("meshing.exact_dag").in_scope(|| ()));
                    }
                    StudyStep::Solve => {
                        tracing::info_span!("solver.factorize").in_scope(|| ());
                    }
                    _ => {}
                });
                let result = match step {
                    StudyStep::Mesh => format!("{} tet10", grouped(12_180)),
                    StudyStep::Assemble => dof_result(117_000),
                    StudyStep::Solve => steps_result(1),
                    StudyStep::Recover => outputs_result(21),
                    _ => assess_result(1, 1),
                };
                finish(&span, &result);
            }
        }
    });
    let spans = capture.0.lock().unwrap().clone();
    let names = spans.iter().map(|span| span.name).collect::<Vec<_>>();
    let case = [
        CASE_SPAN,
        "fea.study.mesh",
        "meshing.native_session",
        "meshing.exact_dag",
        "fea.study.assemble",
        "fea.study.solve",
        "solver.factorize",
        "fea.study.recover",
        "fea.study.assess",
    ];
    let expected = [STUDY_SPAN, "fea.study.check"]
        .into_iter()
        .chain(case)
        .chain(case)
        .collect::<Vec<_>>();
    assert_eq!(names, expected);
    assert!(spans.iter().all(|span| span.closed), "{spans:#?}");

    let study = &spans[0];
    assert_eq!(study.parent, None);
    assert_eq!(study.fields["study"], "/project/bracket.fea");
    assert_eq!(study.fields["run_id"], "run-7");
    assert!(!study.fields.contains_key(STEP_FIELD));

    let parent = |name: &str| {
        spans
            .iter()
            .filter(|span| span.name == name)
            .map(|span| span.parent)
            .collect::<Vec<_>>()
    };
    assert_eq!(parent("fea.study.check"), [Some(STUDY_SPAN)]);
    assert_eq!(parent(CASE_SPAN), [Some(STUDY_SPAN); 2]);
    for step in &StudyStep::ALL[1..] {
        assert_eq!(parent(step.span_name()), [Some(CASE_SPAN); 2]);
    }
    assert_eq!(parent("meshing.native_session"), [Some("fea.study.mesh"); 2]);
    assert_eq!(parent("solver.factorize"), [Some("fea.study.solve"); 2]);

    let steps = spans
        .iter()
        .filter_map(|span| span.fields.get(STEP_FIELD).map(|step| (step.as_str(), span)))
        .collect::<Vec<_>>();
    let labels = steps.iter().map(|(step, _)| *step).collect::<Vec<_>>();
    assert_eq!(
        labels,
        [
            "Check study",
            "Case load=1 kN",
            "Mesh",
            "Assemble",
            "Solve",
            "Recover fields",
            "Assess acceptance",
            "Case load=2 kN",
            "Mesh",
            "Assemble",
            "Solve",
            "Recover fields",
            "Assess acceptance",
        ]
    );
    for (step, span) in steps.iter().filter(|(step, _)| !step.starts_with("Case ")) {
        let result = span.fields.get(RESULT_FIELD).map(String::as_str);
        assert!(result.is_some_and(|result| !result.is_empty()), "{step}: {result:?}");
    }
    let results = spans
        .iter()
        .take(11)
        .filter_map(|span| span.fields.get(RESULT_FIELD).map(String::as_str))
        .collect::<Vec<_>>();
    assert_eq!(
        results,
        ["0 errors", "12,180 tet10", "117,000 dof", "1 step", "21 outputs", "1 of 1 passed"]
    );
}

#[test]
fn every_step_span_has_its_name_and_label() {
    let capture = SpanCapture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());
    tracing::subscriber::with_default(subscriber, || {
        for step in StudyStep::ALL {
            drop(step.span());
        }
    });
    let spans = capture.0.lock().unwrap().clone();
    assert_eq!(spans.len(), StudyStep::ALL.len());
    for (span, step) in spans.iter().zip(StudyStep::ALL) {
        assert_eq!(span.name, step.span_name());
        assert_eq!(span.fields[STEP_FIELD], step.label());
        assert!(!span.fields.contains_key(RESULT_FIELD));
    }
}

#[test]
fn results_read_as_one_line() {
    assert_eq!(grouped(0), "0");
    assert_eq!(grouped(999), "999");
    assert_eq!(grouped(1_000), "1,000");
    assert_eq!(grouped(1_234_567), "1,234,567");
    assert_eq!(check_result(1), "1 error");
    assert_eq!(check_result(2), "2 errors");
    assert_eq!(node_result(41_200), "41,200 nodes");
    assert_eq!(steps_result(3), "3 steps");
    assert_eq!(assess_result(0, 0), "no criteria");
    assert_eq!(assess_result(2, 3), "2 of 3 passed");
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet10,
    );
    assert_eq!(
        mesh_result(&mesh),
        format!("{} tet10", grouped(mesh.topology.volume_elements.len() as u64))
    );
}
