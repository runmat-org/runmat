use super::*;
use futures::channel::oneshot;
use futures::future::{select, Either};
use runmat_filesystem::{self as fs, FsPublicationDisposition};
use staging_fixture::{PausePoint, StageFixture};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;

fn await_stage<F: Future>(started: oneshot::Receiver<()>, call: Pin<&mut F>) {
    match block_on(select(started, call)) {
        Either::Left((Ok(()), _)) => {}
        Either::Left((Err(_), _)) => panic!("selected figure stage was abandoned"),
        Either::Right((_result, _)) => panic!("figure save completed before selected stage"),
    }
}

fn selected_figure() -> f64 {
    let figure = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("figure");
    block_on(plot_builtin(vec![Value::Tensor(tensor(&[1.0, 4.0, 2.0]))])).expect("plot scene");
    figure
}

#[test]
fn public_savefig_retains_one_scene_snapshot_across_pending_write() {
    let _plot = setup();
    let figure = selected_figure();
    let _fs = fs::provider_override_lock();
    let fixture = StageFixture::new(PausePoint::Write);
    let _selected = fs::replace_provider(fixture.provider.clone());
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let mut call = Box::pin(savefig_builtin(vec![
        Value::Num(figure),
        Value::from("/snapshot.fig"),
    ]));
    await_stage(fixture.started, call.as_mut());
    clear_figure(Some(FigureHandle::from(figure as u32))).expect("clear after capture");
    fixture.resume.send(()).expect("resume write");
    assert!(block_on(call).expect("save completes"));
    let archive = fixture
        .published
        .lock()
        .unwrap()
        .clone()
        .expect("published");
    let payloads = block_on(decode_figure_bundle(None, &archive, OPENFIG)).expect("valid archive");
    assert_eq!(payloads.len(), 1);
    #[cfg(feature = "plot-core")]
    block_on(prepare_figure(&payloads[0], OPENFIG)).expect("captured scene remains valid");
    assert_eq!(fixture.drops.load(Ordering::SeqCst), 1);
}

#[test]
fn revoked_authority_during_flush_prevents_publication() {
    let _plot = setup();
    let figure = selected_figure();
    let _fs = fs::provider_override_lock();
    let fixture = StageFixture::new(PausePoint::Flush);
    let _selected = fs::replace_provider(fixture.provider.clone());
    let mut call = Box::pin(savefig_builtin(vec![
        Value::Num(figure),
        Value::from("/revoked.fig"),
    ]));
    await_stage(fixture.started, call.as_mut());
    fixture.lease.revoke();
    fixture.resume.send(()).expect("resume flush");
    let error = block_on(call).expect_err("revoked save must fail");
    assert_eq!(error.identifier(), ERROR_IO.identifier);
    assert!(fixture.published.lock().unwrap().is_none());
    assert_eq!(fixture.drops.load(Ordering::SeqCst), 1);
}

#[test]
fn cancellation_drops_a_pending_private_stage_without_publication() {
    let _plot = setup();
    let figure = selected_figure();
    let _fs = fs::provider_override_lock();
    let fixture = StageFixture::new(PausePoint::Write);
    let _selected = fs::replace_provider(fixture.provider.clone());
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let mut call = Box::pin(execute_savefig(
        Some(runtime),
        Some(0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(figure),
            Value::from("/cancelled.fig"),
        ]),
    ));
    await_stage(fixture.started, call.as_mut());
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(call).expect_err("pending save cancelled");
    assert!(error
        .identifier()
        .is_some_and(|id| id.ends_with(":Cancelled")));
    assert!(fixture.resume.send(()).is_err());
    assert!(fixture.published.lock().unwrap().is_none());
    assert_eq!(fixture.drops.load(Ordering::SeqCst), 1);
}

#[test]
fn cancellation_after_remote_commit_reports_indeterminate_lost_ack() {
    let _plot = setup();
    let figure = selected_figure();
    let _fs = fs::provider_override_lock();
    let fixture = StageFixture::new(PausePoint::Published);
    let _selected = fs::replace_provider(fixture.provider.clone());
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let mut call = Box::pin(execute_savefig(
        Some(runtime),
        Some(0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(figure),
            Value::from("/lost-ack.fig"),
        ]),
    ));
    await_stage(fixture.started, call.as_mut());
    assert!(
        fixture.published.lock().unwrap().is_some(),
        "commit occurred"
    );
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(call).expect_err("lost acknowledgement is not success");
    assert_eq!(
        crate::execution::staged_file::error_disposition(&error),
        Some(FsPublicationDisposition::Indeterminate)
    );
    assert!(fixture.resume.send(()).is_err());
    assert_eq!(fixture.drops.load(Ordering::SeqCst), 1);
}
