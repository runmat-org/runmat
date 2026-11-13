//! Compile-time administrative selection. Ordinary driving cannot retain
//! checkpoint archive/transport futures; numerical transitions stay in Controller.
use super::super::{controller::Quiescence, notice::Frame};
use super::*;
pub(super) trait Mode {
    async fn resume(
        &mut self,
        controller: &mut Controller<'_>,
        observers: &mut Observers,
        stopped: &mut bool,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()>;
    fn record(&mut self, frame: Frame) -> BuiltinResult<()>;
    async fn iteration(
        &mut self,
        controller: &mut Controller<'_>,
        observers: &mut Observers,
        display: OptimDisplay,
        header: &mut bool,
        stopped: &mut bool,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()>;
    async fn done(
        &mut self,
        controller: &mut Controller<'_>,
        observers: &mut Observers,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()>;
}
pub(super) struct Ordinary;
impl Mode for Ordinary {
    async fn resume(
        &mut self,
        _: &mut Controller<'_>,
        _: &mut Observers,
        _: &mut bool,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        Ok(())
    }
    fn record(&mut self, _: Frame) -> BuiltinResult<()> {
        Ok(())
    }
    async fn iteration(
        &mut self,
        _: &mut Controller<'_>,
        _: &mut Observers,
        _: OptimDisplay,
        _: &mut bool,
        _: &mut bool,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        Ok(())
    }
    async fn done(
        &mut self,
        _: &mut Controller<'_>,
        _: &mut Observers,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        Ok(())
    }
}
pub(super) struct Checkpoint<'a> {
    pub session: &'a mut Session,
    pub objective: &'a RepeatableCallable,
    pub resume: bool,
}
impl Mode for Checkpoint<'_> {
    async fn resume(
        &mut self,
        controller: &mut Controller<'_>,
        observers: &mut Observers,
        observer_stopped: &mut bool,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if self.resume && !*observer_stopped {
            {
                let session = &self.session;
                for &frame in session.notices() {
                    let notice = Notice::replay(
                        controller.campaign.history(),
                        frame,
                        controller.constraint_tolerance(),
                    )?;
                    if observers
                        .notify_notice(OutputState::Iter, notice, c)
                        .await?
                        .stop
                    {
                        *observer_stopped = true;
                        controller.request_stop();
                        break;
                    }
                }
            }
            if !*observer_stopped {
                let mut notice = Notice::live(controller)?;
                notice.resume = true;
                notice.frame.resume = true;
                let observed = observers
                    .notify_notice(OutputState::Iter, notice, c)
                    .await?;
                self.session.record_notice(observed.frame)?;
                *observer_stopped = observed.stop;
                if *observer_stopped {
                    controller.request_stop();
                }
                controller.acknowledge_notice();
            }
        }

        Ok(())
    }
    fn record(&mut self, frame: Frame) -> BuiltinResult<()> {
        self.session.record_notice(frame)
    }
    async fn iteration(
        &mut self,
        controller: &mut Controller<'_>,
        observers: &mut Observers,
        display: OptimDisplay,
        header: &mut bool,
        observer_stopped: &mut bool,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        {
            let session = &mut self.session;
            let objective = self.objective;
            if session.due(controller)? {
                while let Quiescence::Iteration = controller.quiesce(c).await? {
                    if !*observer_stopped {
                        if display == OptimDisplay::Iter {
                            report(controller, None, header, c)?;
                        }
                        let observed = observers
                            .notify_notice(OutputState::Iter, Notice::live(controller)?, c)
                            .await?;
                        session.record_notice(observed.frame)?;
                        if observed.stop {
                            *observer_stopped = true;
                            controller.request_stop();
                        }
                        controller.acknowledge_notice();
                    }
                }
                session.save(controller, objective, observers, c).await?;
            }
        }
        Ok(())
    }
    async fn done(
        &mut self,
        controller: &mut Controller<'_>,
        observers: &mut Observers,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        {
            let session = &mut self.session;
            let objective = self.objective;
            if session.enabled() {
                session.save(controller, objective, observers, c).await?;
            }
        }

        Ok(())
    }
}
