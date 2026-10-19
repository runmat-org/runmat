//! Restore temporary source-composition metadata when compilation completes or
//! an awaiting caller is dropped. The scope never owns workspace values.

use crate::session::RunMatSession;

pub(super) struct SourceCompositionScope<'a> {
    session: &'a mut RunMatSession,
    source_name: String,
    source_fullpath: Option<String>,
    companion: Option<crate::session::compile::CompanionSourceDiscovery>,
}

impl<'a> SourceCompositionScope<'a> {
    pub(super) fn enter(session: &'a mut RunMatSession) -> Self {
        Self {
            source_name: std::mem::take(&mut session.active_source_name),
            source_fullpath: session.active_source_fullpath_name.take(),
            companion: session.pending_companion_source_discovery.take(),
            session,
        }
    }

    pub(super) fn session(&mut self) -> &mut RunMatSession {
        self.session
    }
}

impl Drop for SourceCompositionScope<'_> {
    fn drop(&mut self) {
        self.session.active_source_name = std::mem::take(&mut self.source_name);
        self.session.active_source_fullpath_name = self.source_fullpath.take();
        self.session.pending_companion_source_discovery = self.companion.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;

    #[test]
    fn dropped_composition_future_restores_prior_source_and_companion() {
        let mut session = RunMatSession::with_options(false, false).expect("session");
        session.active_source_name = "prior.m".into();
        session.active_source_fullpath_name = Some("/prior/prior.m".into());
        let mut companion = crate::session::compile::CompanionSourceDiscovery::default();
        companion
            .private_function_names
            .insert("retained_prior_helper".into());
        session.pending_companion_source_discovery = Some(companion);

        let mut future = Box::pin(async {
            let mut scope = SourceCompositionScope::enter(&mut session);
            let session = scope.session();
            session.active_source_name = "selected.m".into();
            session.active_source_fullpath_name = Some("/selected/selected.m".into());
            session.pending_companion_source_discovery = Some(Default::default());
            futures::future::pending::<()>().await;
        });
        assert!(future.as_mut().now_or_never().is_none());
        drop(future);

        assert_eq!(session.active_source_name, "prior.m");
        assert_eq!(
            session.active_source_fullpath_name.as_deref(),
            Some("/prior/prior.m")
        );
        let companion = session
            .pending_companion_source_discovery
            .as_ref()
            .expect("prior companion restored");
        assert!(companion
            .private_function_names
            .contains("retained_prior_helper"));
    }
}
