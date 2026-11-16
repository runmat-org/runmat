//! The one log level a subscriber uses for both runtime feeds.
//!
//! The log feed (and the file and collector export beside it) records what the level
//! allows. The trace feed records what the level allows and always INFO and above, so a
//! quiet level keeps a run's spans while "debug" or "trace" adds the detailed ones.

use std::sync::{Arc, RwLock, RwLockReadGuard};

use tracing::subscriber::Interest;
use tracing::{span, Metadata, Subscriber};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};
use tracing_subscriber::layer::{Context, Filter};

/// The trace feed always records spans and events at this level and above. The runtime's
/// spans are INFO.
pub const TRACE_FEED_FLOOR: LevelFilter = LevelFilter::INFO;

/// A reloadable log level shared by the log feed's filter and the trace feed's filter.
/// Clones share the same level.
#[derive(Clone)]
pub struct SharedLogLevel {
    filter: Arc<RwLock<EnvFilter>>,
}

impl SharedLogLevel {
    pub fn new(filter: EnvFilter) -> Self {
        Self {
            filter: Arc::new(RwLock::new(filter)),
        }
    }

    /// Replaces the level with a filter spec such as "error" or "info,wgpu=warn".
    pub fn set(&self, spec: &str) -> Result<(), String> {
        let filter = EnvFilter::try_new(spec).map_err(|err| err.to_string())?;
        *self
            .filter
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = filter;
        // Callsites cached their interest under the old level; ask them again.
        tracing::callsite::rebuild_interest_cache();
        Ok(())
    }

    /// The per-layer filter for the log feed, the file and the collector export.
    pub fn log_filter(&self) -> LogLevelFilter {
        LogLevelFilter {
            level: self.clone(),
        }
    }

    /// The per-layer filter for the trace feed: the level or [`TRACE_FEED_FLOOR`],
    /// whichever lets more through.
    pub fn trace_filter(&self) -> TraceFeedFilter {
        TraceFeedFilter {
            level: self.clone(),
        }
    }

    fn read(&self) -> RwLockReadGuard<'_, EnvFilter> {
        self.filter
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// See [`SharedLogLevel::log_filter`].
pub struct LogLevelFilter {
    level: SharedLogLevel,
}

impl<S: Subscriber> Filter<S> for LogLevelFilter {
    fn enabled(&self, meta: &Metadata<'_>, cx: &Context<'_, S>) -> bool {
        Filter::<S>::enabled(&*self.level.read(), meta, cx)
    }

    fn callsite_enabled(&self, meta: &'static Metadata<'static>) -> Interest {
        Filter::<S>::callsite_enabled(&*self.level.read(), meta)
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Filter::<S>::max_level_hint(&*self.level.read())
    }

    // Span-scoped directives keep their state in the EnvFilter. Only this filter
    // forwards the span callbacks, so the trace feed's filter shares that state
    // without recording it twice.
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, cx: Context<'_, S>) {
        Filter::<S>::on_new_span(&*self.level.read(), attrs, id, cx)
    }

    fn on_record(&self, id: &span::Id, values: &span::Record<'_>, cx: Context<'_, S>) {
        Filter::<S>::on_record(&*self.level.read(), id, values, cx)
    }

    fn on_enter(&self, id: &span::Id, cx: Context<'_, S>) {
        Filter::<S>::on_enter(&*self.level.read(), id, cx)
    }

    fn on_exit(&self, id: &span::Id, cx: Context<'_, S>) {
        Filter::<S>::on_exit(&*self.level.read(), id, cx)
    }

    fn on_close(&self, id: span::Id, cx: Context<'_, S>) {
        Filter::<S>::on_close(&*self.level.read(), id, cx)
    }
}

/// See [`SharedLogLevel::trace_filter`].
pub struct TraceFeedFilter {
    level: SharedLogLevel,
}

impl<S: Subscriber> Filter<S> for TraceFeedFilter {
    fn enabled(&self, meta: &Metadata<'_>, cx: &Context<'_, S>) -> bool {
        *meta.level() <= TRACE_FEED_FLOOR || Filter::<S>::enabled(&*self.level.read(), meta, cx)
    }

    fn callsite_enabled(&self, meta: &'static Metadata<'static>) -> Interest {
        if *meta.level() <= TRACE_FEED_FLOOR {
            return Interest::always();
        }
        Filter::<S>::callsite_enabled(&*self.level.read(), meta)
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Filter::<S>::max_level_hint(&*self.level.read()).map(|hint| hint.max(TRACE_FEED_FLOOR))
    }
}
