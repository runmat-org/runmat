//! One rooted mutable state and the existing exact graphics receiver layout.
//! Unknown user state survives; only this owner's old Best edge retires before
//! append, so a genuine retained predecessor still follows ordinary COW.
use super::super::{
    common::solver_timing::SolverTiming, history::History, invalid, observation::Observers,
    options::Options, ScalarCooperation,
};
use super::{
    frame::{self, How},
    search::{Record, Search},
    state_input::State,
};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_builtins::{OptimGeneticObserverState as Phase, OptimGeneticStateField as Field};
use runmat_value::Value;

pub(super) struct Event {
    pub stopped: bool,
    pub options_changed: bool,
    pub called: bool,
}
pub(super) struct Observing {
    observers: Observers,
    controls: Option<RootedValue>,
    state: Option<RootedValue>,
    history: Option<History>,
    start: Option<Value>,
    record: Option<Record>,
    needed: bool,
    #[cfg(feature = "plot-core")]
    layout: Option<crate::builtins::plotting::state::OptimizationPlotLayout>,
}
impl Observing {
    pub async fn new(
        options: &Options,
        timing: &SolverTiming,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let observers = Observers::select(options.prepared.value(), cooperation)?;
        #[cfg(not(feature = "plot-core"))]
        if observers.plot_count() != 0 {
            return Err(invalid("PlotFcn requires plot-core"));
        }
        let needed = observers.active() || options.mutation.custom().is_some();
        let effective_controls =
            needed || options.crossover.custom().is_some() || options.selection.custom().is_some();
        let controls = if effective_controls {
            Some(options.public(cooperation).await?)
        } else {
            None
        };
        let history = if needed {
            Some(History::new(cooperation)?)
        } else {
            None
        };
        let start = if needed {
            Some(frame::start_token(timing, cooperation)?)
        } else {
            None
        };
        Ok(Self {
            observers,
            controls,
            state: None,
            history,
            start,
            record: None,
            needed,
            #[cfg(feature = "plot-core")]
            layout: None,
        })
    }
    pub fn needed(&self) -> bool {
        self.needed
    }
    pub fn state(&self) -> Option<&RootedValue> {
        self.state.as_ref()
    }
    pub fn controls(&self) -> Option<&RootedValue> {
        self.controls.as_ref()
    }
    pub fn record(&mut self, record: Record) {
        self.record = Some(record);
    }
    pub async fn population_changed(
        &mut self,
        members: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let mut lineage = super::super::POLICY.capacity(members, cooperation.runtime())?;
        let mut ancestors = super::super::POLICY.capacity(members, cooperation.runtime())?;
        for _ in 0..members {
            lineage.push(super::super::lineage::Lineage::Observer);
            ancestors.push(None);
            cooperation.completed(1).await?;
        }
        self.record = Some(Record {
            expectation: None,
            selection: Vec::new(),
            lineage,
            ancestors,
        });
        Ok(())
    }
    pub async fn event_with<G, I: super::super::callbacks::Invocation>(
        &mut self,
        search: &Search<G>,
        phase: Phase,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<std::ops::ControlFlow<I::Stop, Event>> {
        if !self.needed {
            return Ok(std::ops::ControlFlow::Continue(Event {
                stopped: false,
                options_changed: false,
                called: false,
            }));
        }
        let runtime = super::continuation::original(cooperation)?;
        super::continuation::run(&runtime, move || {
            self.event_phase(search, phase, invocation, cooperation)
        })?
        .await
    }
    async fn event_phase<G, I: super::super::callbacks::Invocation>(
        &mut self,
        search: &Search<G>,
        phase: Phase,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<std::ops::ControlFlow<I::Stop, Event>> {
        cooperation.completed(0).await?;
        let append = matches!(phase, Phase::Init | Phase::Iter);
        let previous = match self.state.take() {
            Some(state) if append => Some(frame::retire_history_edge(state, cooperation).await?),
            state => state,
        };
        let stop_flag = previous
            .as_ref()
            .map(State::new)
            .transpose()?
            .and_then(|state| state.get(Field::StopFlag))
            .cloned();
        let history = self
            .history
            .as_mut()
            .ok_or_else(|| invalid("recorded state has no history owner"))?;
        if append {
            history
                .append(
                    search.assessment.best_comparison().unwrap_or(f64::NAN),
                    cooperation,
                )
                .await?;
        }
        let (expectation, selection, how) = if let Some(record) = &self.record {
            (
                record.expectation.as_ref(),
                record.selection.as_slice(),
                How::Evolution(&record.lineage),
            )
        } else {
            (
                None,
                &[][..],
                How::Initial {
                    supplied: search.supplied,
                },
            )
        };
        let state = frame::publish(
            frame::Snapshot {
                population: &search.generation,
                evaluated: &search.evaluated,
                assessment: &search.assessment,
                problem: &search.problem,
                history,
                generation: search.progress.generation,
                calls: search.objective_counts.points,
                start_token: self
                    .start
                    .as_ref()
                    .ok_or_else(|| invalid("recorded state has no stopwatch"))?,
                last_improvement: search.progress.last_improvement,
                last_improvement_time: search.progress.last_improvement_time,
                flags: &search.flags,
                expectation,
                selection,
                how,
                augmented: search.merit.is_some(),
                phase,
                inner_generation: search.progress.inner,
                total_inner_generations: search.progress.total_inner,
                previous: previous.as_ref(),
                stop_flag: stop_flag.as_ref(),
                action: search.action,
                ancestors: self
                    .record
                    .as_ref()
                    .map_or(&[], |record| record.ancestors.as_slice()),
            },
            cooperation,
        )
        .await?;
        drop(previous);
        if !self.observers.active() {
            self.state = Some(state);
            return Ok(std::ops::ControlFlow::Continue(Event {
                stopped: false,
                options_changed: false,
                called: false,
            }));
        }
        cooperation.completed(0).await?;
        if let Some(stop) = invocation.stop(super::super::callbacks::Event::Observer(phase)) {
            self.state = Some(state);
            return Ok(std::ops::ControlFlow::Break(stop));
        }
        #[cfg(feature = "plot-core")]
        let runtime = cooperation.runtime().cloned();
        let plots = self.observers.plot_count();
        #[cfg(feature = "plot-core")]
        if plots != 0 && self.layout.is_none() {
            self.layout = Some(
                crate::builtins::plotting::state::OptimizationPlotLayout::prepare(
                    plots,
                    runtime.as_ref(),
                )?,
            );
        }
        #[cfg(not(feature = "plot-core"))]
        if plots != 0 {
            return Err(invalid("PlotFcn requires plot-core"));
        }
        let controls = self
            .controls
            .take()
            .ok_or_else(|| invalid("observers lack their effective controls"))?;
        #[cfg(feature = "plot-core")]
        let layout = self.layout.as_ref();
        let select = |index| -> BuiltinResult<()> {
            #[cfg(feature = "plot-core")]
            {
                layout
                    .ok_or_else(|| invalid("plot callback layout is missing"))?
                    .select(index, runtime.as_ref())
            }
            #[cfg(not(feature = "plot-core"))]
            {
                let _ = index;
                Err(invalid("PlotFcn requires plot-core"))
            }
        };
        let returned = self
            .observers
            .notify_with(
                state,
                controls,
                phase,
                match phase {
                    Phase::Init | Phase::Done => true,
                    Phase::Iter => search
                        .progress
                        .generation
                        .is_multiple_of(search.options.plot_interval),
                    Phase::Interrupt => search
                        .progress
                        .inner
                        .is_multiple_of(search.options.plot_interval),
                },
                (plots != 0).then_some(&select as &dyn Fn(usize) -> BuiltinResult<()>),
                invocation,
                cooperation,
            )
            .await?;
        let (interrupted, returned) = match returned {
            std::ops::ControlFlow::Continue(returned) => (None, returned),
            std::ops::ControlFlow::Break((stop, returned)) => (Some(stop), returned),
        };
        self.state = Some(returned.state);
        self.controls = Some(returned.controls);
        if let Some(stop) = interrupted {
            return Ok(std::ops::ControlFlow::Break(stop));
        }
        Ok(std::ops::ControlFlow::Continue(Event {
            stopped: returned.stopped,
            options_changed: returned.options_changed,
            called: true,
        }))
    }
    pub async fn adopt_history(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let (Some(state), Some(history)) = (&self.state, &mut self.history) else {
            return Ok(());
        };
        let state = State::new(state)?;
        if let Some(value) = state.get(Field::Best) {
            if !super::state_input::same_numeric(value, &history.value()) {
                history
                    .adopt(state.root(Field::Best, cooperation)?, cooperation)
                    .await?;
            }
        }
        Ok(())
    }
    pub async fn reconfigure(
        &mut self,
        options: &Options,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let observers = Observers::select(options.prepared.value(), cooperation)?;
        #[cfg(not(feature = "plot-core"))]
        if observers.plot_count() != 0 {
            return Err(invalid("PlotFcn requires plot-core"));
        }
        #[cfg(feature = "plot-core")]
        if observers.plot_count() != self.observers.plot_count() {
            self.layout = None;
        }
        self.observers = observers;
        self.needed = self.observers.active() || options.mutation.custom().is_some();
        self.controls = Some(options.public(cooperation).await?);
        Ok(())
    }
    pub fn clear_expectation(&mut self) {
        if let Some(record) = &mut self.record {
            record.expectation = None;
            record.selection.clear();
        }
    }
}
