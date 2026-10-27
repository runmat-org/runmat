//! Typed ingress stages distinguish ordinary execution from a request that
//! needs authenticated checkpoint code/file ownership. There is no fallback
//! that treats a checkpoint filename as an objective name or drops its policy.
use super::{
    controller::{Controller, Setup},
    driver,
    initial::Initial,
    inputs::{Form, Raw},
    observers::{self, SelectedObservers},
    options::Options,
    parameters::Parameters,
    publication, IDENTITY,
};
use crate::builtins::math::optim::common::retained_control::Retained;
use crate::{
    builtins::{
        common::random,
        math::optim::{
            common::{solver_timing::SolverTiming, ScalarCooperation},
            intlinprog::feasibility::{Domain, Input},
        },
    },
    call::descriptor::RepeatableCallable,
    execution::{CallbackBatch, RootedValue, RootedValueSequence},
    BuiltinResult,
};
use runmat_value::Value;

pub(super) enum Plan {
    Ordinary(Retained<Ready>),
    Checkpoint(CheckpointRequest),
    Inconsistent(usize),
}
pub(super) enum CheckpointRequest {
    Start(Retained<CheckpointStart>),
    Resume { path: Value, options: Value },
}
pub(super) struct CheckpointStart {
    pub path: String,
    pub raw: Raw,
    pub options: Options,
}
pub(super) struct Ready {
    pub source: RootedValue,
    pub selected: RepeatableCallable,
    pub observers: SelectedObservers,
    pub parameters: Parameters,
    pub initial: Option<Initial>,
    pub model: Input,
    pub rng: Option<random::RngSnapshot>,
    // Original declared options retain their source roots until the solve's
    // final rooted publication; no callback captures are copied or rebound.
    pub options: Options,
}
pub(super) async fn prepare(form: Form, c: &mut ScalarCooperation) -> BuiltinResult<Plan> {
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::SURROGATEOPT_DESCRIPTOR,
        &runmat_builtins::SURROGATEOPT_ERROR_INVALID_ARGUMENT,
        super::NAME,
        c.runtime(),
    )?;
    crate::context::host_allocation::require_unbounded_workspace(c.runtime(), super::NAME,
        "finite host-memory execution has no complete surrogate numerical, callback and output workspace bound")?;
    let mut raw = match form {
        Form::Checkpoint { path, options } => {
            return Ok(Plan::Checkpoint(CheckpointRequest::Resume {
                path,
                options,
            }))
        }
        Form::Solve(raw) => raw.into_inner(),
    };
    let options = Options::parse(
        std::mem::replace(&mut raw.options, super::public_values::empty(c)?),
        c,
    )
    .await?;
    if let Some(path) = &options.checkpoint {
        return Ok(Plan::Checkpoint(CheckpointRequest::Start(Retained::new(
            CheckpointStart {
                path: path.clone(),
                raw,
                options,
            },
            c,
            super::CONTROL_STORAGE,
        )?)));
    }
    prepare_start(raw, options, c).await
}

pub(super) async fn prepare_start(
    raw: Raw,
    mut options: Options,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Plan> {
    let prepared = raw.prepare(c).await?;
    // Inconsistent original bounds have no feasible source point. This path
    // neither starts a pool/Clock/RNG turn nor invokes any objective/observer.
    if prepared.inconsistent {
        return Ok(Plan::Inconsistent(prepared.model.lower.len()));
    }
    let parameters = options.controls.resolve(prepared.model.lower.len())?;
    // Keep explicit sequencing: every selected numeric initial operand is
    // prepared before source selection, pool startup or UI allocation.
    let initial = match options.initial.take() {
        Some(value) => Some(
            Initial::prepare(
                value,
                prepared.model.lower.len(),
                parameters.nonlinear_tolerance,
                c,
            )
            .await?,
        ),
        None => None,
    };
    let source = RootedValue::in_optional_context(prepared.objective, c.runtime())?;
    let selected = observers::select(source.value(), c)?;
    let rng = prepared
        .rng
        .as_ref()
        .map(|value| {
            let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
            crate::builtins::stats::random::rng::restore_snapshot_value(value, c.runtime())
        })
        .transpose()?;
    let observers = SelectedObservers::prepare(options.output.as_ref(), options.plot.as_ref(), c)?;
    Ok(Plan::Ordinary(Retained::new(
        Ready {
            source,
            selected,
            observers,
            parameters,
            initial,
            model: prepared.model,
            rng,
            options,
        },
        c,
        super::CONTROL_STORAGE,
    )?))
}
impl Ready {
    pub(super) async fn run(
        mut self,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<RootedValueSequence> {
        let clock = SolverTiming::start(c.runtime())?;
        let workers =
            CallbackBatch::select_retained(c.runtime(), &self.selected, self.parameters.parallel)?;
        let mut observers = self.observers.admit_layout(c)?;
        let initial_rng = begin(self.rng.take(), c).await?;
        let domain = Domain::new(self.model, clock.clone(), self.parameters.time, 1e-6, c).await?;
        let mut controller = Controller::new(
            Setup {
                domain,
                source: &self.source,
                selected: &self.selected,
                workers: workers.as_ref(),
                clock,
                parameters: self.parameters,
                linear_tolerance: 1e-6,
                initial: self.initial,
            },
            c,
        )
        .await?;
        let stop = driver::run(&mut controller, &mut observers, self.options.display, c).await?;
        publication::finalize(&controller, stop, &initial_rng, c).await
    }
}
pub(super) async fn begin(
    restored: Option<random::RngSnapshot>,
    c: &ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let runtime = c.runtime().cloned();
    let mut turn = random::acquire_original_turn(runtime.as_ref(), IDENTITY.name).await?;
    if let Some(snapshot) = restored {
        random::apply_snapshot_in_turn(&mut turn, snapshot)?;
    }
    let snapshot = random::snapshot_in_turn(&mut turn)?;
    drop(turn);
    let value = {
        let _entry = runtime.as_ref().map(crate::context::RuntimeContext::enter);
        crate::builtins::stats::random::rng::snapshot_to_value(snapshot)?
    };
    RootedValue::in_optional_context(value, runtime.as_ref())
}
