use runmat_mir::{MirPlace, MirSequenceTarget};
use runmat_runtime::native::NativeValueRef;
use runmat_runtime::object::indexing::ObjectIndexComponent;
use runmat_runtime::sequence::{
    AssignmentStepSpec, PreparedSequenceDestination, SequenceDestinationBuilder,
    SequenceEndpointSpec,
};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use crate::execute::state::HostState;

use crate::execute::call_suspension::{
    begin_sequence_end_stage, begin_sequence_stage, take_completed_sequence_stage,
    SequenceEndAdmission, SequenceStage,
};

enum Phase {
    MaterializePath,
    PushStep(AssignmentStepSpec, runmat_types::AssignmentChildCreation),
    MaterializeEndpoint,
    Finish(SequenceEndpointSpec),
    AwaitPush,
    AwaitFinish,
    Poisoned,
}

enum TargetEndpoint {
    Sequence(MirSequenceTarget),
    Fixed {
        mutation: runmat_types::PlaceMutationKind,
        creation: runmat_types::AssignmentCreationPolicy,
    },
}
pub(in crate::execute::mutation) struct PendingSequenceTarget {
    target: TargetEndpoint,
    root: runmat_mir::MirLocalId,
    path: Vec<MirPlace>,
    builder: Option<SequenceDestinationBuilder>,
    index: usize,
    phase: Phase,
    region: Option<usize>,
    step: usize,
    retained: Vec<NativeValueRef>,
    absent_root: Option<NativeValueRef>,
    original_root: Option<NativeValueRef>,
    preparation_source: Option<super::FixedSourceOwner>,
    phase_roots: Option<runmat_runtime::execution::RootedValueLease>,
    fixed_component: Option<usize>,
    fixed_component_count: usize,
    fixed_end_receiver: Option<runmat_runtime::object::protocol::PreparedSubscriptReceiver>,
    fixed_end_pending: bool,
}

pub(in crate::execute::mutation) struct PreparedSequenceTarget {
    pub root: runmat_mir::MirLocalId,
    pub destination: PreparedSequenceDestination,
    pub absent_root: Option<NativeValueRef>,
    pub root_creation: runmat_mir::SequenceRootCreation,
}

impl PendingSequenceTarget {
    fn current(&self) -> NativeExecutorResult<&Value> {
        self.builder
            .as_ref()
            .map(SequenceDestinationBuilder::current)
            .ok_or_else(|| {
                NativeExecutorError::Host(
                    "sequence builder is owned by its pending callback".into(),
                )
            })
    }
    fn path_length(&self) -> NativeExecutorResult<usize> {
        match &self.preparation_source {
            Some(source) => source
                .descriptor()?
                .path_len()
                .checked_sub(1)
                .ok_or_else(|| NativeExecutorError::Host("fixed path has no endpoint".into())),
            None => Ok(self.path.len()),
        }
    }
    pub(super) fn new(
        state: &mut HostState,
        target: &MirSequenceTarget,
    ) -> NativeExecutorResult<Self> {
        let base = target.base();
        let mut path = Vec::new();
        let root = flatten_root(base, &mut path)?;
        let (root_value, absent_root) = state.sequence_root(root.0, target.root_creation())?;
        let path = path.into_iter().cloned().collect();
        let builder = SequenceDestinationBuilder::new_in_context(&state.runtime, &root_value)?;
        let builder = retain_method_frame(builder, state)?;
        let mut pending = Self {
            target: TargetEndpoint::Sequence(target.clone()),
            root,
            path,
            builder: Some(builder),
            index: 0,
            phase: Phase::MaterializePath,
            region: None,
            step: 0,
            retained: Vec::new(),
            absent_root,
            original_root: None,
            preparation_source: None,
            phase_roots: None,
            fixed_component: None,
            fixed_component_count: 0,
            fixed_end_receiver: None,
            fixed_end_pending: false,
        };
        pending.refresh_retained(state)?;
        Ok(pending)
    }

    pub(in crate::execute::mutation) fn new_fixed(
        state: &mut HostState,
        source: super::FixedSourceOwner,
        original: NativeValueRef,
    ) -> NativeExecutorResult<Self> {
        let owner = source.descriptor()?;
        let place = &owner.mutation.place;
        let mutation = owner.mutation.kind;
        let creation = owner.mutation.creation_policy;
        let root = place.root_local().ok_or_else(|| {
            NativeExecutorError::Host("fixed assignment has no canonical root".into())
        })?;
        if owner.path_len() == 0 {
            return Err(NativeExecutorError::Host(
                "fixed assignment has no endpoint".into(),
            ));
        }
        let root_value = state.arena.get(original)?.clone();
        let builder =
            SequenceDestinationBuilder::from_owned_in_context(&state.runtime, root_value)?
                .with_assignment_creation(creation);
        let builder = retain_method_frame(builder, state)?;
        let mut pending = Self {
            target: TargetEndpoint::Fixed { mutation, creation },
            root,
            // Fixed paths borrow immutable source; only the coordinate is
            // invocation-owned, never a deep MIR clone.
            path: Vec::new(),
            builder: Some(builder),
            index: 0,
            phase: Phase::MaterializePath,
            region: None,
            step: 0,
            retained: Vec::new(),
            absent_root: None,
            original_root: Some(original),
            preparation_source: Some(source),
            phase_roots: None,
            fixed_component: None,
            fixed_component_count: 0,
            fixed_end_receiver: None,
            fixed_end_pending: false,
        };
        pending.refresh_retained(state)?;
        Ok(pending)
    }

    pub(super) fn matches_sequence(&self, target: &MirSequenceTarget) -> bool {
        matches!(&self.target, TargetEndpoint::Sequence(actual) if actual == target)
    }

    pub(super) fn begin_fixed_index(&mut self, path_step: usize) -> NativeExecutorResult<()> {
        if self.index != path_step || !matches!(self.phase, Phase::MaterializePath) {
            return Err(NativeExecutorError::Host(
                "fixed index phase does not match its live path cursor".into(),
            ));
        }
        let source = self
            .preparation_source
            .as_ref()
            .ok_or_else(|| NativeExecutorError::Host("fixed index lost canonical source".into()))?;
        let MirPlace::Index(_, indexing) = source
            .descriptor()?
            .path_step(path_step)
            .ok_or_else(|| NativeExecutorError::Host("fixed index exceeds source path".into()))?
        else {
            return Err(NativeExecutorError::Host(
                "begin-index phase names a non-index path".into(),
            ));
        };
        self.fixed_component_count = indexing.components.len();
        self.fixed_component = None;
        self.fixed_end_receiver = None;
        Ok(())
    }

    pub(super) fn begin_fixed_selector(
        &mut self,
        path_step: usize,
        component: Option<usize>,
        needs_end: bool,
    ) -> NativeExecutorResult<()> {
        if self.index != path_step || !matches!(self.phase, Phase::MaterializePath) {
            return Err(NativeExecutorError::Host(
                "fixed selector does not match its live path cursor".into(),
            ));
        }
        if needs_end && component.is_none_or(|component| component >= self.fixed_component_count) {
            return Err(NativeExecutorError::Host(
                "contextual end has no matching fixed index component".into(),
            ));
        }
        self.fixed_component = needs_end.then_some(component).flatten();
        Ok(())
    }

    /// Only the active source path step advances. Callbacks own the builder
    /// by move while pending; later source selectors are never evaluated here.
    pub(super) fn finish_fixed_step(
        &mut self,
        state: &mut HostState,
        path_step: usize,
    ) -> NativeExecutorResult<Option<PreparedSequenceTarget>> {
        if self.index != path_step {
            return Err(NativeExecutorError::Host(
                "fixed finish does not match its path cursor".into(),
            ));
        }
        self.fixed_component = None;
        loop {
            if let Some(destination) = self.advance_step(state)? {
                return Ok(Some(PreparedSequenceTarget {
                    root: self.root,
                    destination,
                    absent_root: None,
                    root_creation: match &self.target {
                        TargetEndpoint::Fixed { creation, .. } => {
                            runmat_mir::SequenceRootCreation::Ordinary(*creation)
                        }
                        _ => {
                            return Err(NativeExecutorError::Host(
                                "fixed finish lost mutation authority".into(),
                            ))
                        }
                    },
                }));
            }
            self.refresh_retained(state)?;
            if self.index > path_step {
                self.fixed_end_receiver = None;
                self.fixed_component_count = 0;
                return Ok(None);
            }
        }
    }

    pub(super) fn has_fixed_end_context(&self) -> bool {
        self.fixed_component.is_some()
    }

    /// This runs at the actual End rvalue, after all earlier region effects.
    /// Nested own-index contexts take precedence in the ordinary operand owner.
    pub(super) fn resolve_fixed_end(
        &mut self,
        state: &mut HostState,
    ) -> NativeExecutorResult<NativeValueRef> {
        let component = self
            .fixed_component
            .ok_or_else(|| NativeExecutorError::Host("end has no active fixed component".into()))?;
        let stage = if self.fixed_end_pending {
            completed_stage_call(state)?
        } else {
            let runtime = state.runtime.clone();
            let functions = std::sync::Arc::clone(&state.functions);
            let function = state.function.portable_id;
            let component_count = self.fixed_component_count;
            let receiver = self.fixed_end_receiver.clone();
            self.fixed_end_pending = true;
            let builder = &mut self.builder;
            end_stage_call(state, |admission| {
                let builder = builder.take().expect("fixed End owns builder");
                async move {
                    let caller = functions
                        .iter()
                        .find(|candidate| candidate.portable_id == function)
                        .ok_or_else(|| {
                            NativeExecutorError::Host("fixed End lost caller metadata".into())
                        })?;
                    let receiver = match receiver {
                        Some(receiver) => receiver,
                        None => {
                            builder
                                .prepare_current_receiver_for_end_in_context(
                                    &runtime,
                                    Some(&caller.name),
                                )
                                .await?
                        }
                    };
                    let value = runmat_runtime::object::protocol::resolve_subscript_end_in_context(
                        &runtime,
                        &receiver,
                        component,
                        component_count,
                    )
                    .await?;
                    admission.finish(&runtime, builder, receiver, value)
                }
            })?
        };
        let SequenceStage::End(end) = stage else {
            return Err(NativeExecutorError::Host(
                "fixed End completion has wrong typed payload".into(),
            ));
        };
        let (builder, receiver, value) = end.into_parts();
        self.builder = Some(builder);
        self.fixed_end_receiver = Some(receiver);
        self.fixed_end_pending = false;
        value.with_owned(|value| state.arena.insert(value, Some(&state.runtime)))
    }

    pub(in crate::execute::mutation) fn root_references(
        &self,
    ) -> impl Iterator<Item = NativeValueRef> + '_ {
        self.retained.iter().copied().chain(self.absent_root)
    }

    fn refresh_retained(&mut self, state: &mut HostState) -> NativeExecutorResult<()> {
        self.retained.clear();
        let components = match &self.phase {
            Phase::PushStep(AssignmentStepSpec::Parentheses { selectors }, _) => {
                Some(selectors.as_slice())
            }
            Phase::PushStep(AssignmentStepSpec::Braces(selectors), _) => Some(selectors.as_slice()),
            Phase::Finish(SequenceEndpointSpec::CellContents(selectors)) => {
                Some(selectors.as_slice())
            }
            Phase::Finish(SequenceEndpointSpec::Fixed { step, .. }) => match step {
                AssignmentStepSpec::Parentheses { selectors }
                | AssignmentStepSpec::Braces(selectors) => Some(selectors.as_slice()),
                AssignmentStepSpec::Member(_) => None,
            },
            _ => None,
        };
        if self.original_root.is_some() {
            // Builder/destination own their exact charged Runtime root leases.
            // Keep the uncommitted phase's selectors under the same owner,
            // rather than multiplying source-quoted Native arena root slots by
            // a variable-width captured sequence.
            self.phase_roots = components
                .map(|components| {
                    runmat_runtime::execution::RootedValueLease::for_values_iterator_in_context(
                        components.iter().filter_map(|component| match component {
                            ObjectIndexComponent::Value(value) => Some(value),
                            ObjectIndexComponent::Colon | ObjectIndexComponent::End => None,
                        }),
                        &state.runtime,
                    )
                })
                .transpose()?;
            // Original capture registrations remained in their admitted frame
            // slots during every component conversion and successor-root
            // admission above. Retire only this completed selector's leases,
            // and only after the successor has actually been published.
            if components.is_some() {
                let place = self
                    .preparation_source
                    .as_ref()
                    .ok_or_else(|| {
                        NativeExecutorError::Host(
                            "fixed selector roots lost canonical source".into(),
                        )
                    })?
                    .descriptor()?
                    .path_step(self.index)
                    .ok_or_else(|| {
                        NativeExecutorError::Host(
                            "fixed selector roots exceed canonical path".into(),
                        )
                    })?;
                if let MirPlace::Index(_, indexing) = place {
                    for component in &indexing.components {
                        if let runmat_mir::MirIndexComponent::Expansion(region) = component {
                            if let runmat_mir::MirCallArg::CapturedSequence(id) = &region.argument {
                                state.retire_selector_sequence_lease(*id);
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
        if let Some(builder) = &self.builder {
            let mut admission_error = None;
            builder.visit_retained_values(&mut |value| {
                if admission_error.is_some() {
                    return;
                }
                match state.arena.insert(value.clone(), Some(&state.runtime)) {
                    Ok(reference) => self.retained.push(reference),
                    Err(error) => admission_error = Some(error),
                }
            });
            if let Some(error) = admission_error {
                return Err(error);
            }
        }
        if let Some(components) = components {
            for component in components {
                if let ObjectIndexComponent::Value(value) = component {
                    self.retained
                        .push(state.arena.insert(value.clone(), Some(&state.runtime))?);
                }
            }
        }
        Ok(())
    }

    pub(in crate::execute::mutation) fn advance(
        &mut self,
        state: &mut HostState,
    ) -> NativeExecutorResult<Option<PreparedSequenceTarget>> {
        let (region, next_step, _) = state.enter_contextual_region()?;
        match self.region {
            Some(expected) if expected != region || self.step != next_step => {
                return Err(NativeExecutorError::Host(
                    "pending sequence target does not match its exact preparation region".into(),
                ));
            }
            None => self.region = Some(region),
            Some(_) => {}
        }
        loop {
            let step = self.step;
            state.enter_contextual_step(region, step)?;
            let result = self.advance_step(state);
            let next_step = step + usize::from(result.is_ok());
            state.finish_contextual_step(region, step, next_step)?;
            match result {
                Ok(Some(destination)) => {
                    let root = match self.original_root.or(self.absent_root) {
                        Some(seed) => seed,
                        None => state.local_reference(self.root.0)?,
                    };
                    state.finish_contextual_region(region, root);
                    return Ok(Some(PreparedSequenceTarget {
                        root: self.root,
                        destination,
                        absent_root: self.absent_root,
                        root_creation: match &self.target {
                            TargetEndpoint::Sequence(target) => target.root_creation(),
                            TargetEndpoint::Fixed { creation, .. } => {
                                runmat_mir::SequenceRootCreation::Ordinary(*creation)
                            }
                        },
                    }));
                }
                Ok(None) => {
                    self.step += 1;
                    self.refresh_retained(state)?;
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn advance_step(
        &mut self,
        state: &mut HostState,
    ) -> NativeExecutorResult<Option<PreparedSequenceDestination>> {
        match std::mem::replace(&mut self.phase, Phase::Poisoned) {
            Phase::MaterializePath if self.index < self.path_length()? => {
                self.phase = Phase::MaterializePath;
                let place = match &self.preparation_source {
                    Some(source) => {
                        source.descriptor()?.path_step(self.index).ok_or_else(|| {
                            NativeExecutorError::Host(
                                "fixed path cursor exceeds canonical source".into(),
                            )
                        })?
                    }
                    None => &self.path[self.index],
                };
                let step = if self.original_root.is_some() {
                    fixed_assignment_step(state, place)?
                } else {
                    assignment_step(state, self.current()?, place)?
                };
                let child_creation = match &self.preparation_source {
                    Some(source) => {
                        let descriptor = source.descriptor()?;
                        descriptor.mutation.place.assignment_child_creation(
                            self.index + 1,
                            descriptor.mutation.creation_policy,
                        )
                    }
                    None => runmat_types::AssignmentChildCreation::ExistingOnly,
                };
                self.phase = Phase::PushStep(step, child_creation);
                Ok(None)
            }
            Phase::MaterializePath => {
                self.phase = Phase::MaterializeEndpoint;
                Ok(None)
            }
            Phase::PushStep(spec, child_creation) => {
                let runtime = state.runtime.clone();
                let functions = std::sync::Arc::clone(&state.functions);
                let function = state.function.portable_id;
                self.phase = Phase::AwaitPush;
                let builder = &mut self.builder;
                stage_call(state, || {
                    let mut builder = builder
                        .take()
                        .expect("current preparation owns its builder");
                    async move {
                        let caller = functions
                            .iter()
                            .find(|candidate| candidate.portable_id == function)
                            .ok_or_else(|| {
                                NativeExecutorError::Host(
                                    "sequence preparation lost canonical caller".into(),
                                )
                            })?;
                        builder
                            .push_assignment_step_in_context(
                                &runtime,
                                spec,
                                child_creation,
                                Some(&caller.name),
                            )
                            .await?;
                        Ok(SequenceStage::Builder(builder))
                    }
                })
                .map(|_| None)
            }
            Phase::AwaitPush => {
                let stage = completed_stage_call(state)?;
                let SequenceStage::Builder(builder) = stage else {
                    return Err(NativeExecutorError::Host(
                        "sequence step returned destination payload".into(),
                    ));
                };
                self.builder = Some(builder);
                self.index += 1;
                self.phase = Phase::MaterializePath;
                Ok(None)
            }
            Phase::MaterializeEndpoint => {
                self.phase = Phase::MaterializeEndpoint;
                self.phase = Phase::Finish(match &self.target {
                    TargetEndpoint::Fixed { mutation, .. } => SequenceEndpointSpec::Fixed {
                        step: fixed_assignment_step(
                            state,
                            &self
                                .preparation_source
                                .as_ref()
                                .ok_or_else(|| {
                                    NativeExecutorError::Host(
                                        "fixed endpoint lost canonical source".into(),
                                    )
                                })?
                                .descriptor()?
                                .mutation
                                .place,
                        )?,
                        mutation: *mutation,
                    },
                    TargetEndpoint::Sequence(target) => match target {
                        MirSequenceTarget::Member { member, .. } => {
                            let AssignmentStepSpec::Member(member) =
                                AssignmentStepSpec::member_in_context(&state.runtime, &member.0)?
                            else {
                                unreachable!("member factory")
                            };
                            SequenceEndpointSpec::Member(member)
                        }
                        MirSequenceTarget::DynamicMember { member, .. } => {
                            let member =
                                super::super::super::operand::materialize_operand(state, member)?;
                            let AssignmentStepSpec::Member(member) =
                                AssignmentStepSpec::dynamic_member_in_context(
                                    &state.runtime,
                                    &member,
                                )?
                            else {
                                unreachable!("member factory")
                            };
                            SequenceEndpointSpec::Member(member)
                        }
                        MirSequenceTarget::CellContents { indexing, .. } => {
                            SequenceEndpointSpec::CellContents(
                                super::super::super::indexing::materialize_index_components(
                                    state,
                                    self.current()?,
                                    indexing,
                                )?,
                            )
                        }
                    },
                });
                Ok(None)
            }
            Phase::Finish(endpoint) => {
                let runtime = state.runtime.clone();
                self.phase = Phase::AwaitFinish;
                let builder = &mut self.builder;
                stage_call(state, || {
                    let builder = builder
                        .take()
                        .expect("current preparation owns its builder");
                    async move {
                        let destination = builder.finish_in_context(&runtime, endpoint).await?;
                        Ok(SequenceStage::Destination(destination))
                    }
                })
                .map(|_| None)
            }
            Phase::AwaitFinish => {
                let stage = completed_stage_call(state)?;
                let SequenceStage::Destination(destination) = stage else {
                    return Err(NativeExecutorError::Host(
                        "sequence finalization returned builder payload".into(),
                    ));
                };
                Ok(Some(destination))
            }
            Phase::Poisoned => Err(NativeExecutorError::Host(
                "sequence preparation phase reentered without its owner".into(),
            )),
        }
    }
}

fn retain_method_frame(
    builder: SequenceDestinationBuilder,
    state: &HostState,
) -> NativeExecutorResult<SequenceDestinationBuilder> {
    if state.function.class_method_owner.is_some() {
        builder
            .with_method_frame_factory_in_context(&state.runtime, || {
                crate::execute::subscript_path::frame_access_runtime(state)
                    .map_err(|source| *source)
            })
            .map_err(Into::into)
    } else {
        Ok(builder)
    }
}

fn stage_call<F>(
    state: &mut HostState,
    make_future: impl FnOnce() -> F,
) -> NativeExecutorResult<SequenceStage>
where
    F: std::future::Future<Output = NativeExecutorResult<SequenceStage>> + 'static,
{
    stage_call_with(state, |state, embedded| {
        begin_sequence_stage(state, embedded, make_future)
    })
}

fn end_stage_call<F>(
    state: &mut HostState,
    make_future: impl FnOnce(SequenceEndAdmission) -> F,
) -> NativeExecutorResult<SequenceStage>
where
    F: std::future::Future<Output = NativeExecutorResult<SequenceStage>> + 'static,
{
    stage_call_with(state, |state, embedded| {
        begin_sequence_end_stage(state, embedded, make_future)
    })
}

fn stage_call_with(
    state: &mut HostState,
    begin: impl FnOnce(
        &mut HostState,
        Option<crate::execute::state::EmbeddedOperationIdentity>,
    ) -> NativeExecutorResult<SequenceStage>,
) -> NativeExecutorResult<SequenceStage> {
    let embedded = state.enter_embedded_call()?;
    let result = (|| {
        if let Some(stage) = take_completed_sequence_stage(state, embedded.as_ref())? {
            return Ok(stage);
        }
        begin(state, embedded.clone())
    })();
    let finished = state.finish_embedded_call(embedded.as_ref());
    match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(stage), Ok(())) => Ok(stage),
    }
}

fn completed_stage_call(state: &mut HostState) -> NativeExecutorResult<SequenceStage> {
    let embedded = state.enter_embedded_call()?;
    let result = take_completed_sequence_stage(state, embedded.as_ref()).and_then(|stage| {
        stage.ok_or_else(|| {
            NativeExecutorError::Host(
                "pending sequence stage has no matching owned completion".into(),
            )
        })
    });
    let finished = state.finish_embedded_call(embedded.as_ref());
    match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(stage), Ok(())) => Ok(stage),
    }
}

/// Descriptor-only materialization: canonical selector regions have already
/// published receipts in source order. No region cursor or evaluator is
/// entered by FinishStep, and every output slot is admitted before consuming
/// any captured backing.
fn fixed_assignment_step(
    state: &mut HostState,
    place: &MirPlace,
) -> NativeExecutorResult<AssignmentStepSpec> {
    use runmat_mir::{MirCallArg, MirIndexComponent};
    match place {
        MirPlace::Member(_, member) => {
            AssignmentStepSpec::member_in_context(&state.runtime, &member.0).map_err(Into::into)
        }
        MirPlace::DynamicMember(_, member) => {
            let member = super::super::super::operand::materialize_operand(state, member)?;
            AssignmentStepSpec::dynamic_member_in_context(&state.runtime, &member)
                .map_err(Into::into)
        }
        MirPlace::Index(_, indexing) => {
            let width = if indexing.cell_expand_all {
                1
            } else {
                indexing
                    .components
                    .iter()
                    .try_fold(0usize, |total, component| {
                        let count =
                            match component {
                                MirIndexComponent::Colon | MirIndexComponent::Expr(_) => 1,
                                MirIndexComponent::Expansion(region) if region.steps.is_empty() => {
                                    match &region.argument {
                                        MirCallArg::CapturedSequence(id) => {
                                            state.captured_sequence_width(*id)?
                                        }
                                        _ => return Err(NativeExecutorError::Host(
                                            "fixed expansion descriptor has no captured receipt"
                                                .into(),
                                        )),
                                    }
                                }
                                _ => {
                                    return Err(NativeExecutorError::Host(
                                        "fixed selector descriptor retains executable work".into(),
                                    ))
                                }
                            };
                        total.checked_add(count).ok_or_else(|| {
                            NativeExecutorError::Host("fixed selector width overflow".into())
                        })
                    })?
            };
            let mut selectors = runmat_runtime::object::indexing::prepare_components_in_context(
                &state.runtime,
                width,
            )?;
            let mut push = |component| {
                selectors.try_push(component).map_err(|_| {
                    NativeExecutorError::Host("fixed selector exceeded admitted width".into())
                })
            };
            if indexing.cell_expand_all {
                push(ObjectIndexComponent::Colon)?;
            } else {
                for component in &indexing.components {
                    match component {
                        MirIndexComponent::Colon => push(ObjectIndexComponent::Colon)?,
                        MirIndexComponent::Expr(operand) => push(ObjectIndexComponent::Value(
                            super::super::super::operand::materialize_operand(state, operand)?,
                        ))?,
                        MirIndexComponent::Expansion(region) => {
                            let MirCallArg::CapturedSequence(id) = &region.argument else {
                                unreachable!("descriptor preflight")
                            };
                            for value in state.take_selector_sequence(*id)? {
                                push(ObjectIndexComponent::Value(value))?;
                            }
                        }
                        _ => unreachable!("descriptor preflight"),
                    }
                }
            }
            let selectors = selectors.try_finish().map_err(|_| {
                NativeExecutorError::Host(
                    "fixed selector publication has incomplete admitted width".into(),
                )
            })?;
            Ok(match indexing.kind {
                runmat_types::IndexKind::Paren => AssignmentStepSpec::Parentheses { selectors },
                runmat_types::IndexKind::Brace => AssignmentStepSpec::Braces(selectors),
            })
        }
        _ => Err(NativeExecutorError::Host(
            "fixed path step is a root rather than a selector".into(),
        )),
    }
}

fn assignment_step(
    state: &mut HostState,
    current: &Value,
    place: &MirPlace,
) -> NativeExecutorResult<AssignmentStepSpec> {
    match place {
        MirPlace::Member(_, member) => {
            AssignmentStepSpec::member_in_context(&state.runtime, &member.0).map_err(Into::into)
        }
        MirPlace::DynamicMember(_, member) => {
            let member = super::super::super::operand::materialize_operand(state, member)?;
            AssignmentStepSpec::dynamic_member_in_context(&state.runtime, &member)
                .map_err(Into::into)
        }
        MirPlace::Index(_, indexing) => {
            let selectors = if indexing.cell_expand_all {
                let mut selectors =
                    runmat_runtime::object::indexing::prepare_components_in_context(
                        &state.runtime,
                        1,
                    )?;
                selectors
                    .try_push(ObjectIndexComponent::Colon)
                    .map_err(|_| {
                        NativeExecutorError::Host(
                            "whole-cell selector exceeded its admitted width".into(),
                        )
                    })?;
                selectors.try_finish().map_err(|_| {
                    NativeExecutorError::Host(
                        "whole-cell selector did not fill its admitted width".into(),
                    )
                })?
            } else {
                super::super::super::indexing::materialize_index_components(
                    state, current, indexing,
                )?
            };
            Ok(match indexing.kind {
                runmat_types::IndexKind::Paren => AssignmentStepSpec::Parentheses { selectors },
                runmat_types::IndexKind::Brace => AssignmentStepSpec::Braces(selectors),
            })
        }
        _ => Err(NativeExecutorError::Host(
            "destination path contains a root as an interior step".into(),
        )),
    }
}

pub(super) fn root_local(place: &MirPlace) -> NativeExecutorResult<runmat_mir::MirLocalId> {
    let mut path = Vec::new();
    flatten_root(place, &mut path)
}

fn flatten_root<'a>(
    place: &'a MirPlace,
    path: &mut Vec<&'a MirPlace>,
) -> NativeExecutorResult<runmat_mir::MirLocalId> {
    match place {
        MirPlace::Local(local) => Ok(*local),
        MirPlace::Binding(binding) => Err(NativeExecutorError::Host(format!(
            "verified Native IR retains legacy MIR binding place {binding:?}"
        ))),
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            let root = flatten_root(base, path)?;
            path.push(place);
            Ok(root)
        }
    }
}
