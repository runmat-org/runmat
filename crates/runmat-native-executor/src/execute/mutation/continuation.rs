use runmat_mir::{MirLocalId, MirPlace};
use runmat_runtime::execution::{RootedValue, RootedValueSequence};
use runmat_runtime::native::{NativeSiteRequest, NativeValueRef};
use runmat_runtime::RuntimeError;
use runmat_types::{AssignmentCreationPolicy, IndexKind};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::{flatten_place, HostState, PlaceSegment, PreparedPlaceWrite};

#[derive(Clone, Copy)]
enum Phase {
    ReadPrefix(usize),
    Writeback,
}

struct AssignmentPolicy {
    delete: bool,
    creation: AssignmentCreationPolicy,
}

pub(super) fn validate_creation_policy(
    place: &MirPlace,
    delete: bool,
    creation: AssignmentCreationPolicy,
) -> NativeExecutorResult<()> {
    let canonical = place.assignment_creation_policy(delete);
    if creation != canonical && (delete || creation != AssignmentCreationPolicy::ExistingOnly) {
        return Err(NativeExecutorError::Host(
            "native assignment creation policy does not match its place".into(),
        ));
    }
    Ok(())
}

fn root_creation_policy(
    outer: AssignmentCreationPolicy,
    segments: &[PlaceSegment],
) -> AssignmentCreationPolicy {
    if outer == AssignmentCreationPolicy::ExistingOnly {
        return AssignmentCreationPolicy::ExistingOnly;
    }
    match segments.first() {
        Some(PlaceSegment::Member(_) | PlaceSegment::DynamicMember(_)) => {
            AssignmentCreationPolicy::CreateStructFieldPath
        }
        Some(PlaceSegment::Index(indexing)) if indexing.kind == IndexKind::Paren => {
            AssignmentCreationPolicy::CreateArrayByIndex
        }
        _ => AssignmentCreationPolicy::ExistingOnly,
    }
}

fn assignment_root_for_segments(
    state: &mut HostState,
    root: MirLocalId,
    outer: AssignmentCreationPolicy,
    segments: &[PlaceSegment],
) -> NativeExecutorResult<Value> {
    if outer != AssignmentCreationPolicy::ExistingOnly
        && matches!(segments.first(), Some(PlaceSegment::Index(indexing)) if indexing.kind == IndexKind::Brace)
    {
        return state
            .sequence_root(root.0, runmat_mir::SequenceRootCreation::EmptyCell)
            .map(|(value, _retained_seed)| value);
    }
    state.assignment_root(root.0, root_creation_policy(outer, segments))
}

pub(super) fn assignment_root_for_place(
    state: &mut HostState,
    place: &MirPlace,
    creation: AssignmentCreationPolicy,
) -> NativeExecutorResult<Value> {
    validate_creation_policy(place, false, creation)?;
    let root = place.root_local().ok_or_else(|| {
        NativeExecutorError::Host("assignment root is not a canonical Native local".into())
    })?;
    match place.assignment_root_creation(creation) {
        runmat_mir::SequenceRootCreation::Ordinary(policy) => state.assignment_root(root.0, policy),
        intent @ runmat_mir::SequenceRootCreation::OrdinaryContainer(_) => state
            .sequence_root(root.0, intent)
            .map(|(value, _seed)| value),
        runmat_mir::SequenceRootCreation::OrdinaryCell => state
            .sequence_root(root.0, runmat_mir::SequenceRootCreation::EmptyCell)
            .map(|(value, _retained_seed)| value),
        _ => Err(NativeExecutorError::Host(
            "ordinary assignment has no canonical root creation recipe".into(),
        )),
    }
}

/// Capture the actual current arena generation, not a new reference to a
/// cloned payload. The one retained count belongs to the prepared fixed owner
/// and is released on consume/catch; only absent roots allocate a new seed.
pub(super) fn assignment_root_reference_for_place(
    state: &mut HostState,
    place: &MirPlace,
    creation: AssignmentCreationPolicy,
) -> NativeExecutorResult<NativeValueRef> {
    validate_creation_policy(place, false, creation)?;
    let root = place.root_local().ok_or_else(|| {
        NativeExecutorError::Host("assignment root is not a canonical Native local".into())
    })?;
    let recipe = place.assignment_root_creation(creation);
    let value = if matches!(
        recipe,
        runmat_mir::SequenceRootCreation::OrdinaryCell
            | runmat_mir::SequenceRootCreation::OrdinaryContainer(_)
    ) {
        let recipe = if recipe == runmat_mir::SequenceRootCreation::OrdinaryCell {
            runmat_mir::SequenceRootCreation::EmptyCell
        } else {
            recipe
        };
        let (value, seed) = state.sequence_root(root.0, recipe)?;
        if let Some(seed) = seed {
            return Ok(seed);
        }
        value
    } else {
        assignment_root_for_place(state, place, creation)?
    };
    // The canonical helper just refreshed this exact local. Reading its frame
    // slot does not reconcile/reinvoke the workspace or lexical source again.
    let reference = state.locals.get(root.0).copied().ok_or_else(|| {
        NativeExecutorError::Host("assignment root is outside the native frame".into())
    })?;
    if reference.is_null() {
        return state.arena.insert(value, Some(&state.runtime));
    }
    state.arena.retain(reference)?;
    Ok(reference)
}

mod sequence;
pub(in crate::execute) use sequence::{
    assign_prepared_fixed, assign_prepared_sequence, assign_sequence, PendingSequenceAssign,
};

/// The invocation-owned cursor for one statement site. Values are arena
/// references, not copies of a caller's mutable receiver. A completed step is
/// never evaluated again after a suspended child protocol resumes.
pub(in crate::execute) struct PendingAssign {
    request: NativeSiteRequest,
    place: MirPlace,
    root: MirLocalId,
    segments: Vec<PlaceSegment>,
    parents: Vec<(NativeValueRef, PlaceSegment)>,
    current: NativeValueRef,
    rhs: NativeValueRef,
    phase: Phase,
    step: usize,
    region: usize,
    delete: bool,
    creation: AssignmentCreationPolicy,
}

pub(in crate::execute) struct PendingPlacePrepare {
    request: NativeSiteRequest,
    place: MirPlace,
    root: MirLocalId,
    segments: Vec<PlaceSegment>,
    parents: Vec<(NativeValueRef, PlaceSegment)>,
    current: NativeValueRef,
    region: usize,
    step: usize,
}

impl PendingPlacePrepare {
    pub(in crate::execute) fn root_references(&self) -> impl Iterator<Item = NativeValueRef> + '_ {
        std::iter::once(self.current).chain(self.parents.iter().map(|(value, _)| *value))
    }
}

pub(super) fn prepare_place_write(
    state: &mut HostState,
    place: &MirPlace,
) -> NativeExecutorResult<(PreparedPlaceWrite, Value)> {
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("native place preparation has no active rvalue site".into())
    })?;
    let (region, next_step, _) = state.enter_contextual_region()?;
    let mut cursor = match state.pending_place_prepare.take() {
        Some(cursor)
            if cursor.request == request
                && cursor.place == *place
                && cursor.region == region
                && cursor.step == next_step =>
        {
            cursor
        }
        Some(_) => {
            return Err(NativeExecutorError::Host(
                "pending place preparation does not match the exact resumed site".into(),
            ));
        }
        None => {
            let mut segments = Vec::new();
            let root = flatten_place(place, &mut segments)?;
            let current = state.local_reference(root.0)?;
            PendingPlacePrepare {
                request,
                place: place.clone(),
                root,
                segments,
                parents: Vec::new(),
                current,
                region,
                step: 0,
            }
        }
    };
    while cursor.step < cursor.segments.len() {
        let step = cursor.step;
        state.enter_contextual_step(region, step)?;
        let parent = state.arena.get(cursor.current)?.clone();
        let segment = cursor.segments[step].clone();
        let result = read_step(state, parent, &segment, false);
        state.finish_contextual_step(region, step, step + usize::from(result.is_ok()))?;
        match result {
            Ok(value) => {
                cursor.parents.push((cursor.current, segment));
                cursor.current = state.arena.insert(value, Some(&state.runtime))?;
                cursor.step += 1;
            }
            Err(NativeExecutorError::CallSuspended) => {
                state.pending_place_prepare = Some(cursor);
                return Err(NativeExecutorError::CallSuspended);
            }
            Err(error) => return Err(error),
        }
    }
    let current = state.arena.get(cursor.current)?.clone();
    state.finish_contextual_region(region, cursor.current);
    Ok((
        PreparedPlaceWrite {
            root: cursor.root,
            parents: cursor.parents,
        },
        current,
    ))
}

pub(in crate::execute) struct PendingPlainMulti {
    pub(super) request: NativeSiteRequest,
    pub(super) next_target: usize,
}

pub(super) fn skip_completed(state: &mut HostState, place: &MirPlace) -> NativeExecutorResult<()> {
    if matches!(place, MirPlace::Local(_)) {
        return Ok(());
    }
    skip_completed_region(state)
}

pub(in crate::execute) fn skip_completed_region(state: &mut HostState) -> NativeExecutorResult<()> {
    let (_, _, completed) = state.enter_contextual_region()?;
    if completed.is_none() {
        return Err(NativeExecutorError::Host(
            "completed multi-assignment target has no retained result".into(),
        ));
    }
    Ok(())
}

impl PendingAssign {
    pub(in crate::execute) fn root_references(&self) -> impl Iterator<Item = NativeValueRef> + '_ {
        std::iter::once(self.current)
            .chain(std::iter::once(self.rhs))
            .chain(self.parents.iter().map(|(parent, _)| *parent))
    }

    fn new(
        state: &mut HostState,
        request: NativeSiteRequest,
        region: usize,
        place: &MirPlace,
        explicit_root: Option<Value>,
        rhs: Value,
        policy: AssignmentPolicy,
    ) -> NativeExecutorResult<Self> {
        let AssignmentPolicy { delete, creation } = policy;
        validate_creation_policy(place, delete, creation)?;
        let mut segments = Vec::new();
        let root = flatten_place(place, &mut segments)?;
        // MIR describes the outer place; the first segment alone selects the
        // absent root carrier, after that outer policy has been admitted.
        let current = match explicit_root {
            Some(root_value) => root_value,
            None => assignment_root_for_segments(state, root, creation, &segments)?,
        };
        let current = state.arena.insert(current, Some(&state.runtime))?;
        let rhs = state.arena.insert(rhs, Some(&state.runtime))?;
        Ok(Self {
            request,
            place: place.clone(),
            root,
            segments,
            parents: Vec::new(),
            current,
            rhs,
            phase: Phase::ReadPrefix(0),
            step: 0,
            region,
            delete,
            creation,
        })
    }

    fn advance(&mut self, state: &mut HostState) -> NativeExecutorResult<bool> {
        let current = state.arena.get(self.current)?.clone();
        let allow_init = self.creation != AssignmentCreationPolicy::ExistingOnly;
        let (updated, next) = match self.phase {
            Phase::ReadPrefix(index) if index + 1 < self.segments.len() => {
                let segment = self.segments[index].clone();
                let child = read_step(state, current, &segment, allow_init)?;
                let parent = self.current;
                let child = state.arena.insert(child, Some(&state.runtime))?;
                self.parents.push((parent, segment));
                (child, Phase::ReadPrefix(index + 1))
            }
            Phase::ReadPrefix(_) => {
                let rhs = state.arena.get(self.rhs)?.clone();
                let segment = self.segments.last().ok_or_else(|| {
                    NativeExecutorError::Host("nonlocal assignment has no place segment".into())
                })?;
                let value = write_step(state, current, segment, rhs, self.delete)?;
                (
                    state.arena.insert(value, Some(&state.runtime))?,
                    Phase::Writeback,
                )
            }
            Phase::Writeback => {
                let Some((parent, segment)) = self.parents.last() else {
                    return Ok(true);
                };
                let parent_value = state.arena.get(*parent)?.clone();
                let updated = write_step(state, parent_value, segment, current, false)?;
                self.parents.pop();
                (
                    state.arena.insert(updated, Some(&state.runtime))?,
                    Phase::Writeback,
                )
            }
        };
        self.current = updated;
        self.phase = next;
        self.step = self.step.checked_add(1).ok_or_else(|| {
            NativeExecutorError::Host("native mutation step ordinal overflowed".into())
        })?;
        Ok(matches!(self.phase, Phase::Writeback) && self.parents.is_empty())
    }
}

fn field_name(state: &mut HostState, segment: &PlaceSegment) -> NativeExecutorResult<String> {
    match segment {
        PlaceSegment::Member(name) => Ok(name.clone()),
        PlaceSegment::DynamicMember(operand) => {
            let value = super::super::operand::materialize_operand(state, operand)?;
            String::try_from(&value).map_err(|error| {
                NativeExecutorError::from(runmat_runtime::runtime_error::semantic_error(
                    "DynamicFieldName",
                    error,
                ))
            })
        }
        PlaceSegment::Index(_) => Err(NativeExecutorError::Host(
            "indexed mutation step has no member name".into(),
        )),
    }
}

fn member_call<F>(
    state: &mut HostState,
    make_future: impl FnOnce() -> F,
) -> NativeExecutorResult<Value>
where
    F: std::future::Future<Output = Result<RootedValueSequence, RuntimeError>> + 'static,
{
    let embedded = state.enter_embedded_call()?;
    let result = (|| {
        if let Some(mut values) =
            super::super::call_suspension::take_completed(state, embedded.as_ref())?
        {
            if values.len() != 1 {
                return Err(NativeExecutorError::Host(
                    "completed member operation did not produce one value".into(),
                ));
            }
            return Ok(values.remove(0));
        }
        let mut values = super::super::call_suspension::begin_rooted_sequence(
            state,
            embedded.clone(),
            make_future,
            1,
        )?;
        if values.len() != 1 {
            return Err(NativeExecutorError::Host(
                "member operation did not produce one value".into(),
            ));
        }
        Ok(values.remove(0))
    })();
    state.settle_embedded_result(embedded.as_ref(), result, std::slice::from_ref)
}

fn single_member_sequence(value: RootedValue) -> Result<RootedValueSequence, Box<RuntimeError>> {
    value.try_map_owned(
        |value| {
            runmat_value::ValueSequence::single(value)
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
                .map_err(Box::new)
        },
        |error| Box::new(error.into_runtime_error()),
    )
}

fn read_step(
    state: &mut HostState,
    base: Value,
    segment: &PlaceSegment,
    allow_init: bool,
) -> NativeExecutorResult<Value> {
    if let PlaceSegment::Index(indexing) = segment {
        let values = super::super::indexing::read_value(state, base, indexing, 1)?;
        return super::super::call_suspension::own_output_storage(values)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                NativeExecutorError::Host("nested indexing did not produce one value".into())
            });
    }
    let field = field_name(state, segment)?;
    let runtime = state.runtime.clone();
    let caller = state.function.name.clone();
    let access = super::super::subscript_path::frame_access(state)?;
    member_call(state, move || async move {
        let value = runtime
            .scope(runmat_runtime::object::resolve::load_member_with_access(
                Some(&runtime),
                base,
                field,
                allow_init,
                Some(&caller),
                &access,
            ))
            .await?;
        single_member_sequence(value).map_err(|error| *error)
    })
}

fn write_step(
    state: &mut HostState,
    base: Value,
    segment: &PlaceSegment,
    rhs: Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    if let PlaceSegment::Index(indexing) = segment {
        return super::super::indexing::assign(state, base, indexing, rhs, delete);
    }
    let field = field_name(state, segment)?;
    let runtime = state.runtime.clone();
    let caller = state.function.name.clone();
    let access = super::super::subscript_path::frame_access(state)?;
    member_call(state, move || async move {
        let value = runtime
            .scope(
                runmat_runtime::object::resolve::store_member_traced_with_access(
                    &runtime,
                    base,
                    field,
                    rhs,
                    Some(&caller),
                    &access,
                ),
            )
            .await?;
        single_member_sequence(value).map_err(|error| *error)
    })
}

pub(super) fn assign(
    state: &mut HostState,
    place: &MirPlace,
    rhs: Value,
    delete: bool,
    creation: AssignmentCreationPolicy,
) -> NativeExecutorResult<()> {
    let (root, reference) = drive(state, place, None, rhs, delete, creation)?;
    state.set_local(root.0, reference)
}

pub(super) fn assign_to_root(
    state: &mut HostState,
    place: &MirPlace,
    root_value: Value,
    rhs: Value,
    delete: bool,
    creation: AssignmentCreationPolicy,
) -> NativeExecutorResult<(MirLocalId, Value)> {
    let (root, reference) = drive(state, place, Some(root_value), rhs, delete, creation)?;
    Ok((root, state.arena.get(reference)?.clone()))
}

fn drive(
    state: &mut HostState,
    place: &MirPlace,
    explicit_root: Option<Value>,
    rhs: Value,
    delete: bool,
    creation: AssignmentCreationPolicy,
) -> NativeExecutorResult<(MirLocalId, NativeValueRef)> {
    validate_creation_policy(place, delete, creation)?;
    if let MirPlace::Local(local) = place {
        if creation == AssignmentCreationPolicy::ExistingOnly && explicit_root.is_none() {
            state.assignment_root(local.0, creation)?;
        }
        return Ok((*local, state.arena.insert(rhs, Some(&state.runtime))?));
    }
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("native mutation has no active statement site".into())
    })?;
    let (region, next_step, _) = state.enter_contextual_region()?;
    let mut cursor = match state.pending_assign.take() {
        Some(cursor)
            if cursor.request == request
                && cursor.place == *place
                && cursor.region == region
                && cursor.step == next_step
                && cursor.delete == delete
                && cursor.creation == creation =>
        {
            cursor
        }
        Some(_) => {
            return Err(NativeExecutorError::Host(
                "pending mutation does not match the exact resumed place site".into(),
            ));
        }
        None => PendingAssign::new(
            state,
            request,
            region,
            place,
            explicit_root,
            rhs,
            AssignmentPolicy { delete, creation },
        )?,
    };
    loop {
        let step = cursor.step;
        state.enter_contextual_step(region, step)?;
        let result = cursor.advance(state);
        let next_step = if result.is_ok() { cursor.step } else { step };
        state.finish_contextual_step(region, step, next_step)?;
        match result {
            Ok(true) => {
                let reference = cursor.current;
                state.finish_contextual_region(region, reference);
                return Ok((cursor.root, reference));
            }
            Ok(false) => {}
            Err(NativeExecutorError::CallSuspended) => {
                state.pending_assign = Some(cursor);
                return Err(NativeExecutorError::CallSuspended);
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod creation_policy_tests {
    use super::*;
    use runmat_mir::{MirIndexComponent, MirIndexPlan, MirIndexing};
    use runmat_types::{IndexResultContext, MemberName};

    #[test]
    fn typed_outer_policy_rejects_malformed_creation_and_deletion() {
        let indexed = MirPlace::Index(
            Box::new(MirPlace::Local(MirLocalId(0))),
            MirIndexing {
                argument_spans: None,
                kind: IndexKind::Paren,
                plan: MirIndexPlan::Slice,
                components: Vec::<MirIndexComponent>::new(),
                result_context: IndexResultContext::AssignmentTarget,
                cell_expand_all: false,
            },
        );
        assert!(validate_creation_policy(
            &indexed,
            false,
            AssignmentCreationPolicy::CreateArrayByIndex
        )
        .is_ok());
        assert!(
            validate_creation_policy(&indexed, false, AssignmentCreationPolicy::ExistingOnly)
                .is_ok(),
            "an existing indexed root may use the stricter no-creation contract"
        );
        assert!(validate_creation_policy(
            &indexed,
            false,
            AssignmentCreationPolicy::CreateStructFieldPath
        )
        .is_err());
        assert!(validate_creation_policy(
            &indexed,
            true,
            AssignmentCreationPolicy::CreateArrayByIndex
        )
        .is_err());
        assert!(
            validate_creation_policy(&indexed, true, AssignmentCreationPolicy::ExistingOnly)
                .is_ok()
        );
        let MirPlace::Index(_, indexing) = &indexed else {
            unreachable!()
        };
        assert_eq!(
            root_creation_policy(
                AssignmentCreationPolicy::CreateStructFieldPath,
                &[PlaceSegment::Index(indexing.clone())],
            ),
            AssignmentCreationPolicy::CreateArrayByIndex,
            "a nested member's outer policy does not change an indexed root carrier"
        );
        let mut brace = indexing.clone();
        brace.kind = IndexKind::Brace;
        assert_eq!(
            root_creation_policy(
                AssignmentCreationPolicy::CreateArrayByIndex,
                &[PlaceSegment::Index(brace)],
            ),
            AssignmentCreationPolicy::ExistingOnly,
            "brace indexing cannot synthesize an ordinary numeric root"
        );
        assert_eq!(
            root_creation_policy(
                AssignmentCreationPolicy::ExistingOnly,
                &[PlaceSegment::Index(indexing.clone())],
            ),
            AssignmentCreationPolicy::ExistingOnly,
            "deletion and existing-only writes cannot synthesize a root"
        );

        let nested = MirPlace::Member(Box::new(indexed), MemberName("field".into()));
        assert!(validate_creation_policy(
            &nested,
            false,
            AssignmentCreationPolicy::CreateStructFieldPath
        )
        .is_ok());
        assert!(validate_creation_policy(
            &nested,
            false,
            AssignmentCreationPolicy::CreateArrayByIndex
        )
        .is_err());
    }
}
