use std::collections::{BTreeMap, VecDeque};

use crate::{
    MirBody, MirFixedPreparationPhase, MirFixedPreparationState, MirStmtKind, MirTerminatorKind,
};

/// Prove the lifetime of the existing invocation-owned destination slot. The
/// descriptor is source occurrence + typed mutation, never textual place
/// equality. Exceptional entry abandons the slot; Await's normal resume keeps
/// it. Cancellation drops the invocation instead of publishing a destination.
pub(super) fn validate(body: &MirBody) -> Result<(), String> {
    let mut producers = BTreeMap::new();
    let mut consumers = BTreeMap::new();
    let mut ordinary_scalar_writers = BTreeMap::<crate::MirLocalId, usize>::new();
    let scalar_receipts = body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .filter_map(|statement| match &statement.kind {
            MirStmtKind::PrepareFixedAssignment(phase) => match phase.output() {
                Some(crate::MirFixedSelectorDestination::Scalar(local)) => Some(local),
                _ => None,
            },
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    for block in &body.blocks {
        for statement in &block.statements {
            match &statement.kind {
                MirStmtKind::Assign {
                    place: crate::MirPlace::Local(local),
                    ..
                } => {
                    *ordinary_scalar_writers.entry(*local).or_default() += 1;
                }
                MirStmtKind::MultiAssign { targets, .. } => {
                    for target in &targets.targets {
                        if let crate::MirOutputTarget::Place(crate::MirPlace::Local(local)) = target
                        {
                            *ordinary_scalar_writers.entry(*local).or_default() += 1;
                        }
                    }
                }
                _ => {}
            }
        }
        if let MirTerminatorKind::Await {
            result: Some(crate::MirPlace::Local(local)),
            ..
        } = &block.terminator.kind
        {
            *ordinary_scalar_writers.entry(*local).or_default() += 1;
        }
    }
    for statement in body.blocks.iter().flat_map(|block| &block.statements) {
        match &statement.kind {
            MirStmtKind::PreparedAssign { value, .. }
            | MirStmtKind::Assign { value, .. }
            | MirStmtKind::MultiAssign { value, .. }
            | MirStmtKind::SequenceAssign { value, .. }
            | MirStmtKind::Expr(value) => {
                crate::expression_region::validate_fixed_preparation_nesting(value)?
            }
            _ => {}
        }
        match &statement.kind {
            MirStmtKind::PrepareFixedAssignment(owner) => {
                owner.validate()?;
                if matches!(owner.phase, MirFixedPreparationPhase::CaptureReceiver)
                    && producers
                        .insert(
                            owner.assignment.source_statement,
                            (&owner.assignment, statement.span),
                        )
                        .is_some()
                {
                    return Err(
                        "fixed assignment source occurrence has multiple preparation sites".into(),
                    );
                }
            }
            MirStmtKind::PreparedAssign { preparation, value } => {
                preparation.validate_phased()?;
                if matches!(value, crate::MirRvalue::Call(call) if call.requested_outputs.known_count() != Some(1))
                {
                    return Err("prepared fixed assignment must preserve one RHS output".into());
                }
                if consumers
                    .insert(preparation.source_statement, (preparation, statement.span))
                    .is_some()
                {
                    return Err("fixed assignment source occurrence has multiple consumers".into());
                }
            }
            MirStmtKind::Assign { place, .. } if place.requires_prepared_assignment() => {
                return Err(
                    "expanding fixed assignment lacks its canonical preparation/consumer roles"
                        .into(),
                );
            }
            _ => {}
        }
    }
    if producers != consumers {
        return Err("fixed assignment preparation and consumer source contracts disagree".into());
    }
    for statement in body.blocks.iter().flat_map(|block| &block.statements) {
        if let MirStmtKind::PrepareFixedAssignment(phase) = &statement.kind {
            if producers.get(&phase.assignment.source_statement)
                != Some(&(&phase.assignment, statement.span))
            {
                return Err(
                    "fixed preparation phase differs from its exact source occurrence".into(),
                );
            }
            if let Some(crate::MirFixedSelectorDestination::Scalar(local)) = phase.output() {
                if ordinary_scalar_writers.contains_key(&local) {
                    return Err("ordinary work overwrites a reserved fixed selector receipt".into());
                }
                if !body
                    .locals
                    .iter()
                    .any(|entry| entry.id == local && entry.kind == crate::MirLocalKind::Temporary)
                {
                    return Err("fixed selector publishes outside a reserved temporary slot".into());
                }
                if phase.assignment.mutation.place.root_local() == Some(local) {
                    return Err("fixed selector overwrites its original receiver binding".into());
                }
            }
        }
    }
    let Some(entry) = body.blocks.first() else {
        return Ok(());
    };
    let blocks = body
        .blocks
        .iter()
        .map(|block| (block.id, block))
        .collect::<BTreeMap<_, _>>();
    let mut entries = BTreeMap::<_, Option<MirFixedPreparationState<'_>>>::new();
    let mut pending = VecDeque::from([(entry.id, None)]);
    while let Some((id, mut live)) = pending.pop_front() {
        if let Some(prior) = entries.get(&id) {
            live = MirFixedPreparationState::join(prior.clone(), live)?;
            if *prior == live {
                continue;
            }
        }
        entries.insert(id, live.clone());
        let block = blocks
            .get(&id)
            .ok_or("fixed destination lifetime reaches a missing block")?;
        for statement in &block.statements {
            if live.as_ref().is_some_and(|state| !state.is_ready())
                && !matches!(
                    statement.kind,
                    MirStmtKind::PrepareFixedAssignment(_) | MirStmtKind::PreparedAssign { .. }
                )
            {
                return Err(
                    "ordinary statement executes before fixed target preparation completes".into(),
                );
            }
            match &statement.kind {
                MirStmtKind::PrepareFixedAssignment(owner) => {
                    live = Some(MirFixedPreparationState::advance(
                        live,
                        owner,
                        &scalar_receipts,
                    )?);
                }
                MirStmtKind::PreparedAssign { preparation, .. } => {
                    live.take()
                        .ok_or("fixed assignment has no live preparation")?
                        .consume(preparation)?;
                }
                // Other output destination preparation shares the same slot.
                MirStmtKind::MultiAssign { .. } | MirStmtKind::SequenceAssign { .. }
                    if live.is_some() =>
                {
                    return Err(
                        "another output assignment overlaps a live fixed destination".into(),
                    );
                }
                _ => {}
            }
        }
        match &block.terminator.kind {
            MirTerminatorKind::Goto(next) => pending.push_back((*next, live)),
            MirTerminatorKind::Branch {
                then_block,
                else_block,
                ..
            } => {
                pending.extend([(*then_block, live.clone()), (*else_block, live)]);
            }
            MirTerminatorKind::Switch {
                cases, otherwise, ..
            } => {
                pending.extend(cases.iter().map(|(_, next)| (*next, live.clone())));
                pending.push_back((*otherwise, live));
            }
            MirTerminatorKind::For {
                body_block,
                exit_block,
                ..
            }
            | MirTerminatorKind::ParFor {
                body_block,
                exit_block,
                ..
            }
            | MirTerminatorKind::Spmd {
                body_block,
                exit_block,
                ..
            } => {
                pending.extend([(*body_block, live.clone()), (*exit_block, live)]);
            }
            MirTerminatorKind::TryCatch {
                try_block,
                catch_block,
                ..
            } => {
                pending.push_back((*try_block, live));
                pending.push_back((*catch_block, None));
            }
            MirTerminatorKind::Await { resume, .. } => pending.push_back((*resume, live)),
            MirTerminatorKind::Return(_) | MirTerminatorKind::Unreachable => {
                if live.is_some() {
                    return Err("fixed destination remains live at a function boundary".into());
                }
            }
        }
    }
    Ok(())
}
