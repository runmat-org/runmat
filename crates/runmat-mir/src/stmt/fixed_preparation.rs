use serde::{Deserialize, Serialize};

use crate::{
    MirCallArg, MirCallArgumentRegion, MirLocalId, MirOperand, MirPlace, MirPreparedAssignment,
    MirSequenceLocalId,
};

/// A selector evaluation publishes exactly one typed receipt into existing
/// frame slots. A CSL receipt is never a scalar replacement or language value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MirFixedSelectorDestination {
    Scalar(MirLocalId),
    Sequence(MirSequenceLocalId),
}

/// Ordered phases of the single prepared assignment owner. Await remains a
/// CFG terminator BETWEEN completed regions; no region embeds an Await.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirFixedPreparationPhase {
    CaptureReceiver,
    /// Reuses the destination's existing indexed-selector context.
    BeginIndex {
        path_step: usize,
    },
    EvaluateSelector {
        path_step: usize,
        /// None is the dynamic member name; indexed selectors carry their
        /// exact source component so contextual end has a typed receiver.
        component: Option<usize>,
        region: Box<MirCallArgumentRegion>,
        destination: MirFixedSelectorDestination,
    },
    FinishStep {
        path_step: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirFixedPreparation {
    pub assignment: MirPreparedAssignment,
    pub phase: MirFixedPreparationPhase,
}

impl MirPreparedAssignment {
    pub fn path_len(&self) -> usize {
        fn depth(place: &MirPlace) -> usize {
            match place {
                MirPlace::Member(base, _)
                | MirPlace::DynamicMember(base, _)
                | MirPlace::Index(base, _) => 1 + depth(base),
                _ => 0,
            }
        }
        depth(&self.mutation.place)
    }

    /// Borrow one root-to-leaf source step, never allocate a copied path.
    pub fn path_step(&self, index: usize) -> Option<&MirPlace> {
        let skip = self.path_len().checked_sub(index.checked_add(1)?)?;
        let mut place = &self.mutation.place;
        for _ in 0..skip {
            place = match place {
                MirPlace::Member(base, _)
                | MirPlace::DynamicMember(base, _)
                | MirPlace::Index(base, _) => base,
                _ => return None,
            };
        }
        Some(place)
    }
}

impl MirFixedPreparation {
    pub fn validate(&self) -> Result<(), String> {
        self.assignment.validate_phased()?;
        if !self.assignment.steps.is_empty() {
            return Err("phased fixed assignment cannot retain eager lifted target work".into());
        }
        let mut reserved = std::collections::BTreeSet::new();
        for step in 0..self.assignment.path_len() {
            let place = self.assignment.path_step(step).expect("bounded path step");
            for component in 0..selector_count(place) {
                if let Some(destination) = selector_destination(place, component)? {
                    if matches!(destination, MirFixedSelectorDestination::Scalar(local) if Some(local) == self.assignment.mutation.place.root_local())
                        || !reserved.insert(destination)
                    {
                        return Err(
                            "fixed descriptor aliases reserved selector receipts or its receiver"
                                .into(),
                        );
                    }
                }
            }
        }
        match &self.phase {
            MirFixedPreparationPhase::CaptureReceiver => Ok(()),
            MirFixedPreparationPhase::BeginIndex { path_step } => {
                if !matches!(
                    self.assignment.path_step(*path_step),
                    Some(MirPlace::Index(_, _))
                ) {
                    return Err("fixed index context does not identify an indexed path step".into());
                }
                Ok(())
            }
            MirFixedPreparationPhase::FinishStep { path_step } => {
                self.assignment.path_step(*path_step).ok_or_else(|| {
                    "fixed preparation step is outside its source path".to_owned()
                })?;
                Ok(())
            }
            MirFixedPreparationPhase::EvaluateSelector {
                path_step,
                component,
                region,
                destination,
            } => {
                if matches!(destination, MirFixedSelectorDestination::Scalar(local) if Some(*local) == self.assignment.mutation.place.root_local())
                {
                    return Err(
                        "fixed selector intermediate overwrites its original receiver".into(),
                    );
                }
                let invalid_definition = |slot: MirFixedSelectorDestination| {
                    reserved.contains(&slot)
                        || slot == *destination
                        || matches!(slot, MirFixedSelectorDestination::Scalar(local) if Some(local) == self.assignment.mutation.place.root_local())
                };
                let mut aliases = region.steps.iter().any(|step| match step {
                    crate::MirExpressionStep::Let { local, .. } => {
                        invalid_definition(MirFixedSelectorDestination::Scalar(*local))
                    }
                    crate::MirExpressionStep::CaptureSequence { destination, .. } => {
                        invalid_definition(MirFixedSelectorDestination::Sequence(*destination))
                    }
                });
                region.visit_expression_regions(&mut |region| {
                    aliases |= region.defined_locals().any(|local| {
                        invalid_definition(MirFixedSelectorDestination::Scalar(local))
                    });
                    aliases |= region.defined_sequences().any(|sequence| {
                        invalid_definition(MirFixedSelectorDestination::Sequence(sequence))
                    });
                });
                if aliases {
                    return Err("private selector work forges an outer reserved receipt".into());
                }
                let place = self
                    .assignment
                    .path_step(*path_step)
                    .ok_or_else(|| "fixed selector is outside its source path".to_owned())?;
                match (place, component) {
                    (MirPlace::Index(_, indexing), Some(component))
                        if *component < indexing.components.len() => {}
                    (MirPlace::DynamicMember(_, _), None) => {}
                    _ => {
                        return Err(
                            "fixed selector context disagrees with its original path step".into(),
                        )
                    }
                }
                // Frame-owned CSL receipts are proved at the enclosing CFG.
                // Ordinary call argument regions keep their stricter local-only validation.
                let mut external = std::collections::BTreeSet::new();
                region.visit_sequence_locals(&mut |sequence| {
                    external.insert(*sequence);
                });
                region
                    .validate_with_external_sequences(&external)
                    .map_err(str::to_owned)?;
                if matches!(place, MirPlace::Index(_, indexing) if indexing.components.iter().any(|selector| matches!(selector, crate::MirIndexComponent::Expansion(_))))
                    && region.contains_unscoped_end()
                {
                    return Err("contextual end cannot share an index list with variable-width argument expansion".into());
                }
                match (&region.argument, destination) {
                    (MirCallArg::Single(_), MirFixedSelectorDestination::Scalar(_))
                    | (MirCallArg::CapturedSequence(_) | MirCallArg::Expansion(_), MirFixedSelectorDestination::Sequence(_)) => Ok(()),
                    _ => Err("fixed selector publication would scalarize or invent a comma-separated sequence".into()),
                }
            }
        }
    }
}

/// One canonical phase cursor shared by MIR and native CFG validation. It
/// describes compile-time ownership only; it is not an execution registry.
#[derive(Clone, Debug, PartialEq)]
pub struct MirFixedPreparationState<'a> {
    assignment: &'a MirPreparedAssignment,
    path_step: usize,
    index_started: bool,
    selector: usize,
    defined: std::collections::BTreeSet<MirFixedSelectorDestination>,
    available_scalars: std::collections::BTreeSet<MirLocalId>,
    sequences: std::collections::BTreeSet<MirSequenceLocalId>,
}

impl<'a> MirFixedPreparationState<'a> {
    pub fn advance(
        live: Option<Self>,
        preparation: &'a MirFixedPreparation,
        scalar_receipts: &std::collections::BTreeSet<MirLocalId>,
    ) -> Result<Self, String> {
        preparation.validate()?;
        if matches!(preparation.phase, MirFixedPreparationPhase::CaptureReceiver) {
            if live.is_some() {
                return Err("fixed receiver capture overwrites a live destination".into());
            }
            return Ok(Self {
                assignment: &preparation.assignment,
                path_step: 0,
                index_started: false,
                selector: 0,
                defined: std::collections::BTreeSet::new(),
                available_scalars: std::collections::BTreeSet::new(),
                sequences: std::collections::BTreeSet::new(),
            });
        }
        let mut state = live.ok_or("fixed preparation phase has no captured receiver")?;
        if state.assignment != &preparation.assignment {
            return Err("fixed preparation phase changes its source contract".into());
        }
        let mut unavailable = false;
        preparation.visit_external_operands(|operand| {
            if let MirOperand::Local(local) = operand {
                unavailable |=
                    scalar_receipts.contains(local) && !state.available_scalars.contains(local);
            }
        });
        if unavailable {
            return Err(
                "fixed selector reads a scalar receipt unavailable on this control-flow path"
                    .into(),
            );
        }
        let place = state
            .assignment
            .path_step(state.path_step)
            .ok_or("fixed preparation continues after its final path step")?;
        match &preparation.phase {
            MirFixedPreparationPhase::BeginIndex { path_step } => {
                if *path_step != state.path_step
                    || state.index_started
                    || !matches!(place, MirPlace::Index(_, _))
                {
                    return Err("fixed index context begins outside its ordered path step".into());
                }
                state.index_started = true;
            }
            MirFixedPreparationPhase::EvaluateSelector {
                path_step,
                component,
                destination,
                ..
            } => {
                state.skip_colons();
                let expected_component = match place {
                    MirPlace::Index(_, _) if state.index_started => Some(state.selector),
                    MirPlace::DynamicMember(_, _)
                        if !state.index_started && state.selector == 0 =>
                    {
                        None
                    }
                    _ => {
                        return Err(
                            "fixed selector executes outside its prepared path context".into()
                        )
                    }
                };
                if *path_step != state.path_step || *component != expected_component {
                    return Err("fixed selectors execute out of source order".into());
                }
                let expected = selector_destination(place, state.selector)?
                    .ok_or("fixed selector does not identify a value component")?;
                // Intermediate Await-prefix results may use other fresh slots,
                // but may not overwrite a different final selector receipt.
                for step in 0..state.assignment.path_len() {
                    let source = state.assignment.path_step(step).expect("bounded path step");
                    for component in 0..selector_count(source) {
                        if selector_destination(source, component)? == Some(*destination)
                            && (step != state.path_step || component != state.selector)
                        {
                            return Err("fixed selector overwrites another reserved receipt".into());
                        }
                    }
                }
                if *destination == expected {
                    state.selector += 1;
                }
                let mut missing = false;
                preparation.visit_sequence_locals(|sequence| {
                    missing |= !state.sequences.remove(sequence);
                });
                if missing {
                    return Err(
                        "fixed selector consumes a CSL not published by its source occurrence"
                            .into(),
                    );
                }
                if !state.defined.insert(*destination) {
                    return Err(
                        "fixed selector overwrites an already published frame receipt".into(),
                    );
                }
                if let MirFixedSelectorDestination::Sequence(sequence) = destination {
                    state.sequences.insert(*sequence);
                } else if let MirFixedSelectorDestination::Scalar(local) = destination {
                    state.available_scalars.insert(*local);
                }
            }
            MirFixedPreparationPhase::FinishStep { path_step } => {
                state.skip_colons();
                if *path_step != state.path_step
                    || state.selector != selector_count(place)
                    || (matches!(place, MirPlace::Index(_, _)) != state.index_started)
                {
                    return Err("fixed path step finishes before its ordered selectors".into());
                }
                let mut missing = false;
                preparation.visit_sequence_locals(|sequence| {
                    missing |= !state.sequences.remove(sequence);
                });
                if missing {
                    return Err(
                        "fixed path consumes a CSL without its captured source receipt".into(),
                    );
                }
                state.path_step += 1;
                state.index_started = false;
                state.selector = 0;
            }
            MirFixedPreparationPhase::CaptureReceiver => unreachable!("capture handled first"),
        }
        Ok(state)
    }

    /// Join only the same live source occurrence at the same path cursor.
    /// Scalar prefix temporaries may be branch-local: reads require definite
    /// availability, while freshness remembers publications on either path.
    /// CSL receipts are linear owners and must agree exactly, not intersect.
    pub fn join(left: Option<Self>, right: Option<Self>) -> Result<Option<Self>, String> {
        match (left, right) {
            (None, None) => Ok(None),
            (Some(mut left), Some(right))
                if left.assignment == right.assignment
                    && left.path_step == right.path_step
                    && left.index_started == right.index_started
                    && left.selector == right.selector
                    && left.sequences == right.sequences =>
            {
                left.defined.extend(right.defined);
                left.available_scalars
                    .retain(|local| right.available_scalars.contains(local));
                Ok(Some(left))
            }
            _ => Err("fixed destination lifetime disagrees at a control-flow merge".into()),
        }
    }

    fn skip_colons(&mut self) {
        if let Some(MirPlace::Index(_, indexing)) = self.assignment.path_step(self.path_step) {
            while matches!(
                indexing.components.get(self.selector),
                Some(crate::MirIndexComponent::Colon)
            ) {
                self.selector += 1;
            }
        }
    }

    pub fn consume(self, assignment: &MirPreparedAssignment) -> Result<(), String> {
        if self.assignment != assignment
            || self.path_step != assignment.path_len()
            || self.index_started
            || self.selector != 0
            || !self.sequences.is_empty()
        {
            return Err("fixed assignment consumes without its exact completed destination".into());
        }
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        self.path_step == self.assignment.path_len() && !self.index_started && self.selector == 0
    }
}

fn selector_count(place: &MirPlace) -> usize {
    match place {
        MirPlace::Index(_, indexing) => indexing.components.len(),
        MirPlace::DynamicMember(_, _) => 1,
        _ => 0,
    }
}

fn selector_destination(
    place: &MirPlace,
    component: usize,
) -> Result<Option<MirFixedSelectorDestination>, String> {
    match place {
        MirPlace::DynamicMember(_, crate::MirOperand::Local(local)) if component == 0 => {
            Ok(Some(MirFixedSelectorDestination::Scalar(*local)))
        }
        MirPlace::Index(_, indexing) => match indexing.components.get(component) {
            Some(crate::MirIndexComponent::Colon) => Ok(None),
            Some(crate::MirIndexComponent::Expr(crate::MirOperand::Local(local))) => {
                Ok(Some(MirFixedSelectorDestination::Scalar(*local)))
            }
            Some(crate::MirIndexComponent::Expansion(region)) => match &region.argument {
                MirCallArg::CapturedSequence(sequence) if region.steps.is_empty() => {
                    Ok(Some(MirFixedSelectorDestination::Sequence(*sequence)))
                }
                _ => Err("fixed expansion descriptor is not a captured receipt".into()),
            },
            _ => Err("fixed selector descriptor is not a reserved frame slot".into()),
        },
        MirPlace::Member(_, _) => Ok(None),
        _ => Err("fixed selector descriptor has no canonical destination".into()),
    }
}

impl MirFixedPreparation {
    pub fn output(&self) -> Option<MirFixedSelectorDestination> {
        match self.phase {
            MirFixedPreparationPhase::EvaluateSelector { destination, .. } => Some(destination),
            _ => None,
        }
    }

    /// Only this phase's executable work, never the retained source contract.
    pub fn selector_region(&self) -> Option<&MirCallArgumentRegion> {
        match &self.phase {
            MirFixedPreparationPhase::EvaluateSelector { region, .. } => Some(region),
            _ => None,
        }
    }

    pub fn steps(&self) -> &[crate::MirExpressionStep] {
        self.selector_region().map_or(&[], |region| &region.steps)
    }

    pub fn visit_operands(&self, mut visitor: impl FnMut(&crate::MirOperand)) {
        match &self.phase {
            MirFixedPreparationPhase::CaptureReceiver => {
                if let Some(root) = self.assignment.mutation.place.root_local() {
                    visitor(&crate::MirOperand::Local(root));
                }
            }
            MirFixedPreparationPhase::EvaluateSelector { region, .. } => {
                region.visit_operands(&mut visitor);
            }
            MirFixedPreparationPhase::FinishStep { path_step } => {
                if let Some(place) = self.assignment.path_step(*path_step) {
                    match place {
                        MirPlace::DynamicMember(_, member) => visitor(member),
                        MirPlace::Index(_, indexing) => indexing.visit_operands(&mut visitor),
                        _ => {}
                    }
                }
            }
            MirFixedPreparationPhase::BeginIndex { .. } => {}
        }
    }

    pub fn visit_external_operands(&self, mut visitor: impl FnMut(&crate::MirOperand)) {
        if let Some(region) = self.selector_region() {
            region.visit_outer_operands(&mut visitor);
        } else {
            self.visit_operands(visitor);
        }
    }

    /// Rewriters must update the retained descriptor as well as executable
    /// regions so all phases keep exactly the same typed source contract.
    pub fn visit_operands_mut(&mut self, mut visitor: impl FnMut(&mut crate::MirOperand)) {
        self.assignment.visit_operands_mut(&mut visitor);
        if let MirFixedPreparationPhase::EvaluateSelector { region, .. } = &mut self.phase {
            region.visit_operands_mut(&mut visitor);
        }
    }

    pub(crate) fn visit_direct_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(crate::MirExpressionRegionRef<'_>),
    ) {
        if let Some(region) = self.selector_region() {
            region.visit_expression_regions(visitor);
        }
    }

    pub fn visit_expression_regions(
        &self,
        mut visitor: impl FnMut(crate::MirExpressionRegionRef<'_>),
    ) {
        self.visit_direct_expression_regions_dyn(&mut visitor);
    }

    pub fn visit_sequence_locals(&self, mut visitor: impl FnMut(&MirSequenceLocalId)) {
        match &self.phase {
            MirFixedPreparationPhase::EvaluateSelector { region, .. } => {
                region.visit_sequence_locals(&mut visitor);
            }
            MirFixedPreparationPhase::FinishStep { path_step } => {
                if let Some(MirPlace::Index(_, indexing)) = self.assignment.path_step(*path_step) {
                    indexing.visit_sequence_locals(&mut visitor);
                }
            }
            _ => {}
        }
    }

    pub(crate) fn visit_preparation_steps_dyn(
        &self,
        visitor: &mut dyn FnMut(&[crate::MirExpressionStep]),
    ) {
        if let Some(region) = self.selector_region() {
            region.visit_direct_preparation_steps(visitor);
        }
    }
}
