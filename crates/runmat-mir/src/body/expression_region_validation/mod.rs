use std::collections::{BTreeMap, BTreeSet};

use crate::{MirLocalId, MirLocalKind, MirOperand, MirStmtKind};

use super::super::MirBody;

mod traversal;

pub(super) fn validate(body: &MirBody) -> Result<(), String> {
    let local_kinds = body
        .locals
        .iter()
        .map(|local| (local.id, &local.kind))
        .collect::<BTreeMap<_, _>>();
    let mut region_definitions = BTreeMap::<MirLocalId, usize>::new();
    let mut region_sequence_definitions = BTreeMap::new();
    let mut all_sequence_uses = BTreeMap::new();
    let mut outer_sequence_definitions = BTreeSet::new();
    let mut invalid_region = None;

    for block in &body.blocks {
        for (statement_index, statement) in block.statements.iter().enumerate() {
            if let MirStmtKind::PlaceMutation(mutation) = &statement.kind {
                let Some(next) = block.statements.get(statement_index + 1) else {
                    return Err("place mutation annotation has no following assignment".to_owned());
                };
                if !matches!(&next.kind, MirStmtKind::Assign { place, .. } if place == &mutation.place)
                {
                    return Err(
                        "place mutation annotation must be followed by its exact assignment"
                            .to_owned(),
                    );
                }
            }
            if traversal::statement_outer_rvalue(statement)
                .is_some_and(crate::MirRvalue::contains_unscoped_end)
            {
                return Err(
                    "MIR end expression is not enclosed by a contextual index component".to_owned(),
                );
            }
            statement.visit_expression_regions(|region| {
                if invalid_region.is_none() {
                    invalid_region = region.validate().err();
                }
                for local in region.defined_locals() {
                    *region_definitions.entry(local).or_default() += 1;
                }
                for sequence in region.defined_sequences() {
                    *region_sequence_definitions
                        .entry(sequence)
                        .or_insert(0usize) += 1;
                }
                region.visit_sequence_locals(|sequence| {
                    *all_sequence_uses.entry(*sequence).or_insert(0usize) += 1;
                });
            });
            if let MirStmtKind::CaptureSequence { destination, .. } = statement.kind {
                outer_sequence_definitions.insert(destination);
            }
            traversal::visit_outer_statement_sequences(statement, &mut |sequence| {
                *all_sequence_uses.entry(sequence).or_insert(0usize) += 1;
            });
        }
        traversal::visit_terminator_regions(&block.terminator.kind, &mut |region| {
            if invalid_region.is_none() {
                invalid_region = region.validate().err();
            }
            for local in region.defined_locals() {
                *region_definitions.entry(local).or_default() += 1;
            }
        });
    }
    if let Some(error) = invalid_region {
        return Err(error);
    }

    validate_sequence_ownership(
        &region_sequence_definitions,
        &outer_sequence_definitions,
        &all_sequence_uses,
    )?;
    validate_local_ownership(&region_definitions, &local_kinds)?;
    validate_no_local_escape(body, &region_definitions)
}

fn validate_sequence_ownership(
    definitions: &BTreeMap<crate::MirSequenceLocalId, usize>,
    outer_definitions: &BTreeSet<crate::MirSequenceLocalId>,
    uses: &BTreeMap<crate::MirSequenceLocalId, usize>,
) -> Result<(), String> {
    for (&sequence, &count) in definitions {
        if count != 1 {
            return Err(format!(
                "contextual expression sequence {sequence:?} has {count} region definitions; expected exactly one"
            ));
        }
        if outer_definitions.contains(&sequence) {
            return Err(format!(
                "contextual expression sequence {sequence:?} is also defined outside its region"
            ));
        }
        let use_count = uses.get(&sequence).copied().unwrap_or(0);
        if use_count != 1 {
            return Err(format!(
                "contextual expression sequence {sequence:?} has {use_count} body uses; expected its single use inside the defining region"
            ));
        }
    }
    Ok(())
}

fn validate_local_ownership(
    definitions: &BTreeMap<MirLocalId, usize>,
    kinds: &BTreeMap<MirLocalId, &MirLocalKind>,
) -> Result<(), String> {
    for (&local, &count) in definitions {
        let kind = kinds
            .get(&local)
            .ok_or_else(|| format!("contextual expression defines undeclared local {local:?}"))?;
        if !matches!(kind, MirLocalKind::Temporary) {
            return Err(format!(
                "contextual expression local {local:?} must be a temporary"
            ));
        }
        if count != 1 {
            return Err(format!(
                "contextual expression local {local:?} has {count} region definitions; expected exactly one"
            ));
        }
    }
    Ok(())
}

fn validate_no_local_escape(
    body: &MirBody,
    definitions: &BTreeMap<MirLocalId, usize>,
) -> Result<(), String> {
    let region_locals = definitions.keys().copied().collect::<BTreeSet<_>>();
    for block in &body.blocks {
        for statement in &block.statements {
            let mut invalid_definition = None;
            traversal::visit_outer_statement_definitions(statement, &mut |local| {
                if region_locals.contains(&local) && invalid_definition.is_none() {
                    invalid_definition = Some(local);
                }
            });
            if let Some(local) = invalid_definition {
                return Err(format!(
                    "contextual expression local {local:?} is defined outside its region"
                ));
            }
            let mut escaped = None;
            statement.visit_outer_operands_dyn(&mut |operand| {
                if let MirOperand::Local(local) = operand {
                    if region_locals.contains(local) {
                        escaped = Some(*local);
                    }
                }
            });
            if let Some(local) = escaped {
                return Err(format!(
                    "contextual expression local {local:?} is used outside its region"
                ));
            }
        }
        if let Some((local, is_definition)) =
            traversal::terminator_region_local_escape(&block.terminator.kind, &region_locals)
        {
            let action = if is_definition { "defined" } else { "used" };
            return Err(format!(
                "contextual expression local {local:?} is {action} outside its region"
            ));
        }
    }
    Ok(())
}
