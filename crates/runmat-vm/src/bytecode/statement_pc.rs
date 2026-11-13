//! Exact MIR statement boundaries in one function-local bytecode stream.
//!
//! The compiler's resume points are authoritative: position `n` precedes
//! statement `n`, and position `n + 1` follows it. Terminators are emitted
//! after the block-end point and therefore cannot enter a statement range.

use std::collections::BTreeMap;

use runmat_types::ProgramPointId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StatementPcRange {
    pub statement: ProgramPointId,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementPcProjectionError {
    MissingBlockEntry(ProgramPointId),
    MissingStatementBoundary(ProgramPointId),
    PcOutsideInstructions(ProgramPointId),
    NonMonotonicPc(ProgramPointId),
    MixedFunctions(ProgramPointId),
}

impl std::fmt::Display for StatementPcProjectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid MIR statement PC authority: {self:?}")
    }
}

pub(super) fn project_statement_pcs(
    resume_points: &BTreeMap<ProgramPointId, usize>,
    instruction_count: usize,
) -> Result<Vec<StatementPcRange>, StatementPcProjectionError> {
    let mut ranges = Vec::with_capacity(resume_points.len());
    let mut previous: Option<(ProgramPointId, usize)> = None;
    let mut selected_function = None;

    for (&point, &pc) in resume_points {
        if selected_function.is_some_and(|function| function != point.function) {
            return Err(StatementPcProjectionError::MixedFunctions(point));
        }
        selected_function = Some(point.function);
        if pc > instruction_count {
            return Err(StatementPcProjectionError::PcOutsideInstructions(point));
        }
        match previous {
            Some((start_point, start_pc)) if start_point.block == point.block => {
                if start_point.position.checked_add(1) != Some(point.position) {
                    return Err(StatementPcProjectionError::MissingStatementBoundary(point));
                }
                if start_pc > pc {
                    return Err(StatementPcProjectionError::NonMonotonicPc(point));
                }
                ranges.push(StatementPcRange {
                    statement: start_point,
                    start: start_pc,
                    end: pc,
                });
            }
            Some((_, start_pc)) if start_pc > pc => {
                return Err(StatementPcProjectionError::NonMonotonicPc(point));
            }
            _ if point.position != 0 => {
                return Err(StatementPcProjectionError::MissingBlockEntry(point));
            }
            _ => {}
        }
        previous = Some((point, pc));
    }
    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::ProgramFunctionId;

    fn point(function: u32, block: u32, position: u32) -> ProgramPointId {
        ProgramPointId {
            function: ProgramFunctionId(function),
            block,
            position,
        }
    }

    #[test]
    fn preserves_empty_statements_and_excludes_block_terminators() {
        let points = BTreeMap::from([
            (point(2, 0, 0), 3),
            (point(2, 0, 1), 3),
            (point(2, 0, 2), 7),
            (point(2, 1, 0), 9),
            (point(2, 1, 1), 11),
        ]);
        assert_eq!(
            project_statement_pcs(&points, 12).unwrap(),
            vec![
                StatementPcRange {
                    statement: point(2, 0, 0),
                    start: 3,
                    end: 3,
                },
                StatementPcRange {
                    statement: point(2, 0, 1),
                    start: 3,
                    end: 7,
                },
                StatementPcRange {
                    statement: point(2, 1, 0),
                    start: 9,
                    end: 11,
                },
            ]
        );
        let ranges = project_statement_pcs(&points, 12).unwrap();
        assert_eq!(ranges[0].start, ranges[0].end);
    }

    #[test]
    fn rejects_incomplete_or_foreign_resume_authority() {
        for (points, expected) in [
            (
                BTreeMap::from([(point(2, 0, 1), 0)]),
                StatementPcProjectionError::MissingBlockEntry(point(2, 0, 1)),
            ),
            (
                BTreeMap::from([(point(2, 0, 0), 0), (point(2, 0, 2), 1)]),
                StatementPcProjectionError::MissingStatementBoundary(point(2, 0, 2)),
            ),
            (
                BTreeMap::from([(point(2, 0, 0), 2), (point(2, 0, 1), 1)]),
                StatementPcProjectionError::NonMonotonicPc(point(2, 0, 1)),
            ),
            (
                BTreeMap::from([(point(2, 0, 0), 0), (point(3, 0, 0), 1)]),
                StatementPcProjectionError::MixedFunctions(point(3, 0, 0)),
            ),
            (
                BTreeMap::from([(point(2, 0, 0), 2), (point(2, 1, 0), 1)]),
                StatementPcProjectionError::NonMonotonicPc(point(2, 1, 0)),
            ),
            (
                BTreeMap::from([(point(2, 0, 0), 4)]),
                StatementPcProjectionError::PcOutsideInstructions(point(2, 0, 0)),
            ),
        ] {
            assert_eq!(project_statement_pcs(&points, 3), Err(expected));
        }
    }
}
