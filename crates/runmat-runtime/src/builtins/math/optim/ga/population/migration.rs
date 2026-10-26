//! A simultaneous source-row plan over the actual pre-migration generation.
//! Worker evaluation topology is separate; this does not claim worker-resident
//! island evolution or create another worker pool.
use super::super::operators::ranking::RankOrder;
use super::super::{invalid, ScalarCooperation, POLICY};
use super::layout::Layout;
use crate::BuiltinResult;
use runmat_builtins::OptimGeneticMigration;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) struct Copy {
    pub source: usize,
    pub target: usize,
}
pub(in super::super) async fn plan(
    layout: &Layout,
    order: &RankOrder,
    fraction: f64,
    direction: OptimGeneticMigration,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<Copy>> {
    if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) || order.len() != layout.total() {
        return Err(invalid("invalid migration fraction or rank extent"));
    }
    if layout.groups() == 1 || fraction == 0.0 {
        return Ok(Vec::new());
    }
    let mut groups = POLICY.capacity(layout.groups(), cooperation.runtime())?;
    for group in 0..layout.groups() {
        groups.push(POLICY.capacity(layout.range(group).len(), cooperation.runtime())?);
        cooperation.completed(1).await?;
    }
    for &member in order.as_slice() {
        let group = layout
            .group_for(member)
            .ok_or_else(|| invalid("migration rank contains a nonexistent member"))?;
        groups[group].push(member);
        cooperation.completed(1).await?;
    }
    if groups
        .iter()
        .enumerate()
        .any(|(group, rows)| rows.len() != layout.range(group).len())
    {
        return Err(invalid("migration ranks do not cover each subpopulation"));
    }
    let mut copies = POLICY.capacity(layout.total(), cooperation.runtime())?;
    for target in 0..layout.groups() {
        let previous = if target == 0 {
            layout.groups() - 1
        } else {
            target - 1
        };
        let next = if target + 1 == layout.groups() {
            0
        } else {
            target + 1
        };
        let incoming = [
            Some(previous),
            (direction == OptimGeneticMigration::Both && next != previous).then_some(next),
        ];
        let mut quotas = [0usize; 2];
        for (slot, source) in incoming.iter().enumerate() {
            if let Some(source) = source {
                let smaller = groups[*source].len().min(groups[target].len());
                quotas[slot] = ((fraction * smaller as f64).floor() as usize).min(smaller);
            }
        }
        let mut replaced = 0;
        // Round-robin incoming directions prevents the second neighbour from
        // silently erasing the first. Total replacements cannot exceed the
        // destination's actual fixed row count, even for fraction=1.
        for rank in 0..quotas[0].max(quotas[1]) {
            for (slot, source) in incoming.iter().enumerate() {
                if rank < quotas[slot] && replaced < groups[target].len() {
                    let source = source.expect("positive incoming quota");
                    copies.push(Copy {
                        source: groups[source][rank],
                        target: groups[target][groups[target].len() - 1 - replaced],
                    });
                    replaced += 1;
                    cooperation.completed(1).await?;
                }
            }
        }
    }
    Ok(copies)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_forward_and_both_direction_plans_wrap_without_overwriting_two_neighbours() {
        futures::executor::block_on(async {
            let mut cooperation =
                ScalarCooperation::for_builtin(super::super::super::IDENTITY, None);
            let layout = Layout::new(&[4, 4, 4], &mut cooperation).await.unwrap();
            let scores = (0..12)
                .map(|value| super::super::super::operators::ranking::Score::Defined(value as f64))
                .collect::<Vec<_>>();
            let order = super::super::super::operators::ranking::order(&scores, &mut cooperation)
                .await
                .unwrap();
            let forward = plan(
                &layout,
                &order,
                0.25,
                OptimGeneticMigration::Forward,
                &mut cooperation,
            )
            .await
            .unwrap();
            assert_eq!(
                forward,
                [
                    Copy {
                        source: 8,
                        target: 3
                    },
                    Copy {
                        source: 0,
                        target: 7
                    },
                    Copy {
                        source: 4,
                        target: 11
                    }
                ]
            );
            let both = plan(
                &layout,
                &order,
                0.25,
                OptimGeneticMigration::Both,
                &mut cooperation,
            )
            .await
            .unwrap();
            assert_eq!(
                both,
                [
                    Copy {
                        source: 8,
                        target: 3
                    },
                    Copy {
                        source: 4,
                        target: 2
                    },
                    Copy {
                        source: 0,
                        target: 7
                    },
                    Copy {
                        source: 8,
                        target: 6
                    },
                    Copy {
                        source: 4,
                        target: 11
                    },
                    Copy {
                        source: 0,
                        target: 10
                    }
                ]
            );
            let full = plan(
                &layout,
                &order,
                1.0,
                OptimGeneticMigration::Both,
                &mut cooperation,
            )
            .await
            .unwrap();
            assert_eq!(full.len(), 12);
            assert!(full
                .chunks(4)
                .all(|rows| rows[0].source / 4 != rows[1].source / 4));
        });
    }
}
