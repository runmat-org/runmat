//! Simultaneous numeric migration without a second population payload.
//! Dependencies are scheduled once in O(moves + members), then each column
//! needs only one scalar cycle temporary. Retained generations use normal COW.
use super::super::{common, invalid, ScalarCooperation, POLICY};
use super::migration::Copy;
use crate::BuiltinResult;

enum Step {
    Save(usize),
    Move(Copy),
    Restore(usize),
}
pub(in super::super) struct Schedule {
    members: usize,
    steps: Vec<Step>,
}
impl Schedule {
    pub async fn new(
        members: usize,
        copies: &[Copy],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let capacity = members
            .checked_mul(2)
            .ok_or_else(|| invalid("migration schedule extent overflow"))?;
        let mut sources = POLICY.capacity(members, cooperation.runtime())?;
        let mut uses = POLICY.capacity(members, cooperation.runtime())?;
        for _ in 0..members {
            sources.push(None);
            uses.push(0usize);
            cooperation.completed(1).await?;
        }
        for &copy in copies {
            if copy.target >= members || copy.source >= members || sources[copy.target].is_some() {
                return Err(invalid(
                    "migration plan must have unique in-range target rows",
                ));
            }
            if copy.target == copy.source {
                return Err(invalid("migration plan cannot copy a row onto itself"));
            }
            sources[copy.target] = Some(copy.source);
            uses[copy.source] = uses[copy.source]
                .checked_add(1)
                .ok_or_else(|| invalid("migration dependency overflow"))?;
            cooperation.completed(1).await?;
        }
        let mut queue = POLICY.capacity(members, cooperation.runtime())?;
        let mut steps = POLICY.capacity(capacity, cooperation.runtime())?;
        for target in 0..members {
            if sources[target].is_some() && uses[target] == 0 {
                queue.push(target);
            }
            cooperation.completed(1).await?;
        }
        let mut head = 0;
        while head < queue.len() {
            let target = queue[head];
            head += 1;
            let source = sources[target]
                .take()
                .expect("scheduled remaining dependency");
            steps.push(Step::Move(Copy { source, target }));
            uses[source] -= 1;
            if uses[source] == 0 && sources[source].is_some() {
                queue.push(source);
            }
            cooperation.completed(1).await?;
        }
        // Remaining components are disjoint directed cycles. Fan-out trees
        // have already copied their old sources before any cycle is rotated.
        for start in 0..members {
            if sources[start].is_some() {
                steps.push(Step::Save(start));
                let mut target = start;
                loop {
                    let source = sources[target].take().expect("remaining cycle edge");
                    if source == start {
                        steps.push(Step::Restore(target));
                        break;
                    }
                    steps.push(Step::Move(Copy { source, target }));
                    target = source;
                    cooperation.completed(1).await?;
                }
            }
            cooperation.completed(1).await?;
        }
        Ok(Self { members, steps })
    }
    pub async fn apply(
        &self,
        population: &mut common::population::Population,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if population.members != self.members {
            return Err(invalid(
                "migration schedule belongs to a different population extent",
            ));
        }
        if self.steps.is_empty() {
            return Ok(());
        }
        population.writable(cooperation).await?;
        let values = population
            .positions
            .try_as_f64_slice_mut_unique()
            .expect("writable population");
        for column in 0..population.variables {
            let offset = column * self.members;
            let mut saved = None;
            for step in &self.steps {
                match *step {
                    Step::Save(row) => saved = Some(values[offset + row]),
                    Step::Move(Copy { source, target }) => {
                        values[offset + target] = values[offset + source]
                    }
                    Step::Restore(row) => {
                        values[offset + row] = saved.take().expect("scheduled cycle source")
                    }
                }
                cooperation.completed(1).await?;
            }
            debug_assert!(saved.is_none());
        }
        Ok(())
    }
    pub async fn records<T: std::marker::Copy>(
        &self,
        records: &mut [T],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if records.len() != self.members {
            return Err(invalid("migration record extent changed"));
        }
        let mut saved = None;
        for step in &self.steps {
            match *step {
                Step::Save(row) => saved = Some(records[row]),
                Step::Move(Copy { source, target }) => records[target] = records[source],
                Step::Restore(row) => records[row] = saved.take().expect("scheduled cycle source"),
            }
            cooperation.completed(1).await?;
        }
        Ok(())
    }
    pub async fn tensor(
        &self,
        tensor: &mut runmat_value::Tensor,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if tensor.shape().len() != 2
            || tensor.shape()[0] != self.members
            || tensor.as_f64_slice().is_none()
        {
            return Err(invalid(
                "migration tensor requires its original double row extent",
            ));
        }
        if self.steps.is_empty() {
            return Ok(());
        }
        let columns = tensor.shape()[1];
        if tensor.try_as_f64_slice_mut_unique().is_none() {
            let mut values = POLICY.capacity(tensor.len(), cooperation.runtime())?;
            for chunk in tensor.as_f64_slice().expect("double rows").chunks(1024) {
                values.extend_from_slice(chunk);
                cooperation.completed(chunk.len() as u64).await?;
            }
            *tensor =
                runmat_value::Tensor::new(values, vec![self.members, columns]).map_err(invalid)?;
        }
        for column in tensor
            .try_as_f64_slice_mut_unique()
            .expect("writable double rows")
            .chunks_mut(self.members)
        {
            self.records(column, cooperation).await?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simultaneous_cycles_and_fanout_preserve_source_generation() {
        futures::executor::block_on(async {
            let mut cooperation =
                ScalarCooperation::for_builtin(super::super::super::IDENTITY, None);
            let mut population = common::population::Population::new(
                POLICY,
                runmat_value::Tensor::new(
                    vec![10., 20., 30., 40., 50., 100., 200., 300., 400., 500.],
                    vec![5, 2],
                )
                .unwrap(),
                5,
                2,
            )
            .unwrap();
            let old = population.value(); // Real retained predecessor exercises normal COW.
            let schedule = Schedule::new(
                5,
                &[
                    Copy {
                        source: 1,
                        target: 0,
                    },
                    Copy {
                        source: 0,
                        target: 1,
                    },
                    Copy {
                        source: 0,
                        target: 2,
                    },
                    Copy {
                        source: 2,
                        target: 3,
                    },
                ],
                &mut cooperation,
            )
            .await
            .unwrap();
            schedule
                .apply(&mut population, &mut cooperation)
                .await
                .unwrap();
            assert_eq!(
                population.positions.as_f64_slice().unwrap(),
                [20., 10., 10., 30., 50., 200., 100., 100., 300., 500.]
            );
            let runmat_value::Value::Tensor(old) = old else {
                panic!("numeric predecessor")
            };
            assert_eq!(
                old.as_f64_slice().unwrap(),
                [10., 20., 30., 40., 50., 100., 200., 300., 400., 500.]
            );
            assert!(Schedule::new(
                5,
                &[
                    Copy {
                        source: 1,
                        target: 0
                    },
                    Copy {
                        source: 2,
                        target: 0
                    }
                ],
                &mut cooperation
            )
            .await
            .is_err());
        });
    }
}
