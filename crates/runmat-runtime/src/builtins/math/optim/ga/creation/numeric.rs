//! One numeric creation generation. A complete double initial population
//! shares its original backing; only a real modification requests COW.
use super::super::{common, invalid, random, ScalarCooperation, POLICY};
use crate::BuiltinResult;
use common::constrained_numeric::{self, Geometry, Numeric};
use common::population::Population;
use runmat_value::{NumericDType, Tensor, Value};

pub(in super::super) struct Draft {
    population: Population,
    supplied: usize,
}
impl Draft {
    pub fn members(&self) -> usize {
        self.population.members
    }
    pub fn variables(&self) -> usize {
        self.population.variables
    }
    pub async fn new(
        initial: Option<Value>,
        members: usize,
        variables: usize,
        bitstring: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if members == 0 || variables == 0 {
            return Err(invalid(
                "creation requires positive population and variable counts",
            ));
        }
        let length = members
            .checked_mul(variables)
            .ok_or_else(|| invalid("initial population extent overflow"))?;
        let initial = match initial {
            Some(value) if !super::super::inputs::raw::unspecified(&value) => Some(
                constrained_numeric::prepare(
                    value,
                    "InitialPopulationMatrix",
                    Geometry::MatrixAny,
                    &super::super::numeric::INPUT_POLICY,
                    cooperation,
                )
                .await?
                .ok_or_else(|| invalid("InitialPopulationMatrix must be a real matrix"))?,
            ),
            _ => None,
        };
        let (supplied, values) = if let Some(initial) = initial {
            let shape = constrained_numeric::shape(&initial)
                .ok_or_else(|| invalid("InitialPopulationMatrix must be a real matrix"))?;
            if shape.len() != 2 || shape[1] != variables || shape[0] > members {
                return Err(invalid("InitialPopulationMatrix must have nvars columns and at most PopulationSize rows"));
            }
            let supplied = shape[0];
            if let Value::Tensor(tensor) = &initial {
                if supplied == members && tensor.numeric_dtype() == NumericDType::F64 {
                    for &value in tensor.as_f64_slice().expect("double initial population") {
                        admit_coordinate(value, bitstring)?;
                        cooperation.completed(1).await?;
                    }
                    let Value::Tensor(tensor) = initial else {
                        unreachable!("checked tensor");
                    };
                    return Ok(Self {
                        population: Population::new(POLICY, tensor, members, variables)?,
                        supplied,
                    });
                }
            }
            let Numeric::Real { values, .. } = constrained_numeric::materialize_raw(
                initial,
                "InitialPopulationMatrix",
                &super::super::numeric::INPUT_POLICY,
                cooperation,
            )
            .await?
            else {
                return Err(invalid("InitialPopulationMatrix must be a real matrix"));
            };
            for &value in &values {
                admit_coordinate(value, bitstring)?;
                cooperation.completed(1).await?;
            }
            (supplied, values)
        } else {
            (0, Vec::new())
        };
        if supplied == members {
            return Ok(Self {
                population: Population::new(
                    POLICY,
                    Tensor::new(values, vec![members, variables]).map_err(invalid)?,
                    members,
                    variables,
                )?,
                supplied,
            });
        }
        let mut data = POLICY.capacity(length, cooperation.runtime())?;
        for _ in 0..length {
            data.push(0.);
            cooperation.completed(1).await?;
        }
        for column in 0..variables {
            for row in 0..supplied {
                data[row + members * column] = values[row + supplied * column];
                cooperation.completed(1).await?;
            }
        }
        Ok(Self {
            population: Population::new(
                POLICY,
                Tensor::new(data, vec![members, variables]).map_err(invalid)?,
                members,
                variables,
            )?,
            supplied,
        })
    }
    pub fn missing(&self) -> std::ops::Range<usize> {
        self.supplied..self.members()
    }
    pub(super) fn set_supplied(&mut self, supplied: usize) -> BuiltinResult<()> {
        if supplied > self.members() {
            return Err(invalid("supplied population prefix exceeds generation"));
        }
        self.supplied = supplied;
        Ok(())
    }
    pub async fn set(
        &mut self,
        row: usize,
        point: &[f64],
        bitstring: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if row >= self.members() || point.len() != self.variables() {
            return Err(invalid("creation point extent differs from population"));
        }
        let mut changed = false;
        for (column, &value) in point.iter().enumerate() {
            admit_coordinate(value, bitstring)?;
            changed |= self.population.get(row, column).to_bits() != value.to_bits();
            cooperation.completed(1).await?;
        }
        if !changed {
            return Ok(false);
        }
        self.population.writable(cooperation).await?;
        let members = self.members();
        let values = self
            .population
            .positions
            .try_as_f64_slice_mut_unique()
            .expect("writable generation");
        for (column, &value) in point.iter().enumerate() {
            values[row + members * column] = value;
            cooperation.completed(1).await?;
        }
        Ok(true)
    }
    pub async fn row(
        &self,
        row: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<f64>> {
        if row >= self.members() {
            return Err(invalid("creation row is outside population"));
        }
        let mut result = POLICY.capacity(self.variables(), cooperation.runtime())?;
        for column in 0..self.variables() {
            result.push(self.population.get(row, column));
            cooperation.completed(1).await?;
        }
        Ok(result)
    }
    pub async fn fill_uniform(
        &mut self,
        ranges: &[(f64, f64)],
        bitstring: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if ranges.len() != self.variables() {
            return Err(invalid("creation range extent differs from nvars"));
        }
        for &(lower, upper) in ranges {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(invalid("creation ranges must be finite ordered intervals"));
            }
            cooperation.completed(1).await?;
        }
        for row in self.missing() {
            let mut point = random::uniform(self.variables(), cooperation).await?;
            for (column, value) in point.iter_mut().enumerate() {
                *value = if bitstring {
                    if *value >= 0.5 {
                        1.
                    } else {
                        0.
                    }
                } else {
                    let (lower, upper) = ranges[column];
                    super::super::operators::crossover::interpolate(lower, upper, *value)
                        .ok_or_else(|| invalid("creation coordinate cannot be represented"))?
                };
                cooperation.completed(1).await?;
            }
            self.set(row, &point, bitstring, cooperation).await?;
        }
        Ok(())
    }
    pub fn finish(self) -> Population {
        self.population
    }
    pub async fn fill_column_block(
        &mut self,
        column: usize,
        first: usize,
        values: impl ExactSizeIterator<Item = BuiltinResult<f64>>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let end = first
            .checked_add(values.len())
            .ok_or_else(|| invalid("creation column extent overflow"))?;
        if column >= self.variables() || first < self.supplied || end > self.members() {
            return Err(invalid(
                "creation column block is outside the missing population",
            ));
        }
        if first == end {
            return Ok(());
        }
        self.population.writable(cooperation).await?;
        let members = self.members();
        let storage = self
            .population
            .positions
            .try_as_f64_slice_mut_unique()
            .expect("writable generation");
        for (row, value) in (first..end).zip(values) {
            let value = value?;
            admit_coordinate(value, false)?;
            storage[row + members * column] = value;
            cooperation.completed(1).await?;
        }
        Ok(())
    }
}
fn admit_coordinate(value: f64, bitstring: bool) -> BuiltinResult<()> {
    if value.is_finite() && (!bitstring || value == 0. || value == 1.) {
        Ok(())
    } else {
        Err(invalid("population coordinates must be finite and bitstrings require exact zero/one coordinates"))
    }
}
