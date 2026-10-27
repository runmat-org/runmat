//! Complete finite measurements join the single retained callback history.
use super::super::search::rbf::Coordinates;
use super::{Evaluator, Measurement};
use crate::BuiltinResult;
use runmat_value::{ExactArrayStorage, Value};

impl Evaluator<'_> {
    pub(super) async fn complete_batch_measurement(
        &mut self,
        backing: &runmat_value::Tensor,
        axis: super::PointAxis,
        point: usize,
        points: usize,
        dimensions: usize,
        ordinal: usize,
        f: f64,
        inequality: Vec<f64>,
        equality: Vec<f64>,
    ) -> BuiltinResult<Measurement> {
        let coordinates = self
            .state
            .history
            .as_ref()
            .map(|_| Coordinates::batch(backing, axis, point, points, dimensions))
            .transpose()?;
        self.complete_record(coordinates, ordinal, f, inequality, equality)
            .await
    }

    pub(super) async fn complete_measurement(
        &mut self,
        argument: &Value,
        ordinal: usize,
        dimensions: usize,
        f: f64,
        inequality: Vec<f64>,
        equality: Vec<f64>,
    ) -> BuiltinResult<Measurement> {
        let coordinates = self
            .state
            .history
            .as_ref()
            .map(|_| Coordinates::point(argument, dimensions))
            .transpose()?;
        self.complete_record(coordinates, ordinal, f, inequality, equality)
            .await
    }

    async fn complete_record(
        &mut self,
        coordinates: Option<Coordinates>,
        ordinal: usize,
        f: f64,
        inequality: Vec<f64>,
        equality: Vec<f64>,
    ) -> BuiltinResult<Measurement> {
        let Some(history) = &mut self.state.history else {
            return Ok(Measurement {
                f,
                inequality: inequality.into(),
                equality: equality.into(),
            });
        };
        let coordinates = coordinates.expect("retained-history coordinate admitted");
        // Only the retained-history mode shares these existing payloads.
        // Ordinary scalar execution keeps its original Vec storage.
        let inequality = ExactArrayStorage::shared_unbounded(inequality);
        let equality = ExactArrayStorage::shared_unbounded(equality);
        history
            .record(
                ordinal,
                coordinates,
                f,
                inequality.clone(),
                equality.clone(),
                self.cooperation,
            )
            .await?;
        Ok(Measurement {
            f,
            inequality,
            equality,
        })
    }
}
