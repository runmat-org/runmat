//! Selected auxiliary search borrows the current accepted merit generation.
use super::*;
impl Engine {
    pub(super) async fn run_latin(
        &mut self,
        geometry: &Linear,
        controls: &Controls,
        points: usize,
        deadline: &Deadline,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<poll::Outcome> {
        let window = self
            .search_window
            .as_ref()
            .ok_or_else(|| invalid("selected search has no original-bound window"))?;
        let augmented = &self.augmented;
        super::super::search::run(
            &self.point,
            geometry,
            window,
            points,
            controls.complete_search,
            controls.constraint_tolerance,
            deadline,
            async |point, cooperation| match augmented {
                Some(augmented) => augmented.merit(point, cooperation).await,
                None => Ok(Some(point.f)),
            },
            evaluator,
        )
        .await
    }
}

impl Engine {
    pub(super) async fn run_custom(
        &mut self,
        geometry: &Linear,
        options: &super::super::options::Options,
        input: &super::super::input::Input<'_>,
        deadline: &Deadline,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<poll::Outcome> {
        let augmented = &self.augmented;
        super::super::search::run_custom(
            &self.point,
            geometry,
            options,
            input,
            self.iterations,
            self.mesh.size(),
            self.method.name(),
            &self.scaling,
            deadline,
            async |point, cooperation| match augmented {
                Some(augmented) => augmented.merit(point, cooperation).await,
                None => Ok(Some(point.f)),
            },
            evaluator,
        )
        .await
    }
}
