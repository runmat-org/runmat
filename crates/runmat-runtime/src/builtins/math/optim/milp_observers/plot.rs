//! The MILP helper reads bounds/time only; x and unused metadata stay untouched.
use super::*;
use runmat_value::StructValue;
#[cfg(feature = "plot-core")]
const KIND: MilpObserverKind = MilpObserverKind::PlotBounds;
pub(super) async fn update(
    values: &StructValue,
    state: OutputState,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = (values, state, cooperation);
        Err(crate::context::RuntimeCapabilityError::new(
            crate::context::RuntimeCapability::GraphicsPresentation,
            "optimplotmilp requires plot-core graphics support",
        )
        .into_runtime_error())
    }
    #[cfg(feature = "plot-core")]
    {
        let needed = ["time", "fval", "dualbound"].map(|name| {
            values
                .fields
                .get(name)
                .ok_or_else(|| invalid(KIND, format!("optimValues.{name} is required")))
        });
        let [time, primal, dual] = needed;
        let needed = [time?, primal?, dual?];
        for value in needed {
            numeric::preflight(KIND, value)?;
        }
        let mut read = [0.0; 3];
        for (slot, value) in read.iter_mut().zip(needed) {
            *slot = numeric::scalar(KIND, &numeric::gather(value, cooperation).await?)?;
        }
        if !read[0].is_finite() || read[0] < 0.0 {
            return Err(invalid(
                KIND,
                "optimValues.time must be finite and nonnegative",
            ));
        }
        crate::builtins::plotting::state::update_milp_plot(
            state,
            read[0],
            read[1],
            read[2],
            cooperation.runtime(),
        )
    }
}
