use super::*;

pub(super) struct OptimizerState {
    solver: Solver,
    slots: BTreeMap<(usize, &'static str), OptimizerSlot>,
}

impl OptimizerState {
    pub(super) fn new(solver: Solver) -> Self {
        Self {
            solver,
            slots: BTreeMap::new(),
        }
    }

    pub(super) fn update_parameter(
        &mut self,
        values: &mut [f64],
        grad: &[f64],
        config: &TrainingConfig,
        iteration: usize,
        layer_index: usize,
        parameter: &'static str,
        function: &'static str,
    ) -> BuiltinResult<()> {
        if values.len() != grad.len() {
            return Err(deep_learning_error(
                function,
                format!("{function}: optimizer gradient shape does not match {parameter}"),
            ));
        }
        let slot = self
            .slots
            .entry((layer_index, parameter))
            .or_insert_with(|| OptimizerSlot::new(values.len()));
        match self.solver {
            Solver::Sgdm => {
                for idx in 0..values.len() {
                    slot.m[idx] = config.momentum * slot.m[idx] - config.learn_rate * grad[idx];
                    values[idx] += slot.m[idx];
                }
            }
            Solver::Adam => {
                let t = iteration as f64;
                let beta1 = config.gradient_decay_factor;
                let beta2 = config.squared_gradient_decay_factor;
                let m_correction = 1.0 - beta1.powf(t);
                let v_correction = 1.0 - beta2.powf(t);
                if m_correction <= 0.0 || v_correction <= 0.0 {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: Adam bias correction is invalid"),
                    ));
                }
                for idx in 0..values.len() {
                    slot.m[idx] = beta1 * slot.m[idx] + (1.0 - beta1) * grad[idx];
                    slot.v[idx] = beta2 * slot.v[idx] + (1.0 - beta2) * grad[idx] * grad[idx];
                    let m_hat = slot.m[idx] / m_correction;
                    let v_hat = slot.v[idx] / v_correction;
                    values[idx] -= config.learn_rate * m_hat / (v_hat.sqrt() + config.epsilon);
                }
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(deep_learning_error(
                function,
                format!("{function}: optimizer produced non-finite {parameter} values"),
            ));
        }
        Ok(())
    }
}

struct OptimizerSlot {
    m: Vec<f64>,
    v: Vec<f64>,
}

impl OptimizerSlot {
    fn new(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }
}
