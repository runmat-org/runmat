//! Host Adam arithmetic and output reconstruction.
use super::*;

pub(super) struct AdamUpdateEval {
    pub(super) parameters: Value,
    pub(super) average_grad: Value,
    pub(super) average_sq_grad: Value,
}

pub(super) fn evaluate_adamupdate(args: AdamUpdateArgs) -> BuiltinResult<AdamUpdateEval> {
    let count = args.parameters.data.len();
    let mut updated_parameters = Vec::with_capacity(count);
    let mut updated_average_grad = Vec::with_capacity(count);
    let mut updated_average_sq_grad = Vec::with_capacity(count);
    let iteration = args.iteration as f64;
    let grad_correction = 1.0 - args.gradient_decay_factor.powf(iteration);
    let sq_grad_correction = 1.0 - args.squared_gradient_decay_factor.powf(iteration);
    if grad_correction <= 0.0 || sq_grad_correction <= 0.0 {
        return Err(deep_learning_error(
            "adamupdate",
            "adamupdate: decay factors and iteration produced invalid bias correction",
        ));
    }
    for idx in 0..count {
        let grad = args.gradient.data[idx];
        let avg_grad = args.gradient_decay_factor * args.average_grad.data[idx]
            + (1.0 - args.gradient_decay_factor) * grad;
        let avg_sq_grad = args.squared_gradient_decay_factor * args.average_sq_grad.data[idx]
            + (1.0 - args.squared_gradient_decay_factor) * grad * grad;
        let corrected_grad = avg_grad / grad_correction;
        let corrected_sq_grad = avg_sq_grad / sq_grad_correction;
        let step = args.learn_rate * corrected_grad / (corrected_sq_grad.sqrt() + args.epsilon);
        let updated = args.parameters.data[idx] - step;
        if !updated.is_finite() || !avg_grad.is_finite() || !avg_sq_grad.is_finite() {
            return Err(deep_learning_error(
                "adamupdate",
                "adamupdate: update produced a non-finite value",
            ));
        }
        updated_parameters.push(updated);
        updated_average_grad.push(avg_grad);
        updated_average_sq_grad.push(avg_sq_grad);
    }
    Ok(AdamUpdateEval {
        parameters: args.parameters.materialize(updated_parameters)?,
        average_grad: args.average_grad.materialize(updated_average_grad)?,
        average_sq_grad: args.average_sq_grad.materialize(updated_average_sq_grad)?,
    })
}
