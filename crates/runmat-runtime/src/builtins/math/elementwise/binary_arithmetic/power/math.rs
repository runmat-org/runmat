pub(super) fn complex_pow_scalar(
    base_re: f64,
    base_im: f64,
    exp_re: f64,
    exp_im: f64,
) -> (f64, f64) {
    if base_re == 0.0 && base_im == 0.0 {
        if exp_re == 0.0 && exp_im == 0.0 {
            return (1.0, 0.0);
        }
        if exp_im == 0.0 {
            if exp_re > 0.0 {
                return (0.0, 0.0);
            }
            if exp_re < 0.0 {
                return (f64::INFINITY, 0.0);
            }
            return (f64::NAN, f64::NAN);
        }
        if exp_re > 0.0 {
            return (0.0, 0.0);
        }
        if exp_re < 0.0 {
            return (f64::INFINITY, f64::NAN);
        }
        return (f64::NAN, f64::NAN);
    }

    let r = base_re.hypot(base_im);
    if r == 0.0 {
        return (0.0, 0.0);
    }
    let theta = base_im.atan2(base_re);
    let ln_r = r.ln();
    let a = exp_re * ln_r - exp_im * theta;
    let b = exp_re * theta + exp_im * ln_r;
    let mag = a.exp();
    (mag * b.cos(), mag * b.sin())
}

pub(super) fn complex_pow_scalar_f32(
    base_re: f32,
    base_im: f32,
    exp_re: f32,
    exp_im: f32,
) -> (f32, f32) {
    if base_re == 0.0 && base_im == 0.0 {
        if exp_re == 0.0 && exp_im == 0.0 {
            return (1.0, 0.0);
        }
        if exp_im == 0.0 {
            if exp_re > 0.0 {
                return (0.0, 0.0);
            }
            if exp_re < 0.0 {
                return (f32::INFINITY, 0.0);
            }
            return (f32::NAN, f32::NAN);
        }
        if exp_re > 0.0 {
            return (0.0, 0.0);
        }
        if exp_re < 0.0 {
            return (f32::INFINITY, f32::NAN);
        }
        return (f32::NAN, f32::NAN);
    }

    let radius = base_re.hypot(base_im);
    if radius == 0.0 {
        return (0.0, 0.0);
    }
    let theta = base_im.atan2(base_re);
    let log_radius = radius.ln();
    let real = exp_re * log_radius - exp_im * theta;
    let imag = exp_re * theta + exp_im * log_radius;
    let magnitude = real.exp();
    (magnitude * imag.cos(), magnitude * imag.sin())
}
