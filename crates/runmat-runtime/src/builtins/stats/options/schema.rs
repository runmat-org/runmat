use super::*;

const COMMON_STATFUNS: [&str; 48] = [
    "bootci",
    "bootstrp",
    "crossval",
    "factoran",
    "fitglm",
    "fitlm",
    "fitlme",
    "fitnlm",
    "fitrgp",
    "gamfit",
    "gevfit",
    "glmfit",
    "gmdistribution",
    "gpfit",
    "kmeans",
    "kmedoids",
    "lasso",
    "lassoglm",
    "lognfit",
    "mlecustom",
    "mlecov",
    "mvncdf",
    "mvtcdf",
    "nbinfit",
    "nlinfit",
    "nnmf",
    "normfit",
    "parallel",
    "pca",
    "plsregress",
    "ppca",
    "rocmetrics",
    "sequentialfs",
    "tsne",
    "wblfit",
    "copulafit",
    "coxphfit",
    "evfit",
    "fitcox",
    "fitglme",
    "fitlmematrix",
    "mdscale",
    "nlmefitsa",
    "treebagger",
    "candexch",
    "cordexch",
    "daugment",
    "dcovary",
];

pub(super) fn merge_old_into_new(
    oldopts: &mut StructValue,
    newopts: &StructValue,
) -> BuiltinResult<()> {
    let canonical_new = canonicalize_options(newopts)?;
    for field in StatOptionField::ALL {
        if let Some(new_value) = lookup_struct_field(&canonical_new, field.name()) {
            if !is_empty_value(new_value) {
                oldopts.insert(field.name(), new_value.clone());
            }
        }
    }
    Ok(())
}

pub(super) fn canonicalize_options(options: &StructValue) -> BuiltinResult<StructValue> {
    let mut out = empty_options();
    for (name, value) in &options.fields {
        let canonical = canonical_option_name(name)?;
        out.insert(canonical.name(), validate_option_value(canonical, value)?);
    }
    Ok(out)
}

pub(super) fn defaults_for_statfun(statfun: &str) -> BuiltinResult<StructValue> {
    let key = statfun.to_ascii_lowercase();
    if !COMMON_STATFUNS.contains(&key.as_str()) {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_STATFUN,
            format!("statset: unsupported statistics function '{statfun}'"),
        ));
    }
    let mut options = empty_options();
    match key.as_str() {
        "fitglm" | "fitlm" | "fitnlm" | "glmfit" | "lasso" | "lassoglm" | "nlinfit" | "normfit"
        | "wblfit" => {
            options.insert("Display", Value::from("off"));
            options.insert("MaxIter", Value::Num(100.0));
            options.insert("TolX", Value::Num(1.0e-6));
        }
        "factoran" => {
            options.insert("Display", Value::from("off"));
            options.insert("MaxIter", Value::Num(100.0));
            options.insert("TolX", Value::Num(1.0e-8));
        }
        "nbinfit" => {
            options.insert("Display", Value::from("off"));
            options.insert("MaxFunEvals", Value::Num(400.0));
            options.insert("MaxIter", Value::Num(200.0));
            options.insert("TolBnd", Value::Num(1.0e-6));
            options.insert("TolFun", Value::Num(1.0e-6));
            options.insert("TolX", Value::Num(1.0e-6));
        }
        "kmeans" | "tsne" | "pca" | "ppca" | "gmdistribution" | "kmedoids" => {
            options.insert("Display", Value::from("off"));
            options.insert("MaxIter", Value::Num(100.0));
        }
        "bootci" | "bootstrp" | "crossval" | "parallel" | "sequentialfs" => {
            options.insert("UseParallel", Value::Bool(false));
            options.insert("UseSubstreams", Value::Bool(false));
            options.insert("Streams", empty_cell());
        }
        _ => {}
    }
    Ok(options)
}

pub(super) fn empty_options() -> StructValue {
    let mut out = StructValue::new();
    for field in StatOptionField::ALL {
        out.insert(
            field.name(),
            if field == StatOptionField::Streams {
                empty_cell()
            } else {
                empty_numeric()
            },
        );
    }
    out
}
