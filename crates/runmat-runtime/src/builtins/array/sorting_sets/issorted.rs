//! MATLAB-compatible `issorted` builtin with GPU-aware semantics.

use std::cmp::Ordering;

use runmat_builtins::{CharArray, ComplexTensor, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use super::type_resolvers::bool_output_type;
use crate::build_runtime_error;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::issorted")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "issorted",
    op_kind: GpuOpKind::Custom("predicate"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "GPU inputs gather to the host until providers implement dedicated predicate kernels.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::issorted"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "issorted",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Predicate builtin evaluated outside fusion; planner prevents kernel generation.",
};

fn issorted_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("issorted")
        .build()
}

#[runtime_builtin(
    name = "issorted",
    category = "array/sorting_sets",
    summary = "Determine whether an array is already sorted.",
    keywords = "issorted,sorted,monotonic,rows",
    accel = "sink",
    sink = true,
    type_resolver(bool_output_type),
    builtin_path = "crate::builtins::array::sorting_sets::issorted"
)]
async fn issorted_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let input = normalize_input(value).await?;
    let shape = input.shape();
    let args = IssortedArgs::parse(&rest, &shape)?;

    let result = match input {
        InputArray::Real(tensor) => issorted_real(&tensor, &args)?,
        InputArray::Complex(tensor) => issorted_complex(&tensor, &args)?,
        InputArray::String(array) => issorted_string(&array, &args)?,
    };

    Ok(Value::Bool(result))
}

struct IssortedArgs {
    mode: CheckMode,
    direction: Direction,
    comparison: ComparisonMethod,
    missing: MissingPlacement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CheckMode {
    Dimension(usize),
    Rows,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Ascend,
    Descend,
    Monotonic,
    StrictAscend,
    StrictDescend,
    StrictMonotonic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum ComparisonMethod {
    #[default]
    Auto,
    Real,
    Abs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissingPlacement {
    Auto,
    First,
    Last,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissingPlacementResolved {
    First,
    Last,
}

impl MissingPlacement {
    fn resolve(self, direction: SortDirection) -> MissingPlacementResolved {
        match self {
            MissingPlacement::First => MissingPlacementResolved::First,
            MissingPlacement::Last => MissingPlacementResolved::Last,
            MissingPlacement::Auto => match direction {
                SortDirection::Ascend => MissingPlacementResolved::Last,
                SortDirection::Descend => MissingPlacementResolved::First,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SortDirection {
    Ascend,
    Descend,
}

#[derive(Clone, Copy)]
struct OrderSpec {
    direction: SortDirection,
    strict: bool,
}

enum InputArray {
    Real(Tensor),
    Complex(ComplexTensor),
    String(StringArray),
}

impl InputArray {
    fn shape(&self) -> Vec<usize> {
        match self {
            InputArray::Real(t) => t.shape.clone(),
            InputArray::Complex(t) => t.shape.clone(),
            InputArray::String(sa) => sa.shape.clone(),
        }
    }
}

impl IssortedArgs {
    fn parse(args: &[Value], shape: &[usize]) -> crate::BuiltinResult<Self> {
        let mut dim_arg: Option<usize> = None;
        let mut direction: Option<Direction> = None;
        let mut comparison: ComparisonMethod = ComparisonMethod::Auto;
        let mut missing: MissingPlacement = MissingPlacement::Auto;
        let mut mode = CheckMode::Dimension(default_dimension(shape));
        let mut saw_rows = false;

        let mut idx = 0;
        while idx < args.len() {
            let arg = &args[idx];
            if let Some(token) = value_to_string_lower(arg) {
                match token.as_str() {
                    "rows" => {
                        if saw_rows {
                            return Err(issorted_error(
                                "issorted: 'rows' specified more than once",
                            ));
                        }
                        if dim_arg.is_some() {
                            return Err(issorted_error(
                                "issorted: cannot combine 'rows' with a dimension argument",
                            ));
                        }
                        saw_rows = true;
                        mode = CheckMode::Rows;
                        idx += 1;
                        continue;
                    }
                    "ascend" => {
                        ensure_unique_direction(&direction)?;
                        direction = Some(Direction::Ascend);
                        idx += 1;
                        continue;
                    }
                    "descend" => {
                        ensure_unique_direction(&direction)?;
                        direction = Some(Direction::Descend);
                        idx += 1;
                        continue;
                    }
                    "monotonic" => {
                        ensure_unique_direction(&direction)?;
                        direction = Some(Direction::Monotonic);
                        idx += 1;
                        continue;
                    }
                    "strictascend" => {
                        ensure_unique_direction(&direction)?;
                        direction = Some(Direction::StrictAscend);
                        idx += 1;
                        continue;
                    }
                    "strictdescend" => {
                        ensure_unique_direction(&direction)?;
                        direction = Some(Direction::StrictDescend);
                        idx += 1;
                        continue;
                    }
                    "strictmonotonic" => {
                        ensure_unique_direction(&direction)?;
                        direction = Some(Direction::StrictMonotonic);
                        idx += 1;
                        continue;
                    }
                    "comparisonmethod" => {
                        idx += 1;
                        if idx >= args.len() {
                            return Err(issorted_error(
                                "issorted: expected a value for 'ComparisonMethod'",
                            ));
                        }
                        let value = value_to_string_lower(&args[idx]).ok_or_else(|| {
                            issorted_error("issorted: 'ComparisonMethod' expects a string value")
                        })?;
                        comparison = match value.as_str() {
                            "auto" => ComparisonMethod::Auto,
                            "real" => ComparisonMethod::Real,
                            "abs" | "magnitude" => ComparisonMethod::Abs,
                            other => {
                                return Err(issorted_error(format!(
                                    "issorted: unsupported ComparisonMethod '{other}'"
                                )));
                            }
                        };
                        idx += 1;
                        continue;
                    }
                    "missingplacement" => {
                        idx += 1;
                        if idx >= args.len() {
                            return Err(issorted_error(
                                "issorted: expected a value for 'MissingPlacement'",
                            ));
                        }
                        let value = value_to_string_lower(&args[idx]).ok_or_else(|| {
                            issorted_error("issorted: 'MissingPlacement' expects a string value")
                        })?;
                        missing = match value.as_str() {
                            "auto" => MissingPlacement::Auto,
                            "first" => MissingPlacement::First,
                            "last" => MissingPlacement::Last,
                            other => {
                                return Err(issorted_error(format!(
                                    "issorted: unsupported MissingPlacement '{other}'"
                                )));
                            }
                        };
                        idx += 1;
                        continue;
                    }
                    _ => {}
                }
            }

            if !saw_rows && dim_arg.is_none() {
                if let Ok(dim) = tensor::parse_dimension(arg, "issorted") {
                    dim_arg = Some(dim);
                    idx += 1;
                    continue;
                }
            }

            return Err(issorted_error(format!(
                "issorted: unrecognised argument {:?}",
                arg
            )));
        }

        if let Some(dim) = dim_arg {
            mode = CheckMode::Dimension(dim);
        }

        Ok(IssortedArgs {
            mode,
            direction: direction.unwrap_or(Direction::Ascend),
            comparison,
            missing,
        })
    }
}

fn ensure_unique_direction(direction: &Option<Direction>) -> crate::BuiltinResult<()> {
    if direction.is_some() {
        Err(issorted_error(
            "issorted: sorting direction specified more than once",
        ))
    } else {
        Ok(())
    }
}

async fn normalize_input(value: Value) -> crate::BuiltinResult<InputArray> {
    match value {
        Value::Tensor(tensor) => Ok(InputArray::Real(tensor)),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| issorted_error(e))?;
            Ok(InputArray::Real(tensor))
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for("issorted", value)
                .map_err(|e| issorted_error(e))?;
            Ok(InputArray::Real(tensor))
        }
        Value::ComplexTensor(ct) => Ok(InputArray::Complex(ct)),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| issorted_error(format!("issorted: {e}")))?;
            Ok(InputArray::Complex(tensor))
        }
        Value::CharArray(ca) => {
            let tensor = char_array_to_tensor(&ca)?;
            Ok(InputArray::Real(tensor))
        }
        Value::StringArray(sa) => Ok(InputArray::String(sa)),
        Value::String(s) => {
            let array =
                StringArray::new(vec![s], vec![1, 1]).map_err(|e| issorted_error(format!("issorted: {e}")))?;
            Ok(InputArray::String(array))
        }
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            Ok(InputArray::Real(tensor))
        }
        other => Err(issorted_error(format!(
            "issorted: unsupported input type {:?}; expected numeric, logical, complex, char, or string arrays",
            other
        ))),
    }
}

fn issorted_real(tensor: &Tensor, args: &IssortedArgs) -> crate::BuiltinResult<bool> {
    if tensor.data.is_empty() {
        return Ok(true);
    }
    match args.mode {
        CheckMode::Dimension(dim) => Ok(check_real_dimension(tensor, dim, args)),
        CheckMode::Rows => check_real_rows(tensor, args),
    }
}

fn issorted_complex(tensor: &ComplexTensor, args: &IssortedArgs) -> crate::BuiltinResult<bool> {
    if tensor.data.is_empty() {
        return Ok(true);
    }
    match args.mode {
        CheckMode::Dimension(dim) => Ok(check_complex_dimension(tensor, dim, args)),
        CheckMode::Rows => check_complex_rows(tensor, args),
    }
}

fn issorted_string(array: &StringArray, args: &IssortedArgs) -> crate::BuiltinResult<bool> {
    if array.data.is_empty() {
        return Ok(true);
    }
    if !matches!(args.comparison, ComparisonMethod::Auto) {
        return Err(issorted_error(
            "issorted: 'ComparisonMethod' is not supported for string arrays",
        ));
    }
    match args.mode {
        CheckMode::Dimension(dim) => Ok(check_string_dimension(array, dim, args)),
        CheckMode::Rows => check_string_rows(array, args),
    }
}

fn check_real_dimension(tensor: &Tensor, dim: usize, args: &IssortedArgs) -> bool {
    let dim_index = dim.saturating_sub(1);
    if dim_index >= tensor.shape.len() {
        return true;
    }
    let len_dim = tensor.shape[dim_index];
    if len_dim <= 1 {
        return true;
    }

    let before = product(&tensor.shape[..dim_index]);
    let after = product(&tensor.shape[dim_index + 1..]);
    let effective_comp = match args.comparison {
        ComparisonMethod::Auto => ComparisonMethod::Real,
        other => other,
    };
    let mut slice = Vec::with_capacity(len_dim);
    for after_idx in 0..after {
        for before_idx in 0..before {
            slice.clear();
            for k in 0..len_dim {
                let idx = before_idx + k * before + after_idx * before * len_dim;
                slice.push(tensor.data[idx]);
            }
            if !check_real_slice(&slice, args.direction, effective_comp, args.missing) {
                return false;
            }
        }
    }
    true
}

fn check_complex_dimension(tensor: &ComplexTensor, dim: usize, args: &IssortedArgs) -> bool {
    let dim_index = dim.saturating_sub(1);
    if dim_index >= tensor.shape.len() {
        return true;
    }
    let len_dim = tensor.shape[dim_index];
    if len_dim <= 1 {
        return true;
    }
    let before = product(&tensor.shape[..dim_index]);
    let after = product(&tensor.shape[dim_index + 1..]);
    let effective_comp = match args.comparison {
        ComparisonMethod::Auto => ComparisonMethod::Abs,
        other => other,
    };
    let mut slice = Vec::with_capacity(len_dim);
    for after_idx in 0..after {
        for before_idx in 0..before {
            slice.clear();
            for k in 0..len_dim {
                let idx = before_idx + k * before + after_idx * before * len_dim;
                slice.push(tensor.data[idx]);
            }
            if !check_complex_slice(&slice, args.direction, effective_comp, args.missing) {
                return false;
            }
        }
    }
    true
}

fn check_string_dimension(array: &StringArray, dim: usize, args: &IssortedArgs) -> bool {
    let dim_index = dim.saturating_sub(1);
    if dim_index >= array.shape.len() {
        return true;
    }
    let len_dim = array.shape[dim_index];
    if len_dim <= 1 {
        return true;
    }
    let before = product(&array.shape[..dim_index]);
    let after = product(&array.shape[dim_index + 1..]);
    let mut slice = Vec::with_capacity(len_dim);
    for after_idx in 0..after {
        for before_idx in 0..before {
            slice.clear();
            for k in 0..len_dim {
                let idx = before_idx + k * before + after_idx * before * len_dim;
                slice.push(array.data[idx].as_str());
            }
            if !check_string_slice(&slice, args.direction, args.missing) {
                return false;
            }
        }
    }
    true
}

fn check_real_rows(tensor: &Tensor, args: &IssortedArgs) -> crate::BuiltinResult<bool> {
    if tensor.shape.len() > 2 {
        return Err(issorted_error("issorted: 'rows' expects a 2-D matrix"));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    if rows <= 1 || cols == 0 {
        return Ok(true);
    }
    let effective_comp = match args.comparison {
        ComparisonMethod::Auto => ComparisonMethod::Real,
        other => other,
    };
    let orders = direction_orders(args.direction);
    for &order in orders {
        if real_rows_in_order(tensor, rows, cols, order, effective_comp, args.missing) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn check_complex_rows(tensor: &ComplexTensor, args: &IssortedArgs) -> crate::BuiltinResult<bool> {
    if tensor.shape.len() > 2 {
        return Err(issorted_error("issorted: 'rows' expects a 2-D matrix"));
    }
    let rows = tensor.rows;
    let cols = tensor.cols;
    if rows <= 1 || cols == 0 {
        return Ok(true);
    }
    let effective_comp = match args.comparison {
        ComparisonMethod::Auto => ComparisonMethod::Abs,
        other => other,
    };
    let orders = direction_orders(args.direction);
    for &order in orders {
        if complex_rows_in_order(tensor, rows, cols, order, effective_comp, args.missing) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn check_string_rows(array: &StringArray, args: &IssortedArgs) -> crate::BuiltinResult<bool> {
    if array.shape.len() > 2 {
        return Err(issorted_error("issorted: 'rows' expects a 2-D matrix"));
    }
    let rows = array.rows;
    let cols = array.cols;
    if rows <= 1 || cols == 0 {
        return Ok(true);
    }
    let orders = direction_orders(args.direction);
    for &order in orders {
        if string_rows_in_order(array, rows, cols, order, args.missing) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn real_rows_in_order(
    tensor: &Tensor,
    rows: usize,
    cols: usize,
    order: OrderSpec,
    comparison: ComparisonMethod,
    missing: MissingPlacement,
) -> bool {
    if order.strict && tensor.data.iter().any(|v| v.is_nan()) {
        return false;
    }
    let missing_resolved = missing.resolve(order.direction);
    for row in 0..rows - 1 {
        let ord = compare_real_row_pair(
            tensor,
            rows,
            cols,
            row,
            row + 1,
            order.direction,
            comparison,
            missing_resolved,
        );
        if !order_satisfied(ord, order) {
            return false;
        }
    }
    true
}

fn complex_rows_in_order(
    tensor: &ComplexTensor,
    rows: usize,
    cols: usize,
    order: OrderSpec,
    comparison: ComparisonMethod,
    missing: MissingPlacement,
) -> bool {
    if order.strict && tensor.data.iter().any(|v| complex_is_nan(*v)) {
        return false;
    }
    let missing_resolved = missing.resolve(order.direction);
    for row in 0..rows - 1 {
        let ord = compare_complex_row_pair(
            tensor,
            rows,
            cols,
            row,
            row + 1,
            order.direction,
            comparison,
            missing_resolved,
        );
        if !order_satisfied(ord, order) {
            return false;
        }
    }
    true
}

fn string_rows_in_order(
    array: &StringArray,
    rows: usize,
    cols: usize,
    order: OrderSpec,
    missing: MissingPlacement,
) -> bool {
    if order.strict && array.data.iter().any(|s| is_string_missing(s)) {
        return false;
    }
    let missing_resolved = missing.resolve(order.direction);
    for row in 0..rows - 1 {
        let ord = compare_string_row_pair(
            array,
            rows,
            cols,
            row,
            row + 1,
            order.direction,
            missing_resolved,
        );
        if !order_satisfied(ord, order) {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn compare_real_row_pair(
    tensor: &Tensor,
    rows: usize,
    cols: usize,
    a: usize,
    b: usize,
    direction: SortDirection,
    comparison: ComparisonMethod,
    missing: MissingPlacementResolved,
) -> Ordering {
    for col in 0..cols {
        let idx_a = a + col * rows;
        let idx_b = b + col * rows;
        let ord = compare_real_scalars(
            tensor.data[idx_a],
            tensor.data[idx_b],
            direction,
            comparison,
            missing,
        );
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

#[allow(clippy::too_many_arguments)]
fn compare_complex_row_pair(
    tensor: &ComplexTensor,
    rows: usize,
    cols: usize,
    a: usize,
    b: usize,
    direction: SortDirection,
    comparison: ComparisonMethod,
    missing: MissingPlacementResolved,
) -> Ordering {
    for col in 0..cols {
        let idx_a = a + col * rows;
        let idx_b = b + col * rows;
        let ord = compare_complex_scalars(
            tensor.data[idx_a],
            tensor.data[idx_b],
            direction,
            comparison,
            missing,
        );
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_string_row_pair(
    array: &StringArray,
    rows: usize,
    cols: usize,
    a: usize,
    b: usize,
    direction: SortDirection,
    missing: MissingPlacementResolved,
) -> Ordering {
    for col in 0..cols {
        let idx_a = a + col * rows;
        let idx_b = b + col * rows;
        let ord = compare_string_scalars(
            array.data[idx_a].as_str(),
            array.data[idx_b].as_str(),
            direction,
            missing,
        );
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn order_satisfied(ord: Ordering, order: OrderSpec) -> bool {
    match order.direction {
        SortDirection::Ascend => match ord {
            Ordering::Greater => false,
            Ordering::Equal => !order.strict,
            Ordering::Less => true,
        },
        SortDirection::Descend => match ord {
            Ordering::Less => true,
            Ordering::Equal => !order.strict,
            Ordering::Greater => false,
        },
    }
}

fn check_real_slice(
    slice: &[f64],
    direction: Direction,
    comparison: ComparisonMethod,
    missing: MissingPlacement,
) -> bool {
    if slice.len() <= 1 {
        return true;
    }
    let orders = direction_orders(direction);
    for &order in orders {
        if order.strict && slice.iter().any(|v| v.is_nan()) {
            continue;
        }
        let missing_resolved = missing.resolve(order.direction);
        if real_slice_in_order(slice, order, comparison, missing_resolved) {
            return true;
        }
    }
    false
}

fn check_complex_slice(
    slice: &[(f64, f64)],
    direction: Direction,
    comparison: ComparisonMethod,
    missing: MissingPlacement,
) -> bool {
    if slice.len() <= 1 {
        return true;
    }
    let orders = direction_orders(direction);
    for &order in orders {
        if order.strict && slice.iter().any(|v| complex_is_nan(*v)) {
            continue;
        }
        let missing_resolved = missing.resolve(order.direction);
        if complex_slice_in_order(slice, order, comparison, missing_resolved) {
            return true;
        }
    }
    false
}

fn check_string_slice(slice: &[&str], direction: Direction, missing: MissingPlacement) -> bool {
    if slice.len() <= 1 {
        return true;
    }
    let orders = direction_orders(direction);
    for &order in orders {
        if order.strict && slice.iter().any(|s| is_string_missing(s)) {
            continue;
        }
        let missing_resolved = missing.resolve(order.direction);
        if string_slice_in_order(slice, order, missing_resolved) {
            return true;
        }
    }
    false
}

fn real_slice_in_order(
    slice: &[f64],
    order: OrderSpec,
    comparison: ComparisonMethod,
    missing: MissingPlacementResolved,
) -> bool {
    for pair in slice.windows(2) {
        let ord = compare_real_scalars(pair[0], pair[1], order.direction, comparison, missing);
        if !order_satisfied(ord, order) {
            return false;
        }
    }
    true
}

fn complex_slice_in_order(
    slice: &[(f64, f64)],
    order: OrderSpec,
    comparison: ComparisonMethod,
    missing: MissingPlacementResolved,
) -> bool {
    for pair in slice.windows(2) {
        let ord = compare_complex_scalars(pair[0], pair[1], order.direction, comparison, missing);
        if !order_satisfied(ord, order) {
            return false;
        }
    }
    true
}

fn string_slice_in_order(
    slice: &[&str],
    order: OrderSpec,
    missing: MissingPlacementResolved,
) -> bool {
    for pair in slice.windows(2) {
        let ord = compare_string_scalars(pair[0], pair[1], order.direction, missing);
        if !order_satisfied(ord, order) {
            return false;
        }
    }
    true
}

fn compare_real_scalars(
    a: f64,
    b: f64,
    direction: SortDirection,
    comparison: ComparisonMethod,
    missing: MissingPlacementResolved,
) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => match missing {
            MissingPlacementResolved::First => Ordering::Less,
            MissingPlacementResolved::Last => Ordering::Greater,
        },
        (false, true) => match missing {
            MissingPlacementResolved::First => Ordering::Greater,
            MissingPlacementResolved::Last => Ordering::Less,
        },
        (false, false) => compare_real_finite_scalars(a, b, direction, comparison),
    }
}

fn compare_real_finite_scalars(
    a: f64,
    b: f64,
    direction: SortDirection,
    comparison: ComparisonMethod,
) -> Ordering {
    if matches!(comparison, ComparisonMethod::Abs) {
        let abs_cmp = a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal);
        if abs_cmp != Ordering::Equal {
            return match direction {
                SortDirection::Ascend => abs_cmp,
                SortDirection::Descend => abs_cmp.reverse(),
            };
        }
    }
    match direction {
        SortDirection::Ascend => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
        SortDirection::Descend => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
    }
}

fn compare_complex_scalars(
    a: (f64, f64),
    b: (f64, f64),
    direction: SortDirection,
    comparison: ComparisonMethod,
    missing: MissingPlacementResolved,
) -> Ordering {
    match (complex_is_nan(a), complex_is_nan(b)) {
        (true, true) => Ordering::Equal,
        (true, false) => match missing {
            MissingPlacementResolved::First => Ordering::Less,
            MissingPlacementResolved::Last => Ordering::Greater,
        },
        (false, true) => match missing {
            MissingPlacementResolved::First => Ordering::Greater,
            MissingPlacementResolved::Last => Ordering::Less,
        },
        (false, false) => compare_complex_finite_scalars(a, b, direction, comparison),
    }
}

fn compare_complex_finite_scalars(
    a: (f64, f64),
    b: (f64, f64),
    direction: SortDirection,
    comparison: ComparisonMethod,
) -> Ordering {
    match comparison {
        ComparisonMethod::Real => compare_complex_real_first(a, b, direction),
        ComparisonMethod::Abs | ComparisonMethod::Auto => {
            let abs_cmp = complex_abs(a)
                .partial_cmp(&complex_abs(b))
                .unwrap_or(Ordering::Equal);
            if abs_cmp != Ordering::Equal {
                return match direction {
                    SortDirection::Ascend => abs_cmp,
                    SortDirection::Descend => abs_cmp.reverse(),
                };
            }
            compare_complex_real_first(a, b, direction)
        }
    }
}

fn compare_complex_real_first(a: (f64, f64), b: (f64, f64), direction: SortDirection) -> Ordering {
    let real_cmp = match direction {
        SortDirection::Ascend => a.0.partial_cmp(&b.0),
        SortDirection::Descend => b.0.partial_cmp(&a.0),
    }
    .unwrap_or(Ordering::Equal);
    if real_cmp != Ordering::Equal {
        return real_cmp;
    }
    match direction {
        SortDirection::Ascend => a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal),
        SortDirection::Descend => b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal),
    }
}

fn compare_string_scalars(
    a: &str,
    b: &str,
    direction: SortDirection,
    missing: MissingPlacementResolved,
) -> Ordering {
    let missing_a = is_string_missing(a);
    let missing_b = is_string_missing(b);
    match (missing_a, missing_b) {
        (true, true) => Ordering::Equal,
        (true, false) => match missing {
            MissingPlacementResolved::First => Ordering::Less,
            MissingPlacementResolved::Last => Ordering::Greater,
        },
        (false, true) => match missing {
            MissingPlacementResolved::First => Ordering::Greater,
            MissingPlacementResolved::Last => Ordering::Less,
        },
        (false, false) => match direction {
            SortDirection::Ascend => a.cmp(b),
            SortDirection::Descend => b.cmp(a),
        },
    }
}

fn complex_is_nan(value: (f64, f64)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn complex_abs(value: (f64, f64)) -> f64 {
    value.0.hypot(value.1)
}

fn is_string_missing(value: &str) -> bool {
    value.eq_ignore_ascii_case("<missing>")
}

fn direction_orders(direction: Direction) -> &'static [OrderSpec] {
    match direction {
        Direction::Ascend => &[OrderSpec {
            direction: SortDirection::Ascend,
            strict: false,
        }],
        Direction::Descend => &[OrderSpec {
            direction: SortDirection::Descend,
            strict: false,
        }],
        Direction::Monotonic => &[
            OrderSpec {
                direction: SortDirection::Ascend,
                strict: false,
            },
            OrderSpec {
                direction: SortDirection::Descend,
                strict: false,
            },
        ],
        Direction::StrictAscend => &[OrderSpec {
            direction: SortDirection::Ascend,
            strict: true,
        }],
        Direction::StrictDescend => &[OrderSpec {
            direction: SortDirection::Descend,
            strict: true,
        }],
        Direction::StrictMonotonic => &[
            OrderSpec {
                direction: SortDirection::Ascend,
                strict: true,
            },
            OrderSpec {
                direction: SortDirection::Descend,
                strict: true,
            },
        ],
    }
}

fn default_dimension(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn product(slice: &[usize]) -> usize {
    slice
        .iter()
        .copied()
        .fold(1usize, |acc, value| acc.saturating_mul(value.max(1)))
}

fn value_to_string_lower(value: &Value) -> Option<String> {
    match String::try_from(value) {
        Ok(text) => Some(text.trim().to_ascii_lowercase()),
        Err(_) => None,
    }
}

fn char_array_to_tensor(array: &CharArray) -> crate::BuiltinResult<Tensor> {
    let rows = array.rows;
    let cols = array.cols;
    let mut data = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let ch = array.data[r * cols + c];
            let idx = r + c * rows;
            data[idx] = ch as u32 as f64;
        }
    }
    Tensor::new(data, vec![rows, cols]).map_err(|e| issorted_error(format!("issorted: {e}")))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, Type, Value};

    fn issorted_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::issorted_builtin(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_numeric_vector_true() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = issorted_builtin(Value::Tensor(tensor), vec![]).expect("issorted");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn issorted_type_resolver_bool() {
        assert_eq!(bool_output_type(&[Type::tensor()]), Type::Bool);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_numeric_vector_false() {
        let tensor = Tensor::new(vec![3.0, 2.0, 1.0], vec![3, 1]).unwrap();
        let result = issorted_builtin(Value::Tensor(tensor), vec![]).expect("issorted");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_logical_vector() {
        let logical = LogicalArray::new(vec![0, 1, 1], vec![3, 1]).unwrap();
        let result =
            issorted_builtin(Value::LogicalArray(logical), vec![]).expect("issorted logical");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 3.0], vec![2, 2]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_strictascend_rejects_duplicates() {
        let tensor = Tensor::new(vec![1.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let args = vec![Value::from("strictascend")];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_strictmonotonic_true_with_descend() {
        let tensor = Tensor::new(vec![9.0, 4.0, 1.0], vec![3, 1]).unwrap();
        let args = vec![Value::from("strictmonotonic")];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_strictmonotonic_rejects_plateaus() {
        let tensor = Tensor::new(vec![4.0, 4.0, 2.0, 1.0], vec![4, 1]).unwrap();
        let args = vec![Value::from("strictmonotonic")];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_monotonic_accepts_descending() {
        let tensor = Tensor::new(vec![5.0, 4.0, 4.0, 1.0], vec![4, 1]).unwrap();
        let args = vec![Value::from("monotonic")];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_monotonic_rejects_unsorted_data() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0], vec![3, 1]).unwrap();
        let args = vec![Value::from("monotonic")];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_missingplacement_first() {
        let tensor = Tensor::new(vec![f64::NAN, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::from("MissingPlacement"), Value::from("first")];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_missingplacement_first_violation() {
        let tensor = Tensor::new(vec![2.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::from("MissingPlacement"), Value::from("first")];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_missingplacement_auto_descend_prefers_front() {
        let tensor = Tensor::new(vec![f64::NAN, 5.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::from("descend")];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_comparison_abs() {
        let tensor = Tensor::new(vec![-1.0, 1.5, -2.0], vec![3, 1]).unwrap();
        let args = vec![Value::from("ComparisonMethod"), Value::from("abs")];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_complex_abs_method() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 1.0), (2.0, 0.0), (2.0, 3.0)], vec![3, 1]).unwrap();
        let args = vec![Value::from("ComparisonMethod"), Value::from("abs")];
        let result = issorted_builtin(Value::ComplexTensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_complex_real_method() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 1.0), (1.0, 1.0), (2.0, 0.0)], vec![3, 1]).unwrap();
        let args = vec![
            Value::from("ComparisonMethod"),
            Value::from("real"),
            Value::from("strictascend"),
        ];
        let result = issorted_builtin(Value::ComplexTensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_rows_true() {
        let tensor = Tensor::new(vec![1.0, 2.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("rows")];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_rows_dimension_error() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2, 1]).unwrap();
        let result = issorted_builtin(Value::Tensor(tensor), vec![Value::from("rows")]);
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_rows_descend_false() {
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 0.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("rows"), Value::from("descend")];
        let result = issorted_builtin(Value::Tensor(tensor), args).expect("issorted");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_string_dimension() {
        let array = StringArray::new(
            vec![
                "pear".into(),
                "plum".into(),
                "apple".into(),
                "banana".into(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result =
            issorted_builtin(Value::StringArray(array), args).expect("issorted string dim");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_string_missingplacement_last() {
        let array = StringArray::new(
            vec!["apple".into(), "banana".into(), "<missing>".into()],
            vec![3, 1],
        )
        .unwrap();
        let args = vec![Value::from("MissingPlacement"), Value::from("last")];
        let result =
            issorted_builtin(Value::StringArray(array), args).expect("issorted string placement");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_string_missingplacement_last_violation() {
        let array = StringArray::new(vec!["<missing>".into(), "apple".into()], vec![2, 1]).unwrap();
        let args = vec![Value::from("MissingPlacement"), Value::from("last")];
        let result =
            issorted_builtin(Value::StringArray(array), args).expect("issorted string placement");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_string_comparison_method_error() {
        let array = StringArray::new(vec!["apple".into(), "berry".into()], vec![2, 1]).unwrap();
        let args = vec![Value::from("ComparisonMethod"), Value::from("real")];
        let result = issorted_builtin(Value::StringArray(array), args);
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_char_array_input() {
        let chars = CharArray::new(vec!['a', 'c', 'e'], 1, 3).unwrap();
        let result = issorted_builtin(Value::CharArray(chars), vec![]).expect("issorted char");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_duplicate_direction_error() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = vec![Value::from("ascend"), Value::from("descend")];
        let result = issorted_builtin(Value::Tensor(tensor), args);
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issorted_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = issorted_builtin(Value::GpuTensor(handle), vec![]).expect("issorted gpu");
            assert_eq!(result, Value::Bool(true));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn issorted_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let cpu = issorted_builtin(Value::Tensor(tensor.clone()), vec![]).expect("cpu issorted");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let gpu = issorted_builtin(Value::GpuTensor(handle), vec![]).expect("gpu issorted");
        assert_eq!(gpu, cpu);
    }
}
