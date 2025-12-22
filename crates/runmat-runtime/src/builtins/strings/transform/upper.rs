//! MATLAB-compatible `upper` builtin with GPU-aware semantics for RunMat.
use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, uppercase_preserving_missing};
use crate::{gather_if_needed, make_cell};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "upper",
        builtin_path = "crate::builtins::strings::transform::upper"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "upper"
category: "strings/transform"
keywords: ["upper", "uppercase", "convert to uppercase", "string case", "character arrays"]
summary: "Convert strings, character arrays, and cell arrays of character vectors to uppercase."
references:
  - https://www.mathworks.com/help/matlab/ref/upper.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs on the CPU; GPU-resident inputs are gathered before conversion to keep MATLAB parity."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::transform::upper::tests"
  integration: "builtins::strings::transform::upper::tests::upper_cell_array_mixed_content"
---

# What does the `upper` function do in MATLAB / RunMat?
`upper(text)` converts every alphabetic character in `text` to uppercase. It accepts string scalars,
string arrays, character arrays, and cell arrays of character vectors, mirroring MATLAB behaviour.
Non-alphabetic characters are returned unchanged.

## How does the `upper` function behave in MATLAB / RunMat?
- String inputs stay as strings. String arrays preserve their size, orientation, and missing values.
- Character arrays are processed row by row. The result remains a rectangular char array; if any row
  grows after uppercasing (for example `'ß' → "SS"`), the array widens and shorter rows are padded with spaces.
- Cell arrays must contain string scalars or character vectors. The result is a cell array of the same size
  with each element converted to uppercase; other types raise MATLAB-compatible errors.
- Missing string scalars (`string(missing)`) remain `<missing>` so downstream code behaves like MATLAB.
- Inputs that are numeric, logical, structs, or GPU tensors raise MATLAB-compatible type errors.

## `upper` Function GPU Execution Behaviour
`upper` executes on the CPU. Text values currently reside in host memory, so providers do not offer device
kernels for this builtin. When you pass a container that still holds GPU handles (for example, a struct
whose string fields were gathered lazily), RunMat gathers those handles before performing the conversion.
If you store characters as numeric code points on the GPU, gather and convert them to text before calling
`upper`.

## GPU residency in RunMat (Do I need `gpuArray`?)
RunMat keeps text data in host memory, so you typically work with ordinary string and character arrays.
When text originates from GPU computations (for example, numeric code points produced by kernels), gather
those values to the host and convert them to text before calling `upper`.

## Examples of using the `upper` function in MATLAB / RunMat

### Convert a string scalar to uppercase
```matlab
txt = "RunMat";
result = upper(txt);
```
Expected output:
```matlab
result = "RUNMAT"
```

### Uppercase each element of a string array
```matlab
labels = ["north" "South"; "East" "west"];
uppered = upper(labels);
```
Expected output:
```matlab
uppered = 2×2 string
    "NORTH"    "SOUTH"
    "EAST"     "WEST"
```

### Uppercase character array rows while preserving shape
```matlab
animals = char("cat", "doge");
result = upper(animals);
```
Expected output:
```matlab
result =

  2×4 char array

    'CAT '
    'DOGE'
```

### Uppercase a cell array of character vectors
```matlab
C = {'hello', 'World'};
out = upper(C);
```
Expected output:
```matlab
out = 1×2 cell array
    {'HELLO'}    {'WORLD'}
```

### Keep missing strings as missing
```matlab
vals = ["data", string(missing), "gpu"];
converted = upper(vals);
```
Expected output:
```matlab
converted = 1×3 string
    "DATA"    <missing>    "GPU"
```

### Handle text stored on a GPU input
```matlab
codes = gpuArray(uint16('runmat'));
txt = char(gather(codes));
result = upper(txt);
```
Expected output:
```matlab
result = 'RUNMAT'
```

## FAQ

### Does `upper` change non-alphabetic characters?
No. Digits, punctuation, whitespace, and symbols remain untouched. Only alphabetic code points that have
distinct uppercase forms are converted.

### What happens to character array dimensions?
RunMat uppercases each row independently and pads with spaces when an uppercase mapping increases the row
length. This mirrors MATLAB’s behaviour so the result always has rectangular dimensions.

### Can I pass numeric arrays to `upper`?
No. Passing numeric, logical, or struct inputs raises a MATLAB-compatible error. Convert the data to a string
or character array first (for example with `string` or `char`).

### How are missing strings handled?
Missing string scalars remain `<missing>` and are returned unchanged. This matches MATLAB’s handling of
missing values in text processing functions.

### Will `upper` ever execute on the GPU?
Not today. The builtin gathers GPU-resident data automatically and performs the conversion on the CPU so the
results match MATLAB exactly. Providers may add device-side kernels in the future, but behaviour will remain
compatible.

## See Also
[lower](./lower), [string](./string), [char](./char), [regexprep](./regexprep), [strcmpi](./strcmpi)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/transform/upper.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/upper.rs)
- Found an issue? Please [open a GitHub issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::upper")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "upper",
    op_kind: GpuOpKind::Custom("string-transform"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Executes on the CPU; GPU-resident inputs are gathered to host memory before conversion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::upper")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "upper",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; not eligible for fusion and always gathers GPU inputs.",
};

const ARG_TYPE_ERROR: &str =
    "upper: first argument must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "upper: cell array elements must be string scalars or character vectors";

#[runtime_builtin(
    name = "upper",
    category = "strings/transform",
    summary = "Convert strings, character arrays, and cell arrays of character vectors to uppercase.",
    keywords = "upper,uppercase,strings,character array,text",
    accel = "sink",
    builtin_path = "crate::builtins::strings::transform::upper"
)]
fn upper_builtin(value: Value) -> Result<Value, String> {
    let gathered = gather_if_needed(&value).map_err(|e| format!("upper: {e}"))?;
    match gathered {
        Value::String(text) => Ok(Value::String(uppercase_preserving_missing(text))),
        Value::StringArray(array) => upper_string_array(array),
        Value::CharArray(array) => upper_char_array(array),
        Value::Cell(cell) => upper_cell_array(cell),
        _ => Err(ARG_TYPE_ERROR.to_string()),
    }
}

fn upper_string_array(array: StringArray) -> Result<Value, String> {
    let StringArray { data, shape, .. } = array;
    let uppered = data
        .into_iter()
        .map(uppercase_preserving_missing)
        .collect::<Vec<_>>();
    let upper_array = StringArray::new(uppered, shape).map_err(|e| format!("upper: {e}"))?;
    Ok(Value::StringArray(upper_array))
}

fn upper_char_array(array: CharArray) -> Result<Value, String> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 || cols == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut upper_rows = Vec::with_capacity(rows);
    let mut target_cols = cols;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row).to_uppercase();
        let len = text.chars().count();
        target_cols = target_cols.max(len);
        upper_rows.push(text);
    }

    let mut upper_data = Vec::with_capacity(rows * target_cols);
    for row_text in upper_rows {
        let mut chars: Vec<char> = row_text.chars().collect();
        if chars.len() < target_cols {
            chars.resize(target_cols, ' ');
        }
        upper_data.extend(chars.into_iter());
    }

    CharArray::new(upper_data, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| format!("upper: {e}"))
}

fn upper_cell_array(cell: CellArray) -> Result<Value, String> {
    let CellArray {
        data, rows, cols, ..
    } = cell;
    let mut upper_values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let upper = upper_cell_element(&data[idx])?;
            upper_values.push(upper);
        }
    }
    make_cell(upper_values, rows, cols).map_err(|e| format!("upper: {e}"))
}

fn upper_cell_element(value: &Value) -> Result<Value, String> {
    match value {
        Value::String(text) => Ok(Value::String(uppercase_preserving_missing(text.clone()))),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(Value::String(
            uppercase_preserving_missing(sa.data[0].clone()),
        )),
        Value::CharArray(ca) if ca.rows <= 1 => upper_char_array(ca.clone()),
        Value::CharArray(_) => Err(CELL_ELEMENT_ERROR.to_string()),
        _ => Err(CELL_ELEMENT_ERROR.to_string()),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_string_scalar_value() {
        let result = upper_builtin(Value::String("RunMat".into())).expect("upper");
        assert_eq!(result, Value::String("RUNMAT".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_string_array_preserves_shape() {
        let array = StringArray::new(
            vec![
                "gpu".into(),
                "accel".into(),
                "<missing>".into(),
                "MiXeD".into(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let result = upper_builtin(Value::StringArray(array)).expect("upper");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("GPU"),
                        String::from("ACCEL"),
                        String::from("<missing>"),
                        String::from("MIXED")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_char_array_multiple_rows() {
        let data: Vec<char> = vec!['c', 'a', 't', 'd', 'o', 'g'];
        let array = CharArray::new(data, 2, 3).unwrap();
        let result = upper_builtin(Value::CharArray(array)).expect("upper");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 3);
                assert_eq!(ca.data, vec!['C', 'A', 'T', 'D', 'O', 'G']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_char_vector_handles_padding() {
        let array = CharArray::new_row("hello ");
        let result = upper_builtin(Value::CharArray(array)).expect("upper");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 6);
                let expected: Vec<char> = "HELLO ".chars().collect();
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_char_array_unicode_expansion_extends_width() {
        let data: Vec<char> = vec!['ß', 'a'];
        let array = CharArray::new(data, 1, 2).unwrap();
        let result = upper_builtin(Value::CharArray(array)).expect("upper");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 3);
                let expected: Vec<char> = vec!['S', 'S', 'A'];
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("run")),
                Value::String("Mat".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = upper_builtin(Value::Cell(cell)).expect("upper");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("RUN")));
                assert_eq!(second, Value::String("MAT".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_errors_on_invalid_input() {
        let err = upper_builtin(Value::Num(1.0)).unwrap_err();
        assert_eq!(err, ARG_TYPE_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_cell_errors_on_invalid_element() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = upper_builtin(Value::Cell(cell)).unwrap_err();
        assert_eq!(err, CELL_ELEMENT_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_preserves_missing_string() {
        let result = upper_builtin(Value::String("<missing>".into())).expect("upper");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_cell_allows_empty_char_vector() {
        let empty_char = CharArray::new(Vec::new(), 1, 0).unwrap();
        let cell = CellArray::new(vec![Value::CharArray(empty_char.clone())], 1, 1).unwrap();
        let result = upper_builtin(Value::Cell(cell)).expect("upper");
        match result {
            Value::Cell(out) => {
                let element = out.get(0, 0).unwrap();
                assert_eq!(element, Value::CharArray(empty_char));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn upper_gpu_tensor_input_gathers_then_errors() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let data = [1.0f64, 2.0];
        let shape = [2usize, 1usize];
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload");
        let err = upper_builtin(Value::GpuTensor(handle.clone())).unwrap_err();
        assert_eq!(err, ARG_TYPE_ERROR);
        provider.free(&handle).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
