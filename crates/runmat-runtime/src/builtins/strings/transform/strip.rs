//! MATLAB-compatible `strip` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, STRIP_CHARACTER_SET_EXTENSION, STRIP_DIRECTION_ALIAS_EXTENSION,
    STRIP_ERROR_ARG_COUNT, STRIP_ERROR_CELL_ELEMENT, STRIP_ERROR_CHARACTERS, STRIP_ERROR_DIRECTION,
    STRIP_ERROR_INTERNAL, STRIP_ERROR_INVALID_INPUT, STRIP_ERROR_SIZE_MISMATCH,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, StringArray, StringElement, Utf16CodeUnit, Utf8Text, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{
    contains_resident_text_input, is_missing_string, transform_scalar_borrowed,
    transform_string_array_borrowed_indexed,
};
use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::strip")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strip",
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
    notes: "Executes on host text; resident numeric inputs reject before provider access.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::strip")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strip",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Host text transformation; not eligible for fusion. Resident numeric inputs reject.",
};

const BUILTIN_NAME: &str = "strip";

mod character;
mod cell;

fn strip_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn strip_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    strip_error_with_message(error.message, error)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum StripDirection {
    Both,
    Left,
    Right,
}

#[derive(Clone, Copy)]
enum DirectionToken {
    Standard(StripDirection),
    RunMatAlias(StripDirection),
}

enum PatternSpec<'a> {
    Default,
    Scalar(PatternRef<'a>),
    PerElementStrings(&'a StringArray),
    PerElementChars(&'a CharArray),
    PerElementCell(&'a CellArray),
}

#[derive(Clone, Copy)]
enum PatternRef<'a> {
    Default,
    CustomText(&'a str),
    CustomChars(&'a [Utf16CodeUnit]),
}

struct PatternExpectation<'a> {
    len: usize,
    shape: Option<&'a [usize]>,
}

impl<'a> PatternExpectation<'a> {
    fn scalar() -> Self {
        Self {
            len: 1,
            shape: None,
        }
    }

    fn with_len(len: usize) -> Self {
        Self { len, shape: None }
    }

    fn with_shape(len: usize, shape: &'a [usize]) -> Self {
        Self {
            len,
            shape: Some(shape),
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn shape(&self) -> Option<&[usize]> {
        self.shape
    }
}

impl PatternSpec<'_> {
    fn pattern_for_index(&self, idx: usize) -> PatternRef<'_> {
        match self {
            PatternSpec::Default => PatternRef::Default,
            PatternSpec::Scalar(pattern) => *pattern,
            PatternSpec::PerElementStrings(array) => PatternRef::CustomText(
                array.data()[idx]
                    .as_str()
                    .expect("pattern elements validated before selection"),
            ),
            PatternSpec::PerElementChars(array) => {
                let start = idx * array.cols;
                PatternRef::CustomChars(&array.data()[start..start + array.cols])
            }
            PatternSpec::PerElementCell(cell) => pattern_ref_from_value(&cell.data()[idx])
                .expect("pattern elements validated before selection"),
        }
    }
}

#[runtime_builtin(
    name = "strip",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::strip"
)]
async fn strip_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if contains_resident_text_input(&value) || rest.iter().any(contains_resident_text_input) {
        return Err(strip_error(&STRIP_ERROR_INVALID_INPUT));
    }
    match value {
        Value::String(text) => strip_string(runtime.as_ref(), text, rest.view()).await,
        Value::StringArray(array) => strip_string_array(runtime.as_ref(), array, rest.view()).await,
        Value::CharArray(array) => {
            character::strip_char_array(runtime.as_ref(), array, rest.view()).await
        }
        Value::Cell(cell) => cell::strip_cell_array(runtime.as_ref(), cell, rest.view()).await,
        _ => Err(strip_error(&STRIP_ERROR_INVALID_INPUT)),
    }
}

async fn strip_string(
    runtime: Option<&RuntimeContext>,
    text: StringElement,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    if is_missing_string(&text) {
        return Ok(Value::String(text));
    }
    let expectation = PatternExpectation::scalar();
    let (direction, pattern_spec) = parse_arguments(args, &expectation).await?;
    let stripped = transform_scalar_borrowed(
        runtime,
        text,
        |text| strip_slice(text, direction, pattern_spec.pattern_for_index(0)),
        strip_allocation_error,
    )?;
    Ok(Value::String(stripped))
}

async fn strip_string_array(
    runtime: Option<&RuntimeContext>,
    array: StringArray,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let expected_len = array.data().len();
    let expectation = PatternExpectation::with_shape(expected_len, &array.shape);
    let (direction, pattern_spec) = parse_arguments(args, &expectation).await?;
    let result = transform_string_array_borrowed_indexed(
        runtime,
        array,
        |index, text| strip_slice(text, direction, pattern_spec.pattern_for_index(index)),
        strip_allocation_error,
        |detail| {
            strip_error_with_message(format!("{BUILTIN_NAME}: {detail}"), &STRIP_ERROR_INTERNAL)
        },
    )?;
    Ok(Value::StringArray(result))
}

fn strip_allocation_error(runtime: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    build_runtime_error(format!("strip: host text allocation: {cause}"))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(format!(
            "{}:{}",
            RuntimeContext::error_namespace(runtime),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

async fn parse_arguments<'a>(
    args: crate::call::FunctionArgumentView<'a>,
    expectation: &PatternExpectation<'_>,
) -> BuiltinResult<(StripDirection, PatternSpec<'a>)> {
    match args.len() {
        0 => Ok((StripDirection::Both, PatternSpec::Default)),
        1 => {
            if let Some(direction) = try_parse_direction(&args[0], false)? {
                Ok((direction, PatternSpec::Default))
            } else {
                let pattern =
                    parse_pattern(args.get(0).expect("single pattern argument"), expectation)
                        .await?;
                enforce_pattern_extension(&pattern)?;
                Ok((StripDirection::Both, pattern))
            }
        }
        2 => {
            let direction = match try_parse_direction(&args[0], true)? {
                Some(dir) => dir,
                None => return Err(strip_error(&STRIP_ERROR_DIRECTION)),
            };
            let pattern =
                parse_pattern(args.get(1).expect("pattern after direction"), expectation).await?;
            enforce_pattern_extension(&pattern)?;
            Ok((direction, pattern))
        }
        _ => Err(strip_error(&STRIP_ERROR_ARG_COUNT)),
    }
}

fn enforce_pattern_extension(pattern: &PatternSpec<'_>) -> BuiltinResult<()> {
    let extended = match pattern {
        PatternSpec::Default => false,
        PatternSpec::Scalar(PatternRef::Default) => false,
        PatternSpec::Scalar(PatternRef::CustomText(text)) => text.chars().nth(1).is_some(),
        PatternSpec::Scalar(PatternRef::CustomChars(chars)) => chars.len() > 1,
        PatternSpec::PerElementStrings(_)
        | PatternSpec::PerElementChars(_)
        | PatternSpec::PerElementCell(_) => true,
    };
    if extended {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STRIP_CHARACTER_SET_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn try_parse_direction(value: &Value, strict: bool) -> BuiltinResult<Option<StripDirection>> {
    let token = match value {
        Value::String(text) => text.as_str().and_then(direction_token),
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().and_then(direction_token)
        }
        Value::CharArray(array)
            if array.rows <= 1 && array.shape.len() <= 2 && array.validate_layout().is_ok() =>
        {
            direction_char_token(array.data_owner().as_slice())
        }
        _ => None,
    };
    let direction = match token {
        Some(DirectionToken::Standard(direction)) => Some(direction),
        Some(DirectionToken::RunMatAlias(direction)) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &STRIP_DIRECTION_ALIAS_EXTENSION,
                BUILTIN_NAME,
            )?;
            Some(direction)
        }
        _ => {
            if strict {
                return Err(strip_error(&STRIP_ERROR_DIRECTION));
            }
            None
        }
    };
    Ok(direction)
}

fn direction_token(text: &str) -> Option<DirectionToken> {
    let text = text.trim();
    direction_candidates()
        .into_iter()
        .find_map(|(candidate, token)| text.eq_ignore_ascii_case(candidate).then_some(token))
}

fn direction_char_token(chars: &[Utf16CodeUnit]) -> Option<DirectionToken> {
    let mut start = 0;
    let mut end = chars.len();
    while start < end && chars[start].is_whitespace() {
        start += 1;
    }
    while end > start && chars[end - 1].is_whitespace() {
        end -= 1;
    }
    direction_candidates()
        .into_iter()
        .find_map(|(candidate, token)| {
            let mut actual = chars[start..end].iter().map(|unit| unit.raw());
            let matches = candidate.bytes().all(|expected| {
                actual.next().is_some_and(|unit| {
                    unit <= u8::MAX as u16 && (unit as u8).to_ascii_lowercase() == expected
                })
            });
            (matches && actual.next().is_none()).then_some(token)
        })
}

fn direction_candidates() -> [(&'static str, DirectionToken); 5] {
    [
        ("both", DirectionToken::Standard(StripDirection::Both)),
        ("left", DirectionToken::Standard(StripDirection::Left)),
        ("right", DirectionToken::Standard(StripDirection::Right)),
        ("leading", DirectionToken::RunMatAlias(StripDirection::Left)),
        (
            "trailing",
            DirectionToken::RunMatAlias(StripDirection::Right),
        ),
    ]
}

async fn parse_pattern<'a>(
    value: &'a Value,
    expectation: &PatternExpectation<'_>,
) -> BuiltinResult<PatternSpec<'a>> {
    let expected_len = expectation.len();
    match value {
        Value::String(text) => Ok(PatternSpec::Scalar(PatternRef::CustomText(
            text.as_str()
                .ok_or_else(|| strip_error(&STRIP_ERROR_CHARACTERS))?,
        ))),
        Value::StringArray(sa) => {
            if sa.data().iter().any(StringElement::is_missing) {
                return Err(strip_error(&STRIP_ERROR_CHARACTERS));
            }
            if sa.data().len() <= 1 {
                if let Some(first) = sa.data().first() {
                    Ok(PatternSpec::Scalar(PatternRef::CustomText(
                        first
                            .as_str()
                            .ok_or_else(|| strip_error(&STRIP_ERROR_CHARACTERS))?,
                    )))
                } else {
                    Ok(PatternSpec::Scalar(PatternRef::CustomChars(&[])))
                }
            } else if sa.data().len() == expected_len {
                if let Some(shape) = expectation.shape() {
                    if sa.shape.as_slice() != shape {
                        return Err(strip_error(&STRIP_ERROR_SIZE_MISMATCH));
                    }
                }
                Ok(PatternSpec::PerElementStrings(sa))
            } else {
                Err(strip_error(&STRIP_ERROR_SIZE_MISMATCH))
            }
        }
        Value::CharArray(ca) => {
            if ca.shape.len() > 2 || ca.rows.checked_mul(ca.cols) != Some(ca.data().len()) {
                return Err(strip_error(&STRIP_ERROR_CHARACTERS));
            }
            if ca.rows <= 1 {
                if ca.rows == 0 {
                    Ok(PatternSpec::Scalar(PatternRef::CustomChars(&[])))
                } else {
                    Ok(PatternSpec::Scalar(PatternRef::CustomChars(
                        ca.data_owner().as_slice(),
                    )))
                }
            } else if ca.rows == expected_len {
                Ok(PatternSpec::PerElementChars(ca))
            } else {
                Err(strip_error(&STRIP_ERROR_SIZE_MISMATCH))
            }
        }
        Value::Cell(cell) => parse_pattern_cell(cell, expectation).await,
        _ => Err(strip_error(&STRIP_ERROR_CHARACTERS)),
    }
}

async fn parse_pattern_cell<'a>(
    cell: &'a CellArray,
    expectation: &PatternExpectation<'_>,
) -> BuiltinResult<PatternSpec<'a>> {
    let len = cell.data().len();
    if len == 0 {
        return Ok(PatternSpec::Scalar(PatternRef::CustomChars(&[])));
    }
    if len == 1 {
        let pattern = pattern_ref_from_value(&cell.data()[0])
            .ok_or_else(|| strip_error(&STRIP_ERROR_CHARACTERS))?;
        return Ok(PatternSpec::Scalar(pattern));
    }
    if len != expectation.len() {
        return Err(strip_error(&STRIP_ERROR_SIZE_MISMATCH));
    }
    if let Some(shape) = expectation.shape() {
        if cell.shape.as_slice() != shape {
            return Err(strip_error(&STRIP_ERROR_SIZE_MISMATCH));
        }
    }
    for value in cell.data() {
        pattern_ref_from_value(value).ok_or_else(|| strip_error(&STRIP_ERROR_CHARACTERS))?;
    }
    Ok(PatternSpec::PerElementCell(cell))
}

fn pattern_ref_from_value(value: &Value) -> Option<PatternRef<'_>> {
    match value {
        Value::String(text) => text.as_str().map(PatternRef::CustomText),
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().map(PatternRef::CustomText)
        }
        Value::CharArray(array)
            if array.rows <= 1
                && array.shape.len() <= 2
                && array.rows.checked_mul(array.cols) == Some(array.data().len()) =>
        {
            Some(PatternRef::CustomChars(array.data_owner().as_slice()))
        }
        _ => None,
    }
}

fn strip_slice<'a>(text: &'a str, direction: StripDirection, pattern: PatternRef<'_>) -> &'a str {
    match pattern {
        PatternRef::Default => strip_slice_with_predicate(text, direction, |c| {
            crate::builtins::strings::common::trimmable_whitespace(
                c,
                !crate::compatibility::runmat_extensions_enabled(),
            )
        }),
        PatternRef::CustomText(chars) => {
            strip_slice_with_predicate(text, direction, |c| chars.contains(c))
        }
        PatternRef::CustomChars(chars) => strip_slice_with_predicate(text, direction, |c| {
            let mut encoded = [0_u16; 2];
            let encoded = c.encode_utf16(&mut encoded);
            chars.windows(encoded.len()).any(|window| {
                window
                    .iter()
                    .map(|unit| unit.raw())
                    .eq(encoded.iter().copied())
            })
        }),
    }
}

fn strip_slice_with_predicate<F>(text: &str, direction: StripDirection, predicate: F) -> &str
where
    F: Fn(char) -> bool,
{
    match direction {
        StripDirection::Left => text.trim_start_matches(predicate),
        StripDirection::Right => text.trim_end_matches(predicate),
        StripDirection::Both => text.trim_matches(predicate),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn run_strip(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(strip_builtin(
            None,
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_scalar_default() {
        let result = run_strip(Value::String("  RunMat  ".into()), Vec::new()).expect("strip");
        assert_eq!(result, Value::String("RunMat".into()));
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let significant = "  \u{00A0}RunMat\u{00A0}  ";
        assert_eq!(
            run_strip(Value::string(significant), Vec::new()).unwrap(),
            Value::string("\u{00A0}RunMat\u{00A0}"),
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_scalar_direction() {
        let result = run_strip(
            Value::String("...data".into()),
            vec![Value::String("left".into()), Value::String(".".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("data".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_scalar_custom_characters() {
        let result = run_strip(
            Value::String("00052".into()),
            vec![Value::String("left".into()), Value::String("0".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("52".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_scalar_pattern_only() {
        let result = run_strip(
            Value::String("xxaccelerationxx".into()),
            vec![Value::String("x".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("acceleration".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_empty_pattern_returns_original() {
        let result = run_strip(
            Value::String("abc".into()),
            vec![Value::String(String::new().into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("abc".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_supports_leading_synonym() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run_strip(
            Value::String("   data".into()),
            vec![Value::String("leading".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("data".into()));
        drop(_compat);
        let strict = run_strip(
            Value::String("   data".into()),
            vec![Value::String("leading".into())],
        )
        .expect_err("direction alias requires RunMat extension mode");
        assert_eq!(
            strict.identifier(),
            STRIP_DIRECTION_ALIAS_EXTENSION.error_identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_supports_trailing_synonym() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run_strip(
            Value::String("data   ".into()),
            vec![Value::String("trailing".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("data".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_array_per_element_characters() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let strings = StringArray::new(
            vec!["##ok##".into(), "--warn--".into(), "**fail**".into()],
            vec![3, 1],
        )
        .unwrap();
        let chars =
            CharArray::from_unicode_scalars(vec!['#', '#', '-', '-', '*', '*'], 3, 2).unwrap();
        let result = run_strip(
            Value::StringArray(strings),
            vec![Value::String("both".into()), Value::CharArray(chars)],
        )
        .expect("strip");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(
                    sa.data(),
                    vec![
                        String::from("ok"),
                        String::from("warn"),
                        String::from("fail")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
        drop(_compat);
        let strict = run_strip(
            Value::String("##ok##".into()),
            vec![Value::String("both".into()), Value::String("#-".into())],
        )
        .expect_err("multi-character set requires RunMat extension mode");
        assert_eq!(
            strict.identifier(),
            STRIP_CHARACTER_SET_EXTENSION.error_identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_array_cell_pattern_per_element() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let strings =
            StringArray::new(vec!["__pass__".into(), "--warn--".into()], vec![2, 1]).unwrap();
        let patterns = CellArray::new(
            vec![Value::String("_".into()), Value::String("-".into())],
            2,
            1,
        )
        .unwrap();
        let result =
            run_strip(Value::StringArray(strings), vec![Value::Cell(patterns)]).expect("strip");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data(), vec![String::from("pass"), String::from("warn")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_array_preserves_missing() {
        let strings = StringArray::from_elements_unbounded(
            vec!["   data   ".into(), runmat_value::StringElement::Missing],
            vec![2, 1],
        )
        .unwrap();
        let result = run_strip(Value::StringArray(strings), Vec::new()).expect("strip");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data()[0], "data");
                assert!(sa.data()[1].is_missing());
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_char_array_shrinks_width() {
        let source = "  cat  dog  ";
        let chars: Vec<char> = source.chars().collect();
        let array = CharArray::from_unicode_scalars(chars, 1, source.chars().count()).unwrap();
        let result = run_strip(Value::CharArray(array), Vec::new()).expect("strip");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 8);
                let expected = runmat_value::encode_utf16_code_units("cat  dog");
                assert_eq!(ca.data(), expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let row = CharArray::from_unicode_scalars_with_shape("  xy  ".chars().collect(), vec![6])
            .unwrap();
        let Value::CharArray(row) = run_strip(Value::CharArray(row), Vec::new()).unwrap() else {
            panic!("expected character row");
        };
        assert_eq!(row.shape.as_slice(), &[2]);
        assert_eq!(
            row.data(),
            runmat_value::encode_utf16_code_units("xy").as_slice()
        );

        let pages = CharArray::from_unicode_scalars_with_shape(
            [" a   ", " b   ", " cc  ", " d   "]
                .concat()
                .chars()
                .collect(),
            vec![2, 5, 2],
        )
        .unwrap();
        let Value::CharArray(pages) = run_strip(Value::CharArray(pages), Vec::new()).unwrap()
        else {
            panic!("expected character pages");
        };
        assert_eq!(pages.shape.as_slice(), &[2, 2, 2]);
        assert_eq!(
            pages.data(),
            runmat_value::encode_utf16_code_units("a b ccd ").as_slice()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_char_array_supports_trailing_direction() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let array = CharArray::new_row("gpu   ");
        let result = run_strip(
            Value::CharArray(array),
            vec![Value::String("trailing".into())],
        )
        .expect("strip");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 3);
                let expected = runmat_value::encode_utf16_code_units("gpu");
                assert_eq!(ca.data(), expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("  GPU ")),
                Value::String("   Accelerate".into()),
                Value::String("RunMat   ".into()),
            ],
            1,
            3,
        )
        .unwrap();
        let result = run_strip(Value::Cell(cell), Vec::new()).expect("strip");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 3);
                assert_eq!(
                    out.get(0, 0).unwrap(),
                    Value::CharArray(CharArray::new_row("GPU"))
                );
                assert_eq!(out.get(0, 1).unwrap(), Value::String("Accelerate".into()));
                assert_eq!(out.get(0, 2).unwrap(), Value::String("RunMat".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_preserves_missing_string() {
        let result = run_strip(
            Value::String(runmat_value::StringElement::Missing),
            Vec::new(),
        )
        .expect("strip");
        assert_eq!(result, Value::String(runmat_value::StringElement::Missing));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_invalid_input() {
        let err = run_strip(Value::Num(1.0), Vec::new()).unwrap_err();
        assert_eq!(err.to_string(), STRIP_ERROR_INVALID_INPUT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_invalid_pattern_type() {
        let err = run_strip(Value::String("abc".into()), vec![Value::Num(1.0)]).unwrap_err();
        assert_eq!(err.to_string(), STRIP_ERROR_CHARACTERS.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_invalid_direction() {
        let err = run_strip(
            Value::String("abc".into()),
            vec![Value::String("sideways".into()), Value::String("a".into())],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), STRIP_ERROR_DIRECTION.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_pattern_size_mismatch() {
        let strings = StringArray::new(vec!["one".into(), "two".into()], vec![2, 1]).unwrap();
        let pattern =
            StringArray::new(vec!["x".into(), "y".into(), "z".into()], vec![3, 1]).unwrap();
        let err = run_strip(
            Value::StringArray(strings),
            vec![Value::StringArray(pattern)],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), STRIP_ERROR_SIZE_MISMATCH.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_pattern_shape_mismatch() {
        let strings = StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap();
        let pattern = StringArray::new(vec!["x".into(), "y".into()], vec![2, 1]).unwrap();
        let err = run_strip(
            Value::StringArray(strings),
            vec![Value::StringArray(pattern)],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), STRIP_ERROR_SIZE_MISMATCH.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_cell_pattern_shape_mismatch() {
        let strings = StringArray::new(vec!["aa".into(), "bb".into()], vec![1, 2]).unwrap();
        let cell_pattern = CellArray::new(
            vec![Value::String("a".into()), Value::String("b".into())],
            2,
            1,
        )
        .unwrap();
        let err =
            run_strip(Value::StringArray(strings), vec![Value::Cell(cell_pattern)]).unwrap_err();
        assert_eq!(err.to_string(), STRIP_ERROR_SIZE_MISMATCH.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_too_many_arguments() {
        let err = run_strip(
            Value::String("abc".into()),
            vec![
                Value::String("both".into()),
                Value::String("a".into()),
                Value::String("b".into()),
            ],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), STRIP_ERROR_ARG_COUNT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn strip_gpu_tensor_errors() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let host_data = [1.0f64, 2.0];
        let host_shape = [2usize, 1usize];
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &host_data,
                shape: &host_shape,
            })
            .expect("upload");
        let err = run_strip(Value::GpuTensor(handle.clone()), Vec::new()).unwrap_err();
        assert_eq!(err.to_string(), STRIP_ERROR_INVALID_INPUT.message);
        handle
            .allocation_lease()
            .validate_live()
            .expect("borrowed input remains live");
        let physical = handle.allocation_lease().downgrade();
        handle.release().expect("release final fixture reference");
        assert!(
            physical.acquire().is_none(),
            "physical lease must expire after final release"
        );
    }

    #[test]
    fn strip_type_preserves_text() {
        use runmat_types::{
            CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueFact,
            ValueKindFact,
        };
        let input = ValueFact::scalar(ValueKindFact::String);
        let request = CallRequest {
            arguments: vec![input.clone()],
            literals: LiteralContext::new(Vec::new()),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let inferred =
            runmat_builtins::infer_catalog_call(&runmat_builtins::STRIP_CATALOG_ENTRY, &request);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs.len(), 1);
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::String);
        assert_eq!(inferred.outputs[0].shape, input.shape);
    }
}
