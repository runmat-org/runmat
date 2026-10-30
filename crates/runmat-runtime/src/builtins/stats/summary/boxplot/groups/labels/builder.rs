use super::text_copy::{TextCopy, TextCopyProgress};
use super::*;
use crate::builtins::common::grouping_keys::group_key_work_units;
use runmat_value::{LogicalArray, ObjectInstance, StringArray};

pub(super) enum LabelProgress {
    Work(u64),
    Complete(Vec<Option<GroupLabel>>),
}

enum Source<'a> {
    Text(&'a runmat_value::StringElement),
    Strings(&'a StringArray),
    Logical(&'a LogicalArray),
    Characters(&'a CharArray),
    Categorical(&'a ObjectInstance),
    Numeric(&'a Tensor),
    OwnedNumeric(Tensor),
    CellScalar(&'a CellArray),
    CellDiscard(&'a CellArray),
    CellFactors {
        cell: &'a CellArray,
        variables: Vec<Vec<Option<GroupLabel>>>,
        child: Option<Box<GroupLabelBuilder<'a>>>,
        factor: usize,
    },
    CellCombine {
        variables: Vec<Vec<Option<GroupLabel>>>,
        row: usize,
        factor: usize,
        atom: usize,
        parts: GroupLabel,
        missing: bool,
    },
    Done,
}

/// One semantic label decoder for host preflight and cooperative preparation.
/// Recursive cells are resumed a label/factor at a time instead of collecting
/// every nested factor in one uninterrupted pass.
pub(super) struct GroupLabelBuilder<'a> {
    source: Source<'a>,
    expected_len: usize,
    index: usize,
    output: Vec<Option<GroupLabel>>,
    pending_text: Option<TextCopy<'a>>,
}

impl<'a> GroupLabelBuilder<'a> {
    pub(super) fn new(value: &'a Value, expected_len: usize) -> BuiltinResult<Self> {
        let source = match value {
            Value::Cell(cell) if cell.data().len() == expected_len => Source::CellScalar(cell),
            Value::Cell(cell) => Source::CellFactors {
                cell,
                variables: Vec::new(),
                child: None,
                factor: 0,
            },
            Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
                crate::builtins::table::categorical_label_view(object).map_err(|err| {
                    invalid_with_source("boxplot: invalid categorical grouping variable", err)
                })?;
                Source::Categorical(object)
            }
            Value::String(text) if expected_len == 1 => Source::Text(text),
            Value::StringArray(array) if array.data().len() == expected_len => {
                Source::Strings(array)
            }
            Value::LogicalArray(array) if array.len() == expected_len => Source::Logical(array),
            Value::CharArray(array) if array.rows == expected_len => Source::Characters(array),
            Value::Tensor(tensor) => {
                validate_numeric_len(tensor, expected_len)?;
                Source::Numeric(tensor)
            }
            _ => {
                let tensor = tensor::value_into_tensor_for(NAME, value.clone()).map_err(|err| {
                    invalid(format!(
                        "boxplot: grouping variable must be numeric, text, cell, or categorical ({err})"
                    ))
                })?;
                validate_numeric_len(&tensor, expected_len)?;
                Source::OwnedNumeric(tensor)
            }
        };
        Ok(Self {
            source,
            expected_len,
            index: 0,
            output: Vec::new(),
            pending_text: None,
        })
    }

    pub(super) fn step(&mut self) -> BuiltinResult<LabelProgress> {
        loop {
            if let Some(copy) = self.pending_text.as_mut() {
                match copy.step()? {
                    TextCopyProgress::Work(units) => return Ok(LabelProgress::Work(units)),
                    TextCopyProgress::Complete(text) => {
                        self.pending_text = None;
                        push_label(&mut self.output, text.map(|text| vec![KeyAtom::Text(text)]))?;
                        self.index += 1;
                        return Ok(LabelProgress::Work(1));
                    }
                }
            }
            let source = std::mem::replace(&mut self.source, Source::Done);
            match source {
                Source::CellScalar(cell) => {
                    if self.index == cell.data().len() {
                        return Ok(LabelProgress::Complete(std::mem::take(&mut self.output)));
                    }
                    if let Some(copy) = scalar_text_copy(&cell.data()[self.index]) {
                        self.pending_text = Some(copy);
                        self.source = Source::CellScalar(cell);
                        continue;
                    }
                    match scalar_group_label(&cell.data()[self.index]) {
                        Ok(label) => {
                            let units = label.as_ref().map_or(1, |key| group_key_work_units(key));
                            push_label(&mut self.output, label)?;
                            self.index += 1;
                            self.source = Source::CellScalar(cell);
                            return Ok(LabelProgress::Work(units));
                        }
                        Err(_) => {
                            self.source = Source::CellDiscard(cell);
                        }
                    }
                }
                Source::CellDiscard(cell) => {
                    if let Some(label) = self.output.pop() {
                        let units = label.as_ref().map_or(1, |key| group_key_work_units(key));
                        self.source = Source::CellDiscard(cell);
                        return Ok(LabelProgress::Work(units));
                    }
                    self.index = 0;
                    self.source = Source::CellFactors {
                        cell,
                        variables: Vec::new(),
                        child: None,
                        factor: 0,
                    };
                }
                Source::CellFactors {
                    cell,
                    mut variables,
                    mut child,
                    mut factor,
                } => {
                    if let Some(current) = child.as_mut() {
                        match current.step()? {
                            LabelProgress::Work(units) => {
                                self.source = Source::CellFactors {
                                    cell,
                                    variables,
                                    child,
                                    factor,
                                };
                                return Ok(LabelProgress::Work(units));
                            }
                            LabelProgress::Complete(labels) => {
                                if labels.len() != self.expected_len {
                                    return Err(invalid("boxplot: grouping variables in a cell array must share the X length"));
                                }
                                if variables.len() == variables.capacity() {
                                    variables.try_reserve(1).map_err(|error| {
                                        allocation_failure(
                                            "boxplot: grouping factors exceed available memory",
                                            error,
                                        )
                                    })?;
                                }
                                variables.push(labels);
                                child = None;
                                factor += 1;
                                self.source = Source::CellFactors {
                                    cell,
                                    variables,
                                    child,
                                    factor,
                                };
                                return Ok(LabelProgress::Work(1));
                            }
                        }
                    }
                    if factor < cell.data().len() {
                        child = Some(Box::new(Self::new(
                            &cell.data()[factor],
                            self.expected_len,
                        )?));
                        self.source = Source::CellFactors {
                            cell,
                            variables,
                            child,
                            factor,
                        };
                        return Ok(LabelProgress::Work(1));
                    }
                    self.source = Source::CellCombine {
                        variables,
                        row: 0,
                        factor: 0,
                        atom: 0,
                        parts: Vec::new(),
                        missing: false,
                    };
                }
                Source::CellCombine {
                    variables,
                    mut row,
                    mut factor,
                    mut atom,
                    mut parts,
                    mut missing,
                } => {
                    if row == self.expected_len {
                        return Ok(LabelProgress::Complete(std::mem::take(&mut self.output)));
                    }
                    if factor < variables.len() {
                        let units = match &variables[factor][row] {
                            Some(label) if !missing => {
                                if let Some(component) = label.get(atom) {
                                    if parts.len() == parts.capacity() {
                                        parts.try_reserve(1).map_err(|error| {
                                            allocation_failure(
                                                "boxplot: grouping key exceeds available memory",
                                                error,
                                            )
                                        })?;
                                    }
                                    parts.push(component.clone());
                                    atom += 1;
                                    group_key_work_units(std::slice::from_ref(component))
                                } else {
                                    factor += 1;
                                    atom = 0;
                                    1
                                }
                            }
                            None => {
                                missing = true;
                                factor = variables.len();
                                1
                            }
                            Some(_) => {
                                factor = variables.len();
                                1
                            }
                        };
                        self.source = Source::CellCombine {
                            variables,
                            row,
                            factor,
                            atom,
                            parts,
                            missing,
                        };
                        return Ok(LabelProgress::Work(units));
                    }
                    push_label(&mut self.output, (!missing).then_some(parts))?;
                    row += 1;
                    self.source = Source::CellCombine {
                        variables,
                        row,
                        factor: 0,
                        atom: 0,
                        parts: Vec::new(),
                        missing: false,
                    };
                    return Ok(LabelProgress::Work(1));
                }
                Source::Done => {
                    return Err(internal("boxplot: label decoder resumed after completion"))
                }
                leaf => {
                    let leaf_len = match &leaf {
                        Source::Categorical(object) => {
                            crate::builtins::table::categorical_label_view(object)
                                .map_err(|err| {
                                    invalid_with_source(
                                        "boxplot: invalid categorical grouping variable",
                                        err,
                                    )
                                })?
                                .len()
                        }
                        _ => self.expected_len,
                    };
                    if self.index == leaf_len {
                        if leaf_len != self.expected_len {
                            return Err(invalid("boxplot: grouping variable length must match X"));
                        }
                        return Ok(LabelProgress::Complete(std::mem::take(&mut self.output)));
                    }
                    if let Some(copy) = leaf_text_copy(&leaf, self.index)? {
                        self.pending_text = Some(copy);
                        self.source = leaf;
                        continue;
                    }
                    let label = label_at(&leaf, self.index)?;
                    let units = label.as_ref().map_or(1, |key| group_key_work_units(key));
                    push_label(&mut self.output, label)?;
                    self.index += 1;
                    self.source = leaf;
                    return Ok(LabelProgress::Work(units));
                }
            }
        }
    }
}

fn scalar_text_copy(value: &Value) -> Option<TextCopy<'_>> {
    match value {
        Value::String(text) => text.as_str().map(TextCopy::utf8),
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().map(TextCopy::utf8)
        }
        Value::CharArray(array) if array.rows == 1 => Some(TextCopy::character_row(array, 0)),
        _ => None,
    }
}

fn leaf_text_copy<'a>(source: &Source<'a>, index: usize) -> BuiltinResult<Option<TextCopy<'a>>> {
    match source {
        Source::Text(text) => Ok(text.as_str().map(TextCopy::utf8)),
        Source::Strings(array) => Ok(array.data()[index].as_str().map(TextCopy::utf8)),
        Source::Characters(array) => Ok(Some(TextCopy::character_row(array, index))),
        Source::Categorical(object) => {
            let view = crate::builtins::table::categorical_label_view(object).map_err(|err| {
                invalid_with_source("boxplot: invalid categorical grouping variable", err)
            })?;
            view.observation_label_at(index)
                .map(|text| text.map(TextCopy::utf8))
                .map_err(|err| {
                    invalid_with_source("boxplot: invalid categorical grouping variable", err)
                })
        }
        _ => Ok(None),
    }
}

fn validate_numeric_len(tensor: &Tensor, expected_len: usize) -> BuiltinResult<()> {
    if tensor::tensor_element_len(tensor) != expected_len {
        return Err(invalid(
            "boxplot: grouping variable length must match the number of X elements",
        ));
    }
    Ok(())
}

fn label_at(source: &Source<'_>, index: usize) -> BuiltinResult<Option<GroupLabel>> {
    match source {
        Source::Logical(array) => Ok(Some(vec![KeyAtom::Logical(array.data[index] != 0)])),
        Source::Categorical(object) => {
            let view = crate::builtins::table::categorical_label_view(object).map_err(|err| {
                invalid_with_source("boxplot: invalid categorical grouping variable", err)
            })?;
            let label = view.observation_label_at(index).map_err(|err| {
                invalid_with_source("boxplot: invalid categorical grouping variable", err)
            })?;
            if label.is_some() {
                Err(internal(
                    "boxplot: categorical text bypassed the label copier",
                ))
            } else {
                Ok(None)
            }
        }
        Source::Numeric(tensor) => Ok(tensor.numeric_value_at(index).and_then(numeric_group_label)),
        Source::OwnedNumeric(tensor) => {
            Ok(tensor.numeric_value_at(index).and_then(numeric_group_label))
        }
        Source::Text(_) | Source::Strings(_) => Ok(None),
        Source::Characters(_)
        | Source::CellScalar(_)
        | Source::CellDiscard(_)
        | Source::CellFactors { .. }
        | Source::CellCombine { .. }
        | Source::Done => Err(internal("boxplot: invalid label decoder state")),
    }
}

fn push_label(
    output: &mut Vec<Option<GroupLabel>>,
    label: Option<GroupLabel>,
) -> BuiltinResult<()> {
    if output.len() == output.capacity() {
        output.try_reserve(1).map_err(|error| {
            allocation_failure("boxplot: grouping labels exceed available memory", error)
        })?;
    }
    output.push(label);
    Ok(())
}
