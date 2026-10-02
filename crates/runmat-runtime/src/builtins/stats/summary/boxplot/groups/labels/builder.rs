use super::*;
use crate::builtins::common::grouping_keys::group_key_work_units;
use runmat_value::{LogicalArray, ObjectInstance, StringArray};

pub(super) enum LabelProgress {
    Work(u64),
    Complete(Vec<Option<GroupLabel>>),
}

enum Source<'a> {
    Text(&'a str),
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
}

impl<'a> GroupLabelBuilder<'a> {
    pub(super) fn new(value: &'a Value, expected_len: usize) -> BuiltinResult<Self> {
        let source = match value {
            Value::Cell(cell) if cell.data.len() == expected_len => Source::CellScalar(cell),
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
            Value::StringArray(array) if array.data.len() == expected_len => Source::Strings(array),
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
        })
    }

    pub(super) fn step(&mut self) -> BuiltinResult<LabelProgress> {
        loop {
            let source = std::mem::replace(&mut self.source, Source::Done);
            match source {
                Source::CellScalar(cell) => {
                    if self.index == cell.data.len() {
                        return Ok(LabelProgress::Complete(std::mem::take(&mut self.output)));
                    }
                    match scalar_group_label(&cell.data[self.index]) {
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
                                    variables.try_reserve(1).map_err(|_| {
                                        invalid("boxplot: grouping factors exceed available memory")
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
                    if factor < cell.data.len() {
                        child = Some(Box::new(Self::new(&cell.data[factor], self.expected_len)?));
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
                                        parts.try_reserve(1).map_err(|_| {
                                            invalid(
                                                "boxplot: grouping key exceeds available memory",
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
        Source::Text(text) => Ok(Some(vec![KeyAtom::Text((*text).to_string())])),
        Source::Strings(array) => {
            let text = &array.data[index];
            Ok(missing_text_label(text).then(|| vec![KeyAtom::Text(text.clone())]))
        }
        Source::Logical(array) => Ok(Some(vec![KeyAtom::Logical(array.data[index] != 0)])),
        Source::Characters(array) => {
            let text = char_row(array, index);
            Ok(missing_text_label(&text).then_some(vec![KeyAtom::Text(text)]))
        }
        Source::Categorical(object) => {
            let view = crate::builtins::table::categorical_label_view(object).map_err(|err| {
                invalid_with_source("boxplot: invalid categorical grouping variable", err)
            })?;
            view.observation_label_at(index)
                .map(|label| {
                    label
                        .filter(|text| missing_text_label(text))
                        .map(|text| vec![KeyAtom::Text(text.to_string())])
                })
                .map_err(|err| {
                    invalid_with_source("boxplot: invalid categorical grouping variable", err)
                })
        }
        Source::Numeric(tensor) => Ok(tensor.numeric_value_at(index).and_then(numeric_group_label)),
        Source::OwnedNumeric(tensor) => {
            Ok(tensor.numeric_value_at(index).and_then(numeric_group_label))
        }
        Source::CellScalar(_)
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
        output
            .try_reserve(1)
            .map_err(|_| invalid("boxplot: grouping labels exceed available memory"))?;
    }
    output.push(label);
    Ok(())
}
