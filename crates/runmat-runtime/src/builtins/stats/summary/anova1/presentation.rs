use super::*;
pub(super) fn admit(display: bool) -> BuiltinResult<()> {
    if display && !cfg!(feature = "plot-core") {
        return Err(invalid(
            NAME,
            "display='on' requires plot-core; use display='off' for numerical outputs",
        ));
    }
    Ok(())
}
pub(super) async fn show(
    data: &groups::Groups,
    outputs: &[Value],
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<()> {
    let Value::Cell(table) = &outputs[1] else {
        unreachable!("ANOVA cell table")
    };
    for row in 0..4 {
        let line = (0..6)
            .map(|column| match &table.data()[row * 6 + column] {
                Value::Num(value) => Ok(format!("{value:.6}")),
                Value::CharArray(value) => runmat_value::decode_utf16_code_units(value.data())
                    .map_err(|source| {
                        crate::build_runtime_error("anova1: invalid table text")
                            .with_builtin(NAME)
                            .with_source(source)
                            .build()
                    }),
                _ => Ok(String::new()),
            })
            .collect::<BuiltinResult<Vec<_>>>()?
            .join("\t");
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout, line)?;
        cooperation.completed(6).await?;
    }
    #[cfg(feature = "plot-core")]
    {
        let rows = data.values.iter().map(Vec::len).max().unwrap_or(0);
        let columns = data.values.len();
        let len = rows
            .checked_mul(columns)
            .ok_or_else(|| invalid(NAME, "boxplot geometry overflow"))?;
        let mut values = vec![f64::NAN; len];
        for (column, group) in data.values.iter().enumerate() {
            for (chunk, part) in group.chunks(1024).enumerate() {
                values[column * rows + chunk * 1024..column * rows + chunk * 1024 + part.len()]
                    .copy_from_slice(part);
                cooperation.completed(part.len()).await?;
            }
        }
        let labels = Value::Cell(
            CellArray::new(
                data.names
                    .iter()
                    .map(|name| Value::CharArray(CharArray::new_row(name)))
                    .collect(),
                1,
                columns,
            )
            .map_err(|error| invalid(NAME, error))?,
        );
        let args = vec![
            Value::Tensor(
                Tensor::new(values, vec![rows, columns]).map_err(|error| invalid(NAME, error))?,
            ),
            Value::from("Labels"),
            labels,
            Value::from("Notch"),
            Value::from("on"),
        ];
        let args = match runtime {
            Some(runtime) => crate::execution::RootedValues::in_context(args, runtime)?,
            None => crate::execution::RootedValues::new(args)
                .map_err(|error| error.into_runtime_error())?,
        };
        // Reuse the existing lazy, exact continuation owner: constructing a
        // large boxplot future must not enlarge ANOVA's caller poll frame.
        match runtime {
            Some(runtime) => {
                crate::execution::rooted_continuation::admit_domain_continuation_with(
                    runtime,
                    move || async move {
                        super::super::boxplot::boxplot_in_runtime(Some(runtime), args.into())
                            .await?;
                        Ok(())
                    },
                )?
                .await?
            }
            None => {
                crate::execution::rooted_continuation::admit_optional_hostcall_with(
                    None,
                    |source| {
                        crate::context::host_allocation::host_allocation_runtime_error(
                            None,
                            "RunMat:anova1:Continuation",
                            "ANOVA plot continuation admission failed",
                            source,
                        )
                    },
                    move || async move {
                        super::super::boxplot::boxplot_in_runtime(None, args.into()).await?;
                        Ok::<(), crate::RuntimeError>(())
                    },
                )?
                .await?
            }
        }
    }
    #[cfg(not(feature = "plot-core"))]
    let _ = (data, runtime);
    Ok(())
}
