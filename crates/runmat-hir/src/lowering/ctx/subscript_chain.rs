use super::*;

mod end;

enum AstSubscriptStep<'a> {
    Index(&'a [AstExpr], IndexKind),
    Member(&'a str),
    DynamicMember(&'a AstExpr),
    DottedInvoke(&'a str, &'a [AstExpr], Span),
}

impl LoweringCtx {
    pub(super) fn lower_subscript_chain(
        &mut self,
        expression: &AstExpr,
        requested_outputs: RequestedOutputCount,
        indexing_context: runmat_types::ObjectIndexingContext,
    ) -> Result<Option<HirExpr>, HirError> {
        let mut steps = Vec::new();
        let root = self.collect_subscript_chain(expression, &mut steps)?;
        let component_count = steps
            .iter()
            .map(|step| usize::from(matches!(step, AstSubscriptStep::DottedInvoke(..))) + 1)
            .sum::<usize>();
        let contextual_end = steps.iter().any(AstSubscriptStep::contains_end);
        if (component_count < 2 && !contextual_end)
            || self.is_proven_static_namespace_root(root, &steps)?
        {
            return Ok(None);
        }
        let last = steps.len() - 1;
        let mut lowered_steps = Vec::with_capacity(steps.len());
        for (position, step) in steps.into_iter().enumerate() {
            lowered_steps.push(match step {
                AstSubscriptStep::Index(indices, kind) => {
                    let result_context = if kind == IndexKind::Brace && position == last {
                        IndexResultContext::ReadCommaList
                    } else {
                        IndexResultContext::ReadSingle
                    };
                    crate::HirSubscriptStep::Index(if kind == IndexKind::Paren {
                        self.lower_parentheses_arguments(indices)?
                    } else {
                        self.lower_indexing_with_context(indices, kind, result_context)?
                    })
                }
                AstSubscriptStep::Member(member) => {
                    crate::HirSubscriptStep::Member(MemberName(member.to_owned()))
                }
                AstSubscriptStep::DynamicMember(member) => {
                    crate::HirSubscriptStep::DynamicMember(self.lower_expr_semantic(member)?)
                }
                AstSubscriptStep::DottedInvoke(member, indices, span) => {
                    let arguments = self.lower_dotted_invoke_arguments(indices)?;
                    let name = qualified_name(&[member.to_owned()]);
                    // A dotted step may select a method or invoke/index a field.
                    // A scoped function is only its semantic fallback, not proof
                    // that the runtime receiver will select that function.
                    self.hir_index.calls.push(CallResolution {
                        name: name.clone(),
                        callee: HirCallableRef::Unresolved(name),
                        kind: CallKind::Dynamic,
                        syntax: CallSyntax::DottedInvoke,
                        requested_outputs: if position == last {
                            requested_outputs
                        } else {
                            RequestedOutputCount::One
                        },
                        span,
                        source: Some(self.assembly.modules[self.module.0].source_id),
                    });
                    crate::HirSubscriptStep::DottedInvoke {
                        member: MemberName(member.to_owned()),
                        arguments,
                        semantic_fallback: self
                            .resolve_scoped_function_name(member)
                            .map(runmat_types::CallableIdentity::BoundFunction)
                            .or_else(|| {
                                self.external_function_names.get(member).map(|function| {
                                    runmat_types::CallableIdentity::ExternalFunction {
                                        function: *function,
                                        display_name: member.to_owned(),
                                    }
                                })
                            }),
                    }
                }
            });
        }
        Ok(Some(HirExpr {
            id: self.alloc_expr_id(),
            kind: HirExprKind::SubscriptChain(crate::HirSubscriptChain {
                root: Box::new(self.lower_expr_semantic(root)?),
                steps: lowered_steps,
                sequence_use: runmat_types::SequenceUse::from_requested_outputs(requested_outputs),
                context: indexing_context,
            }),
            span: expression.span(),
        }))
    }

    fn lower_dotted_invoke_arguments(
        &mut self,
        arguments: &[AstExpr],
    ) -> Result<Vec<crate::HirDottedInvokeArgument>, HirError> {
        use crate::HirDottedInvokeArgument as Arg;

        let mut lowered = Vec::new();
        for argument in arguments {
            if matches!(argument, AstExpr::Colon(_)) {
                lowered.push(Arg::Colon);
            } else if let Some(offset) = ast_end_relative_offset(argument) {
                lowered.push(Arg::End { offset });
            } else if matches!(argument, AstExpr::NameValueArg(..)) {
                lowered.extend(
                    self.lower_call_arguments(std::slice::from_ref(argument))?
                        .into_iter()
                        .map(Arg::Value),
                );
            } else {
                lowered.push(Arg::Value(
                    self.lower_call_argument(argument, RequestedOutputCount::One)?,
                ));
            }
        }
        Ok(lowered)
    }

    fn collect_subscript_chain<'a>(
        &self,
        expression: &'a AstExpr,
        steps: &mut Vec<AstSubscriptStep<'a>>,
    ) -> Result<&'a AstExpr, HirError> {
        Ok(match expression {
            AstExpr::Index(base, indices, _) => {
                let root = self.collect_subscript_chain(base, steps)?;
                steps.push(AstSubscriptStep::Index(indices, IndexKind::Paren));
                root
            }
            AstExpr::IndexCell(base, indices, _) => {
                let root = self.collect_subscript_chain(base, steps)?;
                steps.push(AstSubscriptStep::Index(indices, IndexKind::Brace));
                root
            }
            AstExpr::Member(base, member, _) => {
                let root = self.collect_subscript_chain(base, steps)?;
                steps.push(AstSubscriptStep::Member(member));
                root
            }
            AstExpr::MemberDynamic(base, member, _) => {
                let root = self.collect_subscript_chain(base, steps)?;
                steps.push(AstSubscriptStep::DynamicMember(member));
                root
            }
            AstExpr::DottedInvoke(base, member, indices, span)
                if !self.dotted_invoke_is_static_boundary(base, member)? =>
            {
                let root = self.collect_subscript_chain(base, steps)?;
                steps.push(AstSubscriptStep::DottedInvoke(member, indices, *span));
                root
            }
            _ => expression,
        })
    }

    fn dotted_invoke_is_static_boundary(
        &self,
        base: &AstExpr,
        _member: &str,
    ) -> Result<bool, HirError> {
        if matches!(base, AstExpr::MetaClass(_, _)) {
            return Ok(true);
        }
        // A wholly unbound dotted base retains the language's qualified-call
        // interpretation. Workspace/external object bindings have already
        // been materialized in the lowering context and therefore enter the
        // typed object path instead.
        Ok(self.unbound_qualified_member_base(base).is_some())
    }

    fn is_proven_static_namespace_root(
        &self,
        root: &AstExpr,
        steps: &[AstSubscriptStep<'_>],
    ) -> Result<bool, HirError> {
        let AstExpr::Ident(name, span) = root else {
            return Ok(false);
        };
        if self.lookup_binding(name).is_some() {
            return Ok(false);
        }
        if self.class_id_for_name(name).is_some()
            || self.external_class_declaration(name).is_some()
            || self
                .resolve_imported_class_name_target(name, *span)?
                .is_some()
        {
            return Ok(true);
        }
        // A package root is not a value binding. Recognize an actual class
        // declaration along the static member prefix, not merely the first
        // identifier, so pkg.Class.Property retains class-reference lowering.
        // Calls, indexing and dynamic members end that namespace prefix.
        let mut qualified = name.clone();
        for step in steps {
            let AstSubscriptStep::Member(member) = step else {
                break;
            };
            qualified.push('.');
            qualified.push_str(member);
            if self.class_id_for_name(&qualified).is_some()
                || self.external_class_declaration(&qualified).is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl AstSubscriptStep<'_> {
    fn contains_end(&self) -> bool {
        match self {
            Self::Index(indices, _) | Self::DottedInvoke(_, indices, _) => {
                indices.iter().any(end::expr_references_end)
            }
            Self::DynamicMember(member) => end::expr_references_end(member),
            Self::Member(_) => false,
        }
    }
}
