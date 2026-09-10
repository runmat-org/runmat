use super::*;

mod end;

enum AstSubscriptStep<'a> {
    Index(&'a [AstExpr], IndexKind),
    Member(&'a str),
    DynamicMember(&'a AstExpr),
    DottedInvoke(&'a str, &'a [AstExpr]),
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
        if (component_count < 2 && !contextual_end) || self.is_proven_static_namespace_root(root)? {
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
                    crate::HirSubscriptStep::Index(self.lower_indexing_with_context(
                        indices,
                        kind,
                        result_context,
                    )?)
                }
                AstSubscriptStep::Member(member) => {
                    crate::HirSubscriptStep::Member(MemberName(member.to_owned()))
                }
                AstSubscriptStep::DynamicMember(member) => {
                    crate::HirSubscriptStep::DynamicMember(self.lower_expr_semantic(member)?)
                }
                AstSubscriptStep::DottedInvoke(member, indices) => {
                    crate::HirSubscriptStep::DottedInvoke {
                        member: MemberName(member.to_owned()),
                        indexing: self.lower_indexing_with_context(
                            indices,
                            IndexKind::Paren,
                            IndexResultContext::ReadSingle,
                        )?,
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
            AstExpr::DottedInvoke(base, member, indices, _)
                if !self.dotted_invoke_is_static_boundary(base, member)? =>
            {
                let root = self.collect_subscript_chain(base, steps)?;
                steps.push(AstSubscriptStep::DottedInvoke(member, indices));
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

    fn is_proven_static_namespace_root(&self, root: &AstExpr) -> Result<bool, HirError> {
        let AstExpr::Ident(name, span) = root else {
            return Ok(false);
        };
        if self.lookup_binding(name).is_some() {
            return Ok(false);
        }
        Ok(self.class_id_for_name(name).is_some()
            || self.external_class_declaration(name).is_some()
            || self
                .resolve_imported_class_name_target(name, *span)?
                .is_some())
    }
}

impl AstSubscriptStep<'_> {
    fn contains_end(&self) -> bool {
        match self {
            Self::Index(indices, _) | Self::DottedInvoke(_, indices) => {
                indices.iter().any(end::expr_references_end)
            }
            Self::DynamicMember(member) => end::expr_references_end(member),
            Self::Member(_) => false,
        }
    }
}
