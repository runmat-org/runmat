//! Unfold and fold of a sheet: the flat pattern in place of the bent sheet, and the rebuild
//! that bends it again.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn unfold(
        &mut self,
        i: usize,
        id: &FeatureId,
        fixed_face: &Reference,
        bends: &[Reference],
        simple: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let h = self.face_handle(i, id, "fixed_face", fixed_face)?;
        let body = h.body as usize;
        if !bends.is_empty() {
            return self.unfold_some(i, id, &h, bends, simple);
        }
        // A folded plate's body can lose its body name; its faces answer to the base flange.
        let Some((maker, _, sec, _)) = self.sheet_of_face(&h) else {
            return Err(self.error(
                id,
                Some("fixed_face"),
                DiagCode::RefKindMismatch,
                "the fixed face is not on a sheet metal body".into(),
                Vec::new(),
            ));
        };
        // A bent sheet's faces come from its base flange and the unfolds and folds since. A
        // plate's flat pattern develops its bends, edge flanges and hems, and a fold replays
        // them.
        if sec.is_some() {
            let rebuilders = self.sheet_rebuilders(i, &maker);
            self.only_made_by(id, body, &rebuilders, "unfold")?;
        } else if self.st.bodies[body].names.faces.iter().flatten().any(|n| {
            matches!(
                self.doc.feature(n.maker()).map(|f| &f.kind),
                Some(FeatureKind::SheetMetal(SheetMetalFeature::MiterFlange { .. }))
            )
        }) {
            return Err(self.not_lowered(
                id,
                "unfolding a plate with a miter flange (its flat pattern develops edge flanges and hems)",
            ));
        }
        // The flat pattern's strip, in place of the bent sheet.
        let flat = SheetMetalFeature::FlatPattern {
            fixed_face: fixed_face.clone(),
            grain: None,
        };
        let compiled = self.sheet_metal(i, id, &flat)?;
        if let Some(FeatureRecord::FlatPattern { in_place, .. }) = self.st.records.get_mut(id) {
            *in_place = true;
        }
        Ok(self.in_place(compiled, body))
    }

    pub(super) fn fold(
        &mut self,
        i: usize,
        id: &FeatureId,
        bends: &[Reference],
        simple: &NamingContext,
    ) -> Result<Compiled, Halt> {
        // The latest unfold before this feature, and the sheet it flattened.
        let unfold = self.doc.features[..i].iter().rev().find(|f| {
            matches!(
                f.kind,
                FeatureKind::SheetMetal(SheetMetalFeature::Unfold { .. })
            ) && (self.st.records.contains_key(&f.id)
                || self
                    .st
                    .bodies
                    .iter()
                    .any(|b| b.names.faces.iter().flatten().any(|n| n.maker() == &f.id)))
        });
        let Some(unfold) = unfold.map(|f| f.id.clone()) else {
            return Err(self.error(
                id,
                None,
                DiagCode::RefKindMismatch,
                "there is no unfolded sheet to fold".into(),
                Vec::new(),
            ));
        };
        // The sheet: the body the unfold, or a fold since, last rebuilt.
        let mut rebuilt = vec![unfold.clone()];
        let from = self.doc.feature_index(&unfold).unwrap_or(i);
        for f in &self.doc.features[from..i] {
            if matches!(
                f.kind,
                FeatureKind::SheetMetal(SheetMetalFeature::Fold { .. })
            ) && !rebuilt.contains(&f.id)
            {
                rebuilt.push(f.id.clone());
            }
        }
        let body = self.st.bodies.iter().position(|b| {
            b.names
                .faces
                .iter()
                .flatten()
                .any(|n| rebuilt.contains(n.maker()))
        });
        let Some(body) = body else {
            return Err(self.error(
                id,
                None,
                DiagCode::RefKindMismatch,
                format!("the sheet `{unfold}` unfolded is gone"),
                Vec::new(),
            ));
        };
        // The sheet: the body's own record, else the base flange the unfold's fixed
        // face was named on (a rebuilt body can lose its body name).
        let from_face = match self.doc.feature(&unfold).map(|f| &f.kind) {
            Some(FeatureKind::SheetMetal(SheetMetalFeature::Unfold {
                fixed_face: Reference::Name { name, .. },
                ..
            })) => Some(name.maker().clone()),
            _ => None,
        };
        let maker = self
            .sheet_of(body)
            .map(|s| s.0)
            .or(from_face)
            .filter(|m| matches!(self.st.records.get(m), Some(FeatureRecord::Sheet { .. })));
        let Some(maker) = maker else {
            return Err(self.not_lowered(id, "folding this sheet"));
        };
        // A partial fold of a plate rebuilds it from its outline with only some bends, which
        // has no place for its flanges' flats.
        if !bends.is_empty() && !self.plate_features_replayed(&maker).is_empty() {
            return Err(self.not_lowered(
                id,
                "folding some bends of a plate with edge flanges, hems, corner features or forms (fold them all: leave `bends` empty)",
            ));
        }
        // A partial fold of a bent sheet: the bends left flat stay flat.
        if !bends.is_empty() {
            if let Some(FeatureRecord::Sheet {
                section: Some(sec), ..
            }) = self.st.records.get(&maker).cloned()
            {
                let keys = Self::strip_bend_keys(&sec, bends).ok_or_else(|| {
                    self.error(
                        id,
                        Some("bends"),
                        DiagCode::RefKindMismatch,
                        "each entry of `bends` names a bend face of the sheet".into(),
                        Vec::new(),
                    )
                })?;
                return self.strip_some(i, id, body, &maker, None, &[], &keys, simple);
            }
        }
        // Faces the rebuild kept may still carry the base flange's names.
        rebuilt.push(maker.clone());
        self.only_made_by(id, body, &rebuilt, "fold")?;
        let Some(base) = self.doc.feature(&maker).cloned() else {
            return Err(self.not_lowered(id, "folding this sheet"));
        };
        let FeatureKind::SheetMetal(base_kind) = &base.kind else {
            return Err(self.not_lowered(id, "folding this sheet"));
        };
        // Step one rebuilds the base flange in place with the base flange's tools;
        // its faces answer to the base flange's names again, so references to the
        // bent sheet's faces resolve after the fold.
        let rebuilt = self.sheet_metal(i, &maker, base_kind)?;
        let Compiled::Op {
            op: rebuilt,
            operands,
            ..
        } = self.in_place(rebuilt, body)
        else {
            return Err(self.not_lowered(id, "folding this sheet"));
        };
        let mut steps = vec![Step {
            op: rebuilt,
            operands,
            naming: simple.at_step(None, vec![Restore::all(&maker)]),
        }];
        // Then each sketched bend and jog that stays bent after this fold, and the plate's
        // flanges, hems, corner features and forms, replayed in feature order: their faces
        // answer to their names again.
        let mut flat = self.flattened_bends(i, &maker);
        if bends.is_empty() {
            flat.clear();
        } else {
            let folded = self.plate_bend_ids(id, &maker, bends)?;
            let folded = self.with_parents(folded);
            flat.retain(|b| !folded.contains(b));
        }
        steps.extend(self.replay_plate(&maker, &flat, body, simple));
        Ok(Compiled::Steps(steps))
    }
}
