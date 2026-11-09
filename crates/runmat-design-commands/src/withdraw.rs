//! Taking commands out of a catalog: a command with no applier behind it is left out of the
//! commands, the rows, the S box, the mode actions and the right-click menus, so no button
//! offers it. A menu left with no entries, and a group left with no items, go too.

use crate::{Catalog, RowItem};

fn keep_items(items: &mut Vec<RowItem>, out: &[&str]) {
    items.retain_mut(|item| match item {
        RowItem::Command { command } => !out.contains(&command.as_str()),
        RowItem::Menu { entries, .. } => {
            entries.retain(|e| !out.contains(&e.command.as_str()));
            !entries.is_empty()
        }
    });
}

impl Catalog {
    /// Removes the commands `out` names everywhere they appear.
    pub fn withdraw(&mut self, out: &[&str]) {
        self.commands.retain(|c| !out.contains(&c.id.as_str()));
        let keep = |ids: &mut Vec<String>| ids.retain(|id| !out.contains(&id.as_str()));
        for rows in &mut self.rows {
            for tab in &mut rows.tabs {
                for group in &mut tab.groups {
                    keep_items(&mut group.items, out);
                }
                tab.groups.retain(|g| !g.items.is_empty());
            }
            keep_items(&mut rows.view_cluster, out);
            keep_items(&mut rows.document_bar, out);
            keep(&mut rows.shortcut_box);
            keep(&mut rows.mode_actions);
        }
        for menu in &mut self.context_menus {
            keep(&mut menu.quick);
            keep(&mut menu.agent);
            for group in &mut menu.groups {
                keep_items(group, out);
            }
            menu.groups.retain(|g| !g.is_empty());
        }
    }
}
