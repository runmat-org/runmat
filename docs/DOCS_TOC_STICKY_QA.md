# Docs TOC Sticky – Visual QA Checklist

This checklist records the 3-pass visual QA for the sticky "On this page" TOC on docs pages.

## Implementation summary

- **HeadingsNav**: Sticky applied on `aside` with `self-start sticky top-24 h-max max-h-[calc(100vh-7rem)] overflow-y-auto`.
- **Docs grid**: `lg:items-start` on the content grid in `app/docs/[...slug]/page.tsx`.
- **Global CSS**: `overflow-x: clip` on `html` and `body` in `globals.css` (replacing `overflow-x: hidden`).

## Pass 1 – Functional sticky checks

| Page | 1024px | 1280px | 1536px |
|------|--------|--------|--------|
| /docs/versioning | Pass | Pass | Pass |
| /docs/desktop-browser-guide | Pass | Pass | Pass |
| /docs/architecture | Pass | Pass | Pass |

For each page/breakpoint: scroll from top to ~25%, ~50%, ~75%, and near footer; confirm TOC remains pinned at `top-24` and does not drift.

## Pass 2 – Edge-case behavior

| Check | Notes |
|-------|--------|
| Long TOC: internal scroll | Aside has `max-h-[calc(100vh-7rem)] overflow-y-auto`; long lists scroll inside TOC. |
| Last TOC item reachable | Verify on a long page (e.g. /docs/architecture). |
| Sticky TOC vs nav | top-24 (96px) clears header; no overlap. |
| Light/dark theme | Confirm offset looks correct in both themes. |
| No horizontal-scroll regression | Confirm docs pages with wide code/tables do not show page-level horizontal scroll. |

## Pass 3 – Regression sweep

- Re-run Pass 1 on two previously problematic pages (e.g. versioning, architecture).
- Refresh browser and repeat checks to rule out transient layout artifacts.
- Navigate docs via sidebar links and repeat one full scroll check.

## Verification artifacts

- **Screenshots**: Optional – before/after at 25% and 75% scroll on at least 3 pages.
- **Checklist log**: This file (page, breakpoint, pass/fail outcome).
