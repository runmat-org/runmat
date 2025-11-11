# Typography Audit - Font Size Usage Across Website

## Summary
This document catalogs all font size usage across the website to identify inconsistencies and guide standardization.

## Font Size Usage by Component Type

### Badges
- **text-xs**: Used in ElementsOfMatlabGrid.tsx (lines 407, 414, 636, 770, 812)
- **text-sm**: Used in app/page.tsx (line 78)
- **text-lg**: Used in ElementsOfMatlabGrid.tsx ListView (lines 717, 724) - INCONSISTENT

**Issue**: Badges use three different sizes. Should standardize to `text-xs`.

### Buttons
- **text-sm**: Default in button.tsx component
- **text-lg**: Used in app/page.tsx hero CTAs (lines 91, 96, 99, 443, 446, 465, 468)

**Issue**: Hero buttons use `text-lg` which is acceptable for prominent CTAs, but should be documented.

### Navigation
- **text-sm**: Used in Navigation.tsx mobile menu (lines 126, 134, 142, 150, 158, 166, 174, 182, 190, 219, 220)
- **text-lg**: Used in Navigation.tsx logo text (lines 29, 68)
- **text-xl**: Used in Navigation.tsx logo text responsive (line 29)

**Issue**: Logo text uses larger sizes which is acceptable for branding, but navigation links should be `text-sm`.

### Headings

#### H1
- **text-3xl sm:text-5xl md:text-6xl lg:text-7xl**: app/page.tsx hero (line 81)
- **text-4xl**: MarkdownRenderer.tsx (line 35)
- **text-4xl sm:text-5xl md:text-6xl lg:text-7xl**: app/benchmarks/page.tsx (line 148)
- **text-3xl md:text-4xl**: app/docs/elements-of-matlab/page.tsx (line 16)

**Issue**: H1 sizes vary. Should standardize to `text-4xl sm:text-5xl md:text-6xl lg:text-7xl` for page titles.

#### H2
- **text-3xl**: MarkdownRenderer.tsx (line 41)
- **text-3xl sm:text-3xl md:text-6xl**: app/page.tsx (line 115) - INCONSISTENT (no sm breakpoint change)
- **text-3xl sm:text-4xl md:text-5xl lg:text-6xl**: app/page.tsx (line 188)
- **text-3xl sm:text-4xl md:text-5xl**: app/page.tsx (lines 320, 390)
- **text-3xl sm:text-3xl md:text-6xl**: app/page.tsx (line 250)
- **text-3xl sm:text-3xl md:text-5xl**: app/page.tsx (line 433)
- **text-3xl sm:text-3xl md:text-6xl**: app/page.tsx (line 458)
- **text-xl**: ElementsOfMatlabGrid.tsx GridView (line 247)
- **text-lg**: ElementsOfMatlabGrid.tsx ListView and TagsView (lines 287, 535)

**Issue**: H2 sizes vary significantly. Should standardize to `text-3xl sm:text-4xl md:text-5xl`.

#### H3
- **text-2xl**: MarkdownRenderer.tsx (line 51)
- **text-xl**: app/benchmarks/page.tsx (line 166)
- **text-lg**: app/benchmarks/page.tsx (line 202), app/benchmarks/[slug]/page.tsx (line 201)

**Issue**: H3 sizes vary. Should standardize to `text-2xl sm:text-3xl`.

#### H4
- **text-xl**: MarkdownRenderer.tsx (line 61)

#### H5
- **text-lg**: MarkdownRenderer.tsx (line 71)

#### H6
- **text-base**: MarkdownRenderer.tsx (line 81)

### Body Text
- **text-sm**: Used extensively for descriptions, captions, metadata
- **text-base**: Default body text (should be primary)
- **text-lg**: Used in some descriptions (app/page.tsx lines 323, 391)
- **text-xl**: Used in hero descriptions (app/page.tsx lines 84, 117, 253, 436, 461), app/benchmarks/page.tsx (line 151)

**Issue**: Body text sizes inconsistent. Should use `text-base` for primary, `text-sm` for secondary.

### Cards
- **text-sm**: CardDescription component (correct)
- **text-base**: ElementsOfMatlabGrid.tsx card titles (line 394, 440, 661)
- **text-lg**: ElementsOfMatlabGrid.tsx ListView items (lines 705, 708, 717, 724)

**Issue**: Card text sizes vary. Should standardize.

### Tables
- **text-xs sm:text-sm**: MarkdownRenderer.tsx (lines 137, 140) - CORRECT

### Footer
- **text-sm**: Used in Footer.tsx (lines 39, 63, 77)
- **text-base**: Used in Footer.tsx (line 35)
- **text-lg**: Used in Footer.tsx (line 35) - responsive

**Issue**: Footer text sizes vary. Should standardize to `text-sm`.

### Other Components
- **text-sm**: Search inputs, form labels, tooltips
- **text-base**: Some card titles, list items

## Files Requiring Updates

### High Priority
1. `website/components/ui/badge.tsx` - Ensure default is `text-xs`
2. `website/components/Navigation.tsx` - Standardize navigation links to `text-sm`
3. `website/components/Footer.tsx` - Standardize to `text-sm`
4. `website/components/ElementsOfMatlabGrid.tsx` - Fix badge sizes and list item text

### Medium Priority
5. `website/app/page.tsx` - Standardize hero and section headings
6. `website/app/benchmarks/page.tsx` - Standardize headings
7. `website/app/docs/page.tsx` - Standardize card text
8. `website/components/MarkdownRenderer.tsx` - Verify and add responsive breakpoints

### Low Priority
9. All other page files with inconsistent font sizes

## Mobile Responsiveness Issues

### Current Issues
1. Some headings lack responsive breakpoints (e.g., `text-3xl sm:text-3xl` - no change)
2. Body text may be too small on mobile (some uses `text-sm` which is 14px)
3. Need to verify minimum 16px for body text on mobile

### Recommendations
1. Ensure all headings have proper responsive breakpoints
2. Use `text-base` (16px) as minimum for body text on mobile
3. Test all breakpoints: 320px, 375px, 414px, 768px, 1024px, 1280px

## Standard Typography Scale (Proposed)

### Headings
- H1: `text-4xl sm:text-5xl md:text-6xl lg:text-7xl` (page titles)
- H1 (section): `text-3xl sm:text-4xl md:text-5xl` (section titles)
- H2: `text-3xl sm:text-4xl md:text-5xl` (section titles)
- H3: `text-2xl sm:text-3xl` (subsection titles)
- H4: `text-xl sm:text-2xl` (sub-subsection titles)
- H5: `text-lg sm:text-xl` (minor headings)
- H6: `text-base sm:text-lg` (smallest headings)

### Body Text
- Primary: `text-base` (16px - mobile friendly)
- Secondary: `text-sm` (14px - descriptions, captions)
- Large: `text-lg sm:text-xl` (hero descriptions, lead paragraphs)

### UI Components
- Buttons: `text-sm` (default)
- Navigation: `text-sm`
- Badges: `text-xs`
- Tables: `text-xs sm:text-sm`
- Cards: `text-sm` for descriptions
- Footer: `text-sm`

