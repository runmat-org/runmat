import type { DisplayCategory } from './builtin-utils';

// Color mapping by category prefix
// Categories with the same prefix share the same color
const PREFIX_COLORS: Record<string, string> = {
  'array': '#1b365e',        // Admiral
  'cells': '#1b5e36',        // Seaweed
  'containers': '#5e361b',   // Cinnamon
  'io': '#1b515e',           // Deep Teal
  'math': '#1b1b5e',         // Indigo
  'logical': '#365e1b',      // Moss
  'stats': '#511b5e',        // Plum
  'strings': '#1b5e51',      // Teal
  'structs': '#5e511b',      // Olive Mustard
  'image': '#361b5e',        // Violet
  'acceleration': '#1b5e1b', // Evergreen
  'general': '#5e1b36',      // Mulberry
  'introspection': '#5e1b51', // Magenta Deep
  'diagnostics': '#5e1b1b',  // Cranberry
  'timing': '#515e1b',       // Lime Olive
};

// Default color for unknown categories
const DEFAULT_COLOR = '#6366F1'; // Indigo

// Get color for a category (uses prefix to determine color)
export function getCategoryColor(category: DisplayCategory): string {
  const prefix = category.split('/')[0];
  return PREFIX_COLORS[prefix] || DEFAULT_COLOR;
}

// Legacy export for backward compatibility (returns a function)
export const CATEGORY_COLORS = new Proxy({} as Record<DisplayCategory, string>, {
  get: (_, prop: string) => getCategoryColor(prop),
});

