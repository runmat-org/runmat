// Single source of truth for formatting JSON category paths to display names
// The actual category names come directly from builtins.json

export type DisplayCategory = string; // JSON category path (e.g., "array/creation")

// Format JSON category path to display-friendly name
// Example: "array/creation" -> "Array Creation"
// Example: "io" -> "I/O" (when prefix only)
export function formatCategoryName(categoryPath: string): string {
  const parts = categoryPath.split('/');
  
  // If it's just a prefix (no subcategory), format it specially
  if (parts.length === 1) {
    if (parts[0] === 'io') return 'I/O';
    if (parts[0] === 'gpu') return 'GPU';
    return parts[0].charAt(0).toUpperCase() + parts[0].slice(1);
  }
  
  // Full path - format each part
  return parts
    .map(part => {
      // Handle special cases
      if (part === 'io') return 'I/O';
      if (part === 'gpu') return 'GPU';
      
      // Capitalize first letter
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(' ');
}

// Group categories: if a prefix has multiple subcategories, group them under the prefix
// If a prefix has only one subcategory, keep the full category path
export function groupCategoriesByPrefix(categories: DisplayCategory[]): Map<DisplayCategory, DisplayCategory[]> {
  const prefixGroups = new Map<string, DisplayCategory[]>();
  
  // Group categories by prefix
  for (const category of categories) {
    const prefix = category.split('/')[0];
    if (!prefixGroups.has(prefix)) {
      prefixGroups.set(prefix, []);
    }
    prefixGroups.get(prefix)!.push(category);
  }
  
  // Create the final grouping
  const result = new Map<DisplayCategory, DisplayCategory[]>();
  
  for (const [prefix, subcategories] of prefixGroups.entries()) {
    if (subcategories.length > 1) {
      // Multiple subcategories - group under prefix
      result.set(prefix, subcategories);
    } else {
      // Single subcategory - keep full path
      result.set(subcategories[0], [subcategories[0]]);
    }
  }
  
  return result;
}

// Get display order for categories (grouped by prefix)
export function getCategoryDisplayOrder(categories: DisplayCategory[]): DisplayCategory[] {
  const sorted = [...new Set(categories)].sort((a, b) => {
    const aParts = a.split('/');
    const bParts = b.split('/');
    
    // Compare prefixes first
    if (aParts[0] !== bParts[0]) {
      return aParts[0].localeCompare(bParts[0]);
    }
    
    // Same prefix, compare full path
    return a.localeCompare(b);
  });
  
  // Preferred prefix order
  const prefixOrder = [
    'array', 'cells', 'containers', 'io', 'math', 'logical', 'stats', 
    'strings', 'structs', 'image', 'acceleration', 'general', 'introspection', 
    'diagnostics', 'timing'
  ];
  
  const grouped: Record<string, DisplayCategory[]> = {};
  for (const cat of sorted) {
    const prefix = cat.split('/')[0];
    if (!grouped[prefix]) {
      grouped[prefix] = [];
    }
    grouped[prefix].push(cat);
  }
  
  const result: DisplayCategory[] = [];
  for (const prefix of prefixOrder) {
    if (grouped[prefix]) {
      result.push(...grouped[prefix]);
    }
  }
  
  // Add any remaining categories
  for (const [prefix, cats] of Object.entries(grouped)) {
    if (!prefixOrder.includes(prefix)) {
      result.push(...cats);
    }
  }
  
  return result;
}
