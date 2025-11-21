"use client";

import { useMemo, useState, useRef, useEffect } from 'react';
import Fuse from 'fuse.js';
import type FuseType from 'fuse.js';
import Link from 'next/link';
import { Grid3x3, List, Tags } from 'lucide-react';
import type { Builtin } from '@/lib/builtins';
import { getBuiltinBadges, getDisplayCategory } from '@/lib/builtin-utils';
import type { DisplayCategory } from '@/lib/builtin-utils';
import { formatCategoryName, getCategoryDisplayOrder, groupCategoriesByPrefix } from '@/lib/display-categories';
import { POPULAR_FUNCTIONS_BY_CATEGORY, getJsonCategoriesForDisplayCategory } from '@/lib/builtin-popularity';
import { getCategoryColor } from '@/lib/category-colors';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';

type GroupedBuiltins = {
  category: DisplayCategory;
  items: Builtin[]; // Limited items (13 or all if <= 13)
  allItems: Builtin[]; // All items in the category (for expansion)
  subcategories?: DisplayCategory[]; // Subcategories if this is a grouped prefix category
  totalCount: number; // Total number of items in this category (before limiting)
};

export default function ElementsOfMatlabGrid({ builtins }: { builtins: Builtin[] }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'tags'>('grid');
  const [expandedCategories, setExpandedCategories] = useState<Set<DisplayCategory>>(new Set());

  // Filter out internal functions
  const publicBuiltins = useMemo(() => {
    return builtins.filter(b => !b.internal);
  }, [builtins]);

  // Setup Fuse.js search
  const fuse = useMemo(() => new Fuse(publicBuiltins, {
    includeScore: true,
    threshold: 0.35,
    keys: [
      { name: 'name', weight: 0.6 },
      { name: 'summary', weight: 0.2 },
      { name: 'category', weight: 0.15 },
      { name: 'keywords', weight: 0.05 },
    ],
  }), [publicBuiltins]);

  // Filter and group builtins
  const groupedBuiltins = useMemo(() => {
    let filtered = publicBuiltins;
    
    // Apply search filter
    if (searchQuery.trim()) {
      filtered = fuse.search(searchQuery).map(r => r.item);
    }

    // First, group by actual JSON category from builtins
    const rawGroups = new Map<DisplayCategory, Builtin[]>();
    for (const builtin of filtered) {
      const category = getDisplayCategory(builtin);
      if (!rawGroups.has(category)) {
        rawGroups.set(category, []);
      }
      rawGroups.get(category)!.push(builtin);
    }

    // Group categories by prefix (combine multiple subcategories under prefix)
    const categoryGrouping = groupCategoriesByPrefix(Array.from(rawGroups.keys()));
    
    // Now group builtins by the grouped display category
    const groupedBuiltinsMap = new Map<DisplayCategory, Builtin[]>();
    for (const [displayCategory, subcategories] of categoryGrouping.entries()) {
      const allItems: Builtin[] = [];
      for (const subcategory of subcategories) {
        const items = rawGroups.get(subcategory) || [];
        allItems.push(...items);
      }
      groupedBuiltinsMap.set(displayCategory, allItems);
    }

    // Convert to array, sort by popularity, and limit to 13 per category (13 functions + 1 "View All" = 14 total cells)
    const result: GroupedBuiltins[] = [];
    for (const [category, items] of groupedBuiltinsMap.entries()) {
      // Get popular functions from all subcategories
      const subcategories = categoryGrouping.get(category) || [category];
      const allPopular: string[] = [];
      for (const subcat of subcategories) {
        const popular = POPULAR_FUNCTIONS_BY_CATEGORY[subcat] || [];
        allPopular.push(...popular);
      }
      
      // Sort: popular functions first, then alphabetically
      const sorted = items.sort((a, b) => {
        const aPopular = allPopular.indexOf(a.name.toLowerCase());
        const bPopular = allPopular.indexOf(b.name.toLowerCase());
        
        // Both popular: sort by popularity order
        if (aPopular !== -1 && bPopular !== -1) {
          return aPopular - bPopular;
        }
        // Only a is popular
        if (aPopular !== -1) return -1;
        // Only b is popular
        if (bPopular !== -1) return 1;
        // Neither popular: alphabetical
        return a.name.localeCompare(b.name);
      });
      
      const totalCount = sorted.length;
      // If there are 13 or fewer items, show all of them (no "View All" needed)
      // Otherwise, limit to 13 (13 functions + 1 "View All" = 14 total cells)
      const limited = totalCount <= 13 ? sorted : sorted.slice(0, 13);
      
      result.push({
        category,
        items: limited,
        allItems: sorted, // Store all items for expansion
        subcategories: categoryGrouping.get(category),
        totalCount,
      });
    }

    // Sort categories by predefined order
    const allCategories = Array.from(groupedBuiltinsMap.keys());
    const categoryOrder = getCategoryDisplayOrder(allCategories);

    return result.sort((a, b) => {
      const aIdx = categoryOrder.indexOf(a.category);
      const bIdx = categoryOrder.indexOf(b.category);
      if (aIdx === -1 && bIdx === -1) return a.category.localeCompare(b.category);
      if (aIdx === -1) return 1;
      if (bIdx === -1) return -1;
      return aIdx - bIdx;
    });
  }, [publicBuiltins, fuse, searchQuery]);

  return (
    <div>
      {/* Search bar and view toggle */}
      <div className="mb-6 flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search tiles (e.g., size, colon, table, plot)"
          className="flex-1 text-sm rounded-md border border-border bg-muted/40 pr-3 pl-3 py-2.5 h-10 focus:outline-none focus:ring-2 focus:ring-primary/60 focus:border-primary placeholder:text-muted-foreground/80 shadow-sm"
        />
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-sm text-muted-foreground whitespace-nowrap">View:</span>
          <div className="flex items-center gap-1 border border-border rounded-md p-1 bg-muted/40 shrink-0">
            <button
              onClick={() => setViewMode('grid')}
              className={`h-8 px-3 text-sm font-medium rounded-md transition-all inline-flex items-center justify-center gap-1.5 ${
                viewMode === 'grid' 
                  ? 'shadow-sm' 
                  : 'hover:bg-accent hover:text-accent-foreground'
              }`}
              style={viewMode === 'grid' ? {
                backgroundColor: '#ffffff',
                color: '#000000',
              } : undefined}
              aria-label="Grid view"
            >
              <Grid3x3 className="h-4 w-4" />
              Cards
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`h-8 px-3 text-sm font-medium rounded-md transition-all inline-flex items-center justify-center gap-1.5 ${
                viewMode === 'list' 
                  ? 'shadow-sm' 
                  : 'hover:bg-accent hover:text-accent-foreground'
              }`}
              style={viewMode === 'list' ? {
                backgroundColor: '#ffffff',
                color: '#000000',
              } : undefined}
              aria-label="List view"
            >
              <List className="h-4 w-4" />
              List
            </button>
            <button
              onClick={() => setViewMode('tags')}
              className={`h-8 px-3 text-sm font-medium rounded-md transition-all inline-flex items-center justify-center gap-1.5 ${
                viewMode === 'tags' 
                  ? 'shadow-sm' 
                  : 'hover:bg-accent hover:text-accent-foreground'
              }`}
              style={viewMode === 'tags' ? {
                backgroundColor: '#ffffff',
                color: '#000000',
              } : undefined}
              aria-label="Chips view"
            >
              <Tags className="h-4 w-4" />
              Chips
            </button>
          </div>
        </div>
      </div>

      {/* Content based on view mode */}
      {viewMode === 'grid' ? (
        <GridView 
          groupedBuiltins={groupedBuiltins} 
          expandedCategories={expandedCategories}
          onToggleExpand={(category) => {
            setExpandedCategories(prev => {
              const newSet = new Set(prev);
              if (newSet.has(category)) {
                newSet.delete(category);
              } else {
                newSet.add(category);
              }
              return newSet;
            });
          }}
        />
      ) : viewMode === 'list' ? (
        <ListView groupedBuiltins={groupedBuiltins} />
      ) : (
        <TagsView builtins={publicBuiltins} searchQuery={searchQuery} fuse={fuse} />
      )}
    </div>
  );
}

function GridView({ 
  groupedBuiltins, 
  expandedCategories, 
  onToggleExpand 
}: { 
  groupedBuiltins: GroupedBuiltins[];
  expandedCategories: Set<DisplayCategory>;
  onToggleExpand: (category: DisplayCategory) => void;
}) {
  return (
    <div className="space-y-12">
      {groupedBuiltins.map((group) => {
        const isExpanded = expandedCategories.has(group.category);
        const itemsToShow = isExpanded ? group.allItems : group.items;
        const showViewAll = !isExpanded && group.totalCount > 13;
        const showCollapse = isExpanded && group.totalCount > 13;
        
        return (
          <div key={group.category}>
            <h2 className="text-2xl font-semibold mb-6 text-foreground sm:text-3xl">{formatCategoryName(group.category)}</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-7 gap-3">
              {itemsToShow.map((builtin) => (
                <ElementTile key={builtin.slug} builtin={builtin} />
              ))}
              {/* "View All" tile - only show if there are more items than displayed and not expanded */}
              {showViewAll && (
                <ViewAllTile 
                  category={group.category} 
                  subcategories={group.subcategories}
                  totalCount={group.totalCount}
                  onClick={() => onToggleExpand(group.category)}
                />
              )}
              {/* "Collapse" tile - only show when expanded */}
              {showCollapse && (
                <CollapseTile 
                  category={group.category}
                  onClick={() => onToggleExpand(group.category)}
                />
              )}
            </div>
          </div>
        );
      })}
      {groupedBuiltins.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          No building blocks match your search.
        </div>
      )}
    </div>
  );
}

function ListView({ groupedBuiltins }: { groupedBuiltins: GroupedBuiltins[] }) {
  return (
    <div className="space-y-8">
      {groupedBuiltins.map((group) => (
        <div key={group.category}>
          <div className="flex items-center gap-3 mb-4">
            <h2 className="text-2xl font-semibold text-foreground sm:text-3xl">{formatCategoryName(group.category)}</h2>
            <div 
              className="h-1 flex-1 rounded"
              style={{ backgroundColor: getCategoryColor(group.category) }}
            />
            {group.totalCount > 13 && (
              <Link
                href={`/docs/reference/builtins?${(group.subcategories && group.subcategories.length > 0 ? group.subcategories : [group.category]).map(c => `category=${encodeURIComponent(c)}`).join('&')}`}
                className="text-lg font-semibold text-muted-foreground hover:text-foreground"
              >
                View All ({group.totalCount}) →
              </Link>
            )}
          </div>
          <div className="space-y-1">
            {group.items.map((builtin) => (
              <ListTile key={builtin.slug} builtin={builtin} />
            ))}
          </div>
        </div>
      ))}
      {groupedBuiltins.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          No building blocks match your search.
        </div>
      )}
    </div>
  );
}

function ElementTile({ builtin }: { builtin: Builtin }) {
  const badges = getBuiltinBadges(builtin);
  const displayCategory = getDisplayCategory(builtin);
  const categoryColor = getCategoryColor(displayCategory);
  const [isHovered, setIsHovered] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState<'top' | 'bottom' | 'left' | 'right'>('bottom');
  const tileRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Get first paragraph - prefer MDX first paragraph, fallback to description/summary
  const firstParagraph = (builtin.firstParagraph ?? builtin.description ?? builtin.summary ?? 'No description available').trim();

  // Calculate position before showing tooltip to prevent jumpiness
  useEffect(() => {
    if (isHovered && tileRef.current) {
      // Small delay to prevent flickering
      hoverTimeoutRef.current = setTimeout(() => {
        if (!tileRef.current) return;
        
        const tileRect = tileRef.current.getBoundingClientRect();
        const viewportHeight = window.innerHeight;
        const viewportWidth = window.innerWidth;

        // Determine best position based on available space
        const spaceAbove = tileRect.top;
        const spaceBelow = viewportHeight - tileRect.bottom;
        const spaceLeft = tileRect.left;
        const spaceRight = viewportWidth - tileRect.right;

        // Estimate tooltip dimensions
        const estimatedTooltipHeight = 250;
        const estimatedTooltipWidth = 300;

        let position: 'top' | 'bottom' | 'left' | 'right' = 'bottom';
        if (spaceBelow >= estimatedTooltipHeight + 20 || spaceBelow > spaceAbove) {
          position = 'bottom';
        } else if (spaceAbove >= estimatedTooltipHeight + 20) {
          position = 'top';
        } else if (spaceRight >= estimatedTooltipWidth + 20) {
          position = 'right';
        } else {
          position = 'left';
        }

        setTooltipPosition(position);
        setShowTooltip(true);
      }, 100); // Small delay to prevent jumpiness
    } else {
      setShowTooltip(false);
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
        hoverTimeoutRef.current = null;
      }
    }

    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
    };
  }, [isHovered]);

  return (
    <div 
      ref={tileRef}
      className="relative"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <Link href={`/docs/reference/builtins/${builtin.slug}`} className="block h-full">
        <Card 
          className="group hover:opacity-90 transition-opacity cursor-pointer h-full flex flex-col gap-2 py-2.5 px-3 border-0 shadow-sm rounded-lg"
          style={{ backgroundColor: categoryColor, minHeight: '139px' }}
        >
          {/* Title */}
          <div className="font-semibold text-base leading-tight text-white">{builtin.name}</div>
          
          {/* Summary */}
          <div className="text-sm text-white/90 line-clamp-2 flex-1">
            {builtin.summary || 'No description available'}
          </div>

          {/* Feature badges - always show for consistent height */}
          <div className="flex items-center gap-1.5 flex-wrap mt-auto pt-1.5 border-t border-white/20 min-h-[20px]">
            {badges.map((badge) => (
              <Badge
                key={badge}
                variant="secondary"
                className="text-xs px-1.5 py-0.5 h-4 font-medium bg-white/20 text-white border-white/30"
              >
                {badge}
              </Badge>
            ))}
          </div>
        </Card>
      </Link>

      {/* Hover Tooltip */}
      {showTooltip && (
        <div
          ref={tooltipRef}
          className={`absolute z-50 pointer-events-none transition-opacity duration-150 ${
            tooltipPosition === 'top' ? 'bottom-full left-1/2 -translate-x-1/2 mb-2' :
            tooltipPosition === 'bottom' ? 'top-full left-1/2 -translate-x-1/2 mt-2' :
            tooltipPosition === 'left' ? 'right-full top-1/2 -translate-y-1/2 mr-2' :
            'left-full top-1/2 -translate-y-1/2 ml-2'
          }`}
          style={{
            maxWidth: '320px',
            minWidth: '280px',
            opacity: showTooltip ? 1 : 0,
          }}
        >
          <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-xl p-4 text-sm relative" style={{ backgroundColor: '#111827' }}>
            {/* Function Name */}
            <div className="font-semibold text-base mb-2 text-white">
              {builtin.name}
            </div>
            
            {/* First Paragraph Description */}
            <div className="text-gray-300 leading-relaxed whitespace-pre-wrap">
              {firstParagraph}
            </div>

            {/* Arrow - positioned based on tooltip position */}
            {tooltipPosition === 'top' && (
              <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px w-0 h-0 border-t-[8px] border-t-gray-900 border-x-[8px] border-x-transparent" />
            )}
            {tooltipPosition === 'bottom' && (
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 -mb-px w-0 h-0 border-b-[8px] border-b-gray-900 border-x-[8px] border-x-transparent" />
            )}
            {tooltipPosition === 'left' && (
              <div className="absolute left-full top-1/2 -translate-y-1/2 -ml-px w-0 h-0 border-l-[8px] border-l-gray-900 border-y-[8px] border-y-transparent" />
            )}
            {tooltipPosition === 'right' && (
              <div className="absolute right-full top-1/2 -translate-y-1/2 -mr-px w-0 h-0 border-r-[8px] border-r-gray-900 border-y-[8px] border-y-transparent" />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function TagsView({ builtins, searchQuery, fuse }: { builtins: Builtin[], searchQuery: string, fuse: FuseType<Builtin> }) {
  // Filter and group all builtins (no 10-item limit)
  const groupedBuiltins = useMemo(() => {
    let filtered = builtins;
    
    // Apply search filter
    if (searchQuery.trim()) {
      filtered = fuse.search(searchQuery).map(r => r.item);
    }

    // First, group by actual JSON category from builtins
    const rawGroups = new Map<DisplayCategory, Builtin[]>();
    for (const builtin of filtered) {
      const category = getDisplayCategory(builtin);
      if (!rawGroups.has(category)) {
        rawGroups.set(category, []);
      }
      rawGroups.get(category)!.push(builtin);
    }

    // Group categories by prefix (combine multiple subcategories under prefix)
    const categoryGrouping = groupCategoriesByPrefix(Array.from(rawGroups.keys()));
    
    // Now group builtins by the grouped display category
    const groupedBuiltinsMap = new Map<DisplayCategory, Builtin[]>();
    for (const [displayCategory, subcategories] of categoryGrouping.entries()) {
      const allItems: Builtin[] = [];
      for (const subcategory of subcategories) {
        const items = rawGroups.get(subcategory) || [];
        allItems.push(...items);
      }
      groupedBuiltinsMap.set(displayCategory, allItems);
    }

    // Convert to array and sort items within each group alphabetically
    const result: GroupedBuiltins[] = [];
    for (const [category, items] of groupedBuiltinsMap.entries()) {
      const sortedItems = items.sort((a, b) => a.name.localeCompare(b.name));
      result.push({
        category,
        items: sortedItems,
        allItems: sortedItems, // TagsView shows all items
        subcategories: categoryGrouping.get(category),
        totalCount: sortedItems.length,
      });
    }

    // Sort categories by predefined order
    const allCategories = Array.from(groupedBuiltinsMap.keys());
    const categoryOrder = getCategoryDisplayOrder(allCategories);

    return result.sort((a, b) => {
      const aIdx = categoryOrder.indexOf(a.category);
      const bIdx = categoryOrder.indexOf(b.category);
      if (aIdx === -1 && bIdx === -1) return a.category.localeCompare(b.category);
      if (aIdx === -1) return 1;
      if (bIdx === -1) return -1;
      return aIdx - bIdx;
    });
  }, [builtins, fuse, searchQuery]);

  return (
    <div className="space-y-6">
      {groupedBuiltins.map((group) => (
        <div key={group.category}>
          <div className="flex items-center gap-3 mb-3">
            <h2 className="text-2xl font-semibold text-foreground sm:text-3xl">{formatCategoryName(group.category)}</h2>
            <div 
              className="h-1 flex-1 rounded"
              style={{ backgroundColor: getCategoryColor(group.category) }}
            />
          </div>
          <div className="flex flex-wrap gap-2">
            {group.items.map((builtin) => (
              <TagChip key={builtin.slug} builtin={builtin} />
            ))}
          </div>
        </div>
      ))}
      {groupedBuiltins.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          No building blocks match your search.
        </div>
      )}
    </div>
  );
}

function TagChip({ builtin }: { builtin: Builtin }) {
  const badges = getBuiltinBadges(builtin);
  const displayCategory = getDisplayCategory(builtin);
  const categoryColor = getCategoryColor(displayCategory);
  const [isHovered, setIsHovered] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState<'top' | 'bottom' | 'left' | 'right'>('bottom');
  const chipRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Get first paragraph - prefer MDX first paragraph, fallback to description/summary
  const firstParagraph = (builtin.firstParagraph ?? builtin.description ?? builtin.summary ?? 'No description available').trim();

  // Calculate position before showing tooltip to prevent jumpiness
  useEffect(() => {
    if (isHovered && chipRef.current) {
      // Small delay to prevent flickering
      hoverTimeoutRef.current = setTimeout(() => {
        if (!chipRef.current) return;
        
        const chipRect = chipRef.current.getBoundingClientRect();
        const viewportHeight = window.innerHeight;
        const viewportWidth = window.innerWidth;

        const spaceAbove = chipRect.top;
        const spaceBelow = viewportHeight - chipRect.bottom;
        const spaceLeft = chipRect.left;
        const spaceRight = viewportWidth - chipRect.right;

        const estimatedTooltipHeight = 250;
        const estimatedTooltipWidth = 300;

        let position: 'top' | 'bottom' | 'left' | 'right' = 'bottom';
        if (spaceBelow >= estimatedTooltipHeight + 20 || spaceBelow > spaceAbove) {
          position = 'bottom';
        } else if (spaceAbove >= estimatedTooltipHeight + 20) {
          position = 'top';
        } else if (spaceRight >= estimatedTooltipWidth + 20) {
          position = 'right';
        } else {
          position = 'left';
        }

        setTooltipPosition(position);
        setShowTooltip(true);
      }, 100); // Small delay to prevent jumpiness
    } else {
      setShowTooltip(false);
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
        hoverTimeoutRef.current = null;
      }
    }

    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
    };
  }, [isHovered]);

  return (
    <div 
      ref={chipRef}
      className="relative inline-block"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <Link href={`/docs/reference/builtins/${builtin.slug}`} className="group">
        <div
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all hover:scale-105 hover:shadow-md cursor-pointer border border-transparent hover:border-white/20"
          style={{ 
            backgroundColor: categoryColor,
            color: 'white'
          }}
        >
          <span>{builtin.name}</span>
          {badges.length > 0 && (
            <span className="text-xs opacity-80">
              {badges.map(b => b.charAt(0)).join('')}
            </span>
          )}
        </div>
      </Link>

      {/* Hover Tooltip */}
      {showTooltip && (
        <div
          ref={tooltipRef}
          className={`absolute z-50 pointer-events-none transition-opacity duration-150 ${
            tooltipPosition === 'top' ? 'bottom-full left-1/2 -translate-x-1/2 mb-2' :
            tooltipPosition === 'bottom' ? 'top-full left-1/2 -translate-x-1/2 mt-2' :
            tooltipPosition === 'left' ? 'right-full top-1/2 -translate-y-1/2 mr-2' :
            'left-full top-1/2 -translate-y-1/2 ml-2'
          }`}
          style={{
            maxWidth: '320px',
            minWidth: '280px',
            opacity: showTooltip ? 1 : 0,
          }}
        >
          <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-xl p-4 text-sm relative" style={{ backgroundColor: '#111827' }}>
            {/* Function Name */}
            <div className="font-semibold text-base mb-2 text-white">
              {builtin.name}
            </div>
            
            {/* First Paragraph Description */}
            <div className="text-gray-300 leading-relaxed whitespace-pre-wrap">
              {firstParagraph}
            </div>

            {/* Arrow - positioned based on tooltip position */}
            {tooltipPosition === 'top' && (
              <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px w-0 h-0 border-t-[8px] border-t-gray-900 border-x-[8px] border-x-transparent" />
            )}
            {tooltipPosition === 'bottom' && (
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 -mb-px w-0 h-0 border-b-[8px] border-b-gray-900 border-x-[8px] border-x-transparent" />
            )}
            {tooltipPosition === 'left' && (
              <div className="absolute left-full top-1/2 -translate-y-1/2 -ml-px w-0 h-0 border-l-[8px] border-l-gray-900 border-y-[8px] border-y-transparent" />
            )}
            {tooltipPosition === 'right' && (
              <div className="absolute right-full top-1/2 -translate-y-1/2 -mr-px w-0 h-0 border-r-[8px] border-r-gray-900 border-y-[8px] border-y-transparent" />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function ListTile({ builtin }: { builtin: Builtin }) {
  const badges = getBuiltinBadges(builtin);
  const displayCategory = getDisplayCategory(builtin);
  const categoryColor = getCategoryColor(displayCategory);

  return (
    <Link href={`/docs/reference/builtins/${builtin.slug}`} className="block">
      <div 
        className="group hover:opacity-90 transition-opacity cursor-pointer rounded-md px-4 py-2.5 flex items-center gap-4"
      >
        <div 
          className="w-2.5 h-2.5 rounded-full shrink-0"
          style={{ backgroundColor: categoryColor }}
        />
        <div className="flex-1 min-w-0">
          <div className="text-base font-semibold text-foreground group-hover:text-primary transition-colors mb-1">
            {builtin.name}
          </div>
          <div className="text-sm text-muted-foreground line-clamp-1">
            {builtin.summary || 'No description available'}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {badges.map((badge) => (
            <Badge
              key={badge}
              variant="secondary"
              className="px-3 py-1 font-semibold"
            >
              {badge}
            </Badge>
          ))}
        </div>
      </div>
    </Link>
  );
}

function ViewAllTile({ 
  category, 
  subcategories, 
  totalCount,
  onClick 
}: { 
  category: DisplayCategory; 
  subcategories?: DisplayCategory[];
  totalCount: number;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="block h-full w-full text-left"
      aria-label={`Expand ${formatCategoryName(category)} to show all ${totalCount} functions`}
    >
      <Card 
        className="group hover:opacity-90 transition-opacity cursor-pointer h-full flex flex-col gap-2 py-2.5 px-3 border-0 shadow-sm border-dashed border-2 rounded-lg"
        style={{ backgroundColor: '#1f2937', borderColor: 'rgba(255, 255, 255, 0.3)', minHeight: '139px' }}
      >
        {/* Title */}
        <div className="font-semibold text-base leading-tight text-white flex items-center gap-1.5">
          <span>View All ({totalCount})</span>
          <span className="text-white/80">→</span>
        </div>
        
        {/* Summary */}
        <div className="text-sm text-white/90 line-clamp-2 flex-1">
          See all {formatCategoryName(category).toLowerCase()} functions
        </div>

        {/* Feature badges - always show for consistent height */}
        <div className="flex items-center gap-1.5 flex-wrap mt-auto pt-1.5 border-t border-white/20 min-h-[20px]">
        </div>
      </Card>
    </button>
  );
}

function CollapseTile({ 
  category,
  onClick 
}: { 
  category: DisplayCategory;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="block h-full w-full text-left"
      aria-label={`Collapse ${formatCategoryName(category)} to show fewer functions`}
    >
      <Card 
        className="group hover:opacity-90 transition-opacity cursor-pointer h-full flex flex-col gap-2 py-2.5 px-3 border-0 shadow-sm border-dashed border-2 rounded-lg"
        style={{ backgroundColor: '#1f2937', borderColor: 'rgba(255, 255, 255, 0.3)', minHeight: '139px' }}
      >
        {/* Title */}
        <div className="font-semibold text-base leading-tight text-white flex items-center gap-1.5">
          <span>Show Less</span>
          <span className="text-white/80">←</span>
        </div>
        
        {/* Summary */}
        <div className="text-sm text-white/90 line-clamp-2 flex-1">
          Collapse {formatCategoryName(category).toLowerCase()} functions
        </div>

        {/* Feature badges - always show for consistent height */}
        <div className="flex items-center gap-1.5 flex-wrap mt-auto pt-1.5 border-t border-white/20 min-h-[20px]">
        </div>
      </Card>
    </button>
  );
}

