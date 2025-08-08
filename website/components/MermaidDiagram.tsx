"use client";

import { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { useTheme } from 'next-themes';

interface MermaidDiagramProps {
  chart: string;
  className?: string;
}

export function MermaidDiagram({ chart, className }: MermaidDiagramProps) {
  const ref = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const renderChart = async () => {
      // Early return if ref is not available
      if (!ref.current) return;

      setIsLoading(true);
      setError(null);

      try {
        // Initialize mermaid with theme
        mermaid.initialize({
          startOnLoad: true,
          theme: theme === 'dark' ? 'dark' : 'default',
          themeVariables: {
            primaryColor: '#3ea7fd',
            primaryTextColor: theme === 'dark' ? '#F4F5F5' : '#202124',
            primaryBorderColor: '#3ea7fd',
            lineColor: theme === 'dark' ? '#AEAFB0' : '#8C8C8C',
            sectionBkgColor: theme === 'dark' ? '#2D2E30' : '#f8f9fa',
            altSectionBkgColor: theme === 'dark' ? '#27282B' : '#e9ecef',
            gridColor: theme === 'dark' ? '#454545' : '#e1e5e9',
            secondaryColor: theme === 'dark' ? '#454545' : '#f1f3f4',
            tertiaryColor: theme === 'dark' ? '#27282B' : '#ffffff',
          },
        });

        // Double-check ref is still available before rendering
        if (ref.current) {
          const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
          const { svg } = await mermaid.render(id, chart);
          
          // Triple-check ref is still available before setting innerHTML
          if (ref.current) {
            ref.current.innerHTML = svg;
            setIsLoading(false);
          }
        }
      } catch (error) {
        console.error('Mermaid rendering error:', error);
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        setError(errorMsg);
        setIsLoading(false);
        
        // Fallback: show the raw chart text if rendering fails
        if (ref.current) {
          ref.current.innerHTML = `<pre style="color: red; padding: 1rem; border: 1px solid red; border-radius: 4px;">Mermaid diagram failed to render: ${errorMsg}</pre>`;
        }
      }
    };

    // Add a small delay to ensure DOM is ready
    const timeoutId = setTimeout(renderChart, 100);
    
    return () => clearTimeout(timeoutId);
  }, [chart, theme]);

  if (error) {
    return (
      <div className={`${className} p-4 border border-red-300 rounded bg-red-50 dark:bg-red-900/20 dark:border-red-700`}>
        <p className="text-red-600 dark:text-red-400">Failed to render diagram: {error}</p>
      </div>
    );
  }

  return (
    <div className={className}>
      {isLoading && (
        <div className="flex items-center justify-center p-8">
          <div className="animate-pulse text-gray-500 dark:text-gray-400">Loading diagram...</div>
        </div>
      )}
      <div ref={ref} className={isLoading ? 'hidden' : ''} />
    </div>
  );
}