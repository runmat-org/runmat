import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowLeft, Calendar, Clock, User } from "lucide-react";

interface BlogLayoutProps {
  children: React.ReactNode;
  title: string;
  description: string;
  date: string;
  readTime: string;
  author: string;
  tags: string[];
  rightAside?: React.ReactNode;
  backLink?: { href: string; text: string };
  descriptionPlacement?: 'beforeMeta' | 'afterMeta';
}

export function BlogLayout({
  children,
  title,
  description,
  date,
  readTime,
  author,
  tags,
  rightAside,
  backLink = { href: '/blog', text: 'Back to Blog' },
  descriptionPlacement = 'beforeMeta'
}: BlogLayoutProps) {
  const descriptionSpacingClass =
    descriptionPlacement === 'afterMeta' ? 'mt-6 mb-6' : 'mb-6';

  const descriptionElement = (
    <p className={`${descriptionSpacingClass} text-base sm:text-xl text-muted-foreground leading-relaxed break-words`}>
      {description}
    </p>
  );

  const headerContent = (
    <>
      <div className="mb-4 flex flex-wrap gap-2">
        {tags.map((tag) => (
          <Badge key={tag} variant="secondary">
            {tag}
          </Badge>
        ))}
      </div>
      
      <h1 className="mb-6 text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl break-words">
        {title}
      </h1>
      
      {descriptionPlacement === 'beforeMeta' && descriptionElement}
      
      <div className="flex flex-wrap items-center gap-6 text-sm text-muted-foreground">
        <div className="flex items-center gap-2">
          <Calendar className="h-4 w-4" />
          {date}
        </div>
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4" />
          {readTime}
        </div>
        <div className="flex items-center gap-2">
          <User className="h-4 w-4" />
          By {author}
        </div>
      </div>

      {descriptionPlacement === 'afterMeta' && descriptionElement}
    </>
  );

  return (
    <div className="min-h-screen bg-background">
      <div className="container px-4 py-8 md:px-6 md:py-16">
        {rightAside ? (
          <div className="mx-auto lg:max-w-7xl grid gap-8 lg:grid-cols-[minmax(0,1fr)_260px]">
            <div className="min-w-0">
              {/* Back Link */}
              <Button variant="ghost" size="sm" className="mb-8 break-words" asChild>
                <Link href={backLink.href}>
                  <ArrowLeft className="mr-2 h-4 w-4 shrink-0" />
                  <span className="break-words">{backLink.text}</span>
                </Link>
              </Button>

              {/* Article Header */}
              <header className="mb-12 max-w-4xl break-words">
                {headerContent}
              </header>

              {/* Article Content */}
              <article className="max-w-4xl min-w-0">
                <div className="blog-content w-full overflow-x-hidden">
                  {children}
                </div>
              </article>
            </div>
            {rightAside}
          </div>
        ) : (
          <div className="mx-auto max-w-4xl min-w-0">
            {/* Back Link */}
            <Button variant="ghost" size="sm" className="mb-8 break-words" asChild>
              <Link href={backLink.href}>
                <ArrowLeft className="mr-2 h-4 w-4 shrink-0" />
                <span className="break-words">{backLink.text}</span>
              </Link>
            </Button>

            {/* Article Header */}
            <header className="mb-12 break-words">
              {headerContent}
            </header>

            {/* Article Content */}
            <article className="max-w-none min-w-0">
              <div className="blog-content w-full overflow-x-hidden">
                {children}
              </div>
            </article>
          </div>
        )}
      </div>
    </div>
  );
}