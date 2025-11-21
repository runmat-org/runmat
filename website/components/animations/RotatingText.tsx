"use client";

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useState,
} from "react";
import { AnimatePresence, motion, type HTMLMotionProps } from "motion/react";

import { cn } from "@/lib/utils";

export type RotatingTextHandle = {
  next: () => void;
  previous: () => void;
  jumpTo: (index: number) => void;
  reset: () => void;
};

type SplitStrategy = "characters" | "words" | "lines" | string | number;

type MotionBaseProps = Omit<HTMLMotionProps<"span">, "initial" | "animate" | "exit" | "transition">;

interface RotatingTextProps extends MotionBaseProps {
  texts: string[];
  transition?: HTMLMotionProps<"span">["transition"];
  initial?: HTMLMotionProps<"span">["initial"];
  animate?: HTMLMotionProps<"span">["animate"];
  exit?: HTMLMotionProps<"span">["exit"];
  animatePresenceMode?: "sync" | "wait" | "popLayout";
  animatePresenceInitial?: boolean;
  rotationInterval?: number;
  staggerDuration?: number;
  staggerFrom?: "first" | "last" | "center" | "random" | number;
  loop?: boolean;
  auto?: boolean;
  splitBy?: SplitStrategy;
  onNext?: (index: number) => void;
  mainClassName?: string;
  splitLevelClassName?: string;
  elementLevelClassName?: string;
}

type WordChunk = {
  characters: string[];
  needsSpace: boolean;
};

const RotatingText = forwardRef<RotatingTextHandle, RotatingTextProps>(
  (
    {
      texts,
      transition = { type: "spring", damping: 25, stiffness: 300 },
      initial = { y: "100%", opacity: 0 },
      animate = { y: 0, opacity: 1 },
      exit = { y: "-120%", opacity: 0 },
      animatePresenceMode = "wait",
      animatePresenceInitial = false,
      rotationInterval = 2000,
      staggerDuration = 0,
      staggerFrom = "first",
      loop = true,
      auto = true,
      splitBy = "characters",
      onNext,
      mainClassName,
      splitLevelClassName,
      elementLevelClassName,
      ...rest
    },
    ref
  ) => {
    const [currentTextIndex, setCurrentTextIndex] = useState(0);

    const splitIntoCharacters = (text: string): string[] => {
      type SegmenterCtor = new (
        locales?: string | string[],
        options?: Intl.SegmenterOptions
      ) => Intl.Segmenter;
      const intlWithSegmenter = Intl as typeof Intl & { Segmenter?: SegmenterCtor };
      if (typeof intlWithSegmenter.Segmenter === "function") {
        const segmenter = new intlWithSegmenter.Segmenter("en", { granularity: "grapheme" });
        return Array.from(segmenter.segment(text), (segment) => segment.segment);
      }
      return Array.from(text);
    };

    const elements = useMemo<WordChunk[]>(() => {
      const currentText = texts[currentTextIndex] ?? "";
      if (splitBy === "characters") {
        const words = currentText.split(" ");
        return words.map((word, idx) => ({
          characters: splitIntoCharacters(word),
          needsSpace: idx !== words.length - 1,
        }));
      }
      if (splitBy === "words") {
        return currentText.split(" ").map((word, idx, arr) => ({
          characters: [word],
          needsSpace: idx !== arr.length - 1,
        }));
      }
      if (splitBy === "lines") {
        return currentText.split("\n").map((line, idx, arr) => ({
          characters: [line],
          needsSpace: idx !== arr.length - 1,
        }));
      }
      const delimiter = typeof splitBy === "number" ? "" : splitBy;
      return currentText.split(delimiter as string).map((part, idx, arr) => ({
        characters: [part],
        needsSpace: idx !== arr.length - 1,
      }));
    }, [texts, currentTextIndex, splitBy]);

    const getStaggerDelay = useCallback(
      (index: number, totalChars: number) => {
        if (staggerFrom === "first") return index * staggerDuration;
        if (staggerFrom === "last") return (totalChars - 1 - index) * staggerDuration;
        if (staggerFrom === "center") {
          const center = Math.floor(totalChars / 2);
          return Math.abs(center - index) * staggerDuration;
        }
        if (staggerFrom === "random") {
          const randomIndex = Math.floor(Math.random() * totalChars);
          return Math.abs(randomIndex - index) * staggerDuration;
        }
        if (typeof staggerFrom === "number") {
          return Math.abs(staggerFrom - index) * staggerDuration;
        }
        return index * staggerDuration;
      },
      [staggerFrom, staggerDuration]
    );

    const handleIndexChange = useCallback(
      (newIndex: number) => {
        setCurrentTextIndex(newIndex);
        onNext?.(newIndex);
      },
      [onNext]
    );

    const next = useCallback(() => {
      if (!texts.length) return;
      const lastIndex = texts.length - 1;
      const nextIndex = currentTextIndex === lastIndex ? (loop ? 0 : lastIndex) : currentTextIndex + 1;
      if (nextIndex !== currentTextIndex) {
        handleIndexChange(nextIndex);
      }
    }, [currentTextIndex, texts.length, loop, handleIndexChange]);

    const previous = useCallback(() => {
      if (!texts.length) return;
      const lastIndex = texts.length - 1;
      const prevIndex = currentTextIndex === 0 ? (loop ? lastIndex : 0) : currentTextIndex - 1;
      if (prevIndex !== currentTextIndex) {
        handleIndexChange(prevIndex);
      }
    }, [currentTextIndex, texts.length, loop, handleIndexChange]);

    const jumpTo = useCallback(
      (index: number) => {
        if (!texts.length) return;
        const validIndex = Math.max(0, Math.min(index, texts.length - 1));
        if (validIndex !== currentTextIndex) {
          handleIndexChange(validIndex);
        }
      },
      [texts.length, currentTextIndex, handleIndexChange]
    );

    const reset = useCallback(() => {
      if (currentTextIndex !== 0) {
        handleIndexChange(0);
      }
    }, [currentTextIndex, handleIndexChange]);

    useImperativeHandle(
      ref,
      () => ({
        next,
        previous,
        jumpTo,
        reset,
      }),
      [next, previous, jumpTo, reset]
    );

    useEffect(() => {
      if (!auto || texts.length <= 1) return;
      const intervalId = window.setInterval(next, rotationInterval);
      return () => window.clearInterval(intervalId);
    }, [auto, next, rotationInterval, texts.length]);

    const totalCharacters = elements.reduce((sum, word) => sum + word.characters.length, 0);

    return (
      <motion.span className={cn("text-rotate", mainClassName)} layout transition={transition} {...rest}>
        <span className="text-rotate-sr-only">{texts[currentTextIndex]}</span>
        <AnimatePresence mode={animatePresenceMode} initial={animatePresenceInitial}>
          <motion.span
            key={currentTextIndex}
            className={cn(splitBy === "lines" ? "text-rotate-lines" : "text-rotate")}
            layout
            aria-hidden="true"
          >
            {elements.map((wordObj, wordIndex, array) => {
              const previousCharsCount = array
                .slice(0, wordIndex)
                .reduce((sum, word) => sum + word.characters.length, 0);
              return (
                <span key={wordIndex} className={cn("text-rotate-word", splitLevelClassName)}>
                  {wordObj.characters.map((char, charIndex) => (
                    <motion.span
                      key={`${wordIndex}-${charIndex}`}
                      initial={initial}
                      animate={animate}
                      exit={exit}
                      transition={{
                        ...transition,
                        delay: getStaggerDelay(previousCharsCount + charIndex, totalCharacters),
                      }}
                      className={cn("text-rotate-element", elementLevelClassName)}
                    >
                      {char}
                    </motion.span>
                  ))}
                  {wordObj.needsSpace && <span className="text-rotate-space"> </span>}
                </span>
              );
            })}
          </motion.span>
        </AnimatePresence>
      </motion.span>
    );
  }
);

RotatingText.displayName = "RotatingText";

export default RotatingText;


