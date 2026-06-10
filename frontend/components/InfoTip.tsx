"use client";

import { GLOSSARY } from "@/lib/copy";

type Align = "left" | "center" | "right";

// Horizontal placement of the tooltip relative to the term. Pick based on where the term
// sits so the bubble never runs off a screen edge: left-column terms open rightward,
// right-edge terms open leftward, mid-line terms center.
const ALIGN: Record<Align, string> = {
  left: "left-0",
  center: "left-1/2 -translate-x-1/2",
  right: "right-0",
};

const TOOLTIP_BASE =
  "pointer-events-none absolute bottom-full z-30 mb-1.5 hidden w-60 rounded-lg bg-slate-900 px-3 py-2 text-left text-xs font-normal leading-snug text-white shadow-xl group-hover:block group-focus:block";

// A term with a dotted underline that reveals a plain-language tooltip on hover/focus.
// Keyboard-accessible (tabIndex) so it's not hover-only. CSS-only (group-hover) — no
// state, no portal — which is plenty for short glossary tips.
export function Term({
  children,
  tip,
  align = "center",
  className = "",
}: {
  children: React.ReactNode;
  tip: string;
  align?: Align;
  className?: string;
}) {
  return (
    <span
      tabIndex={0}
      className={`group relative cursor-help border-b border-dotted border-slate-400 outline-none ${className}`}
    >
      {children}
      <span role="tooltip" className={`${TOOLTIP_BASE} ${ALIGN[align]}`}>
        {tip}
      </span>
    </span>
  );
}

// Convenience: look the tip up in the glossary by `term`. Shows `children` as the visible
// text when provided, otherwise shows the term itself.
export function GlossaryTerm({
  term,
  children,
  align = "center",
  className = "",
}: {
  term: keyof typeof GLOSSARY | string;
  children?: React.ReactNode;
  align?: Align;
  className?: string;
}) {
  const tip = GLOSSARY[term] ?? "";
  return (
    <Term tip={tip} align={align} className={className}>
      {children ?? term}
    </Term>
  );
}
