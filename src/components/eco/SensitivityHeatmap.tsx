import { motion } from "framer-motion";
import type { EcoInputs } from "@/lib/resilience-engine";
import { sensitivityGrid } from "@/lib/resilience-engine";
import { useMemo } from "react";

export function SensitivityHeatmap({ inputs }: { inputs: EcoInputs }) {
  const { size, cells } = useMemo(() => sensitivityGrid(inputs, 14), [inputs]);
  const cellPx = 22;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
        <span>← Low Degradation · High Degradation →</span>
        <span>Slope ↑</span>
      </div>
      <div
        className="grid gap-[2px] p-2 rounded-lg bg-secondary/30 border border-border w-fit"
        style={{ gridTemplateColumns: `repeat(${size}, ${cellPx}px)` }}
      >
        {cells.map((c, i) => {
          const v = c.value / 100;
          const hue = 145; // green
          const bg = v > 0.6
            ? `oklch(${0.55 + v * 0.3} 0.22 ${hue})`
            : v > 0.3
            ? `oklch(0.7 0.18 85)`
            : `oklch(0.55 0.22 28)`;
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.002 }}
              className="rounded-[3px]"
              style={{ width: cellPx, height: cellPx, background: bg, boxShadow: v > 0.7 ? "0 0 8px var(--neon-soft)" : "none" }}
              title={`Resilience: ${c.value.toFixed(0)}`}
            />
          );
        })}
      </div>
      <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1"><i className="w-3 h-3 rounded-sm" style={{ background: "var(--danger)" }} /> Critical</span>
        <span className="flex items-center gap-1"><i className="w-3 h-3 rounded-sm" style={{ background: "var(--warn)" }} /> Vulnerable</span>
        <span className="flex items-center gap-1"><i className="w-3 h-3 rounded-sm" style={{ background: "var(--neon)" }} /> Resilient</span>
      </div>
    </div>
  );
}
