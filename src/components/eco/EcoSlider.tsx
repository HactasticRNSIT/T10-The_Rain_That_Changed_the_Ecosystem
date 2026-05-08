import { motion } from "framer-motion";

interface SliderProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  hint?: string;
}

export function EcoSlider({ label, value, onChange, min, max, step = 1, unit, hint }: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <label className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{label}</label>
        <motion.span
          key={value}
          initial={{ opacity: 0.6, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
          className="font-mono text-sm neon-text"
        >
          {value.toFixed(step < 1 ? 2 : 0)}{unit ?? ""}
        </motion.span>
      </div>
      <div className="relative h-2 rounded-full bg-secondary/60 overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 bg-primary"
          style={{ width: `${pct}%`, boxShadow: "0 0 14px var(--neon-soft)" }}
        />
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
      </div>
      {hint && <p className="text-[10px] text-muted-foreground/70">{hint}</p>}
    </div>
  );
}
