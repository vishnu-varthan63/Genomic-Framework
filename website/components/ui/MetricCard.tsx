'use client'

import { useEffect, useRef, useState } from 'react'
import { motion, useInView } from 'framer-motion'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { cn } from '@/lib/utils'

type TrendDirection = 'up' | 'down' | 'neutral'
type AccentColor = 'sky' | 'violet' | 'green' | 'orange' | 'red'

interface MetricCardProps {
  label: string
  value: number
  unit?: string
  decimals?: number
  trend?: TrendDirection
  trendValue?: string
  accentColor?: AccentColor
  description?: string
  className?: string
  delay?: number
  prefix?: string
}

const accentClasses: Record<AccentColor, { border: string; text: string; glow: string; bg: string }> = {
  sky: {
    border: 'border-sky-500/30',
    text: 'text-sky-400',
    glow: 'shadow-sky-500/20',
    bg: 'bg-sky-500/10',
  },
  violet: {
    border: 'border-violet-500/30',
    text: 'text-violet-400',
    glow: 'shadow-violet-500/20',
    bg: 'bg-violet-500/10',
  },
  green: {
    border: 'border-green-500/30',
    text: 'text-green-400',
    glow: 'shadow-green-500/20',
    bg: 'bg-green-500/10',
  },
  orange: {
    border: 'border-orange-500/30',
    text: 'text-orange-400',
    glow: 'shadow-orange-500/20',
    bg: 'bg-orange-500/10',
  },
  red: {
    border: 'border-red-500/30',
    text: 'text-red-400',
    glow: 'shadow-red-500/20',
    bg: 'bg-red-500/10',
  },
}

function useCountUp(target: number, decimals: number, duration = 1800, start: boolean) {
  const [count, setCount] = useState(0)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    if (!start) return

    const startTime = performance.now()
    const startValue = 0

    const easeOut = (t: number) => 1 - Math.pow(1 - t, 3)

    const tick = (now: number) => {
      const elapsed = now - startTime
      const progress = Math.min(elapsed / duration, 1)
      const easedProgress = easeOut(progress)
      const current = startValue + (target - startValue) * easedProgress
      setCount(parseFloat(current.toFixed(decimals)))
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(tick)
      }
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [target, decimals, duration, start])

  return count
}

export default function MetricCard({
  label,
  value,
  unit = '',
  decimals = 2,
  trend,
  trendValue,
  accentColor = 'sky',
  description,
  className,
  delay = 0,
  prefix = '',
}: MetricCardProps) {
  const ref = useRef<HTMLDivElement>(null)
  const isInView = useInView(ref, { once: true, margin: '-80px' })
  const displayValue = useCountUp(value, decimals, 1600, isInView)
  const accent = accentClasses[accentColor]

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 28, scale: 0.96 }}
      animate={isInView ? { opacity: 1, y: 0, scale: 1 } : {}}
      transition={{
        duration: 0.55,
        delay,
        ease: [0.25, 0.46, 0.45, 0.94],
      }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className={cn(
        'relative overflow-hidden rounded-2xl p-6',
        'glass-dark border',
        accent.border,
        'shadow-lg',
        accent.glow,
        'transition-shadow duration-300',
        className,
      )}
    >
      {/* Background accent glow */}
      <div
        className={cn(
          'absolute -top-6 -right-6 w-24 h-24 rounded-full blur-2xl opacity-30',
          accent.bg,
        )}
      />

      {/* Label */}
      <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-3">
        {label}
      </p>

      {/* Value */}
      <div className="flex items-end gap-1.5 mb-2">
        {prefix && (
          <span className={cn('text-2xl font-bold', accent.text)}>{prefix}</span>
        )}
        <span className={cn('text-4xl font-bold tabular-nums leading-none', accent.text)}>
          {displayValue.toFixed(decimals)}
        </span>
        {unit && (
          <span className="text-lg font-medium text-slate-400 mb-0.5">{unit}</span>
        )}
      </div>

      {/* Trend indicator */}
      {trend && (
        <div
          className={cn(
            'inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full',
            trend === 'up' && 'text-green-400 bg-green-500/10',
            trend === 'down' && 'text-red-400 bg-red-500/10',
            trend === 'neutral' && 'text-slate-400 bg-slate-500/10',
          )}
        >
          {trend === 'up' && <TrendingUp className="w-3 h-3" />}
          {trend === 'down' && <TrendingDown className="w-3 h-3" />}
          {trend === 'neutral' && <Minus className="w-3 h-3" />}
          {trendValue && <span>{trendValue}</span>}
        </div>
      )}

      {/* Description */}
      {description && (
        <p className="mt-2 text-xs text-slate-500 leading-relaxed">{description}</p>
      )}

      {/* Bottom accent bar */}
      <div className={cn('absolute bottom-0 left-0 right-0 h-0.5', accent.bg)} />
    </motion.div>
  )
}
