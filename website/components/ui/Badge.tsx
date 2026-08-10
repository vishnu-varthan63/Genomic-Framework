'use client'

import { cn } from '@/lib/utils'

type BadgeColor =
  | 'default'
  | 'success'
  | 'warning'
  | 'danger'
  | 'info'
  | 'purple'

interface BadgeProps {
  children: React.ReactNode
  color?: BadgeColor
  className?: string
  dot?: boolean
  size?: 'sm' | 'md'
}

const colorClasses: Record<BadgeColor, string> = {
  default:
    'bg-slate-700/60 text-slate-300 border border-slate-600/50',
  success:
    'bg-green-500/15 text-green-400 border border-green-500/30',
  warning:
    'bg-orange-500/15 text-orange-400 border border-orange-500/30',
  danger:
    'bg-red-500/15 text-red-400 border border-red-500/30',
  info:
    'bg-sky-500/15 text-sky-400 border border-sky-500/30',
  purple:
    'bg-violet-500/15 text-violet-400 border border-violet-500/30',
}

const dotColorClasses: Record<BadgeColor, string> = {
  default: 'bg-slate-400',
  success: 'bg-green-400',
  warning: 'bg-orange-400',
  danger: 'bg-red-400',
  info: 'bg-sky-400',
  purple: 'bg-violet-400',
}

const sizeClasses = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-1 text-xs',
}

export default function Badge({
  children,
  color = 'default',
  className,
  dot = false,
  size = 'md',
}: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 font-medium rounded-full',
        colorClasses[color],
        sizeClasses[size],
        className,
      )}
    >
      {dot && (
        <span
          className={cn(
            'w-1.5 h-1.5 rounded-full shrink-0',
            dotColorClasses[color],
          )}
        />
      )}
      {children}
    </span>
  )
}
