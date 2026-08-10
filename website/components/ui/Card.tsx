'use client'

import { motion, type Variants } from 'framer-motion'
import { cn } from '@/lib/utils'

type CardVariant = 'default' | 'featured' | 'metric'

interface CardProps {
  title?: string
  description?: string
  icon?: React.ReactNode
  badge?: React.ReactNode
  variant?: CardVariant
  className?: string
  children?: React.ReactNode
  delay?: number
  onClick?: () => void
  glowColor?: string
}

const cardVariants: Variants = {
  hidden: { opacity: 0, y: 24 },
  visible: (delay: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      delay,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  }),
}

const variantClasses: Record<CardVariant, string> = {
  default: [
    'glass-dark border border-white/8 rounded-2xl p-6',
    'hover:border-sky-500/30 hover:shadow-lg hover:shadow-sky-500/10',
    'transition-all duration-300',
  ].join(' '),
  featured: [
    'relative overflow-hidden rounded-2xl p-6',
    'bg-gradient-to-br from-sky-900/40 via-violet-900/30 to-slate-900/60',
    'border border-sky-500/20 hover:border-sky-400/40',
    'shadow-xl shadow-sky-500/10 hover:shadow-sky-500/20',
    'transition-all duration-300',
  ].join(' '),
  metric: [
    'glass-dark border border-white/8 rounded-xl p-5',
    'hover:border-violet-500/30 hover:shadow-lg hover:shadow-violet-500/10',
    'transition-all duration-300',
  ].join(' '),
}

export default function Card({
  title,
  description,
  icon,
  badge,
  variant = 'default',
  className,
  children,
  delay = 0,
  onClick,
  glowColor,
}: CardProps) {
  return (
    <motion.div
      variants={cardVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-50px' }}
      custom={delay}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      onClick={onClick}
      className={cn(
        variantClasses[variant],
        onClick && 'cursor-pointer',
        className,
      )}
      style={
        glowColor
          ? { boxShadow: `0 0 30px ${glowColor}20` }
          : undefined
      }
    >
      {/* Featured shine overlay */}
      {variant === 'featured' && (
        <div
          className="absolute inset-0 opacity-10 pointer-events-none"
          style={{
            background:
              'radial-gradient(ellipse at top left, rgba(14,165,233,0.4) 0%, transparent 60%)',
          }}
        />
      )}

      {/* Header row */}
      {(icon || badge) && (
        <div className="flex items-start justify-between mb-4">
          {icon && (
            <div className="p-2.5 rounded-xl bg-white/5 border border-white/8 text-sky-400">
              {icon}
            </div>
          )}
          {badge && <div className="ml-auto">{badge}</div>}
        </div>
      )}

      {/* Title */}
      {title && (
        <h3 className="text-base font-semibold text-white mb-2 leading-snug">
          {title}
        </h3>
      )}

      {/* Description */}
      {description && (
        <p className="text-sm text-slate-400 leading-relaxed">{description}</p>
      )}

      {/* Custom children */}
      {children}
    </motion.div>
  )
}
