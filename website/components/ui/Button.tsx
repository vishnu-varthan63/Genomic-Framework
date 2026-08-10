'use client'

import { forwardRef } from 'react'
import { motion, type HTMLMotionProps } from 'framer-motion'
import { cn } from '@/lib/utils'

type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger'
type ButtonSize = 'sm' | 'md' | 'lg'

interface ButtonProps extends Omit<HTMLMotionProps<'button'>, 'size'> {
  variant?: ButtonVariant
  size?: ButtonSize
  isLoading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
  fullWidth?: boolean
}

const variantClasses: Record<ButtonVariant, string> = {
  primary: [
    'bg-sky-500 text-white hover:bg-sky-400',
    'shadow-lg shadow-sky-500/25 hover:shadow-sky-400/30',
    'border border-sky-400/30',
  ].join(' '),
  secondary: [
    'bg-violet-600 text-white hover:bg-violet-500',
    'shadow-lg shadow-violet-500/25 hover:shadow-violet-400/30',
    'border border-violet-400/30',
  ].join(' '),
  outline: [
    'bg-transparent text-sky-400 hover:bg-sky-500/10',
    'border border-sky-500/50 hover:border-sky-400',
  ].join(' '),
  ghost: [
    'bg-transparent text-slate-300 hover:bg-white/5 hover:text-white',
    'border border-transparent hover:border-white/10',
  ].join(' '),
  danger: [
    'bg-red-500 text-white hover:bg-red-400',
    'shadow-lg shadow-red-500/25 hover:shadow-red-400/30',
    'border border-red-400/30',
  ].join(' '),
}

const sizeClasses: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-sm gap-1.5 rounded-lg',
  md: 'px-5 py-2.5 text-sm gap-2 rounded-xl',
  lg: 'px-7 py-3.5 text-base gap-2.5 rounded-xl',
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      isLoading = false,
      leftIcon,
      rightIcon,
      fullWidth = false,
      className,
      children,
      disabled,
      ...props
    },
    ref,
  ) => {
    const isDisabled = disabled || isLoading

    return (
      <motion.button
        ref={ref}
        whileHover={isDisabled ? {} : { scale: 1.03 }}
        whileTap={isDisabled ? {} : { scale: 0.97 }}
        transition={{ type: 'spring', stiffness: 400, damping: 20 }}
        disabled={isDisabled}
        className={cn(
          'relative inline-flex items-center justify-center font-medium',
          'transition-colors duration-200 cursor-pointer select-none',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950',
          'disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none',
          variantClasses[variant],
          sizeClasses[size],
          fullWidth && 'w-full',
          className,
        )}
        {...props}
      >
        {isLoading ? (
          <>
            <svg
              className="animate-spin h-4 w-4 shrink-0"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            <span>Loading…</span>
          </>
        ) : (
          <>
            {leftIcon && (
              <span className="shrink-0 flex items-center">{leftIcon}</span>
            )}
            {children}
            {rightIcon && (
              <span className="shrink-0 flex items-center">{rightIcon}</span>
            )}
          </>
        )}
      </motion.button>
    )
  },
)

Button.displayName = 'Button'

export default Button
