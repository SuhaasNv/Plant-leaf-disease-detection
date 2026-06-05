import { render, screen, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { HeroSection } from '../HeroSection';

// Mock TextScramble component
vi.mock('@/components/ui/text-scramble', () => ({
  TextScramble: ({ children, trigger }: { children: React.ReactNode; trigger?: boolean }) => (
    <span data-testid="mock-scramble" data-trigger={trigger ? 'true' : 'false'}>
      {children}
    </span>
  ),
}));

// Mock RevealWaveImage component
vi.mock('@/components/ui/reveal-wave-image', () => ({
  RevealWaveImage: ({ className }: { className?: string }) => (
    <div data-testid="mock-reveal-wave-image" className={className} />
  ),
}));

// Mock next/link
vi.mock('next/link', () => ({
  default: ({ children, href, className, onMouseEnter, onMouseLeave }: { children: React.ReactNode; href: string; className?: string; onMouseEnter?: () => void; onMouseLeave?: () => void }) => (
    <a
      href={href}
      className={className}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      {children}
    </a>
  ),
}));

describe('HeroSection Component', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('renders title, subtitle, and action buttons', () => {
    render(<HeroSection />);
    
    expect(screen.getByText('Learn More')).toBeInTheDocument();
    expect(screen.getByText('Analyze a Leaf →')).toBeInTheDocument();
    expect(screen.getByText(/Upload a leaf image and get an accurate AI diagnosis/)).toBeInTheDocument();
  });

  it('triggers text scrambles after specific intervals', () => {
    render(<HeroSection />);

    const scrambles = screen.getAllByTestId('mock-scramble');
    expect(scrambles[0]).toHaveAttribute('data-trigger', 'false');

    act(() => {
      vi.advanceTimersByTime(850);
    });
    expect(scrambles[0]).toHaveAttribute('data-trigger', 'true');

    act(() => {
      vi.advanceTimersByTime(150);
    });
    expect(scrambles[1]).toHaveAttribute('data-trigger', 'true');

    act(() => {
      vi.advanceTimersByTime(200);
    });
    expect(scrambles[2]).toHaveAttribute('data-trigger', 'true');
  });
});
