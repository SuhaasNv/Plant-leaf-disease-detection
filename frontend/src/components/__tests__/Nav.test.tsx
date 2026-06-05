import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Nav } from '../Nav';
import { usePathname } from 'next/navigation';

// Mock next/navigation pathname
vi.mock('next/navigation', () => ({
  usePathname: vi.fn(),
}));

// Mock next/link to render standard anchors
vi.mock('next/link', () => ({
  default: ({ children, href, className, onClick }: { children: React.ReactNode; href: string; className?: string; onClick?: () => void }) => (
    <a href={href} className={className} onClick={onClick}>
      {children}
    </a>
  ),
}));

describe('Nav Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const pathnames = ['/', '/disease-recognition', '/about'];

  pathnames.forEach((path) => {
    it(`renders logo and navigation items for path: ${path}`, () => {
      vi.mocked(usePathname).mockReturnValue(path);
      render(<Nav />);
      expect(screen.getByText('LeafScan AI')).toBeInTheDocument();
      expect(screen.getAllByText('Home')).toHaveLength(1);
      expect(screen.getAllByText('Detect')).toHaveLength(1);
      expect(screen.getAllByText('About')).toHaveLength(1);
    });
  });

  it('toggles mobile menu on hamburger click', () => {
    vi.mocked(usePathname).mockReturnValue('/');
    render(<Nav />);

    // Initially mobile dropdown shouldn't be visible since mobileMenuOpen is false
    expect(screen.queryByRole('navigation', { name: '' })).not.toContainElement(screen.queryByText('Try It Free'));

    const toggleButton = screen.getByRole('button', { name: 'Open menu' });
    fireEvent.click(toggleButton);

    // After clicking, button aria-label updates
    expect(screen.getByRole('button', { name: 'Close menu' })).toBeInTheDocument();

    // Close menu again
    fireEvent.click(screen.getByRole('button', { name: 'Close menu' }));
    expect(screen.getByRole('button', { name: 'Open menu' })).toBeInTheDocument();
  });

  it('updates scrolled state on window scroll', () => {
    vi.mocked(usePathname).mockReturnValue('/');
    render(<Nav />);

    // Simulate scroll down
    window.scrollY = 20;
    fireEvent.scroll(window);
    
    // Simulate scroll up
    window.scrollY = 0;
    fireEvent.scroll(window);
  });
});
