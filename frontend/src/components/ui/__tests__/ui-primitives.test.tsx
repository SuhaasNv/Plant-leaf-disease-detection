import { render, screen, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { LampContainer } from '../lamp';
import { TextScramble } from '../text-scramble';
import { NavBar } from '../tubelight-navbar';
import { RevealWaveImage } from '../reveal-wave-image';
import { Home } from 'lucide-react';


describe('UI Primitives Components', () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  describe('LampContainer', () => {
    it('renders lamp layout and child text', () => {
      render(
        <LampContainer>
          <div data-testid="lamp-child">Testing Lamp</div>
        </LampContainer>
      );
      expect(screen.getByTestId('lamp-child')).toBeInTheDocument();
      expect(screen.getByText('Testing Lamp')).toBeInTheDocument();
    });
  });

  describe('NavBar', () => {
    const items = [
      { name: 'Home', url: '/', icon: Home },
      { name: 'Detect', url: '/disease', icon: Home },
    ];

    it('renders the navbar items correctly', () => {
      render(<NavBar items={items} />);
      expect(screen.getByText('Home')).toBeInTheDocument();
      expect(screen.getByText('Detect')).toBeInTheDocument();
    });
  });

  describe('RevealWaveImage', () => {
    it('renders without errors', async () => {
      const { container } = render(
        <RevealWaveImage 
          src="/test.jpg" 
          className="test-image"
          waveSpeed={0.5}
          waveFrequency={1.0}
          waveAmplitude={0.5}
        />
      );
      
      // Wait for window.Image onload mock to fire in next tick and process state change
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 10));
      });
      
      // Canvas with mock-three-canvas should now be rendered
      expect(container.querySelector('.test-image')).toBeInTheDocument();
      expect(screen.getByTestId('mock-three-canvas')).toBeInTheDocument();
    });
  });

  describe('TextScramble', () => {
    it('renders children initially', () => {
      render(<TextScramble>Test Text</TextScramble>);
      expect(screen.getByText('Test Text')).toBeInTheDocument();
    });

    const triggerSpeeds = [
      { speed: 0.01, duration: 0.1 },
      { speed: 0.05, duration: 0.2 },
      { speed: 0.1, duration: 0.3 },
      { speed: 0.2, duration: 0.4 },
      { speed: 0.5, duration: 0.5 },
    ];

    triggerSpeeds.forEach(({ speed, duration }) => {
      it(`handles text scrambling with speed ${speed} and duration ${duration}`, async () => {
        vi.useFakeTimers();
        const completeMock = vi.fn();
        
        const { rerender } = render(
          <TextScramble 
            trigger={false} 
            speed={speed} 
            duration={duration}
            onScrambleComplete={completeMock}
          >
            ScrambleMe
          </TextScramble>
        );

        // Toggle trigger to active
        rerender(
          <TextScramble 
            trigger={true} 
            speed={speed} 
            duration={duration}
            onScrambleComplete={completeMock}
          >
            ScrambleMe
          </TextScramble>
        );

        // Advance timers to trigger completion
        await act(async () => {
          vi.advanceTimersByTime(10000);
        });

        expect(completeMock).toHaveBeenCalled();
        vi.useRealTimers();
      });
    });
  });
});
