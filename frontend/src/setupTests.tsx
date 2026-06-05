import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock ResizeObserver
class MockResizeObserver {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}
global.ResizeObserver = MockResizeObserver;

// Mock FileReader readAsDataURL to execute synchronously
if (typeof window !== 'undefined' && window.FileReader) {
  window.FileReader.prototype.readAsDataURL = function(file: any) {
    Object.defineProperty(this, 'result', {
      value: 'data:image/png;base64,mock-base64-content',
      configurable: true
    });
    if (this.onloadend) {
      this.onloadend();
    }
  };
}

// Mock IntersectionObserver
class MockIntersectionObserver {
  readonly root: Element | Document | null = null;
  readonly rootMargin: string = '';
  readonly thresholds: ReadonlyArray<number> = [];
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
  takeRecords = vi.fn(() => []);
}
global.IntersectionObserver = MockIntersectionObserver as any;

// Mock Canvas getContext
if (typeof window !== 'undefined') {
  HTMLCanvasElement.prototype.getContext = vi.fn(() => {
    return {
      fillRect: vi.fn(),
      clearRect: vi.fn(),
      getImageData: vi.fn(() => ({ data: new Uint8ClampedArray() })),
      putImageData: vi.fn(),
      createImageData: vi.fn(),
      setTransform: vi.fn(),
      drawImage: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      scale: vi.fn(),
      rotate: vi.fn(),
      translate: vi.fn(),
      transform: vi.fn(),
      beginPath: vi.fn(),
      closePath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      rect: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
      clip: vi.fn(),
      measureText: vi.fn(() => ({ width: 0 })),
    } as any;
  }) as any;

  // Mock URL object methods
  window.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
  window.URL.revokeObjectURL = vi.fn();
}

// Mock MediaDevices camera
Object.defineProperty(global.navigator, 'mediaDevices', {
  value: {
    getUserMedia: vi.fn().mockResolvedValue({
      getTracks: () => [
        {
          stop: vi.fn(),
        },
      ],
    }),
  },
  writable: true,
});

// Mock Image onload loading for RevealWaveImage aspect ratio calculation
if (typeof window !== 'undefined') {
  Object.defineProperty(window.Image.prototype, 'src', {
    set(src) {
      this._src = src;
      setTimeout(() => {
        if (this.onload) {
          this.onload();
        }
      }, 0);
    },
    get() {
      return this._src;
    },
    configurable: true
  });
  Object.defineProperty(window.Image.prototype, 'naturalWidth', {
    get() {
      return 1920;
    },
    configurable: true
  });
  Object.defineProperty(window.Image.prototype, 'naturalHeight', {
    get() {
      return 1080;
    },
    configurable: true
  });
}

// Mock Framer Motion
vi.mock('framer-motion', async (importOriginal) => {
  const actual = await importOriginal<typeof import('framer-motion')>();
  return {
    ...actual,
    motion: {
      create: (Component: any) => {
        return ({ children, ...props }: any) => {
          const Tag = Component || 'div';
          return <Tag {...props}>{children}</Tag>;
        };
      },
      div: ({ children, ...props }: any) => {
        const validProps = Object.keys(props)
          .filter(key => !['animate', 'initial', 'transition', 'exit', 'variants', 'whileHover', 'whileTap', 'viewport'].includes(key))
          .reduce((obj: any, key) => {
            obj[key] = props[key];
            return obj;
          }, {});
        return <div {...validProps}>{children}</div>;
      },
      p: ({ children, ...props }: any) => <p {...props}>{children}</p>,
      span: ({ children, ...props }: any) => <span {...props}>{children}</span>,
      h1: ({ children, ...props }: any) => <h1 {...props}>{children}</h1>,
      h2: ({ children, ...props }: any) => <h2 {...props}>{children}</h2>,
      h3: ({ children, ...props }: any) => <h3 {...props}>{children}</h3>,
      button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
      nav: ({ children, ...props }: any) => <nav {...props}>{children}</nav>,
    },
    AnimatePresence: ({ children }: any) => <>{children}</>,
  };
});

// Mock react-three-fiber and react-three-drei
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => <div data-testid="mock-three-canvas">{children}</div>,
  useFrame: vi.fn(),
  useThree: vi.fn(() => ({ size: { width: 100, height: 100 } })),
}));

vi.mock('@react-three/drei', () => ({
  useGLTF: vi.fn(),
  useTexture: vi.fn(() => ({})),
  OrbitControls: () => <div data-testid="mock-orbit-controls" />,
  PerspectiveCamera: () => <div data-testid="mock-perspective-camera" />,
}));
