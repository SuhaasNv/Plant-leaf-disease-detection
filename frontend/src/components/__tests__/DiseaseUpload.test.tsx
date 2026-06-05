import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { DiseaseUpload } from '../DiseaseUpload';

// Mock canvas creation and toBlob for camera capture
const mockToBlob = vi.fn((callback) => {
  callback(new Blob(['test-image'], { type: 'image/jpeg' }));
});

describe('DiseaseUpload Component', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
    vi.useFakeTimers();
    // Setup document element context mocking for camera
    const mockCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn(() => ({
        drawImage: vi.fn(),
      })),
      toBlob: mockToBlob,
    };
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tagName) => {
      if (tagName === 'canvas') {
        return mockCanvas as unknown as HTMLCanvasElement;
      }
      return originalCreateElement(tagName);
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('renders standard state and supports opening camera', async () => {
    render(<DiseaseUpload />);
    
    expect(screen.getByText('Drag & drop or click to upload')).toBeInTheDocument();
    expect(screen.getByText('Open Camera')).toBeInTheDocument();
  });

  it('handles image file uploads and clears properly', async () => {
    const { container } = render(<DiseaseUpload />);
    const file = new File(['dummy content'], 'test-leaf.png', { type: 'image/png' });
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } });
      await vi.advanceTimersByTimeAsync(100);
    });

    expect(screen.getByText('test-leaf.png')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Clear' })).toBeInTheDocument();

    // Clear file selection
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
    expect(screen.queryByText('test-leaf.png')).not.toBeInTheDocument();
  });

  it('shows error message if non-image file is uploaded', async () => {
    const { container } = render(<DiseaseUpload />);
    const file = new File(['text'], 'test-leaf.txt', { type: 'text/plain' });
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } });
      await vi.advanceTimersByTimeAsync(100);
    });

    expect(screen.getByText('Please select an image file (PNG, JPG, JPEG).')).toBeInTheDocument();
  });

  it('runs parallel prediction flow and updates progress', async () => {
    const mockResponse = {
      predictions: [
        { label: 'Tomato___Early_blight', confidence: 0.94, treatment: ['Remove leaves'], prevention: ['Keep dry'] },
        { label: 'Tomato___healthy', confidence: 0.05, treatment: [], prevention: [] }
      ]
    };
    
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    } as unknown as Response);

    const onDiseaseMock = vi.fn();
    const { container } = render(<DiseaseUpload onDisease={onDiseaseMock} />);
    const file = new File(['dummy content'], 'test-leaf.png', { type: 'image/png' });
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } });
      await vi.advanceTimersByTimeAsync(100);
    });

    const analyzeButton = screen.getByRole('button', { name: 'Analyze' });
    
    await act(async () => {
      fireEvent.click(analyzeButton);
    });

    // Verify progress text shows scanning/analyzing
    expect(screen.getByText('Scanning leaf…')).toBeInTheDocument();

    // Run all timers to completion (animation + reveal delay)
    await act(async () => {
      vi.runAllTimers();
    });

    expect(screen.getAllByText('Tomato — Early Blight')[0]).toBeInTheDocument();
    expect(screen.getByText('94%')).toBeInTheDocument();
    expect(screen.getByText('How to Treat:')).toBeInTheDocument();
    expect(onDiseaseMock).toHaveBeenCalledWith(mockResponse.predictions);
  });

  it('handles api errors gracefully during analysis', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: 'No plant leaf detected. Please upload a clear photo of a leaf.' }),
    } as unknown as Response);

    const { container } = render(<DiseaseUpload />);
    const file = new File(['dummy content'], 'test-leaf.png', { type: 'image/png' });
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } });
      await vi.advanceTimersByTimeAsync(100);
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Analyze' }));
    });

    // Run all timers to completion
    await act(async () => {
      vi.runAllTimers();
    });

    expect(screen.getByText('No plant leaf detected. Please upload a clear photo of a leaf.')).toBeInTheDocument();
  });

  it('handles camera stream capture flow and cancels properly', async () => {
    render(<DiseaseUpload />);
    
    const openCameraButton = screen.getByText('Open Camera');
    await act(async () => {
      fireEvent.click(openCameraButton);
    });

    // Camera view active - wait for react updates
    await act(async () => {
      vi.advanceTimersByTime(100);
    });

    expect(screen.getByRole('button', { name: 'Take Photo' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();

    const cancelButton = screen.getByRole('button', { name: 'Cancel' });
    await act(async () => {
      fireEvent.click(cancelButton);
    });

    expect(screen.getByText('Drag & drop or click to upload')).toBeInTheDocument();
  });
});
