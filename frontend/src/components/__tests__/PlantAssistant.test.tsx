import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PlantAssistant } from '../PlantAssistant';

describe('PlantAssistant Component', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
    // Mock scrollIntoView
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it('renders the floating action button (FAB)', () => {
    render(<PlantAssistant detectedPredictions={null} />);
    expect(screen.getByRole('button', { name: 'Open Plant Assistant' })).toBeInTheDocument();
  });

  it('opens and closes the chat panel on FAB click', async () => {
    render(<PlantAssistant detectedPredictions={null} />);
    const fab = screen.getByRole('button', { name: 'Open Plant Assistant' });

    // Open chat
    fireEvent.click(fab);
    expect(screen.getByText('Plant Assistant')).toBeInTheDocument();
    expect(screen.getByText('Analyze a leaf first for context')).toBeInTheDocument();

    // Close chat
    fireEvent.click(screen.getByRole('button', { name: 'Close Plant Assistant' }));
    expect(screen.queryByText('Plant Assistant')).not.toBeInTheDocument();
  });

  it('displays empty state if no disease was analyzed', () => {
    render(<PlantAssistant detectedPredictions={null} />);
    fireEvent.click(screen.getByRole('button', { name: 'Open Plant Assistant' }));
    expect(screen.getByText(/Analyze a leaf first, then I'll help you/)).toBeInTheDocument();
  });

  it('shows disease context and suggestion chips if disease is detected', () => {
    const predictions = [{ label: 'Tomato___Early_blight', confidence: 0.88 }];
    render(<PlantAssistant detectedPredictions={predictions} />);
    fireEvent.click(screen.getByRole('button', { name: 'Open Plant Assistant' }));

    expect(screen.getByText('Advising on')).toBeInTheDocument();
    expect(screen.getByText('Tomato — Early Blight')).toBeInTheDocument();
    
    // Check suggestions rendering
    expect(screen.getByText('How do I treat Tomato — Early Blight?')).toBeInTheDocument();
    expect(screen.getByText('Can Tomato — Early Blight spread to other plants?')).toBeInTheDocument();
  });

  it('submits suggestions or custom messages and updates state on success', async () => {
    const predictions = [{ label: 'Tomato___Early_blight', confidence: 0.88 }];
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ reply: 'This is the mock AI response.' }),
    } as any);

    render(<PlantAssistant detectedPredictions={predictions} />);
    fireEvent.click(screen.getByRole('button', { name: 'Open Plant Assistant' }));

    const chip = screen.getByText('How do I treat Tomato — Early Blight?');
    
    await act(async () => {
      fireEvent.click(chip);
    });

    expect(screen.getByText('How do I treat Tomato — Early Blight?')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('This is the mock AI response.')).toBeInTheDocument();
    });

    // Clear chat
    const clearBtn = screen.getByRole('button', { name: 'Clear chat' });
    fireEvent.click(clearBtn);
    expect(screen.queryByText('This is the mock AI response.')).not.toBeInTheDocument();
  });

  it('handles chat service network failure gracefully', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      json: async () => ({ detail: 'API quota exceeded.' }),
    } as any);

    render(<PlantAssistant detectedPredictions={null} />);
    fireEvent.click(screen.getByRole('button', { name: 'Open Plant Assistant' }));

    const input = screen.getByPlaceholderText('Ask about treatment, prevention…');
    const sendBtn = screen.getByRole('button', { name: 'Send' });

    await act(async () => {
      fireEvent.change(input, { target: { value: 'Help me' } });
      fireEvent.click(sendBtn);
    });

    await waitFor(() => {
      expect(screen.getByText('API quota exceeded.')).toBeInTheDocument();
    });
  });
});
