"use client";

import { useState, useRef, useLayoutEffect, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Message = {
    role: "user" | "assistant";
    text: string;
};

interface PlantAssistantProps {
    detectedDisease: string | null;
}

// Practical pre-prompts that make sense for any plant disease
const SUGGESTIONS = [
    "How serious is this disease?",
    "What should I treat it with?",
    "How do I stop it spreading?",
    "Will it affect my harvest?",
];

const panelVariants = {
    hidden: { opacity: 0, scale: 0.88, y: 12 },
    visible: {
        opacity: 1, scale: 1, y: 0,
        transition: { type: "spring" as const, stiffness: 380, damping: 28, mass: 0.8 },
    },
    exit: {
        opacity: 0, scale: 0.88, y: 12,
        transition: { duration: 0.18, ease: "easeIn" as const },
    },
};

const bubbleVariants = {
    hidden: { opacity: 0, y: 8 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.22, ease: "easeOut" as const } },
};

const chipVariants = {
    hidden: { opacity: 0, y: 6 },
    visible: (i: number) => ({
        opacity: 1, y: 0,
        transition: { duration: 0.2, delay: i * 0.06, ease: "easeOut" as const },
    }),
    exit: { opacity: 0, y: -4, transition: { duration: 0.15 } },
};

export function PlantAssistant({ detectedDisease }: PlantAssistantProps) {
    const [open, setOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);

    const bottomRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    useLayoutEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, loading]);

    useEffect(() => {
        if (open) setTimeout(() => inputRef.current?.focus(), 50);
    }, [open]);

    const sendText = async (text: string) => {
        if (!text.trim() || loading) return;

        const disease = detectedDisease ?? "Unknown disease";
        setMessages((prev) => [...prev, { role: "user", text }]);
        setInput("");
        setLoading(true);

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ disease, message: text }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail ?? "Something went wrong.");
            setMessages((prev) => [...prev, { role: "assistant", text: data.reply }]);
        } catch (e) {
            setMessages((prev) => [
                ...prev,
                {
                    role: "assistant",
                    text: e instanceof Error ? e.message : "Sorry, couldn't get a response. Try again.",
                },
            ]);
        } finally {
            setLoading(false);
        }
    };

    const handleSend = () => sendText(input.trim());

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
    };

    const handleClear = () => {
        setMessages([]);
        setInput("");
    };

    // Show chips when disease is detected and there are no messages yet
    const showChips = !!detectedDisease && messages.length === 0 && !loading;
    const hasUnread = !open && messages.length > 0;

    return (
        <div className="pa-root">

            {/* ── Chat Panel ──────────────────────────────────────────────────────── */}
            <AnimatePresence>
                {open && (
                    <motion.div
                        id="plant-assistant-panel"
                        variants={panelVariants}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        className="pa-panel"
                    >
                        {/* Header */}
                        <div className="pa-header">
                            <span className="pa-header-icon">🌿</span>
                            <div className="pa-header-text">
                                <p className="pa-title">Plant Assistant</p>
                                {detectedDisease ? (
                                    <p className="pa-subtitle">
                                        Advising on{" "}
                                        <span className="pa-disease-tag">{formatLabel(detectedDisease)}</span>
                                    </p>
                                ) : (
                                    <p className="pa-subtitle">Analyze a leaf first for context</p>
                                )}
                            </div>

                            {/* Clear button — only shown when there are messages */}
                            {messages.length > 0 && (
                                <button
                                    id="plant-assistant-clear"
                                    onClick={handleClear}
                                    aria-label="Clear chat"
                                    className="pa-clear-btn"
                                    title="Clear chat"
                                >
                                    {/* Trash icon */}
                                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                                        stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <polyline points="3 6 5 6 21 6" />
                                        <path d="M19 6l-1 14H6L5 6" />
                                        <path d="M10 11v6M14 11v6" />
                                        <path d="M9 6V4h6v2" />
                                    </svg>
                                </button>
                            )}
                        </div>

                        {/* Messages */}
                        <div id="plant-assistant-messages" className="pa-messages">
                            {/* Empty state — only when no disease detected */}
                            {messages.length === 0 && !detectedDisease && (
                                <div className="pa-empty">
                                    <span className="pa-empty-icon">🍃</span>
                                    <p>Analyze a leaf first, then I&apos;ll help you understand the disease.</p>
                                </div>
                            )}

                            {/* Suggestion chips */}
                            <AnimatePresence>
                                {showChips && (
                                    <motion.div
                                        key="chips"
                                        initial="hidden"
                                        animate="visible"
                                        exit="exit"
                                        className="pa-chips-wrap"
                                    >
                                        <p className="pa-chips-label">Try asking:</p>
                                        <div className="pa-chips">
                                            {SUGGESTIONS.map((s, i) => (
                                                <motion.button
                                                    key={s}
                                                    custom={i}
                                                    variants={chipVariants}
                                                    initial="hidden"
                                                    animate="visible"
                                                    exit="exit"
                                                    className="pa-chip"
                                                    onClick={() => sendText(s)}
                                                    disabled={loading}
                                                >
                                                    {s}
                                                </motion.button>
                                            ))}
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            {/* Message bubbles */}
                            <AnimatePresence initial={false}>
                                {messages.map((msg, i) => (
                                    <motion.div
                                        key={i}
                                        variants={bubbleVariants}
                                        initial="hidden"
                                        animate="visible"
                                        className={`pa-row ${msg.role === "user" ? "pa-row-user" : "pa-row-assistant"}`}
                                    >
                                        <div className={`pa-bubble ${msg.role === "user" ? "pa-bubble-user" : "pa-bubble-assistant"}`}>
                                            {msg.text}
                                        </div>
                                    </motion.div>
                                ))}
                            </AnimatePresence>

                            {/* Typing indicator */}
                            <AnimatePresence>
                                {loading && (
                                    <motion.div
                                        key="typing"
                                        variants={bubbleVariants}
                                        initial="hidden"
                                        animate="visible"
                                        exit={{ opacity: 0, transition: { duration: 0.1 } }}
                                        className="pa-row pa-row-assistant"
                                    >
                                        <div className="pa-bubble pa-bubble-assistant pa-typing">
                                            <span /><span /><span />
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            <div ref={bottomRef} />
                        </div>

                        {/* Input */}
                        <div className="pa-input-row">
                            <input
                                ref={inputRef}
                                id="plant-assistant-input"
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Ask about treatment, prevention…"
                                disabled={loading}
                                className="pa-input"
                                aria-label="Type your question"
                            />
                            <button
                                id="plant-assistant-send"
                                onClick={handleSend}
                                disabled={!input.trim() || loading}
                                aria-label="Send"
                                className="pa-send"
                            >
                                <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
                                    stroke="currentColor" strokeWidth="2.2"
                                    strokeLinecap="round" strokeLinejoin="round">
                                    <line x1="22" y1="2" x2="11" y2="13" />
                                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                                </svg>
                            </button>
                        </div>

                        {/* Disclaimer */}
                        <div className="pa-disclaimer" style={{ padding: "0 16px 12px", textAlign: "center", fontSize: "10px", color: "#9ca3af", lineHeight: "1.4" }}>
                            Suggestions generated by Gemini 2.5 Flash. Please consult a real agriculture expert or plant doctor for professional advice.
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── FAB ─────────────────────────────────────────────────────────────── */}
            <motion.button
                id="plant-assistant-fab"
                aria-label={open ? "Close Plant Assistant" : "Open Plant Assistant"}
                onClick={() => setOpen((v) => !v)}
                className="pa-fab"
                whileHover={{ scale: 1.07 }}
                whileTap={{ scale: 0.93 }}
            >
                <AnimatePresence mode="wait" initial={false}>
                    {open ? (
                        <motion.span key="close"
                            initial={{ rotate: -45, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }}
                            exit={{ rotate: 45, opacity: 0 }} transition={{ duration: 0.18 }}
                            style={{ display: "flex" }}
                        >
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                                stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                                <line x1="18" y1="6" x2="6" y2="18" />
                                <line x1="6" y1="6" x2="18" y2="18" />
                            </svg>
                        </motion.span>
                    ) : (
                        <motion.span key="chat"
                            initial={{ rotate: 45, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }}
                            exit={{ rotate: -45, opacity: 0 }} transition={{ duration: 0.18 }}
                            style={{ display: "flex" }}
                        >
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                                stroke="currentColor" strokeWidth="2.2"
                                strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                            </svg>
                        </motion.span>
                    )}
                </AnimatePresence>
                {hasUnread && <span className="pa-unread" aria-hidden />}
            </motion.button>
        </div>
    );
}

function formatLabel(raw: string): string {
    return raw
        .replace(/___/g, " — ")
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase());
}
